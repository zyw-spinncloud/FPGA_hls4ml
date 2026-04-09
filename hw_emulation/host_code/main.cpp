#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"


// Kernel interface sizes
static constexpr int INPUT_ELEMS = 16;          // 16 x ap_fixed<16,6>
static constexpr int INPUT_WORD_BYTES = 32;     // 256 bits

static constexpr int OUTPUT_ELEMS = 64;         // 64 x ap_fixed<34,14>
static constexpr int OUTPUT_WORDS = 9;          // ceil(64*34 / 256) = 9
static constexpr int OUTPUT_WORD_BYTES = OUTPUT_WORDS * 32; // 288 bytes

static constexpr int PARAM_ELEMS = 240;         // 240 x ap_fixed<16,6>
static constexpr int PARAM_BYTES = PARAM_ELEMS * 2;

// ap_fixed<16,6> => 10 fractional bits
static constexpr int IN_FRAC_BITS = 10;
static constexpr int PARAM_FRAC_BITS = 10;

// ap_fixed<34,14> => 20 fractional bits
static constexpr int OUT_FRAC_BITS = 20;


// Fixed-point helpers
static int16_t float_to_apfixed16_6(float x)
{
  // ap_fixed<16,6> => signed, 10 fractional bits
  float scaled = x * static_cast<float>(1 << IN_FRAC_BITS);
  int32_t q = static_cast<int32_t>(std::lround(scaled));

  // Saturate to int16_t range
  if (q > 32767) q = 32767;
  if (q < -32768) q = -32768;

  return static_cast<int16_t>(q);
}

static double apfixed34_14_to_double(int64_t raw_signed_34)
{
  return static_cast<double>(raw_signed_34) / static_cast<double>(1 << OUT_FRAC_BITS);
}


// Bit packing helpers
static void pack_input_16x16_to_256(
    const std::array<int16_t, INPUT_ELEMS>& in_vals,
    std::array<uint8_t, INPUT_WORD_BYTES>& in_bytes)
{
  in_bytes.fill(0);

  // Kernel expects:
  // bits [15:0]   = input[0]
  // bits [31:16]  = input[1]
  // ...
  // bits [255:240]= input[15]
  //
  // Pack little-endian per 16-bit lane.
  for (int i = 0; i < INPUT_ELEMS; ++i) {
    uint16_t raw = static_cast<uint16_t>(in_vals[i]);
    in_bytes[2 * i + 0] = static_cast<uint8_t>(raw & 0xFF);
    in_bytes[2 * i + 1] = static_cast<uint8_t>((raw >> 8) & 0xFF);
  }
}

static void pack_params_240x16(
    const std::array<int16_t, PARAM_ELEMS>& params,
    std::vector<uint8_t>& param_bytes)
{
  param_bytes.resize(PARAM_BYTES);

  for (int i = 0; i < PARAM_ELEMS; ++i) {
    uint16_t raw = static_cast<uint16_t>(params[i]);
    param_bytes[2 * i + 0] = static_cast<uint8_t>(raw & 0xFF);
    param_bytes[2 * i + 1] = static_cast<uint8_t>((raw >> 8) & 0xFF);
  }
}

// Read one bit from packed output byte vector
static uint8_t get_bit(const std::array<uint8_t, OUTPUT_WORD_BYTES>& buf, int bit_index)
{
  int byte_index = bit_index / 8;
  int bit_in_byte = bit_index % 8;
  return (buf[byte_index] >> bit_in_byte) & 0x1;
}

// Extract signed 34-bit field from packed output
static int64_t extract_signed_34(
    const std::array<uint8_t, OUTPUT_WORD_BYTES>& out_bytes,
    int elem_index)
{
  const int start_bit = elem_index * 34;
  uint64_t raw = 0;

  for (int b = 0; b < 34; ++b) {
    raw |= (static_cast<uint64_t>(get_bit(out_bytes, start_bit + b)) << b);
  }

  // Sign-extend 34-bit signed value to int64_t
  if (raw & (1ULL << 33)) {
    raw |= (~0ULL << 34);
  }

  return static_cast<int64_t>(raw);
}

static void decode_output(
    const std::array<uint8_t, OUTPUT_WORD_BYTES>& out_bytes,
    std::array<double, OUTPUT_ELEMS>& out_vals)
{
  for (int i = 0; i < OUTPUT_ELEMS; ++i) {
    int64_t raw34 = extract_signed_34(out_bytes, i);
    out_vals[i] = apfixed34_14_to_double(raw34);
  }
}

// load params from file
// One float per line, total 240 lines
static bool load_params_from_text(
    const std::string& path,
    std::array<int16_t, PARAM_ELEMS>& params_fixed)
{
  std::ifstream fin(path);
  if (!fin)
    return false;

  for (int i = 0; i < PARAM_ELEMS; ++i) {
    float v = 0.0f;
    if (!(fin >> v))
      return false;
    params_fixed[i] = float_to_apfixed16_6(v);
  }

  return true;
}

int main(int argc, char** argv)
{
  try {
    std::string xclbin_path = (argc > 1) ? argv[1] : "binary_container_1.xclbin";

    std::cout << "Using xclbin: " << xclbin_path << "\n";

    // Example input data
    std::array<float, INPUT_ELEMS> input_float = {
      0.10f, 0.20f, 0.30f, 0.40f,
      0.50f, 0.60f, 0.70f, 0.80f,
      0.90f, 1.00f, 1.10f, 1.20f,
      1.30f, 1.40f, 1.50f, 1.60f
    };

    std::array<int16_t, INPUT_ELEMS> input_fixed{};
    for (int i = 0; i < INPUT_ELEMS; ++i) {
      input_fixed[i] = float_to_apfixed16_6(input_float[i]);
    }

    // Params
    // Order must match kernel expectation:
    //   W6[128], B6[64], W7[32], B7[16]
    std::array<int16_t, PARAM_ELEMS> params_fixed{};
    params_fixed.fill(0);

    // Optional: load from text file.
    // One float per line, 240 lines total
    //(void)load_params_from_text(".txt", params_fixed);

    // Pack host buffers
    std::array<uint8_t, INPUT_WORD_BYTES> in_bytes{};
    pack_input_16x16_to_256(input_fixed, in_bytes);

    std::vector<uint8_t> param_bytes;
    pack_params_240x16(params_fixed, param_bytes);

    std::array<uint8_t, OUTPUT_WORD_BYTES> out_bytes{};
    out_bytes.fill(0);


    // Open device and load xclbin
    xrt::device device{0};
    auto uuid = device.load_xclbin(xclbin_path);


    xrt::kernel krnl{device, uuid, "wrapper"};

    // group_id(0) -> in
    // group_id(1) -> out
    // group_id(2) -> params
    xrt::bo bo_in(device, INPUT_WORD_BYTES, krnl.group_id(0));
    xrt::bo bo_out(device, OUTPUT_WORD_BYTES, krnl.group_id(1));
    xrt::bo bo_params(device, PARAM_BYTES, krnl.group_id(2));

    // Write host data to device buffers
    bo_in.write(in_bytes.data(), INPUT_WORD_BYTES);
    bo_params.write(param_bytes.data(), PARAM_BYTES);

    bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_params.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Launch kernel
    auto run = krnl(bo_in, bo_out, bo_params);
    run.wait();


    // Read back output
    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_out.read(out_bytes.data(), OUTPUT_WORD_BYTES);


    // Decode and print output
    std::array<double, OUTPUT_ELEMS> out_vals{};
    decode_output(out_bytes, out_vals);

    std::cout << "Kernel completed.\n";
    std::cout << "Decoded outputs:\n";
    for (int i = 0; i < OUTPUT_ELEMS; ++i) {
      std::cout << "out[" << i << "] = " << out_vals[i] << "\n";
    }

    return 0;
  }
  catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}