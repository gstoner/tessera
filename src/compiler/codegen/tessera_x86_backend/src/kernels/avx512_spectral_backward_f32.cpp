// Native AVX-512 compound-spectral adjoints.
//
// These kernels deliberately consume the dimensions and normalization chosen
// by the content-addressed Schedule -> Tile artifact.  They do not rediscover
// an FFT plan or re-enter Graph IR.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <vector>

extern "C" int tessera_x86_fft_c2r_packed_f32(
    const char *digest, const float *input, float *output, int64_t batch,
    int64_t n);
extern "C" int tessera_x86_fft_r2c_packed_f32(
    const char *digest, const float *input, float *output, int64_t batch,
    int64_t n);

extern "C" void tessera_x86_avx512_spectral_filter_bwd_c64(
    const float *dy, const float *input, const float *filter, float *dx,
    float *dfilter, int64_t elements) {
  for (int64_t i = 0; i < elements; ++i) {
    const float dyr = dy[2 * i], dyi = dy[2 * i + 1];
    const float xr = input[2 * i], xi = input[2 * i + 1];
    const float fr = filter[2 * i], fi = filter[2 * i + 1];
    // Wirtinger VJP for complex multiplication: dy * conj(other).
    dx[2 * i] = dyr * fr + dyi * fi;
    dx[2 * i + 1] = dyi * fr - dyr * fi;
    dfilter[2 * i] = dyr * xr + dyi * xi;
    dfilter[2 * i + 1] = dyi * xr - dyr * xi;
  }
}

extern "C" void tessera_x86_avx512_spectral_conv_bwd_f32(
    const float *dy, const float *input, const float *kernel, float *dx,
    float *dkernel, int64_t batch, int64_t outputLength,
    int64_t inputLength, int64_t kernelLength, float scale) {
  for (int64_t row = 0; row < batch; ++row) {
    const float *rowDy = dy + row * outputLength;
    const float *rowInput = input + row * inputLength;
    const float *rowKernel = kernel + row * kernelLength;
    float *rowDx = dx + row * inputLength;
    float *rowDkernel = dkernel + row * kernelLength;
    for (int64_t i = 0; i < inputLength; ++i) {
      float value = 0.0f;
      for (int64_t j = 0; j < kernelLength; ++j)
        value += rowDy[i + j] * rowKernel[j];
      rowDx[i] = value * scale;
    }
    for (int64_t j = 0; j < kernelLength; ++j) {
      float value = 0.0f;
      for (int64_t i = 0; i < inputLength; ++i)
        value += rowDy[i + j] * rowInput[i];
      rowDkernel[j] = value * scale;
    }
  }
}

namespace {

bool checked_product(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs <= 0 || rhs <= 0 ||
      lhs > std::numeric_limits<int64_t>::max() / rhs)
    return false;
  result = lhs * rhs;
  return static_cast<uint64_t>(result) <=
         std::numeric_limits<size_t>::max();
}

bool checked_framed_length(int64_t frames, int64_t hop, int64_t win,
                           int64_t &samples) {
  if (frames <= 0 || hop <= 0 || win <= 0 ||
      frames - 1 > (std::numeric_limits<int64_t>::max() - win) / hop)
    return false;
  samples = (frames - 1) * hop + win;
  return true;
}

}  // namespace

// Initial AD-TSOL-STFT-BWD-1 envelope: contiguous last-axis, uncentered,
// onesided f32-accumulation core, n_fft == window length. Low-precision real
// storage is converted by the bounded wrappers below. Interior bins carry
// weight 1/2 before the packed C2R because that transform reconstructs their
// Hermitian partners.  Multiplication by N after its normalized inverse then
// gives exactly sum_f Re(dy_f exp(+2*pi*i*f*n/N)): every STORED bin once.
extern "C" int tessera_x86_avx512_stft_bwd_f32(
    const char *digest, const float *dy, const float *input,
    const float *window, float *dx, float *dwindow, int64_t batch,
    int64_t samples, int64_t frames, int64_t win, int64_t hop,
    float forward_scale) {
  int64_t covered = 0, rows = 0, frame_elements = 0, stored_elements = 0;
  int64_t output_elements = 0;
  if (!digest || !dy || !input || !window || !dx || !dwindow || samples <= 0 ||
      !checked_framed_length(frames, hop, win, covered) || covered > samples ||
      !checked_product(batch, frames, rows) ||
      !checked_product(rows, win, frame_elements) ||
      !checked_product(batch, samples, output_elements))
    return 1;
  const int64_t bins = win / 2 + 1;
  if (!checked_product(rows, bins, stored_elements) ||
      stored_elements > std::numeric_limits<int64_t>::max() / 2)
    return 1;
  std::vector<float> frame_grad(static_cast<size_t>(frame_elements));
  if ((win & 1) == 0) {
    const int64_t corrected_elements = 2 * stored_elements;
    std::vector<float> corrected(static_cast<size_t>(corrected_elements));
    std::memcpy(corrected.data(), dy,
                static_cast<size_t>(corrected_elements) * sizeof(float));
    for (int64_t row = 0; row < rows; ++row) {
      // DC and Nyquist are self-conjugate real coordinates. Their stored
      // imaginary cotangents are outside the real-signal image and therefore
      // have zero pullback; the packed C2R ABI requires that explicitly.
      corrected[2 * row * bins + 1] = 0.0f;
      corrected[2 * (row * bins + bins - 1) + 1] = 0.0f;
      for (int64_t bin = 1; bin < bins - 1; ++bin) {
        corrected[2 * (row * bins + bin)] *= 0.5f;
        corrected[2 * (row * bins + bin) + 1] *= 0.5f;
      }
    }
    if (tessera_x86_fft_c2r_packed_f32(
            digest, corrected.data(), frame_grad.data(), rows, win))
      return 2;
    const float scale = static_cast<float>(win) * forward_scale;
    for (float &value : frame_grad) value *= scale;
  } else {
    // The packed-real ABI is even-length. Retain an exact ragged-odd tail
    // rather than silently padding and changing the transform definition.
    constexpr double kTwoPi = 6.283185307179586476925286766559;
    for (int64_t row = 0; row < rows; ++row)
      for (int64_t local = 0; local < win; ++local) {
        double value = 0.0;
        for (int64_t bin = 0; bin < bins; ++bin) {
          const double angle = kTwoPi * static_cast<double>(bin * local) /
                               static_cast<double>(win);
          const float real = dy[2 * (row * bins + bin)];
          const float imag = dy[2 * (row * bins + bin) + 1];
          value += static_cast<double>(real) * std::cos(angle) -
                   static_cast<double>(imag) * std::sin(angle);
        }
        frame_grad[row * win + local] =
            static_cast<float>(value * forward_scale);
      }
  }
  std::fill(dx, dx + output_elements, 0.0f);
  std::fill(dwindow, dwindow + win, 0.0f);
  for (int64_t row = 0; row < batch; ++row)
    for (int64_t frame = 0; frame < frames; ++frame)
      for (int64_t local = 0; local < win; ++local) {
        const int64_t sample = frame * hop + local;
        const float grad = frame_grad[(row * frames + frame) * win + local];
        dx[row * samples + sample] += grad * window[local];
        dwindow[local] += grad * input[row * samples + sample];
      }
  return 0;
}

namespace {

bool valid_backward_storage(int storage) { return storage >= 0 && storage <= 2; }

float load_backward_storage(const void *input, int64_t index, int storage) {
  if (storage == 0) return static_cast<const float *>(input)[index];
  const uint16_t bits = static_cast<const uint16_t *>(input)[index];
  if (storage == 1) return _cvtsh_ss(bits);
  uint32_t wide = uint32_t(bits) << 16;
  float value;
  std::memcpy(&value, &wide, sizeof(value));
  return value;
}

void store_backward_storage(void *output, int64_t index, int storage,
                            float value) {
  if (storage == 0) {
    static_cast<float *>(output)[index] = value;
  } else if (storage == 1) {
    static_cast<uint16_t *>(output)[index] = _cvtss_sh(
        value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  } else {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t rounding = 0x7fff + ((bits >> 16) & 1);
    static_cast<uint16_t *>(output)[index] =
        uint16_t((bits + rounding) >> 16);
  }
}

void unpack_backward_storage(const void *input, std::vector<float> &output,
                             int storage) {
  for (size_t i = 0; i < output.size(); ++i)
    output[i] = load_backward_storage(input, static_cast<int64_t>(i), storage);
}

void pack_backward_storage(const std::vector<float> &input, void *output,
                           int storage) {
  for (size_t i = 0; i < input.size(); ++i)
    store_backward_storage(output, static_cast<int64_t>(i), storage, input[i]);
}

size_t backward_storage_bytes(int storage) {
  return storage == 0 ? sizeof(float) : sizeof(uint16_t);
}

bool checked_backward_bytes(int64_t elements, size_t elementBytes,
                            size_t &bytes) {
  if (elements <= 0 || elementBytes == 0 ||
      static_cast<uint64_t>(elements) >
          std::numeric_limits<size_t>::max() / elementBytes)
    return false;
  bytes = static_cast<size_t>(elements) * elementBytes;
  return true;
}

void pack_backward_axis(const void *input, void *output, int64_t outer,
                        int64_t axis, int64_t inner, size_t bytes) {
  auto *source = static_cast<const unsigned char *>(input);
  auto *target = static_cast<unsigned char *>(output);
  for (int64_t o = 0; o < outer; ++o)
    for (int64_t i = 0; i < inner; ++i)
      for (int64_t a = 0; a < axis; ++a)
        std::memcpy(target + size_t((o * inner + i) * axis + a) * bytes,
                    source + size_t((o * axis + a) * inner + i) * bytes,
                    bytes);
}

void unpack_backward_axis(const void *input, void *output, int64_t outer,
                          int64_t axis, int64_t inner, size_t bytes) {
  pack_backward_axis(input, output, outer, inner, axis, bytes);
}

}  // namespace

extern "C" int tessera_x86_avx512_stft_bwd_storage(
    const char *digest, const float *dy, const void *input,
    const void *window, void *dx, void *dwindow, int64_t batch,
    int64_t samples, int64_t frames, int64_t win, int64_t hop, int storage,
    float forward_scale) {
  int64_t covered = 0, x_elements = 0;
  if (!valid_backward_storage(storage) || !digest || !dy || !input || !window ||
      !dx || !dwindow || samples <= 0 ||
      !checked_framed_length(frames, hop, win, covered) || covered > samples ||
      !checked_product(batch, samples, x_elements))
    return 10;
  if (storage == 0)
    return tessera_x86_avx512_stft_bwd_f32(
        digest, dy, static_cast<const float *>(input),
        static_cast<const float *>(window), static_cast<float *>(dx),
        static_cast<float *>(dwindow), batch, samples, frames, win, hop,
        forward_scale);
  std::vector<float> x(static_cast<size_t>(x_elements));
  std::vector<float> w(static_cast<size_t>(win));
  std::vector<float> x_grad(x.size());
  std::vector<float> w_grad(w.size());
  unpack_backward_storage(input, x, storage);
  unpack_backward_storage(window, w, storage);
  const int rc = tessera_x86_avx512_stft_bwd_f32(
      digest, dy, x.data(), w.data(), x_grad.data(), w_grad.data(), batch,
      samples, frames, win, hop, forward_scale);
  if (!rc) {
    pack_backward_storage(x_grad, dx, storage);
    pack_backward_storage(w_grad, dwindow, storage);
  }
  return rc;
}

// Exact transpose of normalized overlap-add, including the window-energy
// denominator.  The R2C result is weighted by the adjoint of IRFFT: DC and
// Nyquist once, interior bins twice.
extern "C" int tessera_x86_avx512_istft_bwd_f32(
    const char *digest, const float *dy, const float *spectrum,
    const float *window, float *dspectrum, float *dwindow, int64_t batch,
    int64_t frames, int64_t win, int64_t hop, float inverse_scale) {
  int64_t samples = 0, rows = 0, frame_elements = 0, output_elements = 0;
  if (!digest || !dy || !spectrum || !window || !dspectrum || !dwindow ||
      !checked_framed_length(frames, hop, win, samples) ||
      !checked_product(batch, frames, rows) ||
      !checked_product(rows, win, frame_elements) ||
      !checked_product(batch, samples, output_elements))
    return 1;
  const int64_t bins = win / 2 + 1;
  std::vector<float> frame_value(static_cast<size_t>(frame_elements));
  constexpr double kTwoPi = 6.283185307179586476925286766559;
  for (int64_t row = 0; row < rows; ++row)
    for (int64_t local = 0; local < win; ++local) {
      double value = spectrum[2 * row * bins];
      if (win % 2 == 0)
        value += spectrum[2 * (row * bins + bins - 1)] *
                 (local % 2 == 0 ? 1.0 : -1.0);
      const int64_t interior = bins - (win % 2 == 0 ? 1 : 0);
      for (int64_t bin = 1; bin < interior; ++bin) {
        const double angle = kTwoPi * static_cast<double>(bin * local) /
                             static_cast<double>(win);
        const float real = spectrum[2 * (row * bins + bin)];
        const float imag = spectrum[2 * (row * bins + bin) + 1];
        value += 2.0 * (static_cast<double>(real) * std::cos(angle) -
                        static_cast<double>(imag) * std::sin(angle));
      }
      frame_value[row * win + local] =
          static_cast<float>(value * inverse_scale);
    }
  std::vector<float> dframe(static_cast<size_t>(frame_elements), 0.0f);
  std::fill(dwindow, dwindow + win, 0.0f);
  for (int64_t row = 0; row < batch; ++row)
    for (int64_t sample = 0; sample < samples; ++sample) {
      float numerator = 0.0f, weight = 0.0f;
      for (int64_t frame = 0; frame < frames; ++frame) {
        const int64_t local = sample - frame * hop;
        if (local < 0 || local >= win) continue;
        const float w = window[local];
        numerator += frame_value[(row * frames + frame) * win + local] * w;
        weight += w * w;
      }
      const float denom = std::max(weight, 1.0e-12f);
      const float draw = dy[row * samples + sample] / denom;
      const float dweight = weight > 1.0e-12f
                                ? -dy[row * samples + sample] * numerator /
                                      (denom * denom)
                                : 0.0f;
      for (int64_t frame = 0; frame < frames; ++frame) {
        const int64_t local = sample - frame * hop;
        if (local < 0 || local >= win) continue;
        const int64_t at = (row * frames + frame) * win + local;
        dframe[at] += draw * window[local];
        dwindow[local] += draw * frame_value[at] +
                          2.0f * dweight * window[local];
      }
    }
  if (tessera_x86_fft_r2c_packed_f32(
          digest, dframe.data(), dspectrum, rows, win))
    return 3;
  for (int64_t row = 0; row < rows; ++row)
    for (int64_t bin = 0; bin < bins; ++bin) {
      float scale = inverse_scale;
      if (bin > 0 && !(win % 2 == 0 && bin == bins - 1)) scale *= 2.0f;
      dspectrum[2 * (row * bins + bin)] *= scale;
      dspectrum[2 * (row * bins + bin) + 1] *= scale;
    }
  return 0;
}

extern "C" int tessera_x86_avx512_istft_bwd_storage(
    const char *digest, const void *dy, const float *spectrum,
    const void *window, float *dspectrum, void *dwindow, int64_t batch,
    int64_t frames, int64_t win, int64_t hop, int storage,
    float inverse_scale) {
  int64_t samples = 0, y_elements = 0;
  if (!valid_backward_storage(storage) || !digest || !dy || !spectrum ||
      !window || !dspectrum || !dwindow ||
      !checked_framed_length(frames, hop, win, samples) ||
      !checked_product(batch, samples, y_elements))
    return 10;
  if (storage == 0)
    return tessera_x86_avx512_istft_bwd_f32(
        digest, static_cast<const float *>(dy), spectrum,
        static_cast<const float *>(window), dspectrum,
        static_cast<float *>(dwindow), batch, frames, win, hop, inverse_scale);
  std::vector<float> y_grad(static_cast<size_t>(y_elements));
  std::vector<float> w(static_cast<size_t>(win));
  std::vector<float> w_grad(w.size());
  unpack_backward_storage(dy, y_grad, storage);
  unpack_backward_storage(window, w, storage);
  const int rc = tessera_x86_avx512_istft_bwd_f32(
      digest, y_grad.data(), spectrum, w.data(), dspectrum, w_grad.data(),
      batch, frames, win, hop, inverse_scale);
  if (!rc) pack_backward_storage(w_grad, dwindow, storage);
  return rc;
}

extern "C" int tessera_x86_avx512_stft_bwd_policy_storage(
    const char *digest, const float *dy, const void *input,
    const void *window, void *dx, void *dwindow, int64_t batch,
    int64_t samples, int64_t frames, int64_t win, int64_t hop, int storage,
    float forward_scale, int center, int pad_mode) {
  if (!valid_backward_storage(storage) || !digest || !dy || !input || !window ||
      !dx || !dwindow || batch <= 0 || samples <= 0 || win <= 0 || hop <= 0 ||
      (center != 0 && center != 1) || (pad_mode != 0 && pad_mode != 1))
    return 20;
  const int64_t pad = center ? win / 2 : 0;
  if (pad_mode == 1 && samples <= pad) return 21;
  if ((center && win != 16 && win != 18) ||
      samples > std::numeric_limits<int64_t>::max() - 2 * pad)
    return 22;
  const int64_t padded_samples = samples + 2 * pad;
  int64_t covered = 0, x_elements = 0, padded_elements = 0;
  if (!checked_framed_length(frames, hop, win, covered) ||
      padded_samples < win || frames != (padded_samples - win) / hop + 1 ||
      !checked_product(batch, samples, x_elements) ||
      !checked_product(batch, padded_samples, padded_elements))
    return 22;
  std::vector<float> x(static_cast<size_t>(x_elements));
  std::vector<float> w(static_cast<size_t>(win));
  std::vector<float> padded(static_cast<size_t>(padded_elements), 0.0f);
  std::vector<float> padded_dx(padded.size());
  std::vector<float> x_grad(x.size(), 0.0f);
  std::vector<float> w_grad(w.size());
  unpack_backward_storage(input, x, storage);
  unpack_backward_storage(window, w, storage);
  for (int64_t row = 0; row < batch; ++row)
    for (int64_t at = 0; at < padded_samples; ++at) {
      int64_t source = at - pad;
      bool present = source >= 0 && source < samples;
      if (!present && pad_mode == 1) {
        source = source < 0 ? -source : 2 * samples - 2 - source;
        present = source >= 0 && source < samples;
      }
      if (present) padded[row * padded_samples + at] = x[row * samples + source];
    }
  int rc = tessera_x86_avx512_stft_bwd_f32(
      digest, dy, padded.data(), w.data(), padded_dx.data(), w_grad.data(),
      batch, padded_samples, frames, win, hop, forward_scale);
  if (rc) return rc;
  for (int64_t row = 0; row < batch; ++row)
    for (int64_t at = 0; at < padded_samples; ++at) {
      int64_t source = at - pad;
      bool present = source >= 0 && source < samples;
      if (!present && pad_mode == 1) {
        source = source < 0 ? -source : 2 * samples - 2 - source;
        present = source >= 0 && source < samples;
      }
      if (present)
        x_grad[row * samples + source] += padded_dx[row * padded_samples + at];
    }
  pack_backward_storage(x_grad, dx, storage);
  pack_backward_storage(w_grad, dwindow, storage);
  return 0;
}

extern "C" int tessera_x86_avx512_istft_bwd_policy_storage(
    const char *digest, const void *dy, const float *spectrum,
    const void *window, float *dspectrum, void *dwindow, int64_t batch,
    int64_t frames, int64_t win, int64_t hop, int storage,
    float inverse_scale, int center, int64_t output_samples) {
  if (!valid_backward_storage(storage) || !digest || !dy || !spectrum ||
      !window || !dspectrum || !dwindow || batch <= 0 || frames <= 0 ||
      win <= 0 || hop <= 0 || (center != 0 && center != 1))
    return 30;
  int64_t raw_samples = 0;
  if (!checked_framed_length(frames, hop, win, raw_samples)) return 31;
  const int64_t trim = center ? win / 2 : 0;
  const int64_t available = raw_samples - 2 * trim;
  const bool expanded = center || output_samples != raw_samples;
  int64_t cropped_elements = 0, raw_elements = 0;
  if (output_samples <= 0 || output_samples > available ||
      (expanded && win != 16 && win != 18) ||
      !checked_product(batch, output_samples, cropped_elements) ||
      !checked_product(batch, raw_samples, raw_elements))
    return 32;
  std::vector<float> cropped(static_cast<size_t>(cropped_elements));
  std::vector<float> raw(static_cast<size_t>(raw_elements), 0.0f);
  std::vector<float> w(static_cast<size_t>(win));
  std::vector<float> w_grad(w.size());
  unpack_backward_storage(dy, cropped, storage);
  unpack_backward_storage(window, w, storage);
  for (int64_t row = 0; row < batch; ++row)
    std::copy_n(cropped.data() + row * output_samples, output_samples,
                raw.data() + row * raw_samples + trim);
  int rc = tessera_x86_avx512_istft_bwd_f32(
      digest, raw.data(), spectrum, w.data(), dspectrum, w_grad.data(), batch,
      frames, win, hop, inverse_scale);
  if (!rc) pack_backward_storage(w_grad, dwindow, storage);
  return rc;
}

extern "C" int tessera_x86_avx512_stft_bwd_policy_strided_storage(
    const char *digest, const float *dy, const void *input,
    const void *window, void *dx, void *dwindow, int64_t outer,
    int64_t samples, int64_t inner, int64_t frames, int64_t win,
    int64_t hop, int storage, float forward_scale, int center, int pad_mode) {
  int64_t batch = 0, input_elements = 0, spectral_rows = 0;
  if (!valid_backward_storage(storage) || !digest || !dy || !input ||
      !window || !dx || !dwindow || !checked_product(outer, inner, batch) ||
      !checked_product(batch, samples, input_elements) ||
      !checked_product(batch, frames, spectral_rows))
    return 40;
  const int64_t bins = win / 2 + 1;
  int64_t spectral_elements = 0;
  if (!checked_product(spectral_rows, bins, spectral_elements)) return 40;
  const size_t bytes = backward_storage_bytes(storage);
  size_t input_bytes = 0, spectral_bytes = 0;
  if (!checked_backward_bytes(input_elements, bytes, input_bytes) ||
      !checked_backward_bytes(spectral_elements, sizeof(float) * 2,
                              spectral_bytes))
    return 40;
  std::vector<unsigned char> packed_input(input_bytes);
  std::vector<unsigned char> packed_dx(input_bytes);
  std::vector<float> packed_dy(spectral_bytes / sizeof(float));
  pack_backward_axis(input, packed_input.data(), outer, samples, inner, bytes);
  pack_backward_axis(dy, packed_dy.data(), outer, frames * bins, inner,
                     sizeof(float) * 2);
  int rc = tessera_x86_avx512_stft_bwd_policy_storage(
      digest, packed_dy.data(), packed_input.data(), window, packed_dx.data(),
      dwindow, batch, samples, frames, win, hop, storage, forward_scale,
      center, pad_mode);
  if (!rc)
    unpack_backward_axis(packed_dx.data(), dx, outer, samples, inner, bytes);
  return rc;
}

extern "C" int tessera_x86_avx512_istft_bwd_policy_strided_storage(
    const char *digest, const void *dy, const float *spectrum,
    const void *window, float *dspectrum, void *dwindow, int64_t outer,
    int64_t frames, int64_t bins, int64_t inner, int64_t win, int64_t hop,
    int storage, float inverse_scale, int center, int64_t output_samples) {
  int64_t batch = 0, dy_elements = 0, spectral_rows = 0;
  if (!valid_backward_storage(storage) || !digest || !dy || !spectrum ||
      !window || !dspectrum || !dwindow ||
      !checked_product(outer, inner, batch) ||
      !checked_product(batch, output_samples, dy_elements) ||
      !checked_product(batch, frames, spectral_rows))
    return 50;
  int64_t spectral_elements = 0;
  if (!checked_product(spectral_rows, bins, spectral_elements)) return 50;
  const size_t bytes = backward_storage_bytes(storage);
  size_t dy_bytes = 0, spectral_bytes = 0;
  if (!checked_backward_bytes(dy_elements, bytes, dy_bytes) ||
      !checked_backward_bytes(spectral_elements, sizeof(float) * 2,
                              spectral_bytes))
    return 50;
  std::vector<unsigned char> packed_dy(dy_bytes);
  std::vector<float> packed_spectrum(spectral_bytes / sizeof(float));
  std::vector<float> packed_dspectrum(spectral_bytes / sizeof(float));
  pack_backward_axis(dy, packed_dy.data(), outer, output_samples, inner, bytes);
  pack_backward_axis(spectrum, packed_spectrum.data(), outer, frames * bins,
                     inner, sizeof(float) * 2);
  int rc = tessera_x86_avx512_istft_bwd_policy_storage(
      digest, packed_dy.data(), packed_spectrum.data(), window,
      packed_dspectrum.data(), dwindow, batch, frames, win, hop, storage,
      inverse_scale, center, output_samples);
  if (!rc)
    unpack_backward_axis(packed_dspectrum.data(), dspectrum, outer,
                         frames * bins, inner, sizeof(float) * 2);
  return rc;
}

namespace {

// Runtime-layout ABI used by the generalized STFT/ISTFT adjoints.  Strides
// are expressed in elements and may be negative.  The pointer names logical
// element zero, matching NumPy's view ABI; no host-side packing is permitted.
bool valid_layout(int64_t rank, const int64_t *shape, const int64_t *strides) {
  if (rank <= 0 || rank > 8 || !shape || !strides) return false;
  for (int64_t dim = 0; dim < rank; ++dim)
    if (shape[dim] <= 0 || (shape[dim] > 1 && strides[dim] == 0)) return false;
  return true;
}

int64_t layout_elements(int64_t rank, const int64_t *shape) {
  int64_t result = 1;
  for (int64_t dim = 0; dim < rank; ++dim)
    if (!checked_product(result, shape[dim], result)) return -1;
  return result;
}

int64_t layout_offset(int64_t logical, int64_t rank, const int64_t *shape,
                      const int64_t *strides) {
  int64_t offset = 0;
  for (int64_t dim = rank - 1; dim >= 0; --dim) {
    const int64_t coordinate = logical % shape[dim];
    logical /= shape[dim];
    offset += coordinate * strides[dim];
  }
  return offset;
}

int64_t row_without_axis(int64_t logical, int64_t rank, const int64_t *shape,
                         int64_t axis) {
  int64_t row = 0, multiplier = 1;
  for (int64_t dim = rank - 1; dim >= 0; --dim) {
    const int64_t coordinate = logical % shape[dim];
    logical /= shape[dim];
    if (dim == axis) continue;
    row += coordinate * multiplier;
    multiplier *= shape[dim];
  }
  return row;
}

void gather_real_axis(const void *source, int storage, int64_t rank,
                      const int64_t *shape, const int64_t *strides,
                      int64_t axis, std::vector<float> &target) {
  const int64_t elements = layout_elements(rank, shape);
  const int64_t extent = shape[axis];
  for (int64_t logical = 0; logical < elements; ++logical) {
    int64_t cursor = logical;
    int64_t coordinate = 0;
    for (int64_t dim = rank - 1; dim >= 0; --dim) {
      const int64_t current = cursor % shape[dim];
      cursor /= shape[dim];
      if (dim == axis) coordinate = current;
    }
    const int64_t row = row_without_axis(logical, rank, shape, axis);
    target[row * extent + coordinate] =
        load_backward_storage(source, layout_offset(logical, rank, shape, strides),
                              storage);
  }
}

void scatter_real_axis(const std::vector<float> &source, void *target,
                       int storage, int64_t rank, const int64_t *shape,
                       int64_t axis) {
  const int64_t elements = layout_elements(rank, shape);
  const int64_t extent = shape[axis];
  for (int64_t logical = 0; logical < elements; ++logical) {
    int64_t cursor = logical;
    int64_t coordinate = 0;
    for (int64_t dim = rank - 1; dim >= 0; --dim) {
      const int64_t current = cursor % shape[dim];
      cursor /= shape[dim];
      if (dim == axis) coordinate = current;
    }
    const int64_t row = row_without_axis(logical, rank, shape, axis);
    store_backward_storage(target, logical, storage,
                           source[row * extent + coordinate]);
  }
}

void gather_complex_frames(const float *source, int64_t rank,
                           const int64_t *shape, const int64_t *strides,
                           int64_t frame_axis, int64_t bin_axis,
                           std::vector<float> &target) {
  const int64_t elements = layout_elements(rank, shape);
  const int64_t frames = shape[frame_axis], bins = shape[bin_axis];
  for (int64_t logical = 0; logical < elements; ++logical) {
    int64_t cursor = logical, frame = 0, bin = 0, row = 0, multiplier = 1;
    for (int64_t dim = rank - 1; dim >= 0; --dim) {
      const int64_t coordinate = cursor % shape[dim];
      cursor /= shape[dim];
      if (dim == frame_axis) frame = coordinate;
      else if (dim == bin_axis) bin = coordinate;
      else {
        row += coordinate * multiplier;
        multiplier *= shape[dim];
      }
    }
    const int64_t source_at = layout_offset(logical, rank, shape, strides);
    const int64_t target_at = (row * frames + frame) * bins + bin;
    target[2 * target_at] = source[2 * source_at];
    target[2 * target_at + 1] = source[2 * source_at + 1];
  }
}

void scatter_complex_frames(const std::vector<float> &source, float *target,
                            int64_t rank, const int64_t *shape,
                            int64_t frame_axis, int64_t bin_axis) {
  const int64_t elements = layout_elements(rank, shape);
  const int64_t frames = shape[frame_axis], bins = shape[bin_axis];
  for (int64_t logical = 0; logical < elements; ++logical) {
    int64_t cursor = logical, frame = 0, bin = 0, row = 0, multiplier = 1;
    for (int64_t dim = rank - 1; dim >= 0; --dim) {
      const int64_t coordinate = cursor % shape[dim];
      cursor /= shape[dim];
      if (dim == frame_axis) frame = coordinate;
      else if (dim == bin_axis) bin = coordinate;
      else {
        row += coordinate * multiplier;
        multiplier *= shape[dim];
      }
    }
    const int64_t source_at = (row * frames + frame) * bins + bin;
    target[2 * logical] = source[2 * source_at];
    target[2 * logical + 1] = source[2 * source_at + 1];
  }
}

int64_t reflected_index(int64_t index, int64_t samples) {
  while (index < 0 || index >= samples)
    index = index < 0 ? -index : 2 * samples - 2 - index;
  return index;
}

bool broadcast_window_row(int64_t row, const std::vector<int64_t> &batch_shape,
                          int64_t window_rank, const int64_t *window_shape,
                          const int64_t *window_strides, int64_t &source_offset,
                          int64_t &logical_row) {
  if (window_rank < 1 || !window_shape || !window_strides ||
      window_rank - 1 > static_cast<int64_t>(batch_shape.size()))
    return false;
  source_offset = 0;
  logical_row = 0;
  int64_t logical_multiplier = 1;
  int64_t cursor = row;
  for (int64_t batch_dim = static_cast<int64_t>(batch_shape.size()) - 1;
       batch_dim >= 0; --batch_dim) {
    const int64_t coordinate = cursor % batch_shape[batch_dim];
    cursor /= batch_shape[batch_dim];
    const int64_t window_dim =
        batch_dim - (static_cast<int64_t>(batch_shape.size()) - (window_rank - 1));
    if (window_dim < 0) continue;
    const int64_t extent = window_shape[window_dim];
    if (extent != 1 && extent != batch_shape[batch_dim]) return false;
    const int64_t selected = extent == 1 ? 0 : coordinate;
    source_offset += selected * window_strides[window_dim];
    logical_row += selected * logical_multiplier;
    logical_multiplier *= extent;
  }
  return true;
}

}  // namespace

extern "C" int tessera_x86_avx512_stft_bwd_policy_layout_storage(
    const char *digest, const float *dy, const void *input,
    const void *window, void *dx, void *dwindow, int64_t x_rank,
    const int64_t *x_shape, const int64_t *x_strides, int64_t axis,
    int64_t dy_rank, const int64_t *dy_shape, const int64_t *dy_strides,
    int64_t window_rank, const int64_t *window_shape,
    const int64_t *window_strides, int64_t fft_n, int64_t win, int64_t hop,
    int storage, float forward_scale, int center, int pad_mode, int onesided) {
  if (!digest || !dy || !input || !window || !dx || !dwindow ||
      !valid_backward_storage(storage) || !valid_layout(x_rank, x_shape, x_strides) ||
      !valid_layout(dy_rank, dy_shape, dy_strides) ||
      !valid_layout(window_rank, window_shape, window_strides) ||
      dy_rank != x_rank + 1 || axis < 0 || axis >= x_rank || fft_n <= 0 ||
      win <= 0 || win > fft_n || hop <= 0 || (center != 0 && center != 1) ||
      (pad_mode != 0 && pad_mode != 1) || (onesided != 0 && onesided != 1))
    return 60;
  const int64_t samples = x_shape[axis], frames = dy_shape[axis];
  const int64_t bins = dy_shape[axis + 1];
  const int64_t expected_bins = onesided ? fft_n / 2 + 1 : fft_n;
  const int64_t pad = center ? fft_n / 2 : 0;
  const int64_t padded_samples = std::max(samples + 2 * pad, fft_n);
  if (bins != expected_bins || frames != (padded_samples - fft_n) / hop + 1 ||
      (pad_mode == 1 && center && samples <= pad))
    return 61;
  for (int64_t dim = 0; dim < x_rank; ++dim) {
    if (dim < axis && dy_shape[dim] != x_shape[dim]) return 62;
    if (dim > axis && dy_shape[dim + 1] != x_shape[dim]) return 62;
  }
  const int64_t x_elements = layout_elements(x_rank, x_shape);
  const int64_t batch = x_elements / samples;
  int64_t spectral_elements = 0;
  if (x_elements <= 0 || !checked_product(batch * frames, bins, spectral_elements))
    return 63;
  std::vector<float> packed_x(static_cast<size_t>(x_elements));
  std::vector<float> packed_dy(static_cast<size_t>(2 * spectral_elements));
  std::vector<float> packed_dx(static_cast<size_t>(x_elements), 0.0f);
  std::vector<float> padded_x(static_cast<size_t>(batch * padded_samples), 0.0f);
  std::vector<float> padded_dx(padded_x.size(), 0.0f);
  if (window_shape[window_rank - 1] != win) return 64;
  std::vector<int64_t> batch_shape;
  for (int64_t dim = 0; dim < x_rank; ++dim)
    if (dim != axis) batch_shape.push_back(x_shape[dim]);
  const int64_t window_rows = layout_elements(window_rank, window_shape) / win;
  std::vector<float> padded_window(static_cast<size_t>(batch * fft_n), 0.0f);
  std::vector<float> padded_dwindow(static_cast<size_t>(window_rows * fft_n), 0.0f);
  gather_real_axis(input, storage, x_rank, x_shape, x_strides, axis, packed_x);
  gather_complex_frames(dy, dy_rank, dy_shape, dy_strides, axis, axis + 1,
                        packed_dy);
  const int64_t window_offset = (fft_n - win) / 2;
  for (int64_t row = 0; row < batch; ++row) {
    int64_t source_offset = 0, logical_row = 0;
    if (!broadcast_window_row(row, batch_shape, window_rank, window_shape,
                              window_strides, source_offset, logical_row))
      return 65;
    for (int64_t local = 0; local < win; ++local)
      padded_window[row * fft_n + window_offset + local] =
          load_backward_storage(
              window, source_offset + local * window_strides[window_rank - 1],
              storage);
  }
  for (int64_t row = 0; row < batch; ++row)
    for (int64_t at = 0; at < padded_samples; ++at) {
      int64_t source = at - pad;
      bool present = source >= 0 && source < samples;
      if (!present && pad_mode == 1) {
        source = reflected_index(source, samples);
        present = true;
      }
      if (present) padded_x[row * padded_samples + at] = packed_x[row * samples + source];
    }
  constexpr double kTwoPi = 6.283185307179586476925286766559;
  for (int64_t row = 0; row < batch; ++row)
    for (int64_t frame = 0; frame < frames; ++frame)
      for (int64_t local = 0; local < fft_n; ++local) {
        double value = 0.0;
        for (int64_t bin = 0; bin < bins; ++bin) {
          const double angle = kTwoPi * double(bin * local) / double(fft_n);
          const int64_t at = (row * frames + frame) * bins + bin;
          value += double(packed_dy[2 * at]) * std::cos(angle) -
                   double(packed_dy[2 * at + 1]) * std::sin(angle);
        }
        const float grad = static_cast<float>(value * forward_scale);
        const int64_t sample = frame * hop + local;
        padded_dx[row * padded_samples + sample] +=
            grad * padded_window[row * fft_n + local];
        int64_t source_offset = 0, logical_row = 0;
        if (!broadcast_window_row(row, batch_shape, window_rank, window_shape,
                                  window_strides, source_offset, logical_row))
          return 65;
        padded_dwindow[logical_row * fft_n + local] +=
            grad * padded_x[row * padded_samples + sample];
      }
  for (int64_t row = 0; row < batch; ++row)
    for (int64_t at = 0; at < padded_samples; ++at) {
      int64_t source = at - pad;
      bool present = source >= 0 && source < samples;
      if (!present && pad_mode == 1) {
        source = reflected_index(source, samples);
        present = true;
      }
      if (present) packed_dx[row * samples + source] += padded_dx[row * padded_samples + at];
    }
  scatter_real_axis(packed_dx, dx, storage, x_rank, x_shape, axis);
  for (int64_t row = 0; row < window_rows; ++row)
    for (int64_t local = 0; local < win; ++local)
      store_backward_storage(dwindow, row * win + local, storage,
                             padded_dwindow[row * fft_n + window_offset + local]);
  return 0;
}

extern "C" int tessera_x86_avx512_istft_bwd_policy_layout_storage(
    const char *digest, const void *dy, const float *spectrum,
    const void *window, float *dspectrum, void *dwindow, int64_t dy_rank,
    const int64_t *dy_shape, const int64_t *dy_strides, int64_t output_axis,
    int64_t spectrum_rank, const int64_t *spectrum_shape,
    const int64_t *spectrum_strides, int64_t frame_axis, int64_t bin_axis,
    int64_t window_rank, const int64_t *window_shape,
    const int64_t *window_strides, int64_t fft_n, int64_t win, int64_t hop,
    int storage, float inverse_scale, int center, int onesided) {
  if (!digest || !dy || !spectrum || !window || !dspectrum || !dwindow ||
      !valid_backward_storage(storage) || !valid_layout(dy_rank, dy_shape, dy_strides) ||
      !valid_layout(spectrum_rank, spectrum_shape, spectrum_strides) ||
      !valid_layout(window_rank, window_shape, window_strides) ||
      spectrum_rank != dy_rank + 1 || output_axis < 0 || output_axis >= dy_rank ||
      frame_axis < 0 || bin_axis != frame_axis + 1 || bin_axis >= spectrum_rank ||
      fft_n <= 0 || win <= 0 || win > fft_n || hop <= 0 ||
      (center != 0 && center != 1) || (onesided != 0 && onesided != 1))
    return 70;
  const int64_t frames = spectrum_shape[frame_axis], bins = spectrum_shape[bin_axis];
  const int64_t expected_bins = onesided ? fft_n / 2 + 1 : fft_n;
  int64_t raw_samples = 0;
  if (bins != expected_bins || !checked_framed_length(frames, hop, fft_n, raw_samples))
    return 71;
  const int64_t trim = center ? fft_n / 2 : 0;
  const int64_t output_samples = dy_shape[output_axis];
  if (output_samples <= 0 || output_samples > raw_samples - 2 * trim) return 72;
  for (int64_t dim = 0; dim < dy_rank; ++dim) {
    if (dim < output_axis && spectrum_shape[dim] != dy_shape[dim]) return 73;
    if (dim > output_axis && spectrum_shape[dim + 1] != dy_shape[dim]) return 73;
  }
  const int64_t dy_elements = layout_elements(dy_rank, dy_shape);
  const int64_t batch = dy_elements / output_samples;
  int64_t spectral_elements = 0;
  if (dy_elements <= 0 || !checked_product(batch * frames, bins, spectral_elements))
    return 74;
  std::vector<float> packed_dy(static_cast<size_t>(dy_elements));
  std::vector<float> packed_spectrum(static_cast<size_t>(2 * spectral_elements));
  std::vector<float> packed_dspectrum(static_cast<size_t>(2 * spectral_elements), 0.0f);
  if (window_shape[window_rank - 1] != win) return 75;
  std::vector<int64_t> batch_shape;
  for (int64_t dim = 0; dim < dy_rank; ++dim)
    if (dim != output_axis) batch_shape.push_back(dy_shape[dim]);
  const int64_t window_rows = layout_elements(window_rank, window_shape) / win;
  std::vector<float> padded_window(static_cast<size_t>(batch * fft_n), 0.0f);
  std::vector<float> padded_dwindow(static_cast<size_t>(window_rows * fft_n), 0.0f);
  gather_real_axis(dy, storage, dy_rank, dy_shape, dy_strides, output_axis,
                   packed_dy);
  gather_complex_frames(spectrum, spectrum_rank, spectrum_shape,
                        spectrum_strides, frame_axis, bin_axis, packed_spectrum);
  const int64_t window_offset = (fft_n - win) / 2;
  for (int64_t row = 0; row < batch; ++row) {
    int64_t source_offset = 0, logical_row = 0;
    if (!broadcast_window_row(row, batch_shape, window_rank, window_shape,
                              window_strides, source_offset, logical_row))
      return 76;
    for (int64_t local = 0; local < win; ++local)
      padded_window[row * fft_n + window_offset + local] =
          load_backward_storage(
              window, source_offset + local * window_strides[window_rank - 1],
              storage);
  }
  constexpr double kTwoPi = 6.283185307179586476925286766559;
  std::vector<float> frame_value(static_cast<size_t>(batch * frames * fft_n));
  for (int64_t row = 0; row < batch; ++row)
    for (int64_t frame = 0; frame < frames; ++frame)
      for (int64_t local = 0; local < fft_n; ++local) {
        double value = 0.0;
        for (int64_t bin = 0; bin < bins; ++bin) {
          double weight = 1.0;
          if (onesided && bin > 0 && !(fft_n % 2 == 0 && bin == bins - 1))
            weight = 2.0;
          const double angle = kTwoPi * double(bin * local) / double(fft_n);
          const int64_t at = (row * frames + frame) * bins + bin;
          value += weight * (double(packed_spectrum[2 * at]) * std::cos(angle) -
                             double(packed_spectrum[2 * at + 1]) * std::sin(angle));
        }
        frame_value[(row * frames + frame) * fft_n + local] =
            static_cast<float>(value * inverse_scale);
      }
  std::vector<float> dframe(frame_value.size(), 0.0f);
  for (int64_t row = 0; row < batch; ++row)
    for (int64_t output = 0; output < output_samples; ++output) {
      int64_t source_offset = 0, logical_row = 0;
      if (!broadcast_window_row(row, batch_shape, window_rank, window_shape,
                                window_strides, source_offset, logical_row))
        return 76;
      const int64_t sample = output + trim;
      float numerator = 0.0f, weight_sum = 0.0f;
      for (int64_t frame = 0; frame < frames; ++frame) {
        const int64_t local = sample - frame * hop;
        if (local < 0 || local >= fft_n) continue;
        const float w = padded_window[row * fft_n + local];
        numerator += frame_value[(row * frames + frame) * fft_n + local] * w;
        weight_sum += w * w;
      }
      const float denom = std::max(weight_sum, 1.0e-12f);
      const float upstream = packed_dy[row * output_samples + output];
      const float draw = upstream / denom;
      const float dweight = weight_sum > 1.0e-12f
                                ? -upstream * numerator / (denom * denom)
                                : 0.0f;
      for (int64_t frame = 0; frame < frames; ++frame) {
        const int64_t local = sample - frame * hop;
        if (local < 0 || local >= fft_n) continue;
        const int64_t at = (row * frames + frame) * fft_n + local;
        dframe[at] += draw * padded_window[row * fft_n + local];
        padded_dwindow[logical_row * fft_n + local] +=
            draw * frame_value[at] +
            2.0f * dweight * padded_window[row * fft_n + local];
      }
    }
  for (int64_t row = 0; row < batch; ++row)
    for (int64_t frame = 0; frame < frames; ++frame)
      for (int64_t bin = 0; bin < bins; ++bin) {
        double real = 0.0, imag = 0.0;
        for (int64_t local = 0; local < fft_n; ++local) {
          const double angle = kTwoPi * double(bin * local) / double(fft_n);
          const double value = dframe[(row * frames + frame) * fft_n + local];
          real += value * std::cos(angle);
          imag -= value * std::sin(angle);
        }
        double weight = inverse_scale;
        if (onesided && bin > 0 && !(fft_n % 2 == 0 && bin == bins - 1))
          weight *= 2.0;
        const int64_t at = (row * frames + frame) * bins + bin;
        packed_dspectrum[2 * at] = static_cast<float>(real * weight);
        packed_dspectrum[2 * at + 1] = static_cast<float>(imag * weight);
      }
  scatter_complex_frames(packed_dspectrum, dspectrum, spectrum_rank,
                         spectrum_shape, frame_axis, bin_axis);
  for (int64_t row = 0; row < window_rows; ++row)
    for (int64_t local = 0; local < win; ++local)
      store_backward_storage(dwindow, row * win + local, storage,
                             padded_dwindow[row * fft_n + window_offset + local]);
  return 0;
}
