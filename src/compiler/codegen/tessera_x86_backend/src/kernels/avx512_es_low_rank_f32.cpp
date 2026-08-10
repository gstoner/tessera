// AVX-512 physical EGGROLL rank-1 correction.
//
// The kernel reproduces the shared splitmix64 -> Philox-4x32-10 -> Box-Muller
// member stream exactly. It materializes only the O(in+out) rank-1 factors,
// never the O(in*out) perturbation matrix. VNNI is deliberately not used: the
// current Graph contract is Gaussian f32 and has no scale/saturation identity.

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

constexpr uint32_t kPhiloxM0 = 0xD2511F53u;
constexpr uint32_t kPhiloxM1 = 0xCD9E8D57u;
constexpr uint32_t kPhiloxW0 = 0x9E3779B9u;
constexpr uint32_t kPhiloxW1 = 0xBB67AE85u;
constexpr uint64_t kEpochDomain = 0x45535F45504F4348ULL;
constexpr uint64_t kPairDomain = 0x45535F504149525FULL;

uint64_t splitmix64(uint64_t value) {
    uint64_t z = value + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

uint64_t member_seed(uint64_t high, uint64_t low, int64_t epoch,
                     uint64_t pair) {
    const uint64_t root = splitmix64(high ^ splitmix64(low));
    const uint64_t epoch_key =
        splitmix64(root ^ kEpochDomain ^ static_cast<uint64_t>(epoch));
    return splitmix64(epoch_key ^ kPairDomain ^ pair);
}

void philox4x32_10(uint32_t words[4], uint32_t key0, uint32_t key1) {
    for (int round = 0; round < 10; ++round) {
        if (round > 0) {
            key0 += kPhiloxW0;
            key1 += kPhiloxW1;
        }
        const uint64_t product0 = static_cast<uint64_t>(kPhiloxM0) * words[0];
        const uint64_t product1 = static_cast<uint64_t>(kPhiloxM1) * words[2];
        const uint32_t next0 =
            static_cast<uint32_t>(product1 >> 32) ^ words[1] ^ key0;
        const uint32_t next2 =
            static_cast<uint32_t>(product0 >> 32) ^ words[3] ^ key1;
        words[0] = next0;
        words[1] = static_cast<uint32_t>(product1);
        words[2] = next2;
        words[3] = static_cast<uint32_t>(product0);
    }
}

float uniform_at(uint64_t seed, int64_t index, int64_t counter_base) {
    const uint64_t block = static_cast<uint64_t>(counter_base + index / 4);
    uint32_t words[4] = {static_cast<uint32_t>(block),
                         static_cast<uint32_t>(block >> 32), 0u, 0u};
    philox4x32_10(words, static_cast<uint32_t>(seed),
                  static_cast<uint32_t>(seed >> 32));
    return static_cast<float>(words[index & 3]) * (1.0f / 4294967296.0f);
}

float normal_at(uint64_t seed, int64_t index, int64_t count) {
    const int64_t pair = index / 2;
    const int64_t half = (count + 1) / 2;
    const int64_t second_counter_base = (half + 3) / 4 + 1;
    const float u1 = std::max(1.0e-7f, uniform_at(seed, pair, 0));
    const float u2 = uniform_at(seed, pair, second_counter_base);
    const float radius = std::sqrt(-2.0f * std::log(u1));
    const float theta = 6.2831853071795864769f * u2;
    return (index & 1) ? radius * std::sin(theta) : radius * std::cos(theta);
}

float dot_avx512(const float* left, const float* right, int64_t length) {
    __m512 sum = _mm512_setzero_ps();
    int64_t index = 0;
    for (; index + 16 <= length; index += 16)
        sum = _mm512_fmadd_ps(_mm512_loadu_ps(left + index),
                              _mm512_loadu_ps(right + index), sum);
    alignas(64) float lanes[16];
    _mm512_store_ps(lanes, sum);
    float result = 0.0f;
    for (float lane : lanes) result += lane;
    for (; index < length; ++index) result += left[index] * right[index];
    return result;
}

} // namespace

extern "C" int tessera_x86_avx512_es_low_rank_correction_f32(
    const float* x, const int64_t* members, const int64_t* key, float* output,
    int64_t population, int64_t rows, int64_t in_dim, int64_t out_dim,
    int64_t epoch, float sigma, int32_t antithetic) {
    if (!x || !members || !key || !output || population <= 0 || rows <= 0 ||
        in_dim <= 0 || out_dim <= 0 || epoch < 0 || !std::isfinite(sigma) ||
        sigma <= 0.0f || (antithetic != 0 && antithetic != 1))
        return 1;
    thread_local std::vector<float> factors;
    const size_t factor_count = static_cast<size_t>(in_dim + out_dim);
    if (factors.size() < factor_count) factors.resize(factor_count);
    for (int64_t population_index = 0; population_index < population;
         ++population_index) {
        const int64_t member = members[population_index];
        if (member < 0) return 2;
        const uint64_t pair = antithetic
                                  ? static_cast<uint64_t>(member / 2)
                                  : static_cast<uint64_t>(member);
        const float sign =
            (!antithetic || (member & 1) == 0) ? 1.0f : -1.0f;
        const uint64_t seed = member_seed(
            static_cast<uint64_t>(key[0]), static_cast<uint64_t>(key[1]),
            epoch, pair);
        for (size_t index = 0; index < factor_count; ++index)
            factors[index] = normal_at(seed, static_cast<int64_t>(index),
                                       static_cast<int64_t>(factor_count));
        const float* b = factors.data();
        const float* a = factors.data() + in_dim;
        const __m512 scale = _mm512_set1_ps(sign * sigma);
        for (int64_t row = 0; row < rows; ++row) {
            const float* input =
                x + (population_index * rows + row) * in_dim;
            float* destination =
                output + (population_index * rows + row) * out_dim;
            const float dot_value = dot_avx512(input, b, in_dim);
            const __m512 dot = _mm512_set1_ps(dot_value);
            int64_t column = 0;
            for (; column + 16 <= out_dim; column += 16)
                _mm512_storeu_ps(destination + column,
                                 _mm512_mul_ps(scale, _mm512_mul_ps(
                                     dot, _mm512_loadu_ps(a + column))));
            const float scalar = sign * sigma * dot_value;
            for (; column < out_dim; ++column)
                destination[column] = scalar * a[column];
        }
    }
    return 0;
}
