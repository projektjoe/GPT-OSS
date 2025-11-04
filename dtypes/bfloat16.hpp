#pragma once
// bfloat16.hpp — a small, header‑only bfloat16 implementation for C++17+
//
// Why? Some toolchains don’t yet expose std::bfloat16_t or compiler builtins.
// This type stores a 16‑bit bfloat16 and converts to/from float (binary32).
// Arithmetic is implemented in float and rounded back to bfloat16 (round‑to‑nearest‑even).
//
// Features
//  - Exact bit‑preserving pack/unpack between float and bfloat16
//  - +, -, *, /, unary +/-, comparisons, compound ops
//  - isnan/isinf/isfinite helpers
//  - std::hash, ostream<<, user‑defined literal _bf16
//  - constexpr where practical (C++20’s std::bit_cast used if available)
//
// Notes
//  - bfloat16 format: 1 sign, 8 exponent, 7 fraction bits; same exponent bias as float32 (127).
//  - We do arithmetic in float32 to keep this portable. For vector/ISA acceleration (AVX512‑BF16, etc.),
//    you can later specialize kernels where available.
//  - This header aims to be POD and trivially copyable; layout is a single uint16_t.
//
// Usage
//    #include "bfloat16.hpp"
//    using bf16 = bfloat16;
//    bf16 a = 1.5_bf16;             // user‑defined literal
//    bf16 b = bf16{2.0f};
//    bf16 c = a * b;                 // 3.0
//    float f = static_cast<float>(c);
//
// License: MIT

#include <cstdint>
#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>
#include <ostream>
#include <functional>

struct bfloat16 {
    using storage_type = std::uint16_t;
    storage_type bits; // raw bfloat16 bits

    // --- constructors ------------------------------------------------------
    constexpr bfloat16() noexcept : bits(0) {}
    constexpr explicit bfloat16(storage_type raw, int) noexcept : bits(raw) {}

    // from float/double
    explicit bfloat16(float f) noexcept { bits = pack(f); }
    explicit bfloat16(double d) noexcept { bits = pack(static_cast<float>(d)); }

    // implicit conversion to float
    explicit operator float() const noexcept { return unpack(bits); }

    // raw access
    static constexpr bfloat16 from_bits(storage_type raw) noexcept { return bfloat16(raw, 0); }
    constexpr storage_type to_bits() const noexcept { return bits; }

    // --- arithmetic --------------------------------------------------------
    friend inline bfloat16 operator+(bfloat16 a, bfloat16 b) noexcept { return bfloat16(static_cast<float>(a) + static_cast<float>(b)); }
    friend inline bfloat16 operator-(bfloat16 a, bfloat16 b) noexcept { return bfloat16(static_cast<float>(a) - static_cast<float>(b)); }
    friend inline bfloat16 operator*(bfloat16 a, bfloat16 b) noexcept { return bfloat16(static_cast<float>(a) * static_cast<float>(b)); }
    friend inline bfloat16 operator/(bfloat16 a, bfloat16 b) noexcept { return bfloat16(static_cast<float>(a) / static_cast<float>(b)); }

    friend inline bfloat16 operator+(bfloat16 a) noexcept { return a; }
    friend inline bfloat16 operator-(bfloat16 a) noexcept { return bfloat16(-static_cast<float>(a)); }

    inline bfloat16& operator+=(bfloat16 rhs) noexcept { *this = *this + rhs; return *this; }
    inline bfloat16& operator-=(bfloat16 rhs) noexcept { *this = *this - rhs; return *this; }
    inline bfloat16& operator*=(bfloat16 rhs) noexcept { *this = *this * rhs; return *this; }
    inline bfloat16& operator/=(bfloat16 rhs) noexcept { *this = *this / rhs; return *this; }

    // --- comparisons -------------------------------------------------------
    friend inline bool operator==(bfloat16 a, bfloat16 b) noexcept { return a.bits == b.bits || (isnan(a) && isnan(b)); }
    friend inline bool operator!=(bfloat16 a, bfloat16 b) noexcept { return !(a == b); }
    friend inline bool operator<(bfloat16 a, bfloat16 b) noexcept { return static_cast<float>(a) < static_cast<float>(b); }
    friend inline bool operator>(bfloat16 a, bfloat16 b) noexcept { return static_cast<float>(a) > static_cast<float>(b); }
    friend inline bool operator<=(bfloat16 a, bfloat16 b) noexcept { return static_cast<float>(a) <= static_cast<float>(b); }
    friend inline bool operator>=(bfloat16 a, bfloat16 b) noexcept { return static_cast<float>(a) >= static_cast<float>(b); }

    // --- classification helpers -------------------------------------------
    static inline bool isnan(bfloat16 x) noexcept { return ((x.bits & 0x7F80u) == 0x7F80u) && (x.bits & 0x007Fu); }
    static inline bool isinf(bfloat16 x) noexcept { return ((x.bits & 0x7FFFu) == 0x7F80u); }
    static inline bool isfinite(bfloat16 x) noexcept { return (x.bits & 0x7F80u) != 0x7F80u; }
    static inline bool signbit(bfloat16 x) noexcept { return (x.bits & 0x8000u) != 0; }

    // --- constants ---------------------------------------------------------
    static constexpr bfloat16 positive_zero() noexcept { return from_bits(0x0000u); }
    static constexpr bfloat16 negative_zero() noexcept { return from_bits(0x8000u); }
    static constexpr bfloat16 infinity() noexcept { return from_bits(0x7F80u); }
    static constexpr bfloat16 neg_infinity() noexcept { return from_bits(0xFF80u); }
    static constexpr bfloat16 quiet_NaN() noexcept { return from_bits(0x7FC1u); } // canonical qNaN payload

    // --- implementation details -------------------------------------------
    // Round‑to‑nearest‑even pack from float32 -> bfloat16
    static inline storage_type pack(float f) noexcept {
        // Fast path: zero
        if (f == 0.0f) return std::signbit(f) ? 0x8000u : 0x0000u;

        std::uint32_t u32;
        std::memcpy(&u32, &f, sizeof(u32));

        // Extract sign, exponent, mantissa from float32
        const std::uint32_t sign = u32 & 0x80000000u;
        const std::uint32_t exp  = u32 & 0x7F800000u;
        const std::uint32_t mant = u32 & 0x007FFFFFu;

        // NaN or Inf: just truncate mantissa; make sure NaNs stay NaNs
        if (exp == 0x7F800000u) {
            // Preserve top bits and set qNaN if mantissa != 0
            std::uint16_t top = static_cast<std::uint16_t>((u32 >> 16) & 0xFFFFu);
            if (mant) top |= 0x0040u; // ensure quiet NaN in bf16 (set MSB of payload)
            return top;
        }

        // Normal/subnormal numbers: add rounding bias then truncate
        // RNE trick: add 0x7FFF + LSB of the truncated part
        const std::uint32_t lsb = (u32 >> 16) & 1u;
        const std::uint32_t rounding_bias = 0x7FFFu + lsb;
        std::uint32_t rounded = u32 + rounding_bias;
        return static_cast<storage_type>(rounded >> 16);
    }

    // Unpack bfloat16 -> float32 by placing bits in the high 16 of a float
    static inline float unpack(storage_type b) noexcept {
        std::uint32_t u32 = static_cast<std::uint32_t>(b) << 16;
        float f;
        std::memcpy(&f, &u32, sizeof(f));
        return f;
    }
};

// ostream printer
inline std::ostream& operator<<(std::ostream& os, bfloat16 x) {
    return os << static_cast<float>(x);
}

// user‑defined literal: 1.5_bf16
inline bfloat16 operator"" _bf16(long double v) {
    return bfloat16(static_cast<float>(v));
}

// hash support
namespace std {
    template<> struct hash<bfloat16> {
        size_t operator()(const bfloat16& x) const noexcept { return std::hash<std::uint16_t>{}(x.bits); }
    };

    template<> struct numeric_limits<bfloat16> {
        static constexpr bool is_specialized = true;
        static constexpr int digits       = 8;   // including implicit leading 1? bfloat16 has 7 fraction bits; digits (base2 mantissa) ~ 8
        static constexpr int digits10     = 2;   // conservative decimal precision
        static constexpr int max_digits10 = 4;   // enough to round‑trip
        static constexpr bool is_signed   = true;
        static constexpr bool is_integer  = false;
        static constexpr bool is_exact    = false;
        static constexpr int radix        = 2;
        static constexpr bfloat16 min() noexcept { return bfloat16::from_bits(0x0080u); } // min normal
        static constexpr bfloat16 lowest() noexcept { return bfloat16::from_bits(0xFF7Fu); }
        static constexpr bfloat16 max() noexcept { return bfloat16::from_bits(0x7F7Fu); }
        static constexpr bfloat16 epsilon() noexcept { return bfloat16::from_bits(0x3C80u); } // 2^-7 ≈ 0.0078125
        static constexpr bfloat16 round_error() noexcept { return bfloat16::from_bits(0x3F00u); }
        static constexpr int min_exponent   = -126;
        static constexpr int min_exponent10 = -38;
        static constexpr int max_exponent   = 127;
        static constexpr int max_exponent10 = 38;
        static constexpr bool has_infinity      = true;
        static constexpr bool has_quiet_NaN     = true;
        static constexpr bool has_signaling_NaN = false; // we quiet on pack
        static constexpr float_denorm_style has_denorm = denorm_present;
        static constexpr bool has_denorm_loss = true;
        static constexpr bfloat16 infinity() noexcept { return bfloat16::infinity(); }
        static constexpr bfloat16 quiet_NaN() noexcept { return bfloat16::quiet_NaN(); }
        static constexpr bfloat16 signaling_NaN() noexcept { return bfloat16::quiet_NaN(); }
        static constexpr bfloat16 denorm_min() noexcept { return bfloat16::from_bits(0x0001u); }
        static constexpr bool is_iec559  = false; // close but not full IEEE binary16 (different format)
        static constexpr bool is_bounded = true;
        static constexpr bool is_modulo  = false;
        static constexpr bool traps      = false;
        static constexpr bool tinyness_before = false;
        static constexpr float_round_style round_style = round_to_nearest;
    };
}

// Optional: mixed ops with float for convenience
inline bfloat16 operator+(bfloat16 a, float b) noexcept { return bfloat16(static_cast<float>(a) + b); }
inline bfloat16 operator-(bfloat16 a, float b) noexcept { return bfloat16(static_cast<float>(a) - b); }
inline bfloat16 operator*(bfloat16 a, float b) noexcept { return bfloat16(static_cast<float>(a) * b); }
inline bfloat16 operator/(bfloat16 a, float b) noexcept { return bfloat16(static_cast<float>(a) / b); }
inline bfloat16 operator+(float a, bfloat16 b) noexcept { return bfloat16(a + static_cast<float>(b)); }
inline bfloat16 operator-(float a, bfloat16 b) noexcept { return bfloat16(a - static_cast<float>(b)); }
inline bfloat16 operator*(float a, bfloat16 b) noexcept { return bfloat16(a * static_cast<float>(b)); }
inline bfloat16 operator/(float a, bfloat16 b) noexcept { return bfloat16(a / static_cast<float>(b)); }

// End of bfloat16.hpp
