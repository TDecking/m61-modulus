//! Main module for vectorized implementations.
//!
//! Calculating the reduction modulo `2^61 - 1` can be
//! accelerated using SIMD-instructions. The used algorithm
//! is almost the same as described in [`crate::fallback`],
//! with Horner's method being replaced by the generalized
//! Horner's method by W. S. Dorn (see <https://doi.org/10.1147/rd.62.0239>).
//!
//! The generalized Horner's method evaluates a polynomial with coefficients
//! `a_i` at a point `b` by performing the split
//! ```text
//! x =     (a_0 + b^4 a_4 + b^8 a_8 + ...)
//!   + b   (a_1 + b^4 a_5 + b^8 a_9 + ...)
//!   + b^2 (a_2 + b^4 a_6 + b^8 a_10 + ...)
//!   + b^3 (a_3 + b^4 a_7 + b^8 a_11 + ...)
//! ```
//! and evaluating the emerging polynomials using the normal version of
//! Horner's method. The amount of splits may vary, the one shown above
//! is suitable for instructions as provided by AVX2, which has four 64-bit lanes,
//! while other architechtures only have two lanes, making a split into two
//! polynomials the obvious choice.
//!
//! Since the new polynomials are now evaluated at a different point compared to
//! the original algorithm, binary shifts and masking changes accordingly.

cfg_if::cfg_if! {
    if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
        // 1. If nightly features are enabled, and the avx512f target feature
        //    is available by default, use the avx512 version directly.
        // 2. If the avx512 version can't be used directly, and the avx2
        //    target feature is available by default, use the avx2 version directly.
        // 3. Otherwise, use the lookup version which chooses the
        //    implementation at runtime.

        #[cfg(not(target_feature = "avx2"))]
        mod sse2;
        #[cfg(not(target_feature = "avx512f"))]
        mod avx2;
        #[cfg(feature = "nightly")]
        mod avx512;
        #[cfg(any(
            all(not(feature = "nightly"), not(target_feature = "avx2")),
            all(feature = "nightly", not(target_feature = "avx512f")),
        ))]
        mod x86_lookup;

        #[cfg(all(feature = "nightly", target_feature = "avx512f"))]
        pub(crate) use avx512::*;

        #[cfg(all(not(target_feature = "avx512f"), target_feature = "avx2"))]
        pub(crate) use avx2::*;

        #[cfg(any(
            all(feature = "nightly", not(target_feature = "avx512f")),
            all(not(feature = "nightly"), not(target_feature = "avx2")),
        ))]
        pub(crate) use x86_lookup::*;
    } else if #[cfg(any(target_arch = "arm", target_arch = "aarch64"))] {
        mod neon;

        pub(crate) use neon::*;
    } else if #[cfg(any(target_family = "wasm"))] {
        mod wasm_simd128;

        pub(crate) use wasm_simd128::*;
    } else {
        compile_error!("unknown architecture");
    }
}
