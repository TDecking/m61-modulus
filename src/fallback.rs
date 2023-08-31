//! Fallback versions of the reduction functions used
//! for testing or targets without specialized definitions.
//!
//! To calculate the reduction modulo `2^61 - 1`, we employ two
//! algorithms:
//!
//! * Calculate the result using digit sums. This uses the fact
//!   that for every natural number `x` and a modulus `m`, we
//!   have `x = d_(m + 1)(x) (mod m)` with `d_b(y)` being the
//!   digit sum of a number `y` base `b`. This  means that calculating
//!   the reduction modulo `2^61 - 1` can be accomplished by calculating
//!   the digit sum base `2^61`. Since our numerical base is a power of two,
//!   the calculation is very cheap, merely requiring binary shifts,
//!   binary ANDs, and addition.
//! * Calculate the result using polynomial evaluation.
//!   If `b = 2^64` is the base of the number `x` and `a_i` its digit
//!   values, we can use the [Horner's method](https://en.wikipedia.org/wiki/Horner%27s_method)
//!   to evaluate a polynomial with coefficients `a_i` at point `b`,
//!   with addition and multiplication being the modular variants.
//!
//! Putting these two algorithms toegther, we get an implementation
//! of Horner's method where each iteration is a digit sum of a
//! 128-bit unsigned integer, where the low part is a base `b = 2^64` digit
//! of the number `x` and the high part the accumulating variable.
//!
//! If `b` is not equal to `2^64`, several adjacent digits can be bundled
//! together into a 64-bit integer, making the algorithm applicable to
//! other numerical bases.

use crate::definition::{final_reduction, M61, MODULUS};

pub(crate) fn reduce_u8(s: &[u8]) -> M61 {
    let chuncks = s.chunks_exact(8);

    let rem = chuncks.remainder();
    let mut hi = 0;
    for x in rem.iter().copied().rev() {
        hi = (hi << 8) | x as u64;
    }

    for lo in chuncks.rev() {
        let lo = u64::from_le_bytes([lo[0], lo[1], lo[2], lo[3], lo[4], lo[5], lo[6], lo[7]]);
        hi = (lo & MODULUS) + (lo >> 61) + ((hi & (MODULUS >> 3)) << 3) + (hi >> 58);
    }

    final_reduction(hi)
}

pub(crate) fn reduce_u16(s: &[u16]) -> M61 {
    let chuncks = s.chunks_exact(4);

    let rem = chuncks.remainder();
    let mut hi = 0;
    for x in rem.iter().copied().rev() {
        hi = (hi << 16) | x as u64;
    }

    for lo in chuncks.rev() {
        let lo = (lo[0] as u64)
            | ((lo[1] as u64) << 16)
            | ((lo[2] as u64) << 32)
            | ((lo[3] as u64) << 48);
        hi = (lo & MODULUS) + (lo >> 61) + ((hi & (MODULUS >> 3)) << 3) + (hi >> 58);
    }

    final_reduction(hi)
}

pub(crate) fn reduce_u32(s: &[u32]) -> M61 {
    let chuncks = s.chunks_exact(2);

    let rem = chuncks.remainder();
    let mut hi = if let Some(r) = rem.first() {
        *r as u64
    } else {
        0
    };

    for lo in chuncks.rev() {
        let lo = lo[0] as u64 | ((lo[1] as u64) << 32);
        hi = (lo & MODULUS) + (lo >> 61) + ((hi & (MODULUS >> 3)) << 3) + (hi >> 58);
    }

    final_reduction(hi)
}

pub(crate) fn reduce_u64(s: &[u64]) -> M61 {
    let mut hi = 0;

    for lo in s.iter().copied().rev() {
        hi = (lo & MODULUS) + (lo >> 61) + ((hi & (MODULUS >> 3)) << 3) + (hi >> 58);
    }

    final_reduction(hi)
}
