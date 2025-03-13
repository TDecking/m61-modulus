//! Functions for performing arithmetic modulo the 61st Mersenne number.
//! Aimed at testing bignum implementations.
//!
//! ## Usage
//! 
//! The crate comes with a trait [`M61Reduction`] and a type [`M61`].
//! `M61` is an integer in which all arithmetic is performed the
//! 61st Mersenne number, `2^61 - 1`.
//!
//! ```
//! use m61_modulus::*;
//!
//! let x = M61::from(1u64 << 61);
//! let y = M61::from(1u64);
//!
//! assert_eq!(x, y);
//! ```
//! 
//! The trait `M61Reduction` is implemented for unsigned integer slices,
//! providing two functions for reducing the modulo `2^61 - 1`,
//! as if they were digits in a bignum implementation.
//! 
//! ```
//! use m61_modulus::*;
//!
//! let x = [1u16, 734u16, 24u16].reduce_m61();
//! let y = M61::from(1) + M61::from(734 << 16) + M61::from(24u64 << 32);
//! 
//! assert_eq!(x, y);
//! ```
//!
//! The functions are `reduce_m61`, which is single-threaded, and `reduce_m61_parallelized`,
//! which may spawn additional threads.
//!
//! This crate comes with two features:
//! * `nightly`, which enables support for additional nightly-only ISA extensions
//!   like AVX512. Disabled by default.
//! * `std`, which provides access to the `reduce_m61_parallelized` function,
//!   which requires the Rust standard library. If disabled, this crate will
//!   also work on `no-std` targets. Enabled by default.
//!
//! ## Background
//!
//! This crate is designed around verifying the results of bignum implementations
//! (like `num-bigint`) in a cheap manner. By repeating an operation
//! using modular arithmetic one can test the results without having to
//! resort to simpler, but slower implementations involving arbitrary-precision
//! arithmetic.
//!
//! Arithetic modulo the 61st Mersenne number is particulary suitable for this:
//! * It is a prime number, which means the results distribute well given random input.
//! * Its difference of one to the next power of two makes calcuations incredibly cheap.

#![allow(unsafe_op_in_unsafe_fn)]

#![cfg_attr(feature = "nightly", feature(avx512_target_feature, stdsimd))]
#![cfg_attr(not(feature = "std"), no_std)]

mod definition;

cfg_if::cfg_if! {
    if #[cfg(all(
        not(miri),
        target_endian = "little",
        any(
            all(target_arch = "x86", target_feature = "sse2"),
            target_arch = "x86_64",
            all(feature = "nightly", target_arch = "arm", target_feature = "neon"),
            target_arch = "aarch64",
            all(target_family = "wasm", target_feature = "simd128"),
        ),
    ))] {
        #[path = "./simd/mod.rs"]
        mod implementation;
    } else {
        #[path = "./fallback.rs"]
        mod implementation;
    }
}

#[cfg(all(test, not(miri)))]
mod fallback;

#[cfg(feature = "std")]
mod parallelized;

pub use crate::definition::M61;

/// Helper trait for making the fuctions accessible using the dot operator.
pub trait M61Reduction {
    /// Calculates `self mod (2^61 - 1)`, assuming `self` is a number
    /// base `2^Self::BITS`, with digits stored in little-edian ordering.
    #[must_use]
    fn reduce_m61(&self) -> M61;

    /// Calculates `self mod (2^61 - 1)`, assuming `self` is a number
    /// base `2^Self::BITS`, with digits stored in little-edian ordering.
    ///
    /// This function is parallelized, using at most `max_thread_count`
    /// threads to calculate the result.
    #[cfg(feature = "std")]
    #[must_use]
    fn reduce_m61_parallelized(&self, max_thread_count: usize) -> M61;
}

impl M61Reduction for [u8] {
    #[inline(always)]
    fn reduce_m61(&self) -> M61 {
        // SAFETY: The `implementation` module only defers to unsafe
        // versions if their safety conditions are met.
        #[allow(unused_unsafe)]
        unsafe {
            implementation::reduce_u8(self)
        }
    }

    #[cfg(feature = "std")]
    #[inline(always)]
    fn reduce_m61_parallelized(&self, max_thread_count: usize) -> M61 {
        parallelized::reduce_u8(self, max_thread_count)
    }
}

impl M61Reduction for [u16] {
    #[inline(always)]
    fn reduce_m61(&self) -> M61 {
        // SAFETY: The `implementation` module only defers to unsafe
        // versions if their safety conditions are met.
        #[allow(unused_unsafe)]
        unsafe {
            implementation::reduce_u16(self)
        }
    }

    #[cfg(feature = "std")]
    #[inline(always)]
    fn reduce_m61_parallelized(&self, max_thread_count: usize) -> M61 {
        parallelized::reduce_u16(self, max_thread_count)
    }
}

impl M61Reduction for [u32] {
    #[inline(always)]
    fn reduce_m61(&self) -> M61 {
        // SAFETY: The `implementation` module only defers to unsafe
        // versions if their safety conditions are met.
        #[allow(unused_unsafe)]
        unsafe {
            implementation::reduce_u32(self)
        }
    }

    #[cfg(feature = "std")]
    #[inline(always)]
    fn reduce_m61_parallelized(&self, max_thread_count: usize) -> M61 {
        parallelized::reduce_u32(self, max_thread_count)
    }
}

impl M61Reduction for [u64] {
    #[inline(always)]
    fn reduce_m61(&self) -> M61 {
        // SAFETY: The `implementation` module only defers to unsafe
        // versions if their safety conditions are met.
        #[allow(unused_unsafe)]
        unsafe {
            implementation::reduce_u64(self)
        }
    }

    #[cfg(feature = "std")]
    #[inline(always)]
    fn reduce_m61_parallelized(&self, max_thread_count: usize) -> M61 {
        parallelized::reduce_u64(self, max_thread_count)
    }
}

impl M61Reduction for [usize] {
    #[inline(always)]
    fn reduce_m61(&self) -> M61 {
        // SAFETY: Within the body, we turn the input slice into a
        // slice of of the same length and with a identically sized type.
        // Thus, the memory regions are the same.
        unsafe {
            use core::slice::from_raw_parts;
            let (ptr, len) = (self.as_ptr(), self.len());
            match core::mem::size_of::<usize>() {
                2 => from_raw_parts(ptr as *const u16, len).reduce_m61(),
                4 => from_raw_parts(ptr as *const u32, len).reduce_m61(),
                8 => from_raw_parts(ptr as *const u64, len).reduce_m61(),
                _ => unreachable!("an address has only 16, 32 or 64 bits"),
            }
        }
    }

    #[cfg(feature = "std")]
    #[inline(always)]
    fn reduce_m61_parallelized(&self, max_thread_count: usize) -> M61 {
        // SAFETY: Within the body, we turn the input slice into a
        // slice of of the same length and with a identically sized type.
        // Thus, the memory regions are the same.
        unsafe {
            use core::slice::from_raw_parts;
            let (ptr, len) = (self.as_ptr(), self.len());
            match core::mem::size_of::<usize>() {
                2 => from_raw_parts(ptr as *const u16, len).reduce_m61_parallelized(max_thread_count),
                4 => from_raw_parts(ptr as *const u32, len).reduce_m61_parallelized(max_thread_count),
                8 => from_raw_parts(ptr as *const u64, len).reduce_m61_parallelized(max_thread_count),
                _ => unreachable!("an address has only 16, 32 or 64 bits"),
            }
        }
    }
}
