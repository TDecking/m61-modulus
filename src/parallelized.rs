//! Parallelized versions of the reduction function.
//!
//! For a given number `x` in positional notation with base
//! `b` and digits `a_i`, `x` can be split into
//! ```text
//!        a_0 + b a_1 + b^2 a_2 + ... + a_(k - 1) b^(k - 1)
//! + b^k (a_k + b a_(k + 1) + b^2 a_(k + 2) + ... + a_(2 k - 1) b^(2 k - 1))
//! + ...
//! + b^(n k) (a_(n k) + ... + a_m)
//! ```
//!
//! The reduction modulo `m` can thus be parallelized by computing the
//! reduction for each of the chunks in parallel, and accumulating the
//! results by multiplying the reduction of each chunk with the corresponding
//! power of the respective numerical base before performing addition.
//! All operations during the accumulation step are done using standard modular arithmetic.
//!
//! In our case, `b` is a power of two, and `m = 2^61 - 1`. This allows
//! us to simplify the calculation of the powers of `b` by utilizing the
//! fact that `2^u = 2^v (mod m)` iff `u = v (mod 61)`.

use std::thread::{available_parallelism, scope};

use super::*;

#[cfg(not(test))]
const THRESHOLD: usize = 1 << 14;

#[cfg(test)]
const THRESHOLD: usize = 32;

/// Limit the maximum number of threads based
/// on the available parallelism of the system.
fn clamp_thread_count(max_thread_count: usize) -> usize {
    available_parallelism()
        .map(|x| x.get())
        .unwrap_or(1)
        .min(max_thread_count)
        .max(1)
}

/// Helper macro for the creation of the implementations.
macro_rules! make_function {
    ($name:ident, $type:ty) => {
        pub fn $name(mut s: &[$type], max_thread_count: usize) -> M61 {
            if s.len() < THRESHOLD {
                return s.reduce_m61();
            }

            let max_thread_count = clamp_thread_count(max_thread_count);

            scope(|scope| {
                let mut handles = Vec::with_capacity(max_thread_count);

                let mut step = s.len() / max_thread_count;
                if step < THRESHOLD {
                    step = THRESHOLD;
                }

                let scale = M61(1 << ((step * <$type>::BITS as usize) % 61));
                let mut factor = M61::from(1);

                while s.len() > step {
                    let (part, rest) = s.split_at(step);
                    s = rest;
                    handles.push(scope.spawn(move || part.reduce_m61() * factor));
                    factor *= scale;
                }

                let mut result = s.reduce_m61() * factor;

                for handle in handles {
                    result += handle.join().expect("thread function is total");
                }

                result
            })
        }
    };
}

make_function!(reduce_u8, u8);
make_function!(reduce_u16, u16);
make_function!(reduce_u32, u32);
make_function!(reduce_u64, u64);

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(miri, ignore = "the implementation is done using safe Rust")]
    #[test]
    fn reduce_u8_parallelized_correct() {
        for i in 0..1000 {
            let v = vec![1; i];
            assert_eq!(reduce_u8(&v, 16), v.reduce_m61());
        }
    }

    #[cfg_attr(miri, ignore = "the implementation is done using safe Rust")]
    #[test]
    fn reduce_u16_parallelized_correct() {
        for i in 0..1000 {
            let v = vec![1; i];
            assert_eq!(reduce_u16(&v, 16), v.reduce_m61());
        }
    }

    #[cfg_attr(miri, ignore = "the implementation is done using safe Rust")]
    #[test]
    fn reduce_u32_parallelized_correct() {
        for i in 0..1000 {
            let v = vec![1; i];
            assert_eq!(reduce_u32(&v, 16), v.reduce_m61());
        }
    }

    #[cfg_attr(miri, ignore = "the implementation is done using safe Rust")]
    #[test]
    fn reduce_u64_parallelized_correct() {
        for i in 0..1000 {
            let v = vec![1; i];
            assert_eq!(reduce_u64(&v, 16), v.reduce_m61());
        }
    }
}
