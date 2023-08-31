#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;
#[cfg(target_arch = "arm")]
use core::arch::arm::*;

use crate::definition::{final_reduction, M61, MODULUS};

#[target_feature(enable = "neon")]
unsafe fn reduction_core(ptr: *const uint64x2_t, mut len: usize, mut hi: uint64x2_t) -> M61 {
    let mlo = vdupq_n_u64(MODULUS);
    let mhi = vdupq_n_u64(MODULUS >> 6);

    while len > 0 {
        len -= 1;

        let lo = ptr.add(len).read_unaligned();
        let lr = vaddq_u64(vandq_u64(mlo, lo), vshrq_n_u64::<61>(lo));
        let hr = vaddq_u64(vshlq_n_u64::<6>(vandq_u64(hi, mhi)), vshrq_n_u64::<55>(hi));
        hi = vaddq_u64(lr, hr);
    }

    // Last reduction step done using scalar operaions.

    let lo = vgetq_lane_u64::<0>(hi);
    let mut hi = vgetq_lane_u64::<1>(hi);

    hi = (lo & MODULUS) + (lo >> 61) + ((hi & (MODULUS >> 3)) << 3) + (hi >> 58);

    final_reduction(hi)
}

#[target_feature(enable = "neon")]
pub unsafe fn reduce_u8(s: &[u8]) -> M61 {
    let hi = if s.len() & 15 != 0 {
        let mut lo = 0u64;
        let mut hi = 0u64;

        let l = s.len() & !15;
        let mut ptr = s.as_ptr().add(l);

        if s.len() & 8 != 0 {
            lo = (ptr as *const u64).read_unaligned();
            ptr = ptr.add(8);
        }

        let mut tmp = 0;
        for i in (0..(s.len() & 7)).rev() {
            tmp <<= 8;
            tmp |= *ptr.add(i) as u64;
        }

        if s.len() & 8 != 0 {
            hi = tmp;
        } else {
            lo = tmp;
        }

        vsetq_lane_u64::<0>(lo, vdupq_n_u64(hi))
    } else {
        vdupq_n_u64(0)
    };

    reduction_core(s.as_ptr() as *const uint64x2_t, s.len() >> 4, hi)
}

#[target_feature(enable = "neon")]
pub unsafe fn reduce_u16(s: &[u16]) -> M61 {
    let hi = if s.len() & 7 != 0 {
        let mut arr = [0; 8];
        let l = s.len() & !7;

        for i in l..s.len() {
            arr[i - l] = *s.get_unchecked(i);
        }

        (arr.as_ptr() as *const uint64x2_t).read_unaligned()
    } else {
        vdupq_n_u64(0)
    };

    reduction_core(s.as_ptr() as *const uint64x2_t, s.len() >> 3, hi)
}

#[target_feature(enable = "neon")]
pub unsafe fn reduce_u32(s: &[u32]) -> M61 {
    let hi = if s.len() & 3 != 0 {
        let mut arr = [0; 4];
        let l = s.len() & !3;

        for i in l..s.len() {
            arr[i - l] = *s.get_unchecked(i);
        }

        (arr.as_ptr() as *const uint64x2_t).read_unaligned()
    } else {
        vdupq_n_u64(0)
    };

    reduction_core(s.as_ptr() as *const uint64x2_t, s.len() >> 2, hi)
}

#[target_feature(enable = "neon")]
pub unsafe fn reduce_u64(s: &[u64]) -> M61 {
    let hi = if s.len() & 1 != 0 {
        let x = s[s.len() - 1];
        vsetq_lane_u64::<0>((x & MODULUS) + (x >> 61), vdupq_n_u64(0))
    } else {
        vdupq_n_u64(0)
    };

    reduction_core(s.as_ptr() as *const uint64x2_t, s.len() >> 1, hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduce_u8_max() {
        for len in 0..1000 {
            let vec = vec![u8::MAX; len];

            let expected = crate::fallback::reduce_u8(&vec);
            let actual = unsafe { reduce_u8(&vec) };
            assert_eq!(
                expected, actual,
                "expected: {expected:x}, actual: {actual:x}"
            );
        }
    }

    #[test]
    fn reduce_u16_max() {
        for len in 0..1000 {
            let vec = vec![u16::MAX; len];

            let expected = crate::fallback::reduce_u16(&vec);
            let actual = unsafe { reduce_u16(&vec) };
            assert_eq!(
                expected, actual,
                "expected: {expected:x}, actual: {actual:x}"
            );
        }
    }

    #[test]
    fn reduce_u32_max() {
        for len in 0..1000 {
            let vec = vec![u32::MAX; len];

            let expected = crate::fallback::reduce_u32(&vec);
            let actual = unsafe { reduce_u32(&vec) };
            assert_eq!(
                expected, actual,
                "expected: {expected:x}, actual: {actual:x}"
            );
        }
    }

    #[test]
    fn reduce_u64_max() {
        for len in 0..1000 {
            let vec = vec![u64::MAX; len];

            let expected = crate::fallback::reduce_u64(&vec);
            let actual = unsafe { reduce_u64(&vec) };
            assert_eq!(
                expected, actual,
                "expected: {expected:x}, actual: {actual:x}"
            );
        }
    }

    quickcheck::quickcheck! {
        fn reduce_u8_correct(slice: Vec<u8>) -> bool {
            let expected = crate::fallback::reduce_u8(&slice);
            let actual = unsafe { reduce_u8(&slice) };
            expected == actual
        }

        fn reduce_u16_correct(slice: Vec<u16>) -> bool {
            let expected = crate::fallback::reduce_u16(&slice);
            let actual = unsafe { reduce_u16(&slice) };
            expected == actual
        }

        fn reduce_u32_correct(slice: Vec<u32>) -> bool {
            let expected = crate::fallback::reduce_u32(&slice);
            let actual = unsafe { reduce_u32(&slice) };
            expected == actual
        }

        fn reduce_u64_correct(slice: Vec<u64>) -> bool {
            let expected = crate::fallback::reduce_u64(&slice);
            let actual = unsafe { reduce_u64(&slice) };
            expected == actual
        }
    }
}
