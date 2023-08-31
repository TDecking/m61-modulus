#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use core::arch::wasm32::*;
#[cfg(all(target_arch = "wasm64", target_feature = "simd128"))]
use core::arch::wasm64::*;

use crate::definition::{final_reduction, M61, MODULUS};

unsafe fn reduction_core(ptr: *const v128, mut len: usize, mut hi: v128) -> M61 {
    let mlo = u64x2_splat(MODULUS as i64);
    let mhi = u64x2_splat((MODULUS >> 6) as i64);

    hi = _mm_add_epi64(_mm_and_si128(hi, mlo), _mm_srli_epi64::<61>(hi));

    while len > 0 {
        len -= 1;

        let lo = ptr.add(len).read_unaligned();
        let lr = u64x2_add(v128_and(mlo, lo), u64x2_shr(lo, 61));
        let hr = u64x2_add(u64x2_shl(v128_and(hi, mhi), 6), u64x2_shr(hi, 55));
        hi = u64x2_add(lr, hr);
    }

    // Last reduction step done using scalar operaions.

    let lo = u64x2_extract_lane::<0>(hi);
    let mut hi = u64x2_extract_lane::<1>(hi);

    hi = (lo & MODULUS) + (lo >> 61) + ((hi & (MODULUS >> 3)) << 3) + (hi >> 58);

    final_reduction(hi)
}

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

        u64x2(lo, hi)
    } else {
        u64x2(0, 0)
    };

    reduction_core(s.as_ptr() as *const v128, s.len() >> 4, hi)
}

pub unsafe fn reduce_u16(s: &[u16]) -> M61 {
    let hi = if s.len() & 7 != 0 {
        let mut arr = [0; 8];
        let l = s.len() & !7;

        for i in l..s.len() {
            arr[i - l] = *s.get_unchecked(i);
        }

        (arr.as_ptr() as *const v128).read_unaligned()
    } else {
        u64x2(0, 0)
    };

    reduction_core(s.as_ptr() as *const v128, s.len() >> 3, hi)
}

pub unsafe fn reduce_u32(s: &[u32]) -> M61 {
    let hi = if s.len() & 3 != 0 {
        let mut arr = [0; 4];
        let l = s.len() & !3;

        for i in l..s.len() {
            arr[i - l] = *s.get_unchecked(i);
        }

        (arr.as_ptr() as *const v128).read_unaligned()
    } else {
        u64x2(0, 0)
    };

    reduction_core(s.as_ptr() as *const v128, s.len() >> 2, hi)
}

pub unsafe fn reduce_u64(s: &[u64]) -> M61 {
    let hi = if s.len() & 1 != 0 {
        let x = s[s.len() - 1];
        u64x2((x & MODULUS) + (x >> 61), 0)
    } else {
        u64x2(0, 0)
    };

    reduction_core(s.as_ptr() as *const v128, s.len() >> 1, hi)
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
