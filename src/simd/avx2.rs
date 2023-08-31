#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::definition::{final_reduction, M61, MODULUS};

#[target_feature(enable = "avx2")]
unsafe fn reduction_core(ptr: *const __m256i, mut len: usize, mut hi: __m256i) -> M61 {
    let mlo = _mm256_set1_epi64x(MODULUS as i64);
    let mhi = _mm256_set1_epi64x((MODULUS >> 12) as i64);

    // Initial reduction of high elements.
    hi = _mm256_add_epi64(_mm256_and_si256(hi, mlo), _mm256_srli_epi64::<61>(hi));

    while len > 0 {
        len -= 1;

        let lo = ptr.add(len).read_unaligned();
        let lr = _mm256_add_epi64(_mm256_and_si256(mlo, lo), _mm256_srli_epi64::<61>(lo));
        let hr = _mm256_add_epi64(
            _mm256_slli_epi64::<12>(_mm256_and_si256(hi, mhi)),
            _mm256_srli_epi64::<49>(hi),
        );
        hi = _mm256_add_epi64(lr, hr);
    }

    // One reduction step using 128-bit operands
    // halves the problem size.

    let lo = _mm256_castsi256_si128(hi);
    let mut hi = _mm256_extracti128_si256::<1>(hi);

    let mlo = _mm_set1_epi64x(MODULUS as i64);
    let mhi = _mm_set1_epi64x((MODULUS >> 6) as i64);

    let lr = _mm_add_epi64(_mm_and_si128(mlo, lo), _mm_srli_epi64::<61>(lo));
    let hr = _mm_add_epi64(
        _mm_slli_epi64::<6>(_mm_and_si128(hi, mhi)),
        _mm_srli_epi64::<55>(hi),
    );
    hi = _mm_add_epi64(lr, hr);

    // Last reduction step done using scalar operaions.

    let lo = _mm_cvtsi128_si64x(hi) as u64;
    let mut hi = _mm_extract_epi64::<1>(hi) as u64;

    hi = (lo & MODULUS) + (lo >> 61) + ((hi & (MODULUS >> 3)) << 3) + (hi >> 58);

    final_reduction(hi)
}

#[target_feature(enable = "avx2")]
pub unsafe fn reduce_u8(s: &[u8]) -> M61 {
    let hi = if s.len() & 31 != 0 {
        let mut lo = _mm_setzero_si128();
        let mut hi = _mm_setzero_si128();

        let l = s.len() & !31;
        let mut ptr = s.as_ptr().add(l);

        if s.len() & 16 != 0 {
            lo = (ptr as *const __m128i).read_unaligned();
            ptr = ptr.add(16);
        }

        let mut tmp = _mm_setzero_si128();
        for i in (0..(s.len() & 15)).rev() {
            tmp = _mm_bslli_si128::<1>(tmp);
            tmp = _mm_insert_epi8::<0>(tmp, *ptr.add(i) as i32);
        }

        if s.len() & 16 != 0 {
            hi = tmp;
        } else {
            lo = tmp;
        }

        _mm256_set_m128i(hi, lo)
    } else {
        _mm256_setzero_si256()
    };

    reduction_core(s.as_ptr() as *const __m256i, s.len() >> 5, hi)
}

#[target_feature(enable = "avx2")]
pub unsafe fn reduce_u16(s: &[u16]) -> M61 {
    let hi = if s.len() & 15 != 0 {
        let mut lo = _mm_setzero_si128();
        let mut hi = _mm_setzero_si128();

        let l = s.len() & !15;
        let mut ptr = s.as_ptr().add(l);

        if s.len() & 8 != 0 {
            lo = (ptr as *const __m128i).read_unaligned();
            ptr = ptr.add(8);
        }

        let mut tmp = _mm_setzero_si128();
        for i in (0..(s.len() & 7)).rev() {
            tmp = _mm_bslli_si128::<2>(tmp);
            tmp = _mm_insert_epi16::<0>(tmp, *ptr.add(i) as i32);
        }

        if s.len() & 8 != 0 {
            hi = tmp;
        } else {
            lo = tmp;
        }

        _mm256_set_m128i(hi, lo)
    } else {
        _mm256_setzero_si256()
    };

    reduction_core(s.as_ptr() as *const __m256i, s.len() >> 4, hi)
}

#[target_feature(enable = "avx2")]
pub unsafe fn reduce_u32(s: &[u32]) -> M61 {
    let hi = if s.len() & 7 != 0 {
        let mut arr = [0; 8];
        let l = s.len() & !7;

        for i in l..s.len() {
            arr[i - l] = *s.get_unchecked(i);
        }

        (arr.as_ptr() as *const __m256i).read_unaligned()
    } else {
        _mm256_setzero_si256()
    };

    reduction_core(s.as_ptr() as *const __m256i, s.len() >> 3, hi)
}

#[target_feature(enable = "avx2")]
pub unsafe fn reduce_u64(s: &[u64]) -> M61 {
    let hi = if s.len() & 3 != 0 {
        let mut arr = [0; 4];
        let l = s.len() & !3;

        for i in l..s.len() {
            arr[i - l] = *s.get_unchecked(i);
        }

        (arr.as_ptr() as *const __m256i).read_unaligned()
    } else {
        _mm256_setzero_si256()
    };

    reduction_core(s.as_ptr() as *const __m256i, s.len() >> 2, hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduce_u8_max() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }

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
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }

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
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }

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
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }

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
            if !std::arch::is_x86_feature_detected!("avx2") {
                return true;
            }

            let expected = crate::fallback::reduce_u8(&slice);
            let actual = unsafe { reduce_u8(&slice) };
            expected == actual
        }

        fn reduce_u16_correct(slice: Vec<u16>) -> bool {
            if !std::arch::is_x86_feature_detected!("avx2") {
                return true;
            }

            let expected = crate::fallback::reduce_u16(&slice);
            let actual = unsafe { reduce_u16(&slice) };
            expected == actual
        }

        fn reduce_u32_correct(slice: Vec<u32>) -> bool {
            if !std::arch::is_x86_feature_detected!("avx2") {
                return true;
            }

            let expected = crate::fallback::reduce_u32(&slice);
            let actual = unsafe { reduce_u32(&slice) };
            expected == actual
        }

        fn reduce_u64_correct(slice: Vec<u64>) -> bool {
            if !std::arch::is_x86_feature_detected!("avx2") {
                return true;
            }

            let expected = crate::fallback::reduce_u64(&slice);
            let actual = unsafe { reduce_u64(&slice) };
            expected == actual
        }
    }
}
