//! This module leverages the `cpuid`-instruction to dynamically select
//! the used function at runtime based on the available target features.
//!
//! The implementation is simple:
//! 1. Dereference a static variable to obtain its value.
//! 2. Cast the value into a function pointer and call it.
//!
//! The static variable still needs to be initialized with the appropriate
//! version of the function. For this, the initial value of the variable
//! is an initializer variant of the function that queries the available
//! target features and overwrites the variable based on the result,
//! before calling the appropriate implementation.
//!
//! This means that any subsequent calls immediately use
//! the appropriate version.

/// Obtain information about the available target features by
/// using the `is_x86_feature_detected` macro provided by
/// the Rust standard library.
#[cfg(feature = "std")]
mod detection {
    #[inline(always)]
    pub(crate) fn has_avx2() -> bool {
        std::arch::is_x86_feature_detected!("avx2")
    }

    #[cfg(feature = "nightly")]
    #[inline(always)]
    pub(crate) fn has_avx512f() -> bool {
        std::arch::is_x86_feature_detected!("avx512f")
    }
}

/// Obtain information about the available target features by
/// using the `cpuid` intruction. Used in no-std builds.
#[cfg(not(feature = "std"))]
mod detection {
    //__cpuid_count
    #[cfg(target_arch = "x86")]
    use core::arch::x86::{__cpuid_count as cpuid, _xgetbv};
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{__cpuid_count as cpuid, _xgetbv};

    use core::sync::atomic::{AtomicU8, Ordering};

    const UNINIT: u8 = 0;
    const SSE2: u8 = 1;
    const AVX2: u8 = 2;
    const AVX512: u8 = 3;

    static CACHE: AtomicU8 = AtomicU8::new(UNINIT);

    fn get_features() -> u8 {
        let content = CACHE.load(Ordering::Relaxed);
        if content == UNINIT {
            let mut content = SSE2;

            // SAFETY: This will only run on systems with support
            // SSE2, which in turn implies support for CPUID.
            unsafe {
                // Testing for AVX2 or AVX512 requires the following
                // 1. Support for XSAVE by the hardware.
                // 2. Support for XSAVE by the operating system.
                // 3. Registers are enabled when checking XCR0.
                // 4. Support for the instructions.
                if cpuid(0x01, 0x0).ecx & (0b11 << 26) != 0 {
                    // Query the extended control register.
                    let xcr0 = _xgetbv(0);
                    // Query extended features.
                    let ebx = cpuid(0x07, 0x0).ebx;

                    // Support for AVX2
                    if xcr0 & 6 == 6 && ebx & (1 << 5) != 0 {
                        content = AVX2;
                    }

                    // Support for AVX512F
                    if xcr0 & 224 == 224 && ebx & (1 << 16) != 0 {
                        content = AVX512;
                    }
                };
            }

            CACHE.store(content, Ordering::Relaxed);
            content
        } else {
            content
        }
    }

    #[inline(always)]
    pub(crate) fn has_avx2() -> bool {
        get_features() >= AVX2
    }

    #[cfg(feature = "nightly")]
    #[inline(always)]
    pub(crate) fn has_avx512f() -> bool {
        get_features() >= AVX512
    }
}

use detection::*;

use super::avx2;
#[cfg(feature = "nightly")]
use super::avx512;
#[cfg(not(target_feature = "avx2"))]
use super::sse2;

use core::mem::transmute;
use core::sync::atomic::{AtomicPtr, Ordering};

use crate::definition::M61;

// These variables contain fuction pointers to the impementations.

static FUNC8: AtomicPtr<()> = AtomicPtr::new(reduce_u8_init as *mut ());
static FUNC16: AtomicPtr<()> = AtomicPtr::new(reduce_u16_init as *mut ());
static FUNC32: AtomicPtr<()> = AtomicPtr::new(reduce_u32_init as *mut ());
static FUNC64: AtomicPtr<()> = AtomicPtr::new(reduce_u64_init as *mut ());

/// Writes the appropiate versions of the functions into the
/// static variables.
unsafe fn select() {
    #[cfg(feature = "nightly")]
    if has_avx512f() {
        FUNC8.store(avx512::reduce_u8 as *mut (), Ordering::Relaxed);
        FUNC16.store(avx512::reduce_u16 as *mut (), Ordering::Relaxed);
        FUNC32.store(avx512::reduce_u32 as *mut (), Ordering::Relaxed);
        FUNC64.store(avx512::reduce_u64 as *mut (), Ordering::Relaxed);
    }

    if has_avx2() {
        FUNC8.store(avx2::reduce_u8 as *mut (), Ordering::Relaxed);
        FUNC16.store(avx2::reduce_u16 as *mut (), Ordering::Relaxed);
        FUNC32.store(avx2::reduce_u32 as *mut (), Ordering::Relaxed);
        FUNC64.store(avx2::reduce_u64 as *mut (), Ordering::Relaxed);
    }

    #[cfg(not(target_feature = "avx2"))]
    {
        FUNC8.store(sse2::reduce_u8 as *mut (), Ordering::Relaxed);
        FUNC16.store(sse2::reduce_u16 as *mut (), Ordering::Relaxed);
        FUNC32.store(sse2::reduce_u32 as *mut (), Ordering::Relaxed);
        FUNC64.store(sse2::reduce_u64 as *mut (), Ordering::Relaxed);
    }
}

// Helper types used to keep calls to `transmute` clean.

type T8 = unsafe fn(&[u8]) -> M61;
type T16 = unsafe fn(&[u16]) -> M61;
type T32 = unsafe fn(&[u32]) -> M61;
type T64 = unsafe fn(&[u64]) -> M61;

// Definition of the starting values of the static variables.

unsafe fn reduce_u8_init(s: &[u8]) -> M61 {
    select();
    let func = transmute::<_, T8>(FUNC8.load(Ordering::Relaxed));
    func(s)
}

unsafe fn reduce_u16_init(s: &[u16]) -> M61 {
    select();
    let func = transmute::<_, T16>(FUNC16.load(Ordering::Relaxed));
    func(s)
}

unsafe fn reduce_u32_init(s: &[u32]) -> M61 {
    select();
    let func = transmute::<_, T32>(FUNC32.load(Ordering::Relaxed));
    func(s)
}

unsafe fn reduce_u64_init(s: &[u64]) -> M61 {
    select();
    let func = transmute::<_, T64>(FUNC64.load(Ordering::Relaxed));
    func(s)
}

// Definition of the exports.

pub unsafe fn reduce_u8(s: &[u8]) -> M61 {
    let func = transmute::<_, T8>(FUNC8.load(Ordering::Relaxed));
    func(s)
}

pub unsafe fn reduce_u16(s: &[u16]) -> M61 {
    let func = transmute::<_, T16>(FUNC16.load(Ordering::Relaxed));
    func(s)
}

pub unsafe fn reduce_u32(s: &[u32]) -> M61 {
    let func = transmute::<_, T32>(FUNC32.load(Ordering::Relaxed));
    func(s)
}

pub unsafe fn reduce_u64(s: &[u64]) -> M61 {
    let func = transmute::<_, T64>(FUNC64.load(Ordering::Relaxed));
    func(s)
}
