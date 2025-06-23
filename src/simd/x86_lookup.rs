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
//! This means that any subsequent calls immediately use the appropriate version.

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
        if cfg!(not(target_feature = "sse2")) {
            return 0;
        }

        let content = CACHE.load(Ordering::Relaxed);
        if content == UNINIT {
            let mut content = SSE2;

            // SAFETY: This will only run on systems with support
            // for SSE2, which in turn implies support for CPUID.
            unsafe {
                // A precondition for AVX2 or AVX512 support is
                // the presense of the XSAVE instruction _and_ its
                // use by the operating system. The former can be checked
                // by testing bit 26 of the ecx output of CPUID leaf 1, the
                // latter by testing bit 27 of the same output.
                if cpuid(0x01, 0x0).ecx & (0b11 << 26) != 0 {
                    // Query the extended control register. It contains information
                    // about which registers are getting saved. If bit 1 is set then SSE
                    // registers are getting saved. Bit 2 indicates the same thing about
                    // the upper halves introduced with AVX. Bits 5, 6, and 7 are related
                    // to the AVX512 registers.
                    let xcr0 = _xgetbv(0);

                    // Query extended feature flags. Bit 5 indicates support
                    // for AVX2, while bit 16 indicates support for AVX512F.
                    let ebx = cpuid(0x07, 0x0).ebx;

                    // Check support for AVX2.
                    if xcr0 & 6 == 6 && ebx & (1 << 5) != 0 {
                        content = AVX2;
                    }

                    // Check support for AVX512F.
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

// These variables contain the function pointers to the impementations.

static FUNC8: AtomicPtr<()> = AtomicPtr::new(reduce_u8_init as *mut ());
static FUNC16: AtomicPtr<()> = AtomicPtr::new(reduce_u16_init as *mut ());
static FUNC32: AtomicPtr<()> = AtomicPtr::new(reduce_u32_init as *mut ());
static FUNC64: AtomicPtr<()> = AtomicPtr::new(reduce_u64_init as *mut ());

/// Writes the appropiate versions of the functions into the static variables.
unsafe fn select() {
    #[cfg(not(target_feature = "avx2"))]
    {
        FUNC8.store(sse2::reduce_u8 as *mut (), Ordering::Relaxed);
        FUNC16.store(sse2::reduce_u16 as *mut (), Ordering::Relaxed);
        FUNC32.store(sse2::reduce_u32 as *mut (), Ordering::Relaxed);
        FUNC64.store(sse2::reduce_u64 as *mut (), Ordering::Relaxed);
    }

    if has_avx2() {
        FUNC8.store(avx2::reduce_u8 as *mut (), Ordering::Relaxed);
        FUNC16.store(avx2::reduce_u16 as *mut (), Ordering::Relaxed);
        FUNC32.store(avx2::reduce_u32 as *mut (), Ordering::Relaxed);
        FUNC64.store(avx2::reduce_u64 as *mut (), Ordering::Relaxed);
    }

    #[cfg(feature = "nightly")]
    if has_avx512f() {
        FUNC8.store(avx512::reduce_u8 as *mut (), Ordering::Relaxed);
        FUNC16.store(avx512::reduce_u16 as *mut (), Ordering::Relaxed);
        FUNC32.store(avx512::reduce_u32 as *mut (), Ordering::Relaxed);
        FUNC64.store(avx512::reduce_u64 as *mut (), Ordering::Relaxed);
    }
}

/// A helper macro for creating the implementations.
macro_rules! make_implementation {
    ($name:ident, $init_name:ident, $atomic_ptr:ident, $type:ty) => {
        // Used as a starting value for the static variable.
        unsafe fn $init_name(s: &[$type]) -> M61 {
            select();
            let func = transmute::<*mut (), unsafe fn(&[$type]) -> M61>(
                $atomic_ptr.load(Ordering::Relaxed),
            );
            func(s)
        }

        // Definition of the exported function.
        #[inline]
        pub(crate) unsafe fn $name(s: &[$type]) -> M61 {
            let func = transmute::<*mut (), unsafe fn(&[$type]) -> M61>(
                $atomic_ptr.load(Ordering::Relaxed),
            );
            func(s)
        }
    };
}

make_implementation!(reduce_u8, reduce_u8_init, FUNC8, u8);
make_implementation!(reduce_u16, reduce_u16_init, FUNC16, u16);
make_implementation!(reduce_u32, reduce_u32_init, FUNC32, u32);
make_implementation!(reduce_u64, reduce_u64_init, FUNC64, u64);
