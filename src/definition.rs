//! Definition of the [`M61`] type as well as basic operations on it.

use core::fmt;
use core::iter;
use core::ops;

/// The modulus on which arithmetic is performed.
/// Also functions as a bitmask for calculating
/// digit sums base `2^61`.
pub(crate) const MODULUS: u64 = (1 << 61) - 1;

/// When calculating the reduction of an arbitary precision integer
/// using a digit sum, the sum itself must be reduced aswell.
/// This function performs this reduction, assuming that
/// are themselves partially reduced, meaning `x <= 2 * (2^61 - 1)`.
#[inline(always)]
pub(crate) fn final_reduction(mut x: u64) -> M61 {
    if x >= MODULUS {
        x -= MODULUS;
    }

    if x >= MODULUS {
        M61(x - MODULUS)
    } else {
        M61(x)
    }
}

/// A 64-bit integer in which arithmetic is performed modulp `2^61 - 1`.
#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct M61(pub(crate) u64);

impl M61 {
    /// Returns the contained value.
    #[inline(always)]
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Helper macro for the quick generation
/// of formatting trait implementations.
macro_rules! make_fmt_impl {
    ($trait:ident) => {
        impl fmt::$trait for M61 {
            #[inline(always)]
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                <u64 as fmt::$trait>::fmt(&self.0, f)
            }
        }
    };
}

make_fmt_impl!(Display);
make_fmt_impl!(Debug);
make_fmt_impl!(LowerExp);
make_fmt_impl!(UpperExp);
make_fmt_impl!(LowerHex);
make_fmt_impl!(UpperHex);
make_fmt_impl!(Octal);
make_fmt_impl!(Binary);

/// Helper macro for generation of [`From`] implementations
/// where the numerical bounds of the source type are smaller
/// than the modulus.
macro_rules! make_trivial_from {
    ($type:ty) => {
        impl From<$type> for M61 {
            #[inline(always)]
            #[must_use]
            fn from(value: $type) -> Self {
                // rustc warns us against this seemingly
                // useless comparison whenever the argument is
                // an unsigned type. The macro works properly for those
                // types, which means that the warning can be disabled.
                #[allow(unused_comparisons)]
                if value < 0 {
                    Self((value as i64 + MODULUS as i64) as u64)
                } else {
                    Self(value as u64)
                }
            }
        }
    };
}

make_trivial_from!(u8);
make_trivial_from!(u16);
make_trivial_from!(u32);
#[cfg(not(target_pointer_width = "64"))]
make_trivial_from!(usize);

#[cfg(target_pointer_width = "64")]
impl From<usize> for M61 {
    #[inline(always)]
    #[must_use]
    fn from(value: usize) -> Self {
        Self::from(value as u64)
    }
}

make_trivial_from!(i8);
make_trivial_from!(i16);
make_trivial_from!(i32);
#[cfg(not(target_pointer_width = "64"))]
make_trivial_from!(isize);

#[cfg(target_pointer_width = "64")]
impl From<isize> for M61 {
    #[inline(always)]
    #[must_use]
    fn from(value: isize) -> Self {
        Self::from(value as i64)
    }
}

impl From<u64> for M61 {
    #[inline]
    #[must_use]
    fn from(value: u64) -> Self {
        let tmp = (value & MODULUS) + (value >> 61);
        if tmp >= MODULUS {
            Self(tmp - MODULUS)
        } else {
            Self(tmp)
        }
    }
}

impl From<i64> for M61 {
    #[inline]
    #[must_use]
    fn from(mut value: i64) -> Self {
        if value < 0 {
            value = value.wrapping_add(4 * MODULUS as i64);
        }
        if value < 0 {
            value = value.wrapping_add(MODULUS as i64);
        }

        Self::from(value as u64)
    }
}

impl From<u128> for M61 {
    #[inline]
    #[must_use]
    fn from(value: u128) -> Self {
        let mut x = value as u64 & MODULUS;
        x += (value >> 61) as u64 & MODULUS;
        x += (value >> 122) as u64;
        Self::from(x)
    }
}

impl From<i128> for M61 {
    #[inline]
    #[must_use]
    fn from(mut value: i128) -> Self {
        while value < 0 {
            value += 16 * ((1 << 122) - 1);
        }

        Self::from(value as u128)
    }
}

/// Helper macro for the quick implementation
/// of arithmetic operators.
macro_rules! make_arith_impl {
    ($trait:ident, $trait_assign:ident, $func:ident, $func_assign:ident, $op:tt, $impl:expr) => {
        impl ops::$trait for M61 {
            type Output = Self;

            #[inline]
            #[must_use]
            fn $func(self, rhs: Self) -> Self::Output {
                #[allow(clippy::redundant_closure_call)]
                Self($impl(self.0, rhs.0))
            }
        }

        impl<'a> ops::$trait<&'a M61> for M61 {
            type Output = Self;

            #[inline(always)]
            #[must_use]
            fn $func(self, rhs: &Self) -> Self::Output {
                self $op *rhs
            }
        }

        impl ops::$trait_assign for M61 {
            #[inline(always)]
            fn $func_assign(&mut self, rhs: Self) {
                *self = *self $op rhs
            }
        }

        impl<'a> ops::$trait_assign<&'a M61> for M61 {
            #[inline(always)]
            fn $func_assign(&mut self, rhs: &Self) {
                *self = *self $op rhs
            }
        }
    };
}

make_arith_impl!(Add, AddAssign, add, add_assign, +, |a, b| {
    let x = a + b;
    if x >= MODULUS {
        x - MODULUS
    } else {
        x
    }
});
make_arith_impl!(Sub, SubAssign, sub, sub_assign, -, |a, b| {
    a + MODULUS - b
});
make_arith_impl!(Mul, MulAssign, mul, mul_assign, *, |a, b| {
    let x = a as u128 * b as u128;
    let mut hi = (x >> 61) as u64;
    let mut lo = (x as u64) & MODULUS;
    lo = lo.wrapping_add(hi);
    hi = lo.wrapping_sub(MODULUS);
    if lo < MODULUS {
        lo
    } else {
        hi
    }
});
make_arith_impl!(Div, DivAssign, div, div_assign, /, |a, b| {
    if b == 0 {
        panic!("attempt to divide by zero");
    }

    // Calculate the multiplicative inverse
    // using the extended Euclidean algorithm.
    // (https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm)

    let mut r0 = MODULUS;
    let mut r1 = b;
    let mut s0 = 1i64;
    let mut s1 = 0i64;
    let mut t0 = 0i64;
    let mut t1 = 1i64;

    while r1 != 0 {
        let (q, rn) = (r0 / r1, r0 % r1);
        let sn = s0 - q as i64 * s1;
        let tn = t0 - q as i64 * t1;

        r0 = r1;
        r1 = rn;
        s0 = s1;
        s1 = sn;
        t0 = t1;
        t1 = tn;
    }

    debug_assert_eq!(MODULUS as i128 * s0 as i128 + b as i128 * t0 as i128, 1);

    (Self(a) * Self::from(t0)).0
});
//make_arith_impl!(Rem, RemAssign, rem, rem_assign, %, |a, b| {
//    a % b
//});

impl iter::Sum for M61 {
    #[inline(always)]
    #[must_use]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self(0), |a, b| a + b)
    }
}

impl<'a> iter::Sum<&'a M61> for M61 {
    #[inline(always)]
    #[must_use]
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self(0), |a, b| a + b)
    }
}

impl iter::Product for M61 {
    #[inline(always)]
    #[must_use]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self(1), |a, b| a * b)
    }
}

impl<'a> iter::Product<&'a M61> for M61 {
    #[inline(always)]
    #[must_use]
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self(1), |a, b| a * b)
    }
}
