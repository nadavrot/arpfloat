//!
//! ARPFloat is an implementation of arbitrary precision
//![floating point](https://en.wikipedia.org/wiki/IEEE_754) data
//!structures and utilities. The library can be used to emulate floating point
//!operation, in software, or create new floating point data types.

//!### Example
//!```
//!  use arpfloat::Float;
//!  use arpfloat::new_float_type;
//!
//!  // Create a new type: 15 bits exponent, 112 significand.
//!  type FP128 = new_float_type!(15, 112);
//!
//!  // Use Newton-Raphson to find the square root of 5.
//!  let n = FP128::from_u64(5);
//!
//!  let two = FP128::from_u64(2);
//!  let mut x = n.clone();
//!
//!  for _ in 0..20 {
//!      x = (x.clone() + (&n / &x))/two.clone();
//!  }
//!
//!  println!("fp128: {}", x);
//!  println!("fp64:  {}", x.as_f64());
//! ```
//!
//!
//!The program above will print this output:
//!```console
//!fp128: 2.2360679774997896964091736687312763
//!fp64:  2.23606797749979
//!```
//!
//!The library also provides API that exposes rounding modes, and low-level
//!operations.
//!
//!```
//!    use arpfloat::{FP16, FP128, RoundingMode};
//!
//!    let x = FP128::from_u64(1<<53);
//!    let y = FP128::from_f64(1000.0);
//!    let val = FP128::mul_with_rm(&x, &y, RoundingMode::NearestTiesToEven);
//! ```
//!
//! View the internal representation of numbers:
//! ```
//!    use arpfloat::{FP16, FP128, RoundingMode};
//!
//!    let fp = FP16::from_i64(15);
//!    let m = fp.get_mantissa();
//!
//!    // Prints FP[+ E=+3 M=11110000000]
//!    fp.dump();
//!```
//!
//! Control the rounding mode for type conversion:
//!```
//!    use arpfloat::{FP16, FP32, RoundingMode};
//!    let x = FP32::from_u64(2649);
//!    let b : FP16 = x.cast_with_rm(RoundingMode::Zero);
//!    println!("{}", b); // Prints 2648!
//!```

#![no_std]

#[cfg(feature = "std")]
extern crate std;

/// Creates a new Float<> type with a specific number of bits for the exponent and mantissa.
/// The macros selects the appropriate size for the underlying storage.
#[macro_export]
macro_rules! new_float_type {
    ($exponent:expr, $mantissa:expr) => {
        // Allocate twice as many bits for the mantissa, to allow to perform
        // div/mul operations that require shifting of the mantissa to the left.
        Float<$exponent, $mantissa, {($mantissa * 2) / 64 + 1}>
    };
}

mod arithmetic;
mod bigint;
mod cast;
mod float;
mod functions;
mod string;
mod utils;

pub use self::bigint::BigInt;
pub use self::float::Float;
pub use self::float::RoundingMode;
pub use self::float::{FP128, FP16, FP256, FP32, FP64};
