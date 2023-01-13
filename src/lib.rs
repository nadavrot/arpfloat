//!
//! ARPFloat is an implementation of arbitrary precision
//![floating point](https://en.wikipedia.org/wiki/IEEE_754) data
//!structures and utilities. The library can be used to emulate floating point
//!operation, in software, or create new floating point data types.

//!### Example
//!```
//!  use arpfloat::Float;
//!  use arpfloat::FP128;
//!
//!  // Create the number '5' in FP128 format.
//!  let n = Float::from_f64(5.).cast(FP128);
//!
//!  // Use Newton-Raphson to find the square root of 5.
//!  let mut x = n.clone();
//!  for _ in 0..20 {
//!      x += (&n / &x)/2;
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
//!    use arpfloat::FP128;
//!    use arpfloat::RoundingMode::NearestTiesToEven;
//!    use arpfloat::Float;
//!
//!    let x = Float::from_u64(FP128, 1<<53);
//!    let y = Float::from_f64(1000.0).cast(FP128);
//!
//!    let val = Float::mul_with_rm(&x, &y, NearestTiesToEven);
//! ```
//!
//! View the internal representation of numbers:
//! ```
//!    use arpfloat::Float;
//!    use arpfloat::FP16;
//!
//!    let fp = Float::from_i64(FP16, 15);
//!
//!    fp.dump(); // Prints FP[+ E=+3 M=11110000000]
//!
//!    let m = fp.get_mantissa();
//!     m.dump(); // Prints 11110000000
//!```
//!
//! Control the rounding mode for type conversion:
//!```
//!    use arpfloat::{FP16, FP32, RoundingMode, Float};
//!
//!    let x = Float::from_u64(FP32, 2649);
//!    let b = x.cast_with_rm(FP16, RoundingMode::Zero);
//!    println!("{}", b); // Prints 2648!
//!```
//!
//! Define new float formats and use high-precision transcendental functions:
//!```
//!  use arpfloat::{Float, Semantics, RoundingMode};
//!  // Define a new float format with 120 bits of accuracy, and dynamic range
//!  // of 2^10.
//!  let sem = Semantics::new(10, 120, RoundingMode::NearestTiesToEven);
//!
//!  let pi = Float::pi(sem);
//!  let x = Float::exp(&pi);
//!  println!("e^pi = {}", x); // Prints 23.1406926327792....
//!```

#![no_std]

#[cfg(feature = "std")]
extern crate std;

mod arithmetic;
mod bigint;
mod cast;
mod float;
mod functions;
mod string;
mod utils;

pub use self::float::Float;
pub use self::float::RoundingMode;
pub use self::float::Semantics;
pub use self::float::{FP128, FP16, FP256, FP32, FP64};
