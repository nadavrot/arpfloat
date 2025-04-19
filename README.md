
# Arbitrary-Precision Floating-Point Library &emsp; 
[![Latest Version]][crates.io] [![Docs Badge]][docs]

[Latest Version]: https://img.shields.io/crates/v/arpfloat.svg
[crates.io]: https://crates.io/crates/arpfloat
[Docs Badge]: https://docs.rs/arpfloat/badge.svg
[docs]: https://docs.rs/arpfloat

ARPFloat is an implementation of arbitrary precision
[floating point](https://en.wikipedia.org/wiki/IEEE_754) data
structures and utilities. The library can be used to emulate existing floating
point types, such as FP16, and create new floating point types. Floating point
types can scale to hundreds of digits, and perform very accurate calculations.
In ARPFloat the rounding mode is a part of the type-system, and this defines
away a number of problem that show up when using fenv.h.

`no_std` environments are supported by disabling the `std` feature. 
`python` bindings are supported by enabling the `python` feature.

### Example
```rust
  use arpfloat::Float;
  use arpfloat::FP128;

  // Create the number '5' in FP128 format.
  let n = Float::from_f64(5.).cast(FP128);

  // Use Newton-Raphson to find the square root of 5.
  let mut x = n.clone();
  for _ in 0..20 {
      x += (&n / &x)/2;
  }

  println!("fp128: {}", x);
  println!("fp64:  {}", x.as_f64());
 ```


The program above will print this output:
```console
fp128: 2.2360679774997896964091736687312763
fp64:  2.23606797749979
```

The library also provides API that exposes rounding modes, and low-level
operations.

```rust
    use arpfloat::FP128;
    use arpfloat::RoundingMode::NearestTiesToEven;
    use arpfloat::Float;

    let x = Float::from_u64(FP128, 1<<53);
    let y = Float::from_f64(1000.0).cast(FP128);

    let val = Float::mul_with_rm(&x, &y, NearestTiesToEven);
 ```

 View the internal representation of numbers:

 ```rust
    use arpfloat::Float;
    use arpfloat::FP16;

    let fp = Float::from_i64(FP16, 15);

    fp.dump(); // Prints FP[+ E=+3 M=11110000000]

    let m = fp.get_mantissa();
     m.dump(); // Prints 11110000000
```

 Control the rounding mode for type conversion:

```rust
    use arpfloat::{FP16, FP32, RoundingMode, Float};

    let x = Float::from_u64(FP32, 2649);
    let b = x.cast_with_rm(FP16, RoundingMode::Zero);
    println!("{}", b); // Prints 2648!
```

 Define new float formats and use high-precision transcendental functions:

```rust
  use arpfloat::{Float, Semantics};
  // Define a new float format with 120 bits of accuracy, and
  // dynamic range of 2^10.
  let sem = Semantics::new(10, 120);

  let pi = Float::pi(sem);
  let x = Float::exp(&pi);
  println!("e^pi = {}", x); // Prints 23.1406926327792....
```

 Floating point numbers can be converted to
 [Continued Fractions](https://en.wikipedia.org/wiki/Continued_fraction) that
 approximate the value.

 ```rust
  use arpfloat::{Float, FP256, RoundingMode};

  let ln = Float::ln2(FP256);
  println!("ln(2) = {}", ln);
  for i in 1..20 {
    let (p,q) = ln.as_fraction(i);
    println!("{}/{}", p.as_decimal(), q.as_decimal());
  }
 ```
The program above will print this output:
```console
  ln(2) = .6931471805599453094172321214581765680755001343602552.....
  0/1
  1/1
  2/3
  7/10
  9/13
  61/88
  192/277
  253/365
  445/642
  1143/1649
  1588/2291
  2731/3940
  ....
```

The [examples](examples) directory contains a few programs that demonstrate the use of this library.

### Python Bindings

The has python bindings that can be installed with 'pip install -e .'

```python
    >>> from arpfloat import Float, Semantics, FP16, BF16, FP32, fp64, pi

    >>> x = fp64(2.5).cast(FP16)
    >>> y = fp64(1.5).cast(FP16)
    >>> x + y
    4.

    >>> sem = Semantics(10, 10, "NearestTiesToEven")
    >>> sem
    Semantics { exponent: 10, precision: 10, mode: NearestTiesToEven }
    >>> Float(sem, False, 0b1000000001, 0b1100101)
    4.789062

    >>> pi(FP32)
    3.1415927
    >>> pi(FP16)
    3.140625
    >>> pi(BF16)
    3.140625
```

Arpfloat allows you to experiment with new floating point formats. For example,
Nvidia's new [FP8](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
format can be defined as:

```python
    import numpy as np
    from arpfloat import FP32, fp64, Semantics, zero

    # Create two random numpy arrays in the range [0,1)
    A0 = np.random.rand(1000000)
    A1 = np.random.rand(1000000)

    # Calculate the numpy dot product of the two arrays
    print("Using fp32 arithmetic    : ", np.dot(A0, A1))

    # Create the fp8 format (4 exponent bits, 3 mantissa bits + 1 implicit bit)
    FP8 = Semantics(4, 3 + 1, "NearestTiesToEven")

    # Convert the arrays to fp8
    A0 = [fp64(x).cast(FP8) for x in A0]
    A1 = [fp64(x).cast(FP8) for x in A1]

    dot = sum([x.cast(FP32)*y.cast(FP32) for x, y in zip(A0, A1)])
    print("Using fp8/fp32 arithmetic: ", dot)
```

### Resources

There are excellent resources out there, some of which are referenced in the code:

* Books:
    * Handbook of Floating-Point Arithmetic 2010th by Jean-Michel Muller et al.
    * Elementary Functions: Algorithms and Implementation by Jean-Michel Muller.
    * Modern Computer Arithmetic by Brent and Zimmermann.
* Papers:
    * An Accurate Elementary Mathematical Library for the IEEE Floating Point Standard, by Gal and Bachels.
    * How to print floating-point numbers accurately by Steele, White.
    * What Every Computer Scientist Should Know About Floating-Point Arithmetic by David Goldberg.
    * Fast Multiple-Precision Evaluation of Elementary Functions by Richard Brent.
    * Fast Trigonometric functions for Arbitrary Precision number by Henrik Vestermark.
* Other excellent software implementations: APFloat, RYU, libBF, newlib, musl, etc.

### License

Licensed under Apache-2.0
