use arpfloat::{FP256, Float};

///! Calculate the value of PI using the Chudnovsky_algorithm.
///!  cargo run --example calc_pi --release


fn main() {
    // https://en.wikipedia.org/wiki/Chudnovsky_algorithm
    let iterations = 5;

    // Constants:
    let c1 = Float::from_u64(FP256, 10005).sqrt();
    let c2 = Float::from_u64(FP256, 545140134);
    let c3 = Float::from_i64(FP256, -262537412640768000);
    let c16 = Float::from_u64(FP256, 16);
    let c12 = Float::from_u64(FP256, 12);

    // Initial state.
    let mut kc = Float::from_u64(FP256, 6);
    let mut m = Float::from_u64(FP256, 1);
    let mut l = Float::from_u64(FP256, 13591409);
    let mut x = Float::from_u64(FP256, 1);
    let mut s = Float::from_u64(FP256, 13591409);

    for q in 1..iterations + 1 {
        let q3 = Float::from_u64(FP256, q * q * q);
        let k3 = &kc * &(&kc * &kc);
        m = (k3 - (&kc * &c16)) * m / q3;
        l = &l + &c2;
        x = &x * &c3;
        s = s + (&(&m * &l) / &x);
        kc = &kc + &c12;
    }
    let pi = Float::from_u64(FP256, 426880) * (c1 / s);
    println!("pi = {}", pi);
    assert_eq!(pi.as_f64(), std::f64::consts::PI);
}
