// cargo run --example calc_pi --release

use arpfloat::FP256;

fn main() {
    // Calculate the value of PI.
    //https://en.wikipedia.org/wiki/Chudnovsky_algorithm
    type FP = FP256;
    let iterations = 700;

    // Constants:
    let c1 = FP::from_f64(10005f64.sqrt());
    let c2 = FP::from_u64(545140134);
    let c3 = FP::from_i64(-262537412640768000);
    let c16 = FP::from_u64(16);
    let c12 = FP::from_u64(12);

    // Initial state.
    let mut k = FP::from_u64(6);
    let mut m = FP::from_u64(1);
    let mut l = FP::from_u64(13591409);
    let mut x = FP::from_u64(1);
    let mut s = FP::from_u64(13591409);

    for q in 1..iterations + 1 {
        let k3 = FP::from_u64(q * q * q);
        m = (k3 - (k * c16)) * m / k3;
        l = l + c2;
        x = x * c3;
        s = s + (m * l / x);
        k = k + c12;
    }

    let pi = FP::from_u64(426880) * (c1 / s);
    println!("pi = {}", pi.as_f64());
}
