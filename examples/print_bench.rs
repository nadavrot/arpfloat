use arpfloat::{Float, Semantics};

///! Calculates long numbers and prints them.
///!  cargo run --example print_bench --release

fn main() {
    let sem = Semantics::new(32, 5000);
    let val = Float::e(sem);
    println!("F64: {}", val.as_f64());
    println!("FP*: {}", val);
}
