use arpfloat::{Float, RoundingMode, Semantics};

///! Calculates long numbers and prints them.
///!  cargo run --example print_bench --release

fn main() {
    use RoundingMode::NearestTiesToEven as nte;
    let sem = Semantics::new(32, 5000, nte);
    let val = Float::e(sem);
    println!("F64: {}", val.as_f64());
    println!("FP*: {}", val);
}
