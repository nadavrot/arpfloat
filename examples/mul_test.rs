use std::env;

use arpfloat::BigInt;

///! Calculates long numbers and prints them.
///!  cargo run --example mul_test --release 100

fn main() {
    let args: Vec<String> = env::args().collect();

    let digits: usize;

    match args.len() {
        2 => match args[1].parse::<usize>() {
            Ok(x) => digits = x,
            Err(_) => {
                println!("Not an integer");
                return;
            }
        },
        _ => {
            println!("Usage: test_mul [num_digits]");
            return;
        }
    }

    let mut a = BigInt::pseudorandom(digits, 12345);
    let b = BigInt::pseudorandom(digits, 67890);

    println!("Multiplying two {}-bit numbers", digits * 64);
    a.inplace_mul(&b);
    println!("Done");
}
