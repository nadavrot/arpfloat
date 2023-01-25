use arpfloat::{BigInt, Float, RoundingMode, Semantics};

use RoundingMode::NearestTiesToEven as rme;

fn test_e() {
    let sem = Semantics::new(32, 2000, rme);
    black_box(Float::e(sem));
}

fn test_sqrt() {
    let sem = Semantics::new(32, 10000, rme);
    black_box(Float::one(sem, false).scale(1, rme).sqrt());
}

fn test_pi() {
    let sem = Semantics::new(32, 2000, rme);
    black_box(Float::pi(sem));
}

fn test_powi() {
    let a = BigInt::from_u64(1275563424);
    black_box(a.powi(11000));
}

fn test_bigint_as_dec() {
    let a = BigInt::from_u64(197123);
    black_box(a.powi(100).as_decimal());
}

fn test_bigint_div() {
    let a = BigInt::pseudorandom(1000, 12345);
    let b = BigInt::pseudorandom(500, 67890);
    black_box(a / b);
}

fn test_cos() {
    let sem = Semantics::new(32, 90, rme);
    for i in 0..100 {
        let a = Float::from_u64(sem, i).cos();
        black_box(a);
    }
}

fn test_sin() {
    let sem = Semantics::new(32, 90, rme);
    for i in 0..100 {
        let a = Float::from_u64(sem, i).sin();
        black_box(a);
    }
}

fn test_log() {
    let sem = Semantics::new(32, 100, rme);
    for i in 0..100 {
        let a = Float::from_u64(sem, i).log();
        black_box(a);
    }
}

fn test_exp() {
    let sem = Semantics::new(32, 100, rme);
    for i in 0..1000 {
        let a = Float::from_u64(sem, 100 - i).exp();
        let b = Float::from_u64(sem, i).exp();
        black_box(a + b);
    }
}

fn test_bigint_mul_1() {
    let a = BigInt::pseudorandom(1000, 98765);
    let b = BigInt::pseudorandom(1000, 43210);
    black_box(a * b);
}

fn test_bigint_mul_2() {
    let a = BigInt::pseudorandom(10, 98765);
    let b = BigInt::pseudorandom(10, 43210);
    black_box(a * b);
}

fn test_bigint_mul_3() {
    let a = BigInt::pseudorandom(100, 98765);
    let b = BigInt::pseudorandom(100, 43210);
    black_box(a * b);
}

fn test_bigint_mul_4() {
    let a = BigInt::pseudorandom(5000, 98765);
    let b = BigInt::pseudorandom(1, 43210);
    black_box(a * b);
}

fn test_bigint_div_1() {
    let a = BigInt::pseudorandom(1000, 98765);
    let b = BigInt::pseudorandom(1000, 43210);
    black_box(a / b);
}

fn test_bigint_div_2() {
    let a = BigInt::pseudorandom(1000, 98765);
    let b = BigInt::pseudorandom(1, 43210);
    black_box(a / b);
}

use criterion::{black_box, criterion_group, criterion_main, Criterion};

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("pi", |b| b.iter(test_pi));
    c.bench_function("e", |b| b.iter(test_e));
    c.bench_function("sqrt", |b| b.iter(test_sqrt));
    c.bench_function("powi", |b| b.iter(test_powi));
    c.bench_function("bigint_as_dec", |b| b.iter(test_bigint_as_dec));
    c.bench_function("bigint_div", |b| b.iter(test_bigint_div));
    c.bench_function("cos", |b| b.iter(test_cos));
    c.bench_function("sin", |b| b.iter(test_sin));
    c.bench_function("exp", |b| b.iter(test_exp));
    c.bench_function("log", |b| b.iter(test_log));
    c.bench_function("bigint_mul_1", |b| b.iter(test_bigint_mul_1));
    c.bench_function("bigint_mul_2", |b| b.iter(test_bigint_mul_2));
    c.bench_function("bigint_mul_3", |b| b.iter(test_bigint_mul_3));
    c.bench_function("bigint_mul_4", |b| b.iter(test_bigint_mul_4));
    c.bench_function("bigint_div_1", |b| b.iter(test_bigint_div_1));
    c.bench_function("bigint_div_2", |b| b.iter(test_bigint_div_2));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
