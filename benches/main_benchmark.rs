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

fn test_sin_cos() {
    let sem = Semantics::new(32, 90, rme);
    for i in 0..100 {
        let a = Float::from_u64(sem, i).sin();
        let b = Float::from_u64(sem, i).cos();
        black_box(a + b);
    }
}

use criterion::{black_box, criterion_group, criterion_main, Criterion};

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("test_pi", |b| b.iter(test_pi));
    c.bench_function("test_e", |b| b.iter(test_e));
    c.bench_function("test_sqrt", |b| b.iter(test_sqrt));
    c.bench_function("test_powi", |b| b.iter(test_powi));
    c.bench_function("test_bigint_as_dec", |b| b.iter(test_bigint_as_dec));
    c.bench_function("test_bigint_div", |b| b.iter(test_bigint_div));
    c.bench_function("test_sin_cos", |b| b.iter(test_sin_cos));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
