use criterion::{Criterion, criterion_group, criterion_main};
use rsomics_combine_pvalues::{Method, combine_pvalues};
use std::hint::black_box;

fn fixture(n: usize) -> Vec<f64> {
    // A deterministic spread of p-values over (0, 1), avoiding the 0/1 edges.
    (0..n)
        .map(|i| {
            let x = (i as f64 * 0.6180339887498949).fract();
            x * (1.0 - 2e-9) + 1e-9
        })
        .collect()
}

fn bench(c: &mut Criterion) {
    let p = fixture(1_000_000);
    for (name, m) in [
        ("fisher", Method::Fisher),
        ("pearson", Method::Pearson),
        ("mudholkar_george", Method::MudholkarGeorge),
        ("tippett", Method::Tippett),
        ("stouffer", Method::Stouffer),
    ] {
        c.bench_function(name, |b| {
            b.iter(|| combine_pvalues(black_box(&p), m, None).unwrap());
        });
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
