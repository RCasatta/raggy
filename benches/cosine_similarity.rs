use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use raggy::model::{cosine_similarity, dot_product_simd};

fn bench_cosine_similarity(c: &mut Criterion) {
    let a: Vec<f32> = (0..384).map(|i| (i as f32).sin()).collect();
    let b: Vec<f32> = (0..384).map(|i| (i as f32).cos()).collect();

    c.bench_function("cosine_similarity_384d", |bencher| {
        bencher.iter(|| cosine_similarity(black_box(&a), black_box(&b)));
    });

    c.bench_function("dot_product_simd_384d", |bencher| {
        bencher.iter(|| dot_product_simd(black_box(&a), black_box(&b)));
    });
}

criterion_group!(benches, bench_cosine_similarity);
criterion_main!(benches);
