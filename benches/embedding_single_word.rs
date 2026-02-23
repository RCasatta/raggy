use std::hint::black_box;
use std::path::PathBuf;

use criterion::{Criterion, criterion_group, criterion_main};
use raggy::model::{get_embedding, load_model_and_tokenizer};

fn model_dir() -> PathBuf {
    PathBuf::from("models/all-MiniLM-L6-v2")
}

fn bench_single_word_embedding(c: &mut Criterion) {
    let model_dir = model_dir();

    if !model_dir.join("model.safetensors").exists() {
        panic!(
            "Model not found at {}. Please run ./download_model.sh to download the model.",
            model_dir.display()
        );
    }

    let (model, tokenizer) =
        load_model_and_tokenizer(&model_dir).expect("Failed to load model and tokenizer");
    let word = "Riccardo";

    c.bench_function("embedding_single_word", |bencher| {
        bencher.iter(|| {
            let embedding = get_embedding(&model, &tokenizer, black_box(word))
                .expect("Failed to generate embedding for single word");
            black_box(embedding);
        });
    });
}

criterion_group!(benches, bench_single_word_embedding);
criterion_main!(benches);
