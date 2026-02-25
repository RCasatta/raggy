use std::path::PathBuf;
use std::sync::Arc;

use raggy::model::{cosine_similarity, get_embedding, load_model_and_tokenizer, normalize_l2};
use raggy::{Args, RaggyState};

fn model_dir() -> PathBuf {
    PathBuf::from("models/all-MiniLM-L6-v2")
}

fn init_test_tracing_from_env() {
    let Some(level) = std::env::var("RUST_LOG").ok() else {
        return;
    };

    let max_level = match level.trim().to_ascii_lowercase().as_str() {
        "trace" => tracing::Level::TRACE,
        "debug" => tracing::Level::DEBUG,
        "info" => tracing::Level::INFO,
        "warn" => tracing::Level::WARN,
        "error" => tracing::Level::ERROR,
        _ => tracing::Level::INFO,
    };

    let _ = tracing_subscriber::fmt()
        .with_max_level(max_level)
        .with_test_writer()
        .try_init();
}

#[test]
fn test_embedding_generation() {
    let model_dir = model_dir();

    if !model_dir.join("model.safetensors").exists() {
        panic!(
            "Model not found at {}. Please run ./download_model.sh to download the model.",
            model_dir.display()
        );
    }

    let (model, tokenizer) =
        load_model_and_tokenizer(&model_dir).expect("Failed to load model and tokenizer");

    let test_text = "This is a test sentence for embedding generation.";

    let embedding =
        get_embedding(&model, &tokenizer, test_text).expect("Failed to generate embedding");

    assert!(!embedding.is_empty(), "Embedding should not be empty");
    assert!(
        embedding.iter().all(|&x| x.is_finite()),
        "All embedding values should be finite"
    );

    let test_text_2 = "Another different sentence.";
    let embedding_2 = get_embedding(&model, &tokenizer, test_text_2)
        .expect("Failed to generate embedding for second text");

    let embedding = normalize_l2(&embedding);
    let embedding_2 = normalize_l2(&embedding_2);

    let similarity = cosine_similarity(&embedding, &embedding_2);

    assert!(
        (-1.0..=1.0).contains(&similarity),
        "Cosine similarity should be between -1 and 1"
    );

    assert_ne!(
        similarity, 1.0,
        "Different sentences should not have perfect similarity"
    );
}

#[test]
fn test_mcp_integration() {
    init_test_tracing_from_env();

    let model_dir = model_dir();

    if !model_dir.join("model.safetensors").exists() {
        panic!(
            "Model not found at {}. Please run ./download_model.sh to download the model.",
            model_dir.display()
        );
    }

    let args = Args {
        model_dir,
        dir: PathBuf::from("."),
        exclude: Vec::new(),
        chunk_size: 512,
        chunk_overlap: 50,
    };

    let state = Arc::new(RaggyState::new(args));

    state
        .load_model_and_tokenizer()
        .expect("Failed to load model");

    state.index_files().expect("Failed to index files");

    tracing::info!("Making query");
    let result = state.query("How do I launch the project with nix?", 3);

    assert!(result.is_ok(), "Query should succeed");

    let response = result.unwrap();
    assert!(
        !response.results.is_empty(),
        "Should return at least one result"
    );

    tracing::info!(
        "Query returned {} results, first result: {:?}",
        response.results.len(),
        response.results.first()
    );
}

#[test]
fn test_most_characteristic_word_for_phrase() {
    let model_dir = model_dir();

    if !model_dir.join("model.safetensors").exists() {
        panic!(
            "Model not found at {}. Please run ./download_model.sh to download the model.",
            model_dir.display()
        );
    }

    let (model, tokenizer) =
        load_model_and_tokenizer(&model_dir).expect("Failed to load model and tokenizer");

    let phrase = "Who is the person called Riccardo Casatta?";
    let phrase_embedding = get_embedding(&model, &tokenizer, phrase)
        .map(|v| normalize_l2(&v))
        .expect("Failed to generate embedding for phrase");

    let words: Vec<&str> = phrase
        .split_whitespace()
        .map(|word| word.trim_matches(|c: char| !c.is_alphanumeric()).trim())
        .filter(|word| !word.is_empty())
        .collect();

    let (best_word, best_similarity) = words
        .iter()
        .map(|word| {
            let word_embedding = get_embedding(&model, &tokenizer, word)
                .map(|v| normalize_l2(&v))
                .expect("Failed to generate embedding for word");
            let similarity = cosine_similarity(&phrase_embedding, &word_embedding);
            (*word, similarity)
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .expect("Expected at least one word");

    assert!(
        (-1.0..=1.0).contains(&best_similarity),
        "Cosine similarity should be between -1 and 1"
    );
    assert!(
        ["Riccardo", "Casatta"].contains(&best_word),
        "Expected one of the proper names to be the most characteristic word, got '{best_word}' with similarity {best_similarity}"
    );
}
