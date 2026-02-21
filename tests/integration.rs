use std::path::PathBuf;
use std::sync::Arc;

use raggy::model::{cosine_similarity, get_embedding, load_model_and_tokenizer, normalize_l2};
use raggy::{Args, RaggyState};

fn model_dir() -> PathBuf {
    PathBuf::from("models/all-MiniLM-L6-v2")
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
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .try_init();

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
        extensions: ".md".to_string(),
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
