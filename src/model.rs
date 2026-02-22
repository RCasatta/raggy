use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Result;
use candle_core::Tensor;
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use tokenizers::{PaddingParams, Tokenizer};
use wide::f32x8;

use crate::RaggyError;

pub fn model_paths(model_dir: &Path) -> (PathBuf, PathBuf, PathBuf) {
    let model_path = model_dir.join("model.safetensors");
    let tokenizer_path = model_dir.join("tokenizer.json");
    let config_path = model_dir.join("config.json");
    (model_path, tokenizer_path, config_path)
}

pub fn verify_model_exists(model_dir: &Path) -> Result<(), RaggyError> {
    let (model_path, tokenizer_path, config_path) = model_paths(model_dir);

    if !model_path.exists() {
        return Err(RaggyError::MissingModelFile(
            model_path.display().to_string(),
        ));
    }
    if !tokenizer_path.exists() {
        return Err(RaggyError::MissingModelFile(
            tokenizer_path.display().to_string(),
        ));
    }
    if !config_path.exists() {
        return Err(RaggyError::MissingModelFile(
            config_path.display().to_string(),
        ));
    }

    Ok(())
}

pub fn load_model_and_tokenizer(model_dir: &Path) -> Result<(BertModel, Tokenizer)> {
    let (model_path, tokenizer_path, config_path) = model_paths(model_dir);

    let config_str = fs::read_to_string(&config_path)
        .map_err(|e| RaggyError::CorruptModel(format!("Failed to read config: {e}")))?;

    let config: Config = serde_json::from_str(&config_str)
        .map_err(|e| RaggyError::CorruptModel(format!("Failed to parse config: {e}")))?;

    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| RaggyError::CorruptModel(format!("Failed to load tokenizer: {e}")))?;

    let device = candle_core::Device::Cpu;

    let vb = if model_path.extension().map(|e| e == "gguf").unwrap_or(false) {
        return Err(anyhow::anyhow!(
            "GGUF format not yet supported, please use safetensors format"
        ));
    } else {
        unsafe { VarBuilder::from_mmaped_safetensors(&[&model_path], DTYPE, &device)? }
    };

    let model = BertModel::load(vb, &config)?;
    Ok((model, tokenizer))
}

pub fn get_embedding(model: &BertModel, tokenizer: &Tokenizer, text: &str) -> Result<Vec<f32>> {
    let mut tokenizer = tokenizer.clone();

    if let Some(pp) = tokenizer.get_padding_mut() {
        pp.strategy = tokenizers::PaddingStrategy::BatchLongest;
    } else {
        let pp = PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        };
        tokenizer.with_padding(Some(pp));
    }

    let tokens = tokenizer
        .encode(text, true)
        .map_err(|e| anyhow::anyhow!("Tokenization error: {e}"))?;

    let token_ids = tokens.get_ids().to_vec();
    let attention_mask = tokens.get_attention_mask().to_vec();

    let device = &model.device;
    let token_ids_tensor = Tensor::new(&token_ids[..], device)?.unsqueeze(0)?;
    let attention_mask_tensor = Tensor::new(&attention_mask[..], device)?.unsqueeze(0)?;
    let token_type_ids = token_ids_tensor.zeros_like()?;

    let embeddings = model.forward(
        &token_ids_tensor,
        &token_type_ids,
        Some(&attention_mask_tensor),
    )?;

    let attention_mask_for_pooling = attention_mask_tensor.to_dtype(DTYPE)?.unsqueeze(2)?;
    let sum_mask = attention_mask_for_pooling.sum(1)?;
    let pooled = (embeddings.broadcast_mul(&attention_mask_for_pooling)?).sum(1)?;
    let pooled = pooled.broadcast_div(&sum_mask)?.squeeze(0)?;

    let embedding_vec = pooled.to_vec1::<f32>()?;
    Ok(embedding_vec)
}

pub fn normalize_l2(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

/// This is the dot product of two vectors. Since we have normalized vector this is
/// the same as the cosine similarity.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
    // Ensure we only process up to the shortest slice length to prevent panics
    let len = a.len().min(b.len());
    let a = &a[..len];
    let b = &b[..len];

    // f32x8 acts like 8 separate f32 values processed simultaneously
    let mut sum = f32x8::ZERO;

    // Create iterators that yield exactly 8 elements at a time
    let mut chunks_a = a.chunks_exact(8);
    let mut chunks_b = b.chunks_exact(8);

    // 1. The Fast Path (SIMD processing)
    for (chunk_a, chunk_b) in chunks_a.by_ref().zip(chunks_b.by_ref()) {
        // Convert the slice chunks into [f32; 8] arrays, then into SIMD types.
        // The unwrap() is perfectly safe here because chunks_exact(8) guarantees the length.
        let va = f32x8::from(<[f32; 8]>::try_from(chunk_a).unwrap());
        let vb = f32x8::from(<[f32; 8]>::try_from(chunk_b).unwrap());

        // Multiply the 8 pairs and add them to our running totals in a single instruction
        sum += va * vb;
    }

    // Combine the 8 independent parallel accumulation lanes into a single standard float
    let mut total = sum.reduce_add();

    // 2. The Tail Processing (Scalar fallback)
    let remainder_a = chunks_a.remainder();
    let remainder_b = chunks_b.remainder();

    for (x, y) in remainder_a.iter().zip(remainder_b.iter()) {
        total += x * y;
    }

    total
}

#[cfg(test)]
mod tests {
    use super::{cosine_similarity, dot_product_simd};

    #[test]
    fn dot_product_simd_matches_scalar_dot_product() {
        // Include lengths around SIMD boundaries and mismatched inputs.
        for len in [0usize, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 384] {
            let a: Vec<f32> = (0..len).map(|i| (i as f32 * 0.37).sin()).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32 * 0.19).cos()).collect();

            let scalar = cosine_similarity(&a, &b);
            let simd = dot_product_simd(&a, &b);

            assert!(
                (scalar - simd).abs() <= 1e-5,
                "len={len}: scalar={scalar}, simd={simd}"
            );
        }

        // Explicitly verify mismatched lengths (both implementations use the shortest length).
        let a: Vec<f32> = (0..25).map(|i| (i as f32 * 0.11).sin()).collect();
        let b: Vec<f32> = (0..19).map(|i| (i as f32 * 0.23).cos()).collect();

        let scalar = cosine_similarity(&a, &b);
        let simd = dot_product_simd(&a, &b);
        assert!((scalar - simd).abs() <= 1e-5);
    }
}
