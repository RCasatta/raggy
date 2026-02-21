use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Result;
use candle_core::Tensor;
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use tokenizers::{PaddingParams, Tokenizer};

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

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
