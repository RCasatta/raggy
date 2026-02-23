use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{RaggyError, TextChunk};

pub const CACHE_VERSION: u32 = 2;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedIndex {
    pub version: u32,
    pub model_dir: String,
    pub dir: String,
    pub files: HashMap<String, u128>,
    pub chunks: Vec<TextChunk>,
}

pub fn canonicalize_or_original(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

pub fn cache_root_dir() -> Result<PathBuf, RaggyError> {
    if let Some(xdg_cache_home) = env::var_os("XDG_CACHE_HOME") {
        return Ok(PathBuf::from(xdg_cache_home).join("raggy"));
    }

    if let Some(home) = env::var_os("HOME") {
        return Ok(PathBuf::from(home).join(".cache").join("raggy"));
    }

    Err(RaggyError::IndexingError(
        "Cannot resolve cache directory (XDG_CACHE_HOME and HOME are unset)".to_string(),
    ))
}

pub fn cache_file_path(model_dir: &Path, dir: &Path) -> Result<PathBuf, RaggyError> {
    let cache_root = cache_root_dir()?;
    let model_dir = canonicalize_or_original(model_dir);
    let index_dir = canonicalize_or_original(dir);
    let key_input = format!("{}\n{}", model_dir.display(), index_dir.display());

    let hash = Sha256::digest(key_input.as_bytes());
    let hash_hex = hash
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    Ok(cache_root.join(format!("{hash_hex}.bin")))
}

pub fn load_cache(cache_path: &Path) -> Option<CachedIndex> {
    let raw = fs::read(cache_path).ok()?;
    let cache: CachedIndex =
        match bincode::serde::decode_from_slice(&raw, bincode::config::standard()) {
            Ok((cache, _)) => cache,
            Err(e) => {
                tracing::warn!("Ignoring corrupted cache {}: {e}", cache_path.display());
                return None;
            }
        };

    if cache.version != CACHE_VERSION {
        tracing::info!(
            "Ignoring cache {} because version {} != {CACHE_VERSION}",
            cache_path.display(),
            cache.version,
        );
        return None;
    }

    Some(cache)
}

pub fn save_cache(
    cache_path: &Path,
    model_dir: &Path,
    dir: &Path,
    files: HashMap<String, u128>,
    chunks: Vec<TextChunk>,
) -> Result<(), RaggyError> {
    if let Some(parent) = cache_path.parent() {
        fs::create_dir_all(parent).map_err(|e| {
            RaggyError::IndexingError(format!(
                "Failed to create cache dir {}: {e}",
                parent.display()
            ))
        })?;
    }

    let cache = CachedIndex {
        version: CACHE_VERSION,
        model_dir: canonicalize_or_original(model_dir).display().to_string(),
        dir: canonicalize_or_original(dir).display().to_string(),
        files,
        chunks,
    };

    let serialized = bincode::serde::encode_to_vec(&cache, bincode::config::standard())
        .map_err(|e| RaggyError::IndexingError(format!("Failed to serialize cache: {e}")))?;

    let tmp_path = cache_path.with_extension("bin.tmp");
    fs::write(&tmp_path, serialized).map_err(|e| {
        RaggyError::IndexingError(format!("Failed to write cache {}: {e}", tmp_path.display()))
    })?;
    fs::rename(&tmp_path, cache_path).map_err(|e| {
        RaggyError::IndexingError(format!(
            "Failed to finalize cache {} -> {}: {e}",
            tmp_path.display(),
            cache_path.display()
        ))
    })?;

    Ok(())
}
