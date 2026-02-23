pub mod cache;
pub mod chunking;
pub mod model;
pub mod server;

use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::time::{Instant, UNIX_EPOCH};

use candle_transformers::models::bert::BertModel;
use content_inspector::inspect;
use ignore::WalkBuilder;
use ignore::overrides::OverrideBuilder;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use crate::cache::{cache_file_path, load_cache, save_cache};
use crate::chunking::{ChunkStrategy, chunk_text};
use crate::model::{dot_product_simd, get_embedding, normalize_l2};

#[derive(Debug, thiserror::Error)]
pub enum RaggyError {
    #[error("Missing model file: {0}")]
    MissingModelFile(String),

    #[error("Corrupt model: {0}")]
    CorruptModel(String),

    #[error("Empty directory: {0}")]
    EmptyDirectory(String),

    #[error("Unsupported model architecture")]
    UnsupportedModelArchitecture,

    #[error("Indexing error: {0}")]
    IndexingError(String),

    #[error("Query error: {0}")]
    QueryError(String),
}

#[derive(Debug, Clone)]
pub struct Args {
    pub model_dir: std::path::PathBuf,
    pub dir: std::path::PathBuf,
    pub exclude: Vec<String>,
    pub chunk_size: usize,
    pub chunk_overlap: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextChunk {
    pub path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResponse {
    pub results: Vec<SearchResult>,
    pub indexing_status: String,
}

pub struct RaggyState {
    pub args: Args,
    chunks: RwLock<Vec<TextChunk>>,
    model: RwLock<Option<BertModel>>,
    tokenizer: RwLock<Option<Tokenizer>>,
    indexing_status: RwLock<String>,
}

fn short_path(full_path: &str) -> String {
    let parts: Vec<&str> = full_path.rsplit('/').take(2).collect();
    parts.into_iter().rev().collect::<Vec<_>>().join("/")
}

impl RaggyState {
    pub fn new(args: Args) -> Self {
        Self {
            args,
            chunks: RwLock::new(Vec::new()),
            model: RwLock::new(None),
            tokenizer: RwLock::new(None),
            indexing_status: RwLock::new("not_started".to_string()),
        }
    }

    pub fn verify_model_exists(&self) -> Result<(), RaggyError> {
        model::verify_model_exists(&self.args.model_dir)
    }

    pub fn load_model_and_tokenizer(&self) -> anyhow::Result<()> {
        let (bert_model, tok) = model::load_model_and_tokenizer(&self.args.model_dir)?;
        *self.model.write() = Some(bert_model);
        *self.tokenizer.write() = Some(tok);
        Ok(())
    }

    pub fn ensure_model_and_tokenizer_loaded(&self) -> Result<(), RaggyError> {
        let model_is_loaded = self.model.read().is_some();
        let tokenizer_is_loaded = self.tokenizer.read().is_some();
        if model_is_loaded && tokenizer_is_loaded {
            return Ok(());
        }

        self.load_model_and_tokenizer()
            .map_err(|e| RaggyError::IndexingError(format!("Failed to load model: {e}")))?;

        Ok(())
    }

    fn build_walker(&self) -> Result<ignore::Walk, RaggyError> {
        let mut overrides = OverrideBuilder::new(&self.args.dir);
        for pattern in &self.args.exclude {
            overrides.add(&format!("!{pattern}")).map_err(|e| {
                RaggyError::IndexingError(format!("Invalid exclude pattern '{pattern}': {e}"))
            })?;
        }
        let overrides = overrides.build().map_err(|e| {
            RaggyError::IndexingError(format!("Failed to build exclude overrides: {e}"))
        })?;

        Ok(WalkBuilder::new(&self.args.dir)
            .follow_links(true)
            .overrides(overrides)
            .build())
    }

    fn is_text_file(path: &std::path::Path) -> bool {
        let mut buf = [0u8; 1024];
        let bytes_read = match fs::File::open(path).and_then(|mut f| f.read(&mut buf)) {
            Ok(n) => n,
            Err(_) => return false,
        };
        inspect(&buf[..bytes_read]).is_text()
    }

    pub fn index_files(&self) -> Result<(), RaggyError> {
        *self.indexing_status.write() = "in_progress".to_string();

        let mut current_files = Vec::new();
        let walker = self.build_walker()?;
        let progress_interval = std::time::Duration::from_secs(5);
        let mut last_discovery_log = Instant::now();

        for entry in walker {
            let Ok(entry) = entry else {
                continue;
            };

            let path = entry.path();

            if !path.is_file() {
                continue;
            }

            if !Self::is_text_file(path) {
                continue;
            }

            let mtime = match path.metadata().and_then(|m| m.modified()) {
                Ok(modified) => modified
                    .duration_since(UNIX_EPOCH)
                    .map(|duration| duration.as_nanos())
                    .unwrap_or(0),
                Err(_) => continue,
            };

            current_files.push((path.to_path_buf(), path.display().to_string(), mtime));

            if last_discovery_log.elapsed() >= progress_interval {
                tracing::info!(
                    "Index discovery progress: {} text files found so far",
                    current_files.len()
                );
                last_discovery_log = Instant::now();
            }
        }

        if current_files.is_empty() {
            *self.indexing_status.write() = "failed".to_string();
            return Err(RaggyError::EmptyDirectory(
                self.args.dir.display().to_string(),
            ));
        }

        let total_files = current_files.len();
        tracing::info!("Index discovery complete: {total_files} text files to process");

        let model_guard = self.model.read();
        let tokenizer_guard = self.tokenizer.read();

        let model = match model_guard.as_ref() {
            Some(m) => m,
            None => {
                *self.indexing_status.write() = "failed".to_string();
                return Err(RaggyError::IndexingError("Model not loaded".to_string()));
            }
        };

        let tokenizer = match tokenizer_guard.as_ref() {
            Some(t) => t,
            None => {
                *self.indexing_status.write() = "failed".to_string();
                return Err(RaggyError::IndexingError(
                    "Tokenizer not loaded".to_string(),
                ));
            }
        };

        let cp = cache_file_path(&self.args.model_dir, &self.args.dir)?;
        let cached = load_cache(&cp);

        let mut cached_chunks_by_path: HashMap<String, Vec<TextChunk>> = HashMap::new();
        let cached_files = if let Some(c) = cached {
            for chunk in c.chunks {
                cached_chunks_by_path
                    .entry(chunk.path.clone())
                    .or_default()
                    .push(chunk);
            }
            c.files
        } else {
            HashMap::new()
        };

        let mut indexed_chunks = Vec::new();
        let mut indexed_files = HashMap::new();
        let mut reused_files = 0usize;
        let mut reindexed_files = 0usize;
        let mut processed_files = 0usize;
        let mut skipped_files = 0usize;
        let mut last_processing_log = Instant::now();

        for (path_buf, path_string, mtime) in current_files {
            let is_unchanged = cached_files
                .get(&path_string)
                .map(|cached_mtime| *cached_mtime == mtime)
                .unwrap_or(false);

            if is_unchanged && let Some(cached_chunks) = cached_chunks_by_path.remove(&path_string)
            {
                indexed_chunks.extend(cached_chunks);
                indexed_files.insert(path_string, mtime);
                reused_files += 1;
                processed_files += 1;

                if last_processing_log.elapsed() >= progress_interval {
                    tracing::info!(
                        "Index processing progress: {processed_files}/{total_files} files (reused {reused_files}, reindexed {reindexed_files}, skipped {skipped_files})"
                    );
                    last_processing_log = Instant::now();
                }
                continue;
            }

            let content = match fs::read_to_string(&path_buf) {
                Ok(content) => content,
                Err(_) => {
                    skipped_files += 1;
                    processed_files += 1;

                    if last_processing_log.elapsed() >= progress_interval {
                        tracing::info!(
                            "Index processing progress: {processed_files}/{total_files} files (reused {reused_files}, reindexed {reindexed_files}, skipped {skipped_files})"
                        );
                        last_processing_log = Instant::now();
                    }
                    continue;
                }
            };

            let ext = path_buf
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.to_lowercase())
                .unwrap_or_default();
            let strategy = ChunkStrategy::for_extension(&ext);
            let text_chunks = chunk_text(
                &content,
                self.args.chunk_size,
                self.args.chunk_overlap,
                &strategy,
            );
            let prefix = short_path(&path_string);
            for (chunk_text, start_line, end_line) in text_chunks {
                let embedding_text = format!("{prefix}\n{chunk_text}");
                match get_embedding(model, tokenizer, &embedding_text) {
                    Ok(embedding) => {
                        let normalized = normalize_l2(&embedding);
                        indexed_chunks.push(TextChunk {
                            path: path_string.clone(),
                            start_line,
                            end_line,
                            embedding: normalized,
                        });
                    }
                    Err(_) => continue,
                }
            }
            indexed_files.insert(path_string, mtime);
            reindexed_files += 1;
            processed_files += 1;

            if last_processing_log.elapsed() >= progress_interval {
                tracing::info!(
                    "Index processing progress: {processed_files}/{total_files} files (reused {reused_files}, reindexed {reindexed_files}, skipped {skipped_files})"
                );
                last_processing_log = Instant::now();
            }
        }

        if indexed_chunks.is_empty() {
            *self.indexing_status.write() = "failed".to_string();
            return Err(RaggyError::EmptyDirectory(
                self.args.dir.display().to_string(),
            ));
        }

        tracing::info!(
            "Index ready: processed {processed_files}/{total_files} files (reused {reused_files}, reindexed {reindexed_files}, skipped {skipped_files}), total chunks {}",
            indexed_chunks.len()
        );

        if let Err(e) = save_cache(
            &cp,
            &self.args.model_dir,
            &self.args.dir,
            indexed_files,
            indexed_chunks.clone(),
        ) {
            tracing::warn!("Failed to persist cache: {e}");
        }

        *self.chunks.write() = indexed_chunks;
        *self.indexing_status.write() = "complete".to_string();

        Ok(())
    }

    pub fn query(&self, question: &str, top_k: usize) -> Result<QueryResponse, RaggyError> {
        let total_start = Instant::now();
        let status = self.indexing_status.read().clone();

        let model_guard = self.model.read();
        let tokenizer_guard = self.tokenizer.read();

        let model = match model_guard.as_ref() {
            Some(m) => m,
            None => return Err(RaggyError::QueryError("Model not loaded".to_string())),
        };

        let tokenizer = match tokenizer_guard.as_ref() {
            Some(t) => t,
            None => return Err(RaggyError::QueryError("Tokenizer not loaded".to_string())),
        };

        let embedding_start = Instant::now();
        let query_embedding = get_embedding(model, tokenizer, question)
            .map_err(|e| RaggyError::QueryError(e.to_string()))?;
        tracing::info!(
            "Timing: query_get_embedding {:?}",
            embedding_start.elapsed()
        );

        let query_embedding = normalize_l2(&query_embedding);

        let chunks = self.chunks.read();
        let scoring_start = Instant::now();
        let mut scores: Vec<(usize, f32)> = chunks
            .iter()
            .enumerate()
            .map(|(idx, chunk)| {
                let score = dot_product_simd(&query_embedding, &chunk.embedding);
                (idx, score)
            })
            .collect();
        tracing::info!(
            "Timing: query_score_chunks {:?} for {} chunks",
            scoring_start.elapsed(),
            chunks.len()
        );

        let sort_start = Instant::now();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);
        tracing::info!("Timing: query_sort_and_truncate {:?}", sort_start.elapsed());

        let results_start = Instant::now();
        let results: Vec<SearchResult> = scores
            .into_iter()
            .map(|(idx, score)| {
                let chunk = &chunks[idx];
                SearchResult {
                    path: chunk.path.clone(),
                    start_line: chunk.start_line,
                    end_line: chunk.end_line,
                    score,
                }
            })
            .collect();
        tracing::info!("Timing: query_build_results {:?}", results_start.elapsed());
        tracing::info!("Timing: query_total {:?}", total_start.elapsed());

        Ok(QueryResponse {
            results,
            indexing_status: status,
        })
    }

    pub fn load_cached_chunks(&self) -> Result<(), RaggyError> {
        let cp = cache_file_path(&self.args.model_dir, &self.args.dir)?;
        let cached = load_cache(&cp).ok_or_else(|| {
            RaggyError::QueryError(format!(
                "No index found at {}. Run `raggy index --model-dir {} --dir {}` first.",
                cp.display(),
                self.args.model_dir.display(),
                self.args.dir.display()
            ))
        })?;

        if cached.chunks.is_empty() {
            return Err(RaggyError::QueryError(format!(
                "Index at {} is empty. Rebuild it with `raggy index --model-dir {} --dir {}`.",
                cp.display(),
                self.args.model_dir.display(),
                self.args.dir.display()
            )));
        }

        *self.chunks.write() = cached.chunks;
        *self.indexing_status.write() = "complete".to_string();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_short_path_with_parent() {
        let result = short_path("/home/casatta/gb/kb/docs/2026-02-06_Pagella.txt");
        assert_eq!(result, "docs/2026-02-06_Pagella.txt");
    }

    #[test]
    fn test_short_path_single_component() {
        let result = short_path("file.txt");
        assert_eq!(result, "file.txt");
    }

    #[test]
    fn test_short_path_two_components() {
        let result = short_path("dir/file.txt");
        assert_eq!(result, "dir/file.txt");
    }

    #[test]
    fn test_query_fails_when_cache_missing() {
        let args = Args {
            model_dir: PathBuf::from("models/all-MiniLM-L6-v2"),
            dir: PathBuf::from("definitely-non-existing-dir-for-cache-key"),
            exclude: Vec::new(),
            chunk_size: 512,
            chunk_overlap: 50,
        };
        let state = RaggyState::new(args);

        let err = state
            .load_cached_chunks()
            .expect_err("query cache load should fail when no index exists");
        assert!(
            err.to_string().contains("No index found"),
            "Expected clear missing index error, got: {err}"
        );
    }
}
