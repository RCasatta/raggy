use anyhow::Result;
use candle_core::Tensor;
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use clap::Parser;
use parking_lot::RwLock;
use rmcp::{
    RoleServer,
    ServiceExt,
    handler::server::ServerHandler,
    model::{
        CallToolRequestParam, CallToolResult, Content, ErrorData, Implementation,
        InitializeRequestParam, InitializeResult, ListToolsResult, PaginatedRequestParam,
        ProtocolVersion, ServerCapabilities, Tool,
    },
    schemars,
    service::RequestContext,
    transport::stdio,
};
use serde::{Deserialize, Serialize};
use schemars::JsonSchema;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;
use tokenizers::{PaddingParams, Tokenizer};
use walkdir::WalkDir;

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

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    
    #[arg(long)]
    dir: PathBuf,
    
    #[arg(long, default_value = ".txt,.md")]
    extensions: String,
    
    #[arg(long, default_value = "512")]
    chunk_size: usize,
    
    #[arg(long, default_value = "50")]
    chunk_overlap: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct QueryParams {
    #[schemars(description = "The search query")]
    question: String,
    
    #[schemars(description = "Number of results to return (default: 10)")]
    top_k: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct IndexParams {}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SearchResult {
    path: String,
    chunk: String,
    start_line: usize,
    end_line: usize,
    score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct QueryResponse {
    results: Vec<SearchResult>,
    indexing_status: String,
}

#[derive(Debug, Clone)]
struct TextChunk {
    path: String,
    chunk: String,
    start_line: usize,
    end_line: usize,
    embedding: Vec<f32>,
}

struct RaggyState {
    args: Args,
    chunks: RwLock<Vec<TextChunk>>,
    model: RwLock<Option<BertModel>>,
    tokenizer: RwLock<Option<Tokenizer>>,
    indexing_status: RwLock<String>,
}

impl RaggyState {
    fn new(args: Args) -> Self {
        Self {
            args,
            chunks: RwLock::new(Vec::new()),
            model: RwLock::new(None),
            tokenizer: RwLock::new(None),
            indexing_status: RwLock::new("not_started".to_string()),
        }
    }
    
    fn verify_model_exists(&self) -> Result<(), RaggyError> {
        if !self.args.model.exists() {
            return Err(RaggyError::MissingModelFile(
                self.args.model.display().to_string()
            ));
        }
        Ok(())
    }
    
    fn load_model_and_tokenizer(&self) -> Result<()> {
        let model_path = &self.args.model;
        
        let tokenizer_path = model_path.parent()
            .map(|p| p.join("tokenizer.json"))
            .unwrap_or_else(|| PathBuf::from("tokenizer.json"));
        
        let config_path = model_path.parent()
            .map(|p| p.join("config.json"))
            .unwrap_or_else(|| PathBuf::from("config.json"));
        
        let config_str = fs::read_to_string(&config_path)
            .map_err(|e| RaggyError::CorruptModel(format!("Failed to read config: {}", e)))?;
        
        let config: Config = serde_json::from_str(&config_str)
            .map_err(|e| RaggyError::CorruptModel(format!("Failed to parse config: {}", e)))?;
        
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| RaggyError::CorruptModel(format!("Failed to load tokenizer: {}", e)))?;
        
        let device = candle_core::Device::Cpu;
        
        let vb = if model_path.extension().map(|e| e == "gguf").unwrap_or(false) {
            return Err(anyhow::anyhow!("GGUF format not yet supported, please use safetensors format"));
        } else {
            unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], DTYPE, &device)? }
        };
        
        let model = BertModel::load(vb, &config)?;
        
        *self.model.write() = Some(model);
        *self.tokenizer.write() = Some(tokenizer);
        
        Ok(())
    }
    
    fn get_extensions(&self) -> Vec<String> {
        self.args.extensions
            .split(',')
            .map(|s| s.trim().to_lowercase())
            .collect()
    }
    
    fn chunk_text(&self, text: &str) -> Vec<(String, usize, usize)> {
        let lines: Vec<&str> = text.lines().collect();
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut start_line = 1;
        let mut current_line = 1;
        
        for (idx, line) in lines.iter().enumerate() {
            current_line = idx + 1;
            
            if current_chunk.len() + line.len() + 1 > self.args.chunk_size && !current_chunk.is_empty() {
                chunks.push((current_chunk.clone(), start_line, current_line - 1));
                
                let overlap_text = find_overlap(&current_chunk, self.args.chunk_overlap);
                current_chunk = overlap_text;
                start_line = current_line;
            }
            
            if !current_chunk.is_empty() {
                current_chunk.push('\n');
            }
            current_chunk.push_str(line);
        }
        
        if !current_chunk.is_empty() {
            chunks.push((current_chunk, start_line, current_line));
        }
        
        chunks
    }
    
    fn index_files(&self) -> Result<(), RaggyError> {
        *self.indexing_status.write() = "in_progress".to_string();
        
        let extensions = self.get_extensions();
        let mut all_chunks = Vec::new();
        
        for entry in WalkDir::new(&self.args.dir)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();
            
            if !path.is_file() {
                continue;
            }
            
            let ext = path.extension()
                .and_then(|e| e.to_str())
                .map(|e| e.to_lowercase())
                .unwrap_or_default();
            
            if !extensions.iter().any(|e| e.trim_start_matches('.') == ext) {
                continue;
            }
            
            let content = match fs::read_to_string(path) {
                Ok(c) => c,
                Err(_) => continue,
            };
            
            let text_chunks = self.chunk_text(&content);
            
            for (chunk_text, start_line, end_line) in text_chunks {
                all_chunks.push((path.display().to_string(), chunk_text, start_line, end_line));
            }
        }
        
        if all_chunks.is_empty() {
            *self.indexing_status.write() = "complete".to_string();
            return Err(RaggyError::EmptyDirectory(self.args.dir.display().to_string()));
        }
        
        let model_guard = self.model.read();
        let tokenizer_guard = self.tokenizer.read();
        
        let model = match model_guard.as_ref() {
            Some(m) => m,
            None => {
                *self.indexing_status.write() = "not_started".to_string();
                return Err(RaggyError::IndexingError("Model not loaded".to_string()));
            }
        };
        
        let tokenizer = match tokenizer_guard.as_ref() {
            Some(t) => t,
            None => {
                *self.indexing_status.write() = "not_started".to_string();
                return Err(RaggyError::IndexingError("Tokenizer not loaded".to_string()));
            }
        };
        
        let mut indexed_chunks = Vec::new();
        
        for (path, chunk_text, start_line, end_line) in all_chunks {
            match get_embedding(model, tokenizer, &chunk_text) {
                Ok(embedding) => {
                    let normalized = normalize_l2(&embedding);
                    indexed_chunks.push(TextChunk {
                        path,
                        chunk: chunk_text,
                        start_line,
                        end_line,
                        embedding: normalized,
                    });
                }
                Err(_) => continue,
            }
        }
        
        *self.chunks.write() = indexed_chunks;
        *self.indexing_status.write() = "complete".to_string();
        
        Ok(())
    }
    
    fn query(&self, question: &str, top_k: usize) -> Result<QueryResponse, RaggyError> {
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
        
        let query_embedding = get_embedding(model, tokenizer, question)
            .map_err(|e| RaggyError::QueryError(e.to_string()))?;
        
        let query_embedding = normalize_l2(&query_embedding);
        
        let chunks = self.chunks.read();
        let mut scores: Vec<(usize, f32)> = chunks
            .iter()
            .enumerate()
            .map(|(idx, chunk)| {
                let score = cosine_similarity(&query_embedding, &chunk.embedding);
                (idx, score)
            })
            .collect();
        
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);
        
        let results: Vec<SearchResult> = scores
            .into_iter()
            .map(|(idx, score)| {
                let chunk = &chunks[idx];
                SearchResult {
                    path: chunk.path.clone(),
                    chunk: chunk.chunk.clone(),
                    start_line: chunk.start_line,
                    end_line: chunk.end_line,
                    score,
                }
            })
            .collect();
        
        Ok(QueryResponse {
            results,
            indexing_status: status,
        })
    }
}

fn find_overlap(text: &str, overlap_size: usize) -> String {
    if text.len() <= overlap_size {
        return text.to_string();
    }
    
    let chars: Vec<char> = text.chars().collect();
    let start_pos = chars.len().saturating_sub(overlap_size);
    
    for i in start_pos..chars.len() {
        if chars[i] == '\n' {
            let overlap_start = i + 1;
            if overlap_start < chars.len() {
                return chars[overlap_start..].iter().collect();
            }
            return String::new();
        }
    }
    
    chars[chars.len().saturating_sub(overlap_size)..].iter().collect()
}

fn get_embedding(model: &BertModel, tokenizer: &Tokenizer, text: &str) -> Result<Vec<f32>> {
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
        .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;
    
    let token_ids = tokens.get_ids().to_vec();
    let attention_mask = tokens.get_attention_mask().to_vec();
    
    let device = &model.device;
    let token_ids_tensor = Tensor::new(&token_ids[..], device)?.unsqueeze(0)?;
    let attention_mask_tensor = Tensor::new(&attention_mask[..], device)?.unsqueeze(0)?;
    let token_type_ids = token_ids_tensor.zeros_like()?;
    
    let embeddings = model.forward(&token_ids_tensor, &token_type_ids, Some(&attention_mask_tensor))?;
    
    let attention_mask_for_pooling = attention_mask_tensor.to_dtype(DTYPE)?.unsqueeze(2)?;
    let sum_mask = attention_mask_for_pooling.sum(1)?;
    let pooled = (embeddings.broadcast_mul(&attention_mask_for_pooling)?).sum(1)?;
    let pooled = pooled.broadcast_div(&sum_mask)?;
    
    let embedding_vec = pooled.to_vec1::<f32>()?;
    
    Ok(embedding_vec)
}

fn normalize_l2(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x * y)
        .sum()
}

#[derive(Clone)]
struct RaggyServer {
    state: Arc<RaggyState>,
}

impl RaggyServer {
    fn new(args: Args) -> Self {
        Self {
            state: Arc::new(RaggyState::new(args)),
        }
    }
}

impl ServerHandler for RaggyServer {
    async fn list_tools(
        &self,
        _params: Option<PaginatedRequestParam>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        use std::sync::Arc;

        let query_schema = schemars::schema_for!(QueryParams);
        let query_input_schema = rmcp::serde_json::to_value(query_schema).map_err(|e| {
            ErrorData::internal_error(format!("Failed to serialize schema: {}", e), None)
        })?;

        let query_input_schema_map =
            if let rmcp::serde_json::Value::Object(map) = query_input_schema {
                Arc::new(map)
            } else {
                return Err(ErrorData::internal_error("Schema is not an object", None));
            };

        let index_schema = schemars::schema_for!(IndexParams);
        let index_input_schema = rmcp::serde_json::to_value(index_schema).map_err(|e| {
            ErrorData::internal_error(format!("Failed to serialize schema: {}", e), None)
        })?;

        let index_input_schema_map =
            if let rmcp::serde_json::Value::Object(map) = index_input_schema {
                Arc::new(map)
            } else {
                return Err(ErrorData::internal_error("Schema is not an object", None));
            };

        Ok(ListToolsResult {
            tools: vec![
                Tool {
                    name: "raggy_query".into(),
                    title: None,
                    description: Some("Search for relevant text chunks using semantic similarity. Returns documents matching the query.".into()),
                    input_schema: query_input_schema_map,
                    output_schema: None,
                    annotations: None,
                    icons: None,
                },
                Tool {
                    name: "raggy_index".into(),
                    title: None,
                    description: Some("Trigger re-indexing of the configured directory. Indexing happens in the background.".into()),
                    input_schema: index_input_schema_map,
                    output_schema: None,
                    annotations: None,
                    icons: None,
                }
            ],
            next_cursor: None,
        })
    }

    async fn call_tool(
        &self,
        params: CallToolRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, ErrorData> {
        let tool_name = params.name.as_ref();

        match tool_name {
            "raggy_query" => {
                let args = params.arguments.unwrap_or_default();
                let args_value = rmcp::serde_json::Value::Object(args);
                let query_params: QueryParams = rmcp::serde_json::from_value(args_value)
                    .map_err(|e| {
                        ErrorData::invalid_request(format!("Invalid parameters: {}", e), None)
                    })?;

                let top_k = query_params.top_k.unwrap_or(10) as usize;
                
                match self.state.query(&query_params.question, top_k) {
                    Ok(response) => {
                        let json = serde_json::to_string_pretty(&response)
                            .map_err(|e| {
                                ErrorData::internal_error(format!("Failed to serialize response: {}", e), None)
                            })?;
                        Ok(CallToolResult::success(vec![Content::text(json)]))
                    }
                    Err(e) => {
                        Ok(CallToolResult::success(vec![Content::text(format!("Error: {}", e))]))
                    }
                }
            }
            "raggy_index" => {
                let state = Arc::clone(&self.state);
                thread::spawn(move || {
                    let _ = state.index_files();
                });
                
                Ok(CallToolResult::success(vec![Content::text("Indexing started in background.".to_string())]))
            }
            _ => {
                Err(ErrorData::invalid_request(
                    format!("Unknown tool: {}", tool_name),
                    None,
                ))
            }
        }
    }

    async fn initialize(
        &self,
        _params: InitializeRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> Result<InitializeResult, ErrorData> {
        if let Err(e) = self.state.verify_model_exists() {
            return Err(ErrorData::internal_error(format!("Model verification failed: {}", e), None));
        }
        
        let state = Arc::clone(&self.state);
        thread::spawn(move || {
            if state.load_model_and_tokenizer().is_ok() {
                let _ = state.index_files();
            }
        });

        Ok(InitializeResult {
            protocol_version: ProtocolVersion::default(),
            capabilities: ServerCapabilities {
                tools: Some(Default::default()),
                ..Default::default()
            },
            server_info: Implementation {
                name: "Raggy RAG MCP Server".to_string(),
                title: None,
                version: "0.1.0".to_string(),
                icons: None,
                website_url: None,
            },
            instructions: Some("Use raggy_query to search for relevant text chunks in indexed documents. Use raggy_index to trigger re-indexing.".to_string()),
        })
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    
    tracing_subscriber::fmt::init();
    
    let server = RaggyServer::new(args);
    let rt = tokio::runtime::Runtime::new()?;
    
    rt.block_on(async {
        let running_service = server.serve(stdio()).await?;
        let _quit_reason = running_service.waiting().await?;
        Ok(())
    })
}
