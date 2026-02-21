use std::sync::Arc;
use std::thread;

use rmcp::{
    RoleServer, ServiceExt,
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
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{Args, RaggyState};

pub const DEFAULT_QUERY_TOOL_DESCRIPTION: &str = "Search for relevant text chunks using semantic similarity. Returns documents matching the query.";
pub const DEFAULT_INDEX_TOOL_DESCRIPTION: &str =
    "Trigger re-indexing of the configured directory. Indexing happens in the background.";
pub const DEFAULT_MCP_INSTRUCTIONS: &str = "Use raggy_query to search for relevant text chunks in indexed documents. Use raggy_index to trigger re-indexing.";

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct QueryParams {
    #[schemars(description = "The search query")]
    question: String,

    #[schemars(description = "Number of results to return (default: 10)")]
    top_k: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct IndexParams {}

#[derive(Clone)]
pub struct RaggyServer {
    state: Arc<RaggyState>,
    query_tool_description: String,
}

impl RaggyServer {
    pub fn new(args: Args, query_tool_description: Option<String>) -> Self {
        Self {
            state: Arc::new(RaggyState::new(args)),
            query_tool_description: query_tool_description
                .unwrap_or_else(|| DEFAULT_QUERY_TOOL_DESCRIPTION.to_string()),
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
            ErrorData::internal_error(format!("Failed to serialize schema: {e}"), None)
        })?;

        let query_input_schema_map =
            if let rmcp::serde_json::Value::Object(map) = query_input_schema {
                Arc::new(map)
            } else {
                return Err(ErrorData::internal_error("Schema is not an object", None));
            };

        let index_schema = schemars::schema_for!(IndexParams);
        let index_input_schema = rmcp::serde_json::to_value(index_schema).map_err(|e| {
            ErrorData::internal_error(format!("Failed to serialize schema: {e}"), None)
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
                    description: Some(self.query_tool_description.clone().into()),
                    input_schema: query_input_schema_map,
                    output_schema: None,
                    annotations: None,
                    icons: None,
                },
                Tool {
                    name: "raggy_index".into(),
                    title: None,
                    description: Some(DEFAULT_INDEX_TOOL_DESCRIPTION.into()),
                    input_schema: index_input_schema_map,
                    output_schema: None,
                    annotations: None,
                    icons: None,
                },
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
                let query_params: QueryParams =
                    rmcp::serde_json::from_value(args_value).map_err(|e| {
                        ErrorData::invalid_request(format!("Invalid parameters: {e}"), None)
                    })?;

                let top_k = query_params.top_k.unwrap_or(10) as usize;

                match self.state.query(&query_params.question, top_k) {
                    Ok(response) => {
                        let json = serde_json::to_string_pretty(&response).map_err(|e| {
                            ErrorData::internal_error(
                                format!("Failed to serialize response: {e}"),
                                None,
                            )
                        })?;
                        Ok(CallToolResult::success(vec![Content::text(json)]))
                    }
                    Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                        "Error: {e}"
                    ))])),
                }
            }
            "raggy_index" => {
                self.state.verify_model_exists().map_err(|e| {
                    ErrorData::internal_error(
                        format!("Cannot start indexing, model check failed: {e}"),
                        None,
                    )
                })?;

                self.state
                    .ensure_model_and_tokenizer_loaded()
                    .map_err(|e| {
                        ErrorData::internal_error(format!("Cannot start indexing: {e}"), None)
                    })?;

                let state = Arc::clone(&self.state);
                thread::spawn(move || {
                    if let Err(e) = state.index_files() {
                        tracing::error!("Background indexing failed: {e}");
                    }
                });

                Ok(CallToolResult::success(vec![Content::text(
                    "Indexing started in background.".to_string(),
                )]))
            }
            _ => Err(ErrorData::invalid_request(
                format!("Unknown tool: {tool_name}"),
                None,
            )),
        }
    }

    async fn initialize(
        &self,
        _params: InitializeRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> Result<InitializeResult, ErrorData> {
        if let Err(e) = self.state.verify_model_exists() {
            return Err(ErrorData::internal_error(
                format!("Model verification failed: {e}"),
                None,
            ));
        }

        let state = Arc::clone(&self.state);
        thread::spawn(move || {
            if let Err(e) = state.load_model_and_tokenizer() {
                tracing::error!("Failed to load model: {e}");
                return;
            }
            tracing::info!("Model loaded, starting indexing...");
            if let Err(e) = state.index_files() {
                tracing::error!("Indexing failed: {e}");
                return;
            }
            tracing::info!("Indexing complete");
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
            instructions: Some(DEFAULT_MCP_INSTRUCTIONS.to_string()),
        })
    }
}

pub fn run_mcp(args: Args, query_tool_description: Option<String>) -> anyhow::Result<()> {
    tracing::info!("Starting Raggy MCP Server");
    tracing::debug!(
        "Model dir: {}, Dir: {}",
        args.model_dir.display(),
        args.dir.display()
    );

    let server = RaggyServer::new(args, query_tool_description);
    let rt = tokio::runtime::Runtime::new()?;

    rt.block_on(async {
        let running_service = server.serve(stdio()).await?;
        let _quit_reason = running_service.waiting().await?;
        Ok(())
    })
}
