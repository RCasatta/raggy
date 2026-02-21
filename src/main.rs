use std::io;
use std::path::PathBuf;

use clap::{Args as ClapArgs, Parser, Subcommand};

use raggy::{Args, RaggyState};

#[derive(ClapArgs, Debug, Clone)]
struct CommonArgs {
    #[arg(long)]
    model_dir: PathBuf,

    #[arg(long)]
    dir: PathBuf,

    #[arg(long, default_value = ".txt,.md")]
    extensions: String,

    #[arg(long, default_value = "512")]
    chunk_size: usize,

    #[arg(long, default_value = "50")]
    chunk_overlap: usize,
}

impl From<CommonArgs> for Args {
    fn from(value: CommonArgs) -> Self {
        Self {
            model_dir: value.model_dir,
            dir: value.dir,
            extensions: value.extensions,
            chunk_size: value.chunk_size,
            chunk_overlap: value.chunk_overlap,
        }
    }
}

#[derive(ClapArgs, Debug, Clone)]
struct QueryCommandArgs {
    #[command(flatten)]
    common: CommonArgs,

    #[arg(long)]
    question: String,

    #[arg(long, default_value_t = 10)]
    top_k: usize,
}

#[derive(ClapArgs, Debug, Clone)]
struct McpCommandArgs {
    #[command(flatten)]
    common: CommonArgs,

    #[arg(
        long,
        help = "Override raggy_query tool description shown to MCP clients"
    )]
    query_tool_description: Option<String>,
}

#[derive(Subcommand, Debug, Clone)]
enum Command {
    #[command(about = "Start the MCP server")]
    Mcp(McpCommandArgs),
    #[command(about = "Index files in the configured directory")]
    Index(CommonArgs),
    #[command(about = "Run a semantic query against indexed files")]
    Query(QueryCommandArgs),
}

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

fn run_index(args: Args) -> anyhow::Result<()> {
    tracing::info!(
        "Starting index build for model dir {} and source dir {}",
        args.model_dir.display(),
        args.dir.display()
    );
    let state = RaggyState::new(args);
    state.verify_model_exists()?;
    state.ensure_model_and_tokenizer_loaded()?;
    state.index_files()?;
    tracing::info!("Indexing complete");
    Ok(())
}

fn run_query(args: Args, question: &str, top_k: usize) -> anyhow::Result<()> {
    tracing::info!(
        "Running query for model dir {} and source dir {}",
        args.model_dir.display(),
        args.dir.display()
    );
    let state = RaggyState::new(args);
    state.load_cached_chunks()?;
    state.verify_model_exists()?;
    state.ensure_model_and_tokenizer_loaded()?;
    let response = state.query(question, top_k)?;
    let json = serde_json::to_string_pretty(&response)?;
    println!("{json}");
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let _ = tracing_subscriber::fmt()
        .with_ansi(false)
        .with_writer(io::stderr)
        .try_init();

    match cli.command {
        Command::Mcp(mcp_args) => {
            raggy::server::run_mcp(mcp_args.common.into(), mcp_args.query_tool_description)
        }
        Command::Index(common_args) => run_index(common_args.into()),
        Command::Query(query_args) => run_query(
            query_args.common.into(),
            &query_args.question,
            query_args.top_k,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_parsing_subcommands() {
        let cli = Cli::try_parse_from([
            "raggy",
            "mcp",
            "--model-dir",
            "models/all-MiniLM-L6-v2",
            "--dir",
            ".",
        ])
        .expect("mcp subcommand should parse");
        match cli.command {
            Command::Mcp(mcp_args) => {
                assert_eq!(mcp_args.common.extensions, ".txt,.md");
                assert_eq!(mcp_args.query_tool_description, None);
            }
            _ => panic!("Expected mcp subcommand"),
        }

        let cli = Cli::try_parse_from([
            "raggy",
            "mcp",
            "--model-dir",
            "models/all-MiniLM-L6-v2",
            "--dir",
            ".",
            "--query-tool-description",
            "Search my local notes and docs using semantic similarity.",
        ])
        .expect("mcp with custom tool description should parse");
        match cli.command {
            Command::Mcp(mcp_args) => {
                assert_eq!(
                    mcp_args.query_tool_description,
                    Some("Search my local notes and docs using semantic similarity.".to_string())
                );
            }
            _ => panic!("Expected mcp subcommand"),
        }

        let cli = Cli::try_parse_from([
            "raggy",
            "index",
            "--model-dir",
            "models/all-MiniLM-L6-v2",
            "--dir",
            ".",
        ])
        .expect("index subcommand should parse");
        assert!(matches!(cli.command, Command::Index(_)));

        let cli = Cli::try_parse_from([
            "raggy",
            "query",
            "--model-dir",
            "models/all-MiniLM-L6-v2",
            "--dir",
            ".",
            "--question",
            "How do I run tests?",
            "--top-k",
            "3",
        ])
        .expect("query subcommand should parse");
        match cli.command {
            Command::Query(query_args) => {
                assert_eq!(query_args.question, "How do I run tests?");
                assert_eq!(query_args.top_k, 3);
                assert_eq!(query_args.common.extensions, ".txt,.md");
            }
            _ => panic!("Expected query subcommand"),
        }
    }

    #[test]
    fn test_cli_parsing_without_subcommand_fails() {
        let err = Cli::try_parse_from([
            "raggy",
            "--model-dir",
            "models/all-MiniLM-L6-v2",
            "--dir",
            ".",
        ])
        .expect_err("root args without subcommand should fail");
        assert!(
            err.to_string().contains("Usage: raggy <COMMAND>"),
            "Expected subcommand parsing error, got: {err}"
        );
    }
}
