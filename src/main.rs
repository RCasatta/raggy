use std::fs;
use std::io;
use std::path::PathBuf;
use std::time::Instant;

use clap::{Args as ClapArgs, Parser, Subcommand, ValueEnum};
use grep_matcher::{Match, Matcher};
use grep_regex::RegexMatcherBuilder;
use ignore::WalkBuilder;

use raggy::chunking::{ChunkStrategy, chunk_text};
use raggy::{Args, RaggyState};

#[derive(ClapArgs, Debug, Clone)]
struct CommonArgs {
    #[arg(long)]
    model_dir: PathBuf,

    #[arg(long)]
    dir: PathBuf,

    #[arg(long, help = "Glob patterns to exclude from indexing (repeatable)")]
    exclude: Vec<String>,

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
            exclude: value.exclude,
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

#[derive(ValueEnum, Debug, Clone)]
enum StrategyArg {
    Auto,
    Default,
    Markdown,
    Rust,
}

#[derive(ClapArgs, Debug, Clone)]
struct ChunksCommandArgs {
    /// File to split into chunks
    file: PathBuf,

    #[arg(long, default_value = "512")]
    chunk_size: usize,

    #[arg(long, default_value = "50")]
    chunk_overlap: usize,

    /// Chunking strategy (auto-detected from extension by default)
    #[arg(long, default_value = "auto")]
    strategy: StrategyArg,
}

#[derive(ClapArgs, Debug, Clone)]
struct RgCommandArgs {
    /// Directory to search recursively
    dir: PathBuf,

    /// Single word to search for (case-insensitive)
    #[arg(value_parser = parse_single_word)]
    word: String,
}

fn parse_single_word(value: &str) -> Result<String, String> {
    if value.is_empty() {
        return Err("word cannot be empty".to_string());
    }
    if value.chars().any(char::is_whitespace) {
        return Err("word must be a single token without spaces".to_string());
    }
    Ok(value.to_string())
}

#[derive(Subcommand, Debug, Clone)]
enum Command {
    #[command(about = "Start the MCP server")]
    Mcp(McpCommandArgs),
    #[command(about = "Index files in the configured directory")]
    Index(CommonArgs),
    #[command(about = "Run a semantic query against indexed files")]
    Query(QueryCommandArgs),
    #[command(about = "Split a file into chunks and print them")]
    Chunks(ChunksCommandArgs),
    #[command(about = "Recursively search for a word and rank files by occurrences")]
    Rg(RgCommandArgs),
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
    let total_start = Instant::now();
    tracing::info!(
        "Running query for model dir {} and source dir {}",
        args.model_dir.display(),
        args.dir.display()
    );
    let state = RaggyState::new(args);

    let step_start = Instant::now();
    state.load_cached_chunks()?;
    tracing::info!("Timing: load_cached_chunks {:?}", step_start.elapsed());

    let step_start = Instant::now();
    state.verify_model_exists()?;
    tracing::info!("Timing: verify_model_exists {:?}", step_start.elapsed());

    let step_start = Instant::now();
    state.ensure_model_and_tokenizer_loaded()?;
    tracing::info!(
        "Timing: ensure_model_and_tokenizer_loaded {:?}",
        step_start.elapsed()
    );

    let step_start = Instant::now();
    let response = state.query(question, top_k)?;
    tracing::info!("Timing: state.query {:?}", step_start.elapsed());

    let step_start = Instant::now();
    let json = serde_json::to_string_pretty(&response)?;
    tracing::info!(
        "Timing: serialize_query_response {:?}",
        step_start.elapsed()
    );

    println!("{json}");
    tracing::info!("Timing: run_query_total {:?}", total_start.elapsed());
    Ok(())
}

fn run_chunks(args: ChunksCommandArgs) -> anyhow::Result<()> {
    let content = fs::read_to_string(&args.file)?;

    let strategy = match args.strategy {
        StrategyArg::Auto => {
            let ext = args
                .file
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.to_lowercase())
                .unwrap_or_default();
            ChunkStrategy::for_extension(&ext)
        }
        StrategyArg::Default => ChunkStrategy::Default,
        StrategyArg::Markdown => ChunkStrategy::Markdown,
        StrategyArg::Rust => ChunkStrategy::Rust,
    };

    let chunks = chunk_text(&content, args.chunk_size, args.chunk_overlap, &strategy);

    for (i, (text, start_line, end_line)) in chunks.iter().enumerate() {
        println!(
            "--- chunk {} | lines {start_line}-{end_line} | {} chars ---",
            i + 1,
            text.len()
        );
        println!("{text}");
        println!();
    }

    println!("Total: {} chunks", chunks.len());

    Ok(())
}

fn run_rg(args: RgCommandArgs) -> anyhow::Result<()> {
    let matcher = RegexMatcherBuilder::new()
        .case_insensitive(true)
        .build(&args.word)?;

    let mut files_with_counts: Vec<(PathBuf, usize)> = Vec::new();

    for entry in WalkBuilder::new(&args.dir).follow_links(true).build() {
        let Ok(entry) = entry else {
            continue;
        };

        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        let content = match fs::read(path) {
            Ok(content) => content,
            Err(_) => continue,
        };

        let mut occurrences = 0usize;
        matcher.find_iter(&content, |_: Match| {
            occurrences += 1;
            true
        })?;

        if occurrences > 0 {
            files_with_counts.push((path.to_path_buf(), occurrences));
        }
    }

    files_with_counts.sort_by(|(path_a, count_a), (path_b, count_b)| {
        count_b.cmp(count_a).then_with(|| path_a.cmp(path_b))
    });

    for (path, count) in files_with_counts {
        println!("{count}\t{}", path.display());
    }

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
        Command::Chunks(chunks_args) => run_chunks(chunks_args),
        Command::Rg(rg_args) => run_rg(rg_args),
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
                assert!(mcp_args.common.exclude.is_empty());
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
                assert!(query_args.common.exclude.is_empty());
            }
            _ => panic!("Expected query subcommand"),
        }
    }

    #[test]
    fn test_cli_parsing_chunks_subcommand() {
        let cli = Cli::try_parse_from(["raggy", "chunks", "src/main.rs"])
            .expect("chunks subcommand should parse with defaults");
        match cli.command {
            Command::Chunks(args) => {
                assert_eq!(args.file, PathBuf::from("src/main.rs"));
                assert_eq!(args.chunk_size, 512);
                assert_eq!(args.chunk_overlap, 50);
                assert!(matches!(args.strategy, StrategyArg::Auto));
            }
            _ => panic!("Expected chunks subcommand"),
        }

        let cli = Cli::try_parse_from([
            "raggy",
            "chunks",
            "README.md",
            "--chunk-size",
            "256",
            "--chunk-overlap",
            "20",
            "--strategy",
            "markdown",
        ])
        .expect("chunks with explicit options should parse");
        match cli.command {
            Command::Chunks(args) => {
                assert_eq!(args.chunk_size, 256);
                assert_eq!(args.chunk_overlap, 20);
                assert!(matches!(args.strategy, StrategyArg::Markdown));
            }
            _ => panic!("Expected chunks subcommand"),
        }
    }

    #[test]
    fn test_cli_parsing_rg_subcommand() {
        let cli =
            Cli::try_parse_from(["raggy", "rg", "src", "query"]).expect("rg subcommand parses");
        match cli.command {
            Command::Rg(args) => {
                assert_eq!(args.dir, PathBuf::from("src"));
                assert_eq!(args.word, "query");
            }
            _ => panic!("Expected rg subcommand"),
        }
    }

    #[test]
    fn test_cli_parsing_rg_requires_single_word() {
        let err = Cli::try_parse_from(["raggy", "rg", "src", "two words"])
            .expect_err("rg should reject words with spaces");
        assert!(
            err.to_string().contains("single token"),
            "Expected single token error, got: {err}"
        );
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
