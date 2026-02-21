# Raggy

A RAG (Retrieval Augmented Generation) MCP Server that provides semantic search over documents using BERT embeddings.

## Running Tests

Tests require the `all-MiniLM-L6-v2` BERT model to be downloaded.

### Download the Model

Run the provided script to download the model files:

```bash
./download_model.sh
```

This will download the model to `models/all-MiniLM-L6-v2/` (approximately 90MB).

### Run Tests

```bash
cargo test
```

## Running the Application

```bash
cargo run --release -- mcp --model-dir <path> --dir <path>
```

Required arguments:
- `--model-dir`: Path to the model directory containing `model.safetensors`, `tokenizer.json`, and `config.json`
- `--dir`: Directory to index for semantic search

Optional arguments:
- `--extensions`: Comma-separated file extensions to index (default: ".txt,.md")
- `--chunk_size`: Size of text chunks (default: 512)
- `--chunk_overlap`: Overlap between chunks (default: 50)

## Shell-friendly Commands

Build/update the index and exit:

```bash
raggy index --model-dir ./models/all-MiniLM-L6-v2 --dir .
```

Run a one-shot query using an existing index and print JSON to stdout:

```bash
raggy query --model-dir ./models/all-MiniLM-L6-v2 --dir . --question "how do I run the project with nix?" --top-k 5
```

If no index exists for the selected `model-dir` + `dir`, `raggy query` fails and asks you to run `raggy index` first.

## Manual Query from Terminal

You can query Raggy manually using the MCP Inspector:

```bash
npx -y @modelcontextprotocol/inspector ./target/release/raggy mcp --model-dir ./models/all-MiniLM-L6-v2 --dir .
```

Then open the local URL printed by Inspector, call `raggy_query`, and pass JSON input like:

```json
{"question":"how do I run the project with nix?","top_k":5}
```

## Development

```bash
cargo build          # Debug build
cargo build --release # Release build
cargo check          # Type-check without building
cargo clippy         # Run lints
cargo fmt            # Format code
```
