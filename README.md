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
cargo run --release -- --model <path> --dir <path>
```

Required arguments:
- `--model`: Path to the BERT model file (safetensors format)
- `--dir`: Directory to index for semantic search

Optional arguments:
- `--extensions`: Comma-separated file extensions to index (default: ".txt,.md")
- `--chunk_size`: Size of text chunks (default: 512)
- `--chunk_overlap`: Overlap between chunks (default: 50)

## Development

```bash
cargo build          # Debug build
cargo build --release # Release build
cargo check          # Type-check without building
cargo clippy         # Run lints
cargo fmt            # Format code
```
