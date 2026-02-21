# Raggy

A RAG (Retrieval Augmented Generation) MCP Server that provides semantic search over documents using BERT embeddings.

```json
{
  "mcpServers": {
    "raggy": {
      "command": "raggy",
      "args": [
        "mcp",
        "--model-dir",
        "/home/user/models/all-MiniLM-L6-v2",
        "--dir",
        "/home/user/knowledge-base",
        "--query-tool-description",
        "Search my local notes and docs using semantic similarity."
      ]
    }
  }
}
```

## Model

The system need an embedding BERT model to operate.

Tests require the `all-MiniLM-L6-v2` BERT model to be downloaded.
Run the provided script to download the model files:

```bash
./download_model.sh
```

This will download the model to `models/all-MiniLM-L6-v2/` (approximately 90MB).

## Run Tests

```bash
cargo test --release # release mode suggested because computing embedding is expensive 
```

## Running the MCP

```bash
cargo run --release -- mcp --model-dir <path> --dir <path>

# or with nix
nix run -- mcp --model-dir <path> --dir <path>
```

Required arguments:
- `--model-dir`: Path to the model directory containing `model.safetensors`, `tokenizer.json`, and `config.json`
- `--dir`: Directory to index for semantic search

Optional arguments:
- `--extensions`: Comma-separated file extensions to index (default: ".txt,.md")
- `--chunk_size`: Size of text chunks (default: 512)
- `--chunk_overlap`: Overlap between chunks (default: 50)

## Shell-friendly Commands
While the main purpose of this is using it via MCP protocol, it's fully testable with a couple of shell commands

Build/update the index and exit:

```bash
cargo run --release -- index --model-dir ./models/all-MiniLM-L6-v2 --dir .
```

Run a one-shot query using an existing index and print JSON to stdout:

```bash
cargo run --release -- query --model-dir ./models/all-MiniLM-L6-v2 --dir . --question "how do I run the project with nix?" --top-k 5
```

If no index exists for the selected `model-dir` + `dir`, `raggy query` fails and asks you to index first.
