# AGENTS.md - Raggy Development Guide

This file provides guidelines for agents working on the Raggy codebase.

## Project Overview

Raggy is a RAG (Retrieval Augmented Generation) MCP Server that provides semantic search over documents using BERT embeddings. It exposes tools via the MCP (Model Context Protocol) protocol.

## Build Commands

### Standard Build
```bash
cargo build          # Debug build
cargo build --release # Release build
```

### Running the Application
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

### Development Tools

```bash
cargo check          # Type-check without building
cargo clippy         # Run lints
cargo fmt            # Format code
cargo fmt --check    # Check formatting without changes
```

### Running Tests

This project has a test that requires a BERT model. The model must be downloaded before running tests.

```bash
./download_model.sh   # Download the model (first time only)
cargo test           # Run all tests
```

### Using Nix (Optional)

```bash
nix develop         # Enter development shell
nix build           # Build the package
```

## Code Style Guidelines

### General Principles

- Write clear, readable code with descriptive names
- Keep functions small and focused on a single responsibility
- Use early returns to reduce nesting
- Prefer composition over inheritance

### Imports

Group imports in the following order with blank lines between groups:

1. Standard library (`std::`)
2. External crates (alphabetical)
3. Local modules (`crate::`, `super::`)

```rust
use std::fs;
use std::path::PathBuf;

use anyhow::Result;
use candle_core::Tensor;
use serde::{Deserialize, Serialize};

use crate::module::Item;
```

### Formatting

- Use `cargo fmt` for automatic formatting
- Maximum line length: 100 characters (default rustfmt)
- Use 4 spaces for indentation
- Use trailing commas in multi-line collections
- Place braces on the same line for function definitions

### Types

- Use explicit type annotations for public API parameters and return types
- Prefer idiomatic Rust types: `&str` over `&String`, `&[T]` over `&Vec<T>`
- Use `Arc` for shared ownership across threads
- Use `RwLock` for interior mutability with multiple readers

```rust
// Good
fn process_data(input: &str) -> Result<Vec<ProcessingItem>>

// Avoid
fn process_data(input: &String) -> Vec<ProcessingItem>
```

### Naming Conventions

- **Variables/Functions**: `snake_case` (e.g., `load_model`, `chunk_size`)
- **Types/Structs/Enums**: `PascalCase` (e.g., `RaggyState`, `QueryParams`)
- **Constants**: `SCREAMING_SNAKE_CASE` (e.g., `MAX_CHUNK_SIZE`)
- **Modules**: `snake_case` (e.g., `mod indexer`)
- **Traits**: `PascalCase` with `-er` suffix when applicable (e.g., `Handler`)

### Error Handling

- Use `anyhow::Result<()>` for application-level error handling with context
- Use `thiserror` for defining custom error types with meaningful messages
- Use `?` operator for propagating errors
- Return `Result<T, ErrorData>` for MCP protocol errors

```rust
#[derive(Debug, thiserror::Error)]
pub enum RaggyError {
    #[error("Missing model file: {0}")]
    MissingModelFile(String),
    
    #[error("Indexing error: {0}")]
    IndexingError(String),
}

fn load_config(&self) -> Result<Config, RaggyError> {
    let content = fs::read_to_string(&self.path)
        .map_err(|e| RaggyError::IndexingError(format!("Failed to read: {}", e)))?;
    
    serde_json::from_str(&content)
        .map_err(|e| RaggyError::IndexingError(format!("Failed to parse: {}", e)))
}
```

### Async/Tokio

- Use `tokio::spawn` for background tasks
- Use `thread::spawn` when blocking operations are needed
- Be explicit about async vs sync boundaries

### Documentation

- Add doc comments (`///`) for public API items
- Keep documentation concise but informative
- Document error conditions for fallible functions

### Dependencies

This project uses:
- `anyhow` - Application error handling
- `thiserror` - Custom error enum derivation
- `clap` - CLI argument parsing
- `candle-core`, `candle-nn`, `candle-transformers` - ML framework
- `tokenizers` - Text tokenization
- `rmcp` - MCP protocol server
- `serde` - Serialization
- `tracing` - Logging

### Common Patterns

#### Using `parking_lot::RwLock`
```rust
struct State {
    data: RwLock<Vec<Item>>,
}

impl State {
    fn read_items(&self) -> Vec<Item> {
        self.data.read().clone()
    }
    
    fn update_items(&self, items: Vec<Item>) {
        *self.data.write() = items;
    }
}
```

#### Pattern Matching with Early Return
```rust
let value = match option {
    Some(v) => v,
    None => return Err(Error::MissingValue),
};
```

#### Using `?` with Context
```rust
let config: Config = serde_json::from_str(&content)
    .map_err(|e| MyError::ParseError(format!("Config error: {}", e)))?;
```

## Architecture Notes

- `main.rs` - Entry point, CLI parsing, server setup
- `RaggyState` - Central state management with RwLock for thread safety
- `RaggyServer` - MCP protocol handler implementation
- Text chunking with configurable overlap for retrieval
- BERT-based embeddings using mean pooling

## Development Workflow

1. Make changes to source code
2. Run `cargo fmt` to format
3. Run `cargo clippy` to catch common mistakes
4. Run `cargo check` to verify compilation
5. Test functionality with `cargo run --release`

## File Locations

- Source: `src/main.rs`
- Config: `Cargo.toml`, `rust-toolchain.toml`
- Models: Place in `models/<model-name>/` directory (e.g., `models/all-MiniLM-L6-v2/model.safetensors`)
