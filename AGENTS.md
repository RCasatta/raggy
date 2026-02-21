# AGENTS.md - Raggy Development Guide

This file provides guidelines for agents working on the Raggy codebase.

## Project Overview

Raggy is a RAG (Retrieval Augmented Generation) MCP Server that provides semantic search over documents using BERT embeddings. It exposes tools via the MCP (Model Context Protocol) protocol.

## Build Commands

### Standard Build
```bash
cargo build           # Debug build
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
cargo test --release  # Run all tests, some tests are demanding and release flag is suggested
```

### Nix / direnv Environment

This project provides a Nix development shell via `direnv` / `nix-direnv`. Some agents may already operate inside an activated Nix environment where tools like `cargo` are available directly. Others (e.g., sandboxed agents) may not.

If a build command fails with missing toolchain or dependency errors, retry it through the cached Nix environment:

```bash
direnv exec . <command>   # e.g. direnv exec . cargo check
```

`direnv exec .` uses the cached `nix-direnv` environment and is effectively instant — there is no need to run `nix develop` interactively.

```bash
nix build           # Build the Nix package directly
```

When creating bash scripts in this project, use `#!/usr/bin/env bash` as the shebang for NixOS compatibility.

## Code Style Guidelines

### General Principles

- Write clear, readable code with descriptive names
- Keep functions small and focused on a single responsibility
- Use early returns to reduce nesting
- Prefer composition over inheritance
- Avoid adding new dependencies unless explicitly asked; prefer standard library implementations
- Use interpolated variables in format strings: `format!("msg {var}")` instead of `format!("msg {}", var)`

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

### Logging

This project uses the `tracing` crate. Use `tracing::info!`, `tracing::warn!`, etc. instead of `println!` for runtime output.

### Dependencies

Avoid adding new dependencies unless explicitly asked. Prefer standard library implementations when possible.

This project uses:
- `anyhow` - Application error handling
- `thiserror` - Custom error enum derivation
- `clap` - CLI argument parsing
- `candle-core`, `candle-nn`, `candle-transformers` - ML framework
- `tokenizers` - Text tokenization
- `rmcp` - MCP protocol server
- `serde` - Serialization
- `tracing` - Logging

## Architecture Notes

- `main.rs` - Entry point, CLI parsing, server setup
- `RaggyState` - Central state management with RwLock for thread safety
- `RaggyServer` - MCP protocol handler implementation
- Text chunking with configurable overlap for retrieval
- BERT-based embeddings using mean pooling

## Development Workflow

1. Make changes to source code
2. Run `cargo fmt` to format
3. Run `cargo clippy` to catch common mistakes — verify zero warnings on new code
4. Run `cargo check --tests` to verify compilation (preferred over `cargo build` for speed)
5. Test functionality with `cargo test --release`

## Git Commit Conventions

- **Format:** `context: <description>`
- **Context:** Use the specific crate/directory name, or categories like `ci`, `fix`, `feat`, `docs`, or `refactor`.
- **Breaking Changes:** Append `!` after the context (e.g., `feat!: rewrite api`).
- **Title (first line):**
  - Max 50 characters
  - Use imperative mood ("add", not "added")
  - No period at the end
- **Body:**
  - Separate from title with a blank line
  - Focus on "why"; use bullet points for multiple items

## File Locations

- Source: `src/main.rs`
- Config: `Cargo.toml`, `rust-toolchain.toml`
- Models: Place in `models/<model-name>/` directory (e.g., `models/all-MiniLM-L6-v2/model.safetensors`)
