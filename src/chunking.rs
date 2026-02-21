use tree_sitter::Parser;

pub enum ChunkStrategy {
    Default,
    Markdown,
    Rust,
}

impl ChunkStrategy {
    pub fn for_extension(ext: &str) -> Self {
        match ext {
            "md" | "markdown" => Self::Markdown,
            "rs" => Self::Rust,
            _ => Self::Default,
        }
    }
}

struct Section {
    text: String,
    start_line: usize,
    end_line: usize,
}

pub fn chunk_text(
    text: &str,
    chunk_size: usize,
    chunk_overlap: usize,
    strategy: &ChunkStrategy,
) -> Vec<(String, usize, usize)> {
    let sections = split_into_sections(text, strategy);
    if sections.is_empty() {
        return Vec::new();
    }
    merge_sections(sections, chunk_size, chunk_overlap)
}

fn split_into_sections(text: &str, strategy: &ChunkStrategy) -> Vec<Section> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }
    match strategy {
        ChunkStrategy::Default => split_by_blank_lines(&lines),
        ChunkStrategy::Markdown => split_by_headings(&lines),
        ChunkStrategy::Rust => split_by_rust_ast(text, &lines),
    }
}

// --- Default: split on blank lines (paragraph-aware) ---

fn split_by_blank_lines(lines: &[&str]) -> Vec<Section> {
    let mut sections = Vec::new();
    let mut buf: Vec<&str> = Vec::new();
    let mut start = 1usize;

    for (idx, &line) in lines.iter().enumerate() {
        let ln = idx + 1;
        if line.trim().is_empty() {
            if !buf.is_empty() {
                sections.push(Section {
                    text: buf.join("\n"),
                    start_line: start,
                    end_line: ln - 1,
                });
                buf.clear();
            }
        } else {
            if buf.is_empty() {
                start = ln;
            }
            buf.push(line);
        }
    }

    if !buf.is_empty() {
        sections.push(Section {
            text: buf.join("\n"),
            start_line: start,
            end_line: lines.len(),
        });
    }

    sections
}

// --- Markdown: split on heading lines ---

fn split_by_headings(lines: &[&str]) -> Vec<Section> {
    let mut sections = Vec::new();
    let mut buf: Vec<&str> = Vec::new();
    let mut start = 1usize;

    for (idx, &line) in lines.iter().enumerate() {
        let ln = idx + 1;

        if is_markdown_heading(line) && !buf.is_empty() {
            emit_section(&mut sections, &buf, start);
            buf.clear();
        }

        if buf.is_empty() && line.trim().is_empty() {
            continue;
        }

        if buf.is_empty() {
            start = ln;
        }
        buf.push(line);
    }

    if !buf.is_empty() {
        emit_section(&mut sections, &buf, start);
    }

    sections
}

fn is_markdown_heading(line: &str) -> bool {
    let trimmed = line.trim_start_matches('#');
    line.len() != trimmed.len() && trimmed.starts_with(' ')
}

// --- Rust: tree-sitter AST-aware splitting ---
// Parses the source with tree-sitter and splits at top-level item boundaries.
// Comments (extras) are accumulated as preamble and attached to the next item.
// Falls back to a heuristic splitter if parsing fails.

fn split_by_rust_ast(text: &str, lines: &[&str]) -> Vec<Section> {
    let mut parser = Parser::new();
    let lang: tree_sitter::Language = tree_sitter_rust::LANGUAGE.into();
    if parser.set_language(&lang).is_err() {
        return split_by_rust_heuristic(lines);
    }
    let Some(tree) = parser.parse(text, None) else {
        return split_by_rust_heuristic(lines);
    };

    let root = tree.root_node();
    let mut sections = Vec::new();
    let mut cursor = root.walk();

    // Byte offset + row of the first accumulated preamble comment
    let mut preamble_start: Option<(usize, usize)> = None;

    for child in root.children(&mut cursor) {
        if !child.is_named() {
            continue;
        }

        let is_preamble =
            child.is_extra() || matches!(child.kind(), "attribute_item" | "inner_attribute_item");

        if is_preamble {
            if preamble_start.is_none() {
                preamble_start = Some((child.start_byte(), child.start_position().row));
            }
            continue;
        }

        let (start_byte, start_row) =
            preamble_start.unwrap_or((child.start_byte(), child.start_position().row));
        let end_byte = child.end_byte();
        let end_row = child.end_position().row;

        let section_text = text[start_byte..end_byte].trim_end();
        if !section_text.is_empty() {
            sections.push(Section {
                text: section_text.to_string(),
                start_line: start_row + 1,
                end_line: end_row + 1,
            });
        }

        preamble_start = None;
    }

    // Trailing comments with no following item
    if let Some((start_byte, start_row)) = preamble_start {
        let section_text = text[start_byte..].trim_end();
        if !section_text.is_empty() {
            let end_line = start_row + section_text.lines().count();
            sections.push(Section {
                text: section_text.to_string(),
                start_line: start_row + 1,
                end_line,
            });
        }
    }

    if sections.is_empty() {
        return split_by_rust_heuristic(lines);
    }

    sections
}

/// Heuristic fallback: split at blank lines followed by non-indented Rust item
/// keywords. Used when tree-sitter parsing fails.
fn split_by_rust_heuristic(lines: &[&str]) -> Vec<Section> {
    let mut sections = Vec::new();
    let mut buf: Vec<&str> = Vec::new();
    let mut start = 1usize;
    let mut prev_blank = false;

    for (idx, &line) in lines.iter().enumerate() {
        let ln = idx + 1;
        let is_blank = line.trim().is_empty();

        if !is_blank && prev_blank && is_rust_item_start(line) && !buf.is_empty() {
            emit_section(&mut sections, &buf, start);
            buf.clear();
        }

        if buf.is_empty() && is_blank {
            prev_blank = true;
            continue;
        }

        if buf.is_empty() {
            start = ln;
        }
        buf.push(line);
        prev_blank = is_blank;
    }

    if !buf.is_empty() {
        emit_section(&mut sections, &buf, start);
    }

    sections
}

const RUST_ITEM_PREFIXES: &[&str] = &[
    "fn ",
    "pub fn ",
    "pub(crate) fn ",
    "pub(super) fn ",
    "async fn ",
    "pub async fn ",
    "pub(crate) async fn ",
    "impl ",
    "impl<",
    "struct ",
    "pub struct ",
    "pub(crate) struct ",
    "enum ",
    "pub enum ",
    "pub(crate) enum ",
    "mod ",
    "pub mod ",
    "pub(crate) mod ",
    "trait ",
    "pub trait ",
    "pub(crate) trait ",
    "type ",
    "pub type ",
    "pub(crate) type ",
    "const ",
    "pub const ",
    "pub(crate) const ",
    "static ",
    "pub static ",
    "macro_rules!",
    "use ",
    "pub use ",
    "pub(crate) use ",
    "#[",
    "/// ",
    "//! ",
];

fn is_rust_item_start(line: &str) -> bool {
    if line.starts_with(' ') || line.starts_with('\t') {
        return false;
    }
    RUST_ITEM_PREFIXES.iter().any(|p| line.starts_with(p))
}

// --- Shared helpers ---

fn emit_section(sections: &mut Vec<Section>, buf: &[&str], start: usize) {
    let mut end_offset = buf.len();
    while end_offset > 0 && buf[end_offset - 1].trim().is_empty() {
        end_offset -= 1;
    }
    if end_offset > 0 {
        sections.push(Section {
            text: buf[..end_offset].join("\n"),
            start_line: start,
            end_line: start + end_offset - 1,
        });
    }
}

/// Merge small consecutive sections into chunks up to `chunk_size`.
/// Sections that exceed `chunk_size` on their own are sub-split line-by-line.
fn merge_sections(
    sections: Vec<Section>,
    chunk_size: usize,
    chunk_overlap: usize,
) -> Vec<(String, usize, usize)> {
    let mut chunks = Vec::new();
    let mut current_text = String::new();
    let mut current_start = 0usize;
    let mut current_end = 0usize;

    for section in sections {
        if section.text.len() > chunk_size {
            if !current_text.is_empty() {
                chunks.push((current_text, current_start, current_end));
                current_text = String::new();
            }
            chunks.extend(split_lines_with_overlap(
                &section.text,
                chunk_size,
                chunk_overlap,
                section.start_line,
            ));
            continue;
        }

        let sep_len = if current_text.is_empty() { 0 } else { 2 };
        let new_len = current_text.len() + sep_len + section.text.len();

        if new_len > chunk_size && !current_text.is_empty() {
            chunks.push((current_text, current_start, current_end));
            current_text = section.text;
            current_start = section.start_line;
            current_end = section.end_line;
        } else {
            if current_text.is_empty() {
                current_start = section.start_line;
            } else {
                current_text.push_str("\n\n");
            }
            current_text.push_str(&section.text);
            current_end = section.end_line;
        }
    }

    if !current_text.is_empty() {
        chunks.push((current_text, current_start, current_end));
    }

    chunks
}

/// Fallback line-by-line splitter for oversized sections.
fn split_lines_with_overlap(
    text: &str,
    chunk_size: usize,
    chunk_overlap: usize,
    base_line: usize,
) -> Vec<(String, usize, usize)> {
    let lines: Vec<&str> = text.lines().collect();
    let mut chunks = Vec::new();
    let mut current_chunk = String::new();
    let mut start_line = base_line;
    let mut current_line = base_line;

    for (idx, &line) in lines.iter().enumerate() {
        current_line = base_line + idx;

        if current_chunk.len() + line.len() + 1 > chunk_size && !current_chunk.is_empty() {
            chunks.push((current_chunk.clone(), start_line, current_line - 1));
            let overlap_text = find_overlap(&current_chunk, chunk_overlap);
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

pub fn find_overlap(text: &str, overlap_size: usize) -> String {
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

    chars[chars.len().saturating_sub(overlap_size)..]
        .iter()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_for_extension() {
        assert!(matches!(
            ChunkStrategy::for_extension("md"),
            ChunkStrategy::Markdown
        ));
        assert!(matches!(
            ChunkStrategy::for_extension("markdown"),
            ChunkStrategy::Markdown
        ));
        assert!(matches!(
            ChunkStrategy::for_extension("rs"),
            ChunkStrategy::Rust
        ));
        assert!(matches!(
            ChunkStrategy::for_extension("txt"),
            ChunkStrategy::Default
        ));
        assert!(matches!(
            ChunkStrategy::for_extension("py"),
            ChunkStrategy::Default
        ));
    }

    // --- Default strategy ---

    #[test]
    fn test_default_merges_small_paragraphs() {
        let text = "Para one\n\nPara two\n\nPara three";
        let chunks = chunk_text(text, 1000, 0, &ChunkStrategy::Default);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].0.contains("Para one"));
        assert!(chunks[0].0.contains("Para three"));
    }

    #[test]
    fn test_default_splits_when_exceeding_size() {
        let text = "First paragraph here\n\nSecond paragraph here\n\nThird paragraph here";
        let chunks = chunk_text(text, 50, 0, &ChunkStrategy::Default);
        assert!(chunks.len() >= 2, "got {} chunks", chunks.len());
    }

    #[test]
    fn test_default_empty_text() {
        let chunks = chunk_text("", 512, 50, &ChunkStrategy::Default);
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_default_single_paragraph() {
        let text = "Just one line";
        let chunks = chunk_text(text, 512, 0, &ChunkStrategy::Default);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].0, "Just one line");
        assert_eq!(chunks[0].1, 1);
        assert_eq!(chunks[0].2, 1);
    }

    // --- Markdown strategy ---

    #[test]
    fn test_markdown_splits_on_headings() {
        let text =
            "# Title\n\nIntro text\n\n## Section A\n\nContent A\n\n## Section B\n\nContent B";
        let chunks = chunk_text(text, 40, 0, &ChunkStrategy::Markdown);
        assert!(chunks.len() >= 2, "got {} chunks", chunks.len());
        assert!(chunks[0].0.starts_with("# Title"));
    }

    #[test]
    fn test_markdown_merges_small_sections() {
        let text = "# Title\n\nIntro\n\n## A\n\nSmall\n\n## B\n\nSmall";
        let chunks = chunk_text(text, 1000, 0, &ChunkStrategy::Markdown);
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn test_markdown_preserves_content_before_first_heading() {
        let text = "Some preamble text\n\n# Heading\n\nBody";
        let chunks = chunk_text(text, 30, 0, &ChunkStrategy::Markdown);
        assert!(chunks[0].0.contains("preamble"));
    }

    #[test]
    fn test_markdown_heading_detection() {
        assert!(is_markdown_heading("# Title"));
        assert!(is_markdown_heading("## Sub"));
        assert!(is_markdown_heading("### Deep"));
        assert!(!is_markdown_heading("#notatitle"));
        assert!(!is_markdown_heading("not a heading"));
        assert!(!is_markdown_heading("#[derive(Debug)]"));
    }

    // --- Rust strategy (tree-sitter) ---

    #[test]
    fn test_rust_keeps_function_body_together() {
        let text = "\
fn foo() {
    let x = 1;

    let y = 2;
}";
        let lines: Vec<&str> = text.lines().collect();
        let sections = split_by_rust_ast(text, &lines);
        assert_eq!(
            sections.len(),
            1,
            "blank line inside function body should not split"
        );
        assert!(sections[0].text.contains("let y = 2;"));
    }

    #[test]
    fn test_rust_splits_between_items() {
        let text = "\
fn foo() {
    let x = 1;
}

fn bar() {
    let y = 2;
}";
        let lines: Vec<&str> = text.lines().collect();
        let sections = split_by_rust_ast(text, &lines);
        assert_eq!(sections.len(), 2);
        assert!(sections[0].text.contains("foo"));
        assert!(sections[1].text.contains("bar"));
    }

    #[test]
    fn test_rust_doc_comment_stays_with_item() {
        let text = "\
fn foo() {}

/// Documentation for bar
fn bar() {}";
        let lines: Vec<&str> = text.lines().collect();
        let sections = split_by_rust_ast(text, &lines);
        assert_eq!(sections.len(), 2);
        assert!(
            sections[1].text.contains("/// Documentation for bar"),
            "doc comment should be in same section as item"
        );
        assert!(sections[1].text.contains("fn bar"));
    }

    #[test]
    fn test_rust_attribute_stays_with_item() {
        let text = "\
fn foo() {}

#[derive(Debug)]
struct Bar {
    x: i32,
}";
        let lines: Vec<&str> = text.lines().collect();
        let sections = split_by_rust_ast(text, &lines);
        assert_eq!(sections.len(), 2);
        assert!(sections[1].text.contains("#[derive(Debug)]"));
        assert!(sections[1].text.contains("struct Bar"));
    }

    #[test]
    fn test_rust_use_groups_split() {
        let text = "\
use std::fs;
use std::path::PathBuf;

use anyhow::Result;";
        let lines: Vec<&str> = text.lines().collect();
        let sections = split_by_rust_ast(text, &lines);
        assert_eq!(sections.len(), 3);
        assert!(sections[0].text.contains("std::fs"));
        assert!(sections[2].text.contains("anyhow"));
    }

    #[test]
    fn test_rust_impl_block_stays_together() {
        let text = "\
struct Foo;

impl Foo {
    fn new() -> Self {
        Foo
    }

    fn method(&self) -> i32 {
        42
    }
}";
        let lines: Vec<&str> = text.lines().collect();
        let sections = split_by_rust_ast(text, &lines);
        assert_eq!(sections.len(), 2);
        assert!(sections[0].text.contains("struct Foo"));
        assert!(sections[1].text.contains("impl Foo"));
        assert!(
            sections[1].text.contains("fn method"),
            "entire impl block should be one section"
        );
    }

    #[test]
    fn test_rust_chunks_respect_size() {
        let text = "\
fn foo() {
    let x = 1;
}

fn bar() {
    let y = 2;
}";
        let chunks = chunk_text(text, 30, 0, &ChunkStrategy::Rust);
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].0.contains("foo"));
        assert!(chunks[1].0.contains("bar"));
    }

    #[test]
    fn test_rust_module_doc_comment() {
        let text = "\
//! Module documentation
//! Second line

use std::fs;

fn main() {}";
        let lines: Vec<&str> = text.lines().collect();
        let sections = split_by_rust_ast(text, &lines);
        assert!(sections.len() >= 2);
        assert!(
            sections[0].text.contains("//! Module documentation"),
            "module doc should be in first section"
        );
    }

    // --- Line numbers ---

    #[test]
    fn test_line_numbers_default() {
        let text = "line one\n\nline three\n\nline five";
        let chunks = chunk_text(text, 22, 0, &ChunkStrategy::Default);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].1, 1);
        assert_eq!(chunks[0].2, 3);
        assert_eq!(chunks[1].1, 5);
        assert_eq!(chunks[1].2, 5);
    }

    #[test]
    fn test_line_numbers_rust() {
        let text = "\
fn foo() {
    let x = 1;
}

fn bar() {
    let y = 2;
}";
        let chunks = chunk_text(text, 30, 0, &ChunkStrategy::Rust);
        assert_eq!(chunks[0].1, 1); // foo starts line 1
        assert_eq!(chunks[0].2, 3); // foo ends line 3
        assert_eq!(chunks[1].1, 5); // bar starts line 5
        assert_eq!(chunks[1].2, 7); // bar ends line 7
    }

    // --- Oversized section fallback ---

    #[test]
    fn test_oversized_section_uses_line_fallback() {
        let text = "# Big heading\n\nline a\nline b\nline c\nline d\nline e\nline f";
        let chunks = chunk_text(text, 20, 0, &ChunkStrategy::Markdown);
        assert!(
            chunks.len() >= 2,
            "oversized section should be sub-split, got {} chunks",
            chunks.len()
        );
    }
}
