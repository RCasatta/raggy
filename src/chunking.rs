pub fn chunk_text(
    text: &str,
    chunk_size: usize,
    chunk_overlap: usize,
) -> Vec<(String, usize, usize)> {
    let lines: Vec<&str> = text.lines().collect();
    let mut chunks = Vec::new();
    let mut current_chunk = String::new();
    let mut start_line = 1;
    let mut current_line = 1;

    for (idx, line) in lines.iter().enumerate() {
        current_line = idx + 1;

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
