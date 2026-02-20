Create an MCP server written in rust. 
It's a RAG system, it will be launched with these parameters:

raggy --model embedding-model.gguf --dir /directory-to-index

Optional parameters:
--extensions .txt,.md           # comma-separated list of file extensions to index (default: .txt,.md)
--chunk-size 512               # size of text chunks in characters (default: 512)
--chunk-overlap 50             # overlap between chunks in characters (default: 50)

Phase 2 (not implemented yet):
--re-ranker reranker-model.gguf # path to re-ranker model

Dependencies:
- Candle framework with: candle-core, candle-nn, candle-transformers
- candle-transformers provides quantized_nn and quantized_var_builder for GGUF support
- rmcp for MCP protocol

Initialization:
- MCP initialize must be fast - only verify the model file exists
- Model loading and indexing happen in a background thread
- This ensures the MCP server passes availability tests from MCP clients

Indexing:
- During initialization spawns a background thread for indexing
- Recursive search for text files in the given directory
- Supported extensions: .txt and .md (configurable via --extensions)
- Text is split into chunks using newlines as delimiter
- Chunks have configurable size (default: 512 chars)
- Overlaps are sized to end at newlines when possible (default: ~50 chars)
- Create embeddings using the given model and save the data in memory
- Embeddings are L2-normalized before storage
- The index tool has the same logic as the initialization indexing

MCP Tools:

1. raggy_query
   Parameters:
   - question: String (required) - the search query
   - top_k: Integer (optional, default: 10) - number of results to return
   
   Behavior:
   - If indexing is in progress, returns partial results with what is indexed so far
   - Includes a message indicating indexing status
   - Question goes through the same embedding model
   - Embedding is L2-normalized
   - Cosine similarity is computed against all indexed chunks
   - Returns top_k results sorted by similarity score

2. raggy_index
   Parameters: none
   
   Behavior:
   - Triggers re-indexing of the directory
   - Same logic as initialization indexing
   - Returns immediately, indexing happens in background

Response format (JSON):
{
  "results": [
    {
      "path": "/path/to/file.md",
      "chunk": "The relevant text chunk...",
      "start_line": 42,
      "end_line": 58,
      "score": 0.95
    },
    ...
  ],
  "indexing_status": "complete" | "in_progress" | "not_started"
}

Re-ranker (Phase 2):
- Takes top 50 results from embedding cosine similarity
- Re-ranks using cross-encoder scoring with the re-ranker model

Error handling:
- Define an error enum to handle: MissingModelFile, CorruptModel, EmptyDirectory, UnsupportedModelArchitecture, IndexingError, QueryError

It will have a flake.nix and the repository will be pushed at https://github.com/RCasatta/raggy so that it can be launched like that `nix run github:RCasatta/raggy --model xxx --dir yyy`
