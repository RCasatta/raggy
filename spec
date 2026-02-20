Create an MCP server written in rust. For the structure have a look at ./easy-memory-mcp
It's a RAG system, it will be launched with these parameters:

raggy --model embedding-model.gguf --dir /directory-to-index

Optional parameters:
--extensions .txt,.md           # comma-separated list of file extensions to index (default: .txt,.md)
--chunk-size 512               # size of text chunks in characters (default: 512)
--chunk-overlap 50             # overlap between chunks in characters (default: 50)

Phase 2 (not implemented yet):
--re-ranker reranker-model.gguf # path to re-ranker model

It will use Candle framework to run GGUF models.

Indexing:
- During initialization will spawn a background thread doing the indexing
- Recursive search for text files in the given directory
- Supported extensions: .txt and .md (configurable via --extensions)
- Text is split into chunks using newlines as delimiter
- Chunks have configurable size (default: 512 chars) and overlap (default: 50 chars)
- Create embeddings using the given model and save the data in memory
- The index tool has the same logic as the initialization indexing

Query endpoint:
- Takes a question as input
- The question goes through the same embedding model
- Cosine similarity is computed to find best matches
- Returns the list of the top 10 results with better scores
- Each result contains: file path, chunk text, chunk start line number, similarity score

Response format (JSON):
{
  "results": [
    {
      "path": "/path/to/file.md",
      "chunk": "The relevant text chunk...",
      "line": 42,
      "score": 0.95
    },
    ...
  ]
}

Re-ranker (Phase 2):
- Takes top 50 results from embedding cosine similarity
- Re-ranks using cross-encoder scoring with the re-ranker model

Error handling:
- Define an error enum to handle: MissingModelFile, CorruptModel, EmptyDirectory, UnsupportedModelArchitecture, IndexingError, QueryError

It will have a flake.nix and the repository will be pushed at https://github.com/RCasatta/raggy so that it can be launched like that `nix run github:RCasatta/raggy --model xxx --dir yyy`
