# Embedded vector store with fixed BGE-M3 / 1024-dim embeddings

We store row embeddings in an embedded `sqlite_vec` virtual table (`vec0`) co-located with the data table inside the same SQLite database, and we fix the embedding model at BGE-M3 producing 1024-dimensional vectors. This keeps the entire application self-contained in a single file with no separate vector service to deploy or maintain, and the fixed 1024-dim size is a hard contract that the `vec0(embedding float[1024])` declaration depends on.

**Considered Options**
- Embedded `sqlite_vec` (chosen): zero extra infrastructure, single-file portability, trivial backup.
- Separate vector database (e.g. Milvus, Qdrant, pgvector): more scalable and queryable, but adds a service, deployment, and a sync boundary we don't need at this scale.
- Variable / configurable embedding dimension: rejected because the `vec0` schema is compiled against a fixed width; changing the model would require rebuilding every vector table.

**Consequences**
- Swapping the embedding model later means re-ingesting all tables (rebuild every `vec_` table).
- The app has no external vector dependency, which is why `llm.embed` is the only network call besides NL2SQL.
