# Spreadsheet Database

An application that turns uploaded spreadsheet files into a queryable SQLite table plus a companion vector store, then lets users search and query that data with AI (semantic + keyword fusion, and natural-language-to-SQL).

## Storage

**Table (表)**
A dataset imported from one spreadsheet file, stored as a SQLite table identified by a sanitized name.
_Avoid_: sheet, dataset, worksheet

**Row (行)**
One record within a table, uniquely identified by `row_id`. Carries the original column values plus a derived `__row_text`.
_Avoid_: record

**Column (列)**
A named, typed field of a table. Types map to `INTEGER`, `REAL`, or `TEXT`.
_Avoid_: field, attribute

**Row Text (`__row_text`)**
The derived, concatenated text of a row (each column rendered as `col: value` and joined by ` | `). It is the source for embeddings and the BM25 corpus, and is not a user-facing column.
_Avoid_: row content, cell text

**Vector Table (`vec_` table)**
The `sqlite_vec` virtual table paired with a data table, holding each row's embedding keyed by `row_id`. Excluded from the list of user-facing tables.
_Avoid_: embedding table

## Ingestion

**Ingestion / Import (导入)**
The pipeline that reads a spreadsheet file, cleans it, creates a table, and builds embeddings. Driven by an ingest job.
_Avoid_: upload

**Spreadsheet File (电子表格文件)**
The xlsx / xls / csv source artifact for a table. `txt` is explicitly unsupported.
_Avoid_: upload, document

**Ingest Job (导入任务)**
A background unit of work tracking one import's progress (0–100%), status message, completion flag, and error.
_Avoid_: task, import process

**Embedding (向量)**
A 1024-dimensional float representation of a row's `__row_text`, stored in the row's vector table and used for semantic search.
_Avoid_: vector (use only as the Chinese gloss)

## Search

**Hybrid Search (混合搜索)**
The default search mode. Fuses semantic and BM25 rankings into one result list.
_Avoid_: combined search

**Semantic Search (语义搜索)**
Retrieval by vector similarity (distance) between the query embedding and row embeddings.

**BM25 Search (关键词搜索)**
Keyword ranking over tokenized row text using the BM25 algorithm.

**Reciprocal Rank Fusion / RRF**
The fusion method that combines multiple ranked result lists by summing `1 / (K + rank)`. `K = 60` here.
_Avoid_: rank blending

**Tokenize (分词)**
Splits text into individual CJK characters and latin/alphanumeric tokens, lowercased. This is the unit for BM25.
_Avoid_: split, segment

**Selected Tables (参与查询的表)**
The subset of imported tables the user checks to include in a search or NL2SQL query.

## Query & LLM

**Ask / NL2SQL (自然语言转SQL)**
A mode where a natural-language question is translated to SQL by an LLM, executed, and retried on error.
_Avoid_: chat, Q&A

**SQL Mode (手写SQL)**
Direct execution of user-written SQL against the selected tables.

**Schema (表结构)**
The columns and sample rows of a table, supplied to the LLM to ground NL2SQL generation.
_Avoid_: metadata, structure

**LLM (大模型)**
The external model services used by the app. Chat / NL2SQL / RAG route to a self-hosted custom platform (JAC); embedding and rerank route to SiliconFlow for inference speed. These are the only components that call outside the local process.
_Avoid_: model, AI
