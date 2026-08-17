# WattBot

WattBot is a hybrid retrieval-augmented generation system that answers quantitative questions about the environmental cost of AI.
It was built for the UW-Madison Environmental AI Kaggle competition.

The competition gives you 32 source documents and a set of questions such as "What were the net CO2e emissions from training the GShard-600B model?" or "By what factor did inference energy per token drop between the two models?".
Answers have to be exact.
A question is only scored correct if the extracted value, its unit, and the document it came from all match the reference, so the system is built around precise numeric extraction rather than fluent prose.

## Results

87% accuracy on the 282 held-out test questions.

The interesting constraint was supervision.
`data/raw/train_QA.csv` contains 40 labelled question and answer pairs, which is far too few to fine-tune anything.
The accuracy therefore comes from retrieval quality and prompt structure rather than from model training, and the 40 labelled rows are spent as few-shot exemplars, selected by question type at inference time.

## How it works

The pipeline has four stages: ingest, index, retrieve, answer.

### Ingest

`src/data/pdf_processor.py` reads `data/raw/metadata.csv` and downloads all 32 source PDFs from their published URLs.
31 are research papers and one is the 2023 Amazon Sustainability Report.

Most of the answers live in tables and figures rather than in body text, so text extraction alone is not enough.
Each PDF goes through PyPDF2 for prose and Camelot for tables, running both the `lattice` and `stream` strategies and de-duplicating the overlap, since neither strategy alone handles every layout in the corpus.
Extracted tables are rendered as Markdown rather than raw CSV, which keeps the row and column relationships legible to the language model downstream.

### Chunking

`src/data/chunker.py` splits documents into roughly 1200-character chunks.

Chunking is section-aware.
The chunker first locates standard paper headings (Abstract, Introduction, Methods, Results, Discussion and so on) and chunks within those boundaries, so a chunk never straddles two sections.
Splits happen on paragraph boundaries, and each new chunk carries the last two sentences of its predecessor as overlap so that a fact split across a boundary is still retrievable.

Tables are treated as atomic.
A table is kept whole where it fits, and where it does not, it is split with its header row repeated into every piece so no fragment loses its column labels.

Every chunk is tagged with a `contains_numeric` flag, set by a regular expression covering the scientific units this corpus actually uses: kWh, MWh, GWh, TWh, CO2e, tCO2e, percent, lbs, kg, tons, GPU, TPU, FLOPs, PUE, WUE, watts, hours, days and years.
That flag is later used to bias retrieval toward chunks that can plausibly contain a numeric answer.

Chunks are written to `data/chunks/<doc_id>_chunks.json`.
The chunked corpus is committed, so you do not have to re-run ingestion to use the pipeline.

### Retrieval

Retrieval is hybrid, in `src/retrieval/hybrid_search.py`.

Two independent indexes are built over the same chunks:

1. A dense index of `BAAI/bge-large-en-v1.5` embeddings (1024 dimensions), L2-normalised so cosine similarity reduces to a dot product.
2. A BM25 index over Porter-stemmed tokens, via `rank_bm25`.

#### The dense index

Dense lookups go through an HNSW graph built with `hnswlib`, in `src/retrieval/dense_index.py`.
The graph is built in cosine space with `M=16` and `ef_construction=200`, and queried with `ef_search=100`.
All three are set in the `retrieval.vector_search.hnsw` block of `configs/config.yaml`.
Because the vectors are unit length, hnswlib's cosine distance converts back to a cosine similarity as `1 - distance`, so HNSW results arrive on exactly the same scale as the exhaustive scan and fusion downstream does not need to know which path produced them.

The index is persisted under `data/cache/hnsw/`, next to the embedding cache, as a `.bin` graph plus a small `.json` of its metadata.
Both are named after a fingerprint that hashes the embedding matrix together with the metric, `M` and `ef_construction`.
A changed corpus or a changed build parameter therefore produces a different fingerprint and a fresh build, so a stale graph can never be served against new embeddings.
`ef_search` is deliberately left out of the fingerprint, since it only widens the search at query time and does not change the graph that was built.
Building the index over the 4,498 chunks takes about a second, so a missing or superseded index is simply rebuilt rather than treated as an error.

The exhaustive numpy scan is still there as a second backend, selected with `retrieval.vector_search.backend: exact`.
It serves three purposes: it is the automatic fallback when `hnswlib` is not installed, it is the ground truth the recall tests measure HNSW against, and it is what document-filtered search in `src/retrieval/embeddings.py` uses, since that filter has to be applied before the top-k cut rather than after it.

#### Retrieval microbenchmark

The numbers below are a retrieval-layer microbenchmark on synthetic vectors at corpus scale.
They are not an end to end measurement.
They time a single dense lookup once a query vector already exists, so they exclude query encoding, BM25, fusion and generation, which together dominate real question latency.
The vectors are synthetic, generated to have the low rank structure, topic clusters and shared direction that sentence embeddings have, because reproducing the benchmark should not require downloading the embedding model.
Reproduce it with `python scripts/benchmark_hnsw.py`.

Measured on one core of a 12-core Qualcomm arm64 laptop CPU under WSL2, Python 3.11.9, numpy 2.3.1, hnswlib 0.8.0, 200 queries at `top_k=10`, median per-query latency:

| vectors | exact scan | HNSW | HNSW recall@10 |
| --- | --- | --- | --- |
| 4,498 (this corpus) | 0.25 ms | 1.20 ms | 0.999 |
| 25,000 | 2.06 ms | 1.37 ms | 0.989 |
| 100,000 | 9.28 ms | 1.71 ms | 0.946 |

At the size of this corpus HNSW is the slower of the two, by roughly a factor of five.
4,498 by 1024 floats is 18 MB, and a BLAS dot product over a block that size is fast and perfectly sequential, while a graph walk pays for scattered memory access and per-query Python overhead it cannot amortise.
The crossover sits somewhere above 20,000 vectors, and by 100,000 the graph is about five times faster than the scan.
So HNSW is here for the corpus growing, not for the corpus as it stands, and the exact backend remains a reasonable choice at this size.

Recall is measured against the exhaustive scan on the same synthetic vectors, and it is a property of the vector distribution as much as of the parameters, so it should not be read as a guarantee for real BGE embeddings.
The 0.999 at corpus scale is what `tests/test_dense_index.py` asserts a 0.95 floor on.

Both are needed.
Dense search handles paraphrase, where the question says "carbon footprint" and the paper says "net emissions".
BM25 handles the opposite failure, where the answer hinges on a rare literal token such as `GShard-600B` or `A100_80GB` that an embedding will happily smooth away.

The two ranked lists are merged with Reciprocal Rank Fusion at k=60.
RRF is used rather than a weighted score blend because BM25 scores and cosine similarities are not on a comparable scale, and fusing on rank sidesteps the need to normalise them.
A weighted-blend path is also implemented and selectable via the `method` argument.

On top of fusion, each question is expanded into up to three query variants before retrieval: the original question, a simplified form with the leading interrogative stripped, and either an entity-plus-metric keyword query or a table-oriented variant for questions that look numeric.
Every variant is retrieved separately and the results are unioned, with chunks found by more than one variant boosted.
The intent is that a chunk which surfaces under several phrasings of the same question is more likely to be the right one.

### Answer generation

`src/vector_pipeline.py` classifies each question into one of four types (true or false, named entity, calculation, numeric) and builds a type-specific prompt.
Each prompt carries two few-shot examples drawn from the training set for that same question type, plus the top 8 retrieved chunks with their source document tags.
Generation uses `claude-3-5-sonnet-20241022` at temperature 0.1 and is constrained to return JSON.

The output is then normalised into the competition schema.
True and false answers are coerced to `1` and `0`, units are stripped out of `answer_value` into `answer_unit`, ranges are collapsed to `[low,high]`, and cited `ref_id` values are validated against the ids in `metadata.csv` so the system cannot cite a document that does not exist.
Every field falls back to the sentinel string `is_blank`.

Knowing when *not* to answer matters here, because a confident wrong number scores worse than an abstention.
Two guards handle this.
Retrieval-time, a question whose key entities appear nowhere in the retrieved context is refused before an API call is made.
Generation-time, the prompt instructs the model to return `is_blank` rather than guess, and if `answer_value` comes back blank the post-processor cascades that blank through the dependent fields so no orphaned unit or citation survives.

## Running it

Requires an Anthropic API key, and Python 3.9 to 3.11.
The pinned numpy, torch and pandas versions have no wheels for 3.12.

```bash
pip install -r requirements.txt

cp .env.template .env
# then put your ANTHROPIC_API_KEY in .env
```

The chunked corpus is already in `data/chunks/`, so you can go straight to running the pipeline:

```bash
python run.py
```

`verify_setup.py` checks that the data, config and packages are all in place if something looks wrong:

```bash
python verify_setup.py
```

### Rebuilding the corpus from source

You only need this if you want to change the chunking strategy or refresh the source documents.
It downloads all 32 PDFs and re-chunks them, overwriting `data/chunks/`:

```bash
python src/data/pdf_processor.py    # download PDFs into data/raw/
python src/data/chunker.py          # chunk them into data/chunks/
```

`pdf_processor.py` also writes a flat text extraction per document to `data/processed/`, which is useful for inspecting what came out of a PDF.
The chunker does not read those files, it re-extracts from the PDFs itself so it can keep tables separate from prose.

`run.py` builds the index, reports accuracy against the training questions, and then prompts before processing the full test set.
The submission is written to `data/processed/submission.csv`, with checkpoints every 10 questions so a long run can survive an interruption.

Model, retrieval and path settings live in `configs/config.yaml`.

## Data layout

```
data/
  raw/          metadata.csv, train_QA.csv, test_Q.csv, and downloaded PDFs
  processed/    extracted text and the final submission.csv
  chunks/       per-document chunk JSON, one file per source document
  cache/        cached embedding matrices and the persisted HNSW index
```

The three CSVs and the chunked corpus are tracked in git, which is enough to run the pipeline after a clone.
`data/chunks/` holds 4,498 chunks across the 32 documents, 532 of them tables.
The PDFs, the extracted text, the embedding caches and the HNSW index are all derived and are ignored, since they are regenerated by the rebuild steps above.

`metadata.csv` is the source of truth for the corpus, with one row per document giving its id, type, title, year, citation and URL.
`train_QA.csv` has 40 fully labelled rows.
`test_Q.csv` has 282 rows where only `id`, `question` and the expected `answer_unit` are populated.
Both share the submission schema: `id`, `question`, `answer`, `answer_value`, `answer_unit`, `ref_id`, `ref_url`, `supporting_materials`, `explanation`.

## Layout

```
run.py                          entrypoint
verify_setup.py                 preflight check for data, config and packages
configs/config.yaml             model, retrieval and path settings
scripts/benchmark_hnsw.py       exact against HNSW retrieval microbenchmark
src/
  vector_pipeline.py            the pipeline used for the submission
  data/                         PDF download, extraction and chunking
  retrieval/                    hybrid BM25 and dense search, embedding cache
  retrieval/dense_index.py      HNSW index with an exact numpy fallback
  llm/                          answer generation and prompt construction
  knowledge_graph/              networkx entity and relation graph
  routing/                      question type classification
  pipeline/hybrid_pipeline.py   graph-augmented pipeline (in progress)
tests/
```

Two retrievers exist.
`src/retrieval/hybrid_search.py` is the one the submission pipeline uses.
`src/retrieval/embeddings.py` is a variant that adds document and section prefixes to each chunk before encoding, applies the BGE query instruction prefix, and caches encoded matrices to `data/cache/embeddings/` keyed by a hash of the input, so re-indexing an unchanged corpus skips the encode entirely.
The test suite is written against this second one.
Both share `src/retrieval/dense_index.py`, so both get the HNSW path and the exact fallback from the same place and the same config block.

`src/knowledge_graph/graph_builder.py` builds a networkx multigraph of typed entities (model, metric, value, hardware, method) and the relations between them (`TRAINED_ON`, `PRODUCES`, `REQUIRES`, `HAS_PUE`), extracted by pattern matching.
It is consumed by `src/pipeline/hybrid_pipeline.py`, which routes numeric and comparison questions through the graph first and falls back to plain retrieval.
That path is incomplete and was not used for the submitted results.

## Testing

```bash
pytest tests/test_dense_index.py -v
```

`tests/test_dense_index.py` holds 14 tests for the dense index: recall@10 of HNSW against the exhaustive scan, score agreement between the two backends, the persistence round trip, rebuild on a changed corpus or changed build parameters, the fallback when `hnswlib` is missing, and config parsing.
It runs on synthetic vectors at corpus scale, so it needs neither the embedding model nor the embedding cache, and it is self contained rather than part of the suite below.

```bash
pytest tests/ -v
```

`tests/test_embeddings.py` holds 48 tests covering the chunk data model, indexing and embedding normalisation, dense and hybrid search, document filtering, index save and load, cache hits, and retrieval behaviour on the real question shapes from the training set.
Coverage settings are in `.coveragerc`.

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

## Known issues

The committed chunks in `data/chunks/` were produced by an earlier revision of the chunker and carry no `section` labels.
The loader handles this, since it recomputes the `contains_numeric` flag from the chunk text and treats a missing section as `None`, so retrieval works as described.
The only loss is the small section-match boost in reranking and the section tag in the prompt context.
Re-running `src/data/chunker.py` regenerates them with section metadata.

`tests/test_embeddings.py` imports `ScientificPaperRetriever`, a SciBERT-backed subclass that was removed from `src/retrieval/embeddings.py` in a later commit.
The suite will not collect until that class is restored or the tests that use it are updated.

`src/pipeline.py` and the `src/pipeline/` package share a name, which makes `src.pipeline` ambiguous.
`src/pipeline.py` is an unused stub and should be deleted.

Several modules under `src/llm/` and `src/evaluation/` are stubs with unimplemented method bodies.
