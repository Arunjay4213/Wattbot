# WattBot

WattBot is a hybrid retrieval-augmented generation system that answers quantitative questions about the environmental cost of AI.
It was built for the UW-Madison Environmental AI Kaggle competition.

The competition gives you 32 source documents and a set of questions such as "What were the net CO2e emissions from training the GShard-600B model?" or "By what factor did inference energy per token drop between the two models?".
Answers have to be exact.
A question only counts as correct if the extracted value, its unit and the document it came from all match the reference, so most of the system is about pulling out the right number and citing the right source.

## Results

87% accuracy on the 282 held-out test questions.

There was very little labelled data to work with.
`data/raw/train_QA.csv` has 40 labelled question and answer pairs, far too few to fine-tune anything, so the accuracy comes from retrieval quality and prompt structure.
The 40 labelled rows are used as few-shot examples instead, picked by question type at inference time.

## How it works

The pipeline has four stages: ingest, index, retrieve, answer.

### Ingest

`src/data/pdf_processor.py` reads `data/raw/metadata.csv` and downloads all 32 source PDFs from their published URLs.
31 are research papers and one is the 2023 Amazon Sustainability Report.

Most of the answers are in tables and figures, so extracting the body text alone is not enough.
Each PDF goes through PyPDF2 for prose and Camelot for tables.
Camelot runs with both the `lattice` and `stream` strategies and the overlap is de-duplicated, because neither strategy handles every layout in the corpus on its own.
Extracted tables are written out as Markdown instead of raw CSV so the language model can still tell which value belongs to which row and column.

### Chunking

`src/data/chunker.py` splits documents into chunks of roughly 1200 characters.
It is section-aware: it first finds the standard paper headings (Abstract, Introduction, Methods, Results, Discussion and so on) and chunks inside those boundaries, so a chunk never spans two sections.
Splits happen on paragraph boundaries.
Each new chunk repeats the last two sentences of the one before it, so a fact that falls across a boundary can still be retrieved.

Tables are kept whole where they fit.
A table that does not fit is split with its header row repeated in every piece, so no fragment loses its column labels.

Every chunk gets a `contains_numeric` flag from a regular expression over the units this corpus actually uses: kWh, MWh, GWh, TWh, CO2e, tCO2e, percent, lbs, kg, tons, GPU, TPU, FLOPs, PUE, WUE, watts, hours, days and years.
Retrieval later uses the flag to favour chunks that could plausibly hold a numeric answer.

Chunks are written to `data/chunks/<doc_id>_chunks.json`.
The chunked corpus is committed, so you do not have to re-run ingestion to use the pipeline.

### Retrieval

Retrieval, in `src/retrieval/hybrid_search.py`, is hybrid.
It builds two independent indexes over the same chunks:

1. A dense index of `BAAI/bge-large-en-v1.5` embeddings (1024 dimensions), L2-normalised so cosine similarity reduces to a dot product.
2. A BM25 index over Porter-stemmed tokens, via `rank_bm25`.

Each one covers a failure of the other.
Dense search handles paraphrase, as when the question says "carbon footprint" and the paper says "net emissions".
BM25 handles questions where the answer hinges on a rare literal token such as `GShard-600B` or `A100_80GB`, which an embedding tends to smooth away.

The two ranked lists are merged with Reciprocal Rank Fusion at k=60.
BM25 scores and cosine similarities are on different scales, and fusing on rank avoids having to normalise them.
A weighted score blend is also implemented and can be selected with the `method` argument.

Before retrieval, each question is expanded into up to three query variants: the original question, a simplified form with the leading question word removed, and either an entity-plus-metric keyword query or a table-oriented variant for questions that look numeric.
Each variant is retrieved separately and the results are merged, with a boost for chunks that more than one variant found.
The idea is that a chunk which turns up under several phrasings of the same question is more likely to be the right one.

#### The dense index

Dense lookups go through an HNSW graph built with `hnswlib`, in `src/retrieval/dense_index.py`.
The graph is built in cosine space with `M=16` and `ef_construction=200` and queried with `ef_search=100`; all three are set in the `retrieval.vector_search.hnsw` block of `configs/config.yaml`.
Because the vectors have unit length, hnswlib's cosine distance converts back to cosine similarity as `1 - distance`.
HNSW results therefore come out on the same scale as the exhaustive scan, and the fusion step does not need to know which backend produced them.

The index is saved under `data/cache/hnsw/`, next to the embedding cache, as a `.bin` graph plus a small `.json` file of metadata.
Both files are named after a fingerprint that hashes the embedding matrix together with the metric, `M` and `ef_construction`.
If the corpus or a build parameter changes, the fingerprint changes and the graph is rebuilt, so a stale graph is never served against new embeddings.
`ef_search` is left out of the fingerprint on purpose, because it only widens the search at query time and does not change the graph that was built.
Building the index over the 4,498 chunks takes about a second, so a missing or outdated index is rebuilt instead of being treated as an error.

The exhaustive numpy scan is still there as a second backend, selected with `retrieval.vector_search.backend: exact`.
It is the automatic fallback when `hnswlib` is not installed and the ground truth the recall tests compare HNSW against.
Document-filtered search in `src/retrieval/embeddings.py` also uses it, since that filter has to be applied before the top-k cut.

#### Retrieval microbenchmark

The numbers below come from a microbenchmark of the retrieval layer on synthetic vectors at corpus scale, and they do not measure end-to-end latency.
Each one times a single dense lookup after the query vector already exists, so query encoding, BM25, fusion and generation are excluded, and those make up most of the time a real question takes.
The vectors are synthetic so that reproducing the benchmark does not require downloading the embedding model; they are generated with the low-rank structure, topic clusters and shared direction that sentence embeddings have.
Reproduce it with `python scripts/benchmark_hnsw.py`.

Measured on one core of a 12-core Qualcomm arm64 laptop CPU under WSL2, Python 3.11.9, numpy 2.3.1, hnswlib 0.8.0, 200 queries at `top_k=10`, median per-query latency:

| vectors | exact scan | HNSW | HNSW recall@10 |
| --- | --- | --- | --- |
| 4,498 (this corpus) | 0.25 ms | 1.20 ms | 0.999 |
| 25,000 | 2.06 ms | 1.37 ms | 0.989 |
| 100,000 | 9.28 ms | 1.71 ms | 0.946 |

At this corpus size HNSW is the slower of the two, by roughly a factor of five.
4,498 by 1024 floats is 18 MB, and a BLAS dot product over a block that size is fast and reads memory in order, while a graph walk pays for scattered memory access and per-query Python overhead that it cannot amortise.
The crossover is somewhere above 20,000 vectors, and at 100,000 the graph is about five times faster than the scan.
HNSW is there for when the corpus grows, and at the current size the exact backend is still a reasonable choice.

Recall is measured against the exhaustive scan on the same synthetic vectors.
It depends on the vector distribution as much as on the parameters, so it is not a guarantee for real BGE embeddings.
`tests/test_dense_index.py` asserts a 0.95 floor on the corpus-scale figure, which measures 0.999 here.

### Answer generation

`src/vector_pipeline.py` sorts each question into one of four types (true or false, named entity, calculation, numeric) and builds a prompt for that type.
The prompt carries two few-shot examples of the same type from the training set, plus the top 8 retrieved chunks with their source document tags.
Generation uses `claude-3-5-sonnet-20241022` at temperature 0.1 and is constrained to return JSON.

The output is then normalised into the competition schema.
True and false answers become `1` and `0`, units are moved out of `answer_value` into `answer_unit`, ranges are collapsed to `[low,high]`, and cited `ref_id` values are checked against the ids in `metadata.csv` so the system cannot cite a document that does not exist.
Every field falls back to the sentinel string `is_blank`.

Knowing when *not* to answer matters here, because a confident wrong number scores worse than an abstention.
There are two guards for this.
At retrieval time, a question whose key entities appear nowhere in the retrieved context is refused before any API call is made.
At generation time, the prompt tells the model to return `is_blank` instead of guessing, and if `answer_value` comes back blank, the post-processor blanks the dependent fields as well so no orphaned unit or citation is left behind.

## Running it

Requires an Anthropic API key and Python 3.9 to 3.11, because the pinned numpy, torch and pandas versions have no wheels for 3.12.

```bash
pip install -r requirements.txt

cp .env.template .env
# then put your ANTHROPIC_API_KEY in .env
```

The chunked corpus is already in `data/chunks/`, so you can go straight to running the pipeline:

```bash
python run.py
```

If something looks wrong, `verify_setup.py` checks that the data, config and packages are in place:

```bash
python verify_setup.py
```

### Rebuilding the corpus from source

You only need this to change the chunking strategy or refresh the source documents.
It downloads all 32 PDFs and re-chunks them, overwriting `data/chunks/`:

```bash
python src/data/pdf_processor.py    # download PDFs into data/raw/
python src/data/chunker.py          # chunk them into data/chunks/
```

`pdf_processor.py` also writes a plain text extraction of each document to `data/processed/`, which is useful for checking what came out of a PDF.
The chunker does not read those files; it re-extracts from the PDFs itself so it can keep tables separate from prose.

`run.py` builds the index, reports accuracy against the training questions, and then asks before processing the full test set.
It writes the submission to `data/processed/submission.csv` with a checkpoint every 10 questions, so a long run is not lost if it gets interrupted.

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
`data/chunks/` holds 4,498 chunks across the 32 documents, and 532 of them are tables.
The PDFs, extracted text, embedding caches and HNSW index are derived files and are git-ignored, since the rebuild steps above regenerate them.

`metadata.csv` is the source of truth for the corpus, with one row per document giving its id, type, title, year, citation and URL.
`train_QA.csv` has 40 fully labelled rows.
`test_Q.csv` has 282 rows where only `id`, `question` and the expected `answer_unit` are filled in.
Both use the submission schema: `id`, `question`, `answer`, `answer_value`, `answer_unit`, `ref_id`, `ref_url`, `supporting_materials`, `explanation`.

## Code layout

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

There are two retrievers.
The submission pipeline uses `src/retrieval/hybrid_search.py`.
`src/retrieval/embeddings.py` is a variant that prefixes each chunk with its document and section before encoding, applies the BGE query instruction prefix, and caches encoded matrices in `data/cache/embeddings/` keyed by a hash of the input, so re-indexing an unchanged corpus skips encoding entirely.
The test suite is written against this second one.
Both use `src/retrieval/dense_index.py`, so they get the HNSW path and the exact fallback from the same code and the same config block.

`src/knowledge_graph/graph_builder.py` builds a networkx multigraph of typed entities (model, metric, value, hardware, method) and the relations between them (`TRAINED_ON`, `PRODUCES`, `REQUIRES`, `HAS_PUE`), extracted by pattern matching.
`src/pipeline/hybrid_pipeline.py` uses it to route numeric and comparison questions through the graph first, falling back to plain retrieval.
That path is incomplete and was not used for the submitted results.

## Testing

```bash
pytest tests/test_dense_index.py -v
```

`tests/test_dense_index.py` tests the dense index: recall@10 of HNSW against the exhaustive scan, score agreement between the two backends, the persistence round trip, rebuilds after a change to the corpus or the build parameters, the fallback when `hnswlib` is missing, and config parsing.
It runs on synthetic vectors at corpus scale, so it needs neither the embedding model nor the embedding cache, and it runs on its own, separate from the suite below.

```bash
pytest tests/ -v
```

`tests/test_embeddings.py` covers the chunk data model, indexing and embedding normalisation, dense and hybrid search, document filtering, index save and load, cache hits, and retrieval on the real question shapes from the training set.
Coverage settings are in `.coveragerc`.

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

## Known issues

The committed chunks in `data/chunks/` were produced by an earlier version of the chunker and have no `section` labels.
The loader copes with this: it recomputes the `contains_numeric` flag from the chunk text and treats a missing section as `None`, so retrieval works as described.
What is lost is the small section-match boost in reranking and the section tag in the prompt context.
Re-running `src/data/chunker.py` regenerates the chunks with section metadata.

`tests/test_embeddings.py` imports `ScientificPaperRetriever`, a SciBERT-backed subclass that a later commit removed from `src/retrieval/embeddings.py`.
The suite will not collect until that class is restored or the tests that use it are updated.

`src/pipeline.py` and the `src/pipeline/` package share a name, which makes `src.pipeline` ambiguous.
`src/pipeline.py` is an unused stub and should be deleted.

Several modules under `src/llm/` and `src/evaluation/` are stubs with unimplemented method bodies.
