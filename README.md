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

### Retrieval

Retrieval is hybrid, in `src/retrieval/hybrid_search.py`.

Two independent indexes are built over the same chunks:

1. A dense index of `BAAI/bge-large-en-v1.5` embeddings (1024 dimensions), L2-normalised so cosine similarity reduces to a dot product.
2. A BM25 index over Porter-stemmed tokens, via `rank_bm25`.

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

Build the corpus, which downloads the 32 PDFs and chunks them:

```bash
python src/data/pdf_processor.py    # download PDFs into data/raw/
python src/data/chunker.py          # chunk them into data/chunks/
```

`pdf_processor.py` also writes a flat text extraction per document to `data/processed/`, which is useful for inspecting what came out of a PDF.
The chunker does not read those files, it re-extracts from the PDFs itself so it can keep tables separate from prose.

Check that everything is in place:

```bash
python verify_setup.py
```

Then run the pipeline:

```bash
python run.py
```

`run.py` builds the index, reports accuracy against the training questions, and then prompts before processing the full test set.
The submission is written to `data/processed/submission.csv`, with checkpoints every 10 questions so a long run can survive an interruption.

Model, retrieval and path settings live in `configs/config.yaml`.

## Data layout

```
data/
  raw/          metadata.csv, train_QA.csv, test_Q.csv, and downloaded PDFs
  processed/    extracted text and the final submission.csv
  chunks/       per-document chunk JSON produced by the chunker
  cache/        cached embedding matrices
```

Only the three CSVs are tracked in git.
PDFs, extracted text, chunks and caches are all derived and are regenerated by the ingest steps above.

`metadata.csv` is the source of truth for the corpus, with one row per document giving its id, type, title, year, citation and URL.
`train_QA.csv` has 40 fully labelled rows.
`test_Q.csv` has 282 rows where only `id`, `question` and the expected `answer_unit` are populated.
Both share the submission schema: `id`, `question`, `answer`, `answer_value`, `answer_unit`, `ref_id`, `ref_url`, `supporting_materials`, `explanation`.

## Layout

```
run.py                          entrypoint
verify_setup.py                 preflight check for data, config and packages
configs/config.yaml             model, retrieval and path settings
src/
  vector_pipeline.py            the pipeline used for the submission
  data/                         PDF download, extraction and chunking
  retrieval/                    hybrid BM25 and dense search, embedding cache
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

`src/knowledge_graph/graph_builder.py` builds a networkx multigraph of typed entities (model, metric, value, hardware, method) and the relations between them (`TRAINED_ON`, `PRODUCES`, `REQUIRES`, `HAS_PUE`), extracted by pattern matching.
It is consumed by `src/pipeline/hybrid_pipeline.py`, which routes numeric and comparison questions through the graph first and falls back to plain retrieval.
That path is incomplete and was not used for the submitted results.

## Testing

```bash
pytest tests/ -v
```

`tests/test_embeddings.py` holds 48 tests covering the chunk data model, indexing and embedding normalisation, dense and hybrid search, document filtering, index save and load, cache hits, and retrieval behaviour on the real question shapes from the training set.
Coverage settings are in `.coveragerc`.

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

## Known issues

`tests/test_embeddings.py` imports `ScientificPaperRetriever`, a SciBERT-backed subclass that was removed from `src/retrieval/embeddings.py` in a later commit.
The suite will not collect until that class is restored or the tests that use it are updated.

`src/pipeline.py` and the `src/pipeline/` package share a name, which makes `src.pipeline` ambiguous.
`src/pipeline.py` is an unused stub and should be deleted.

Several modules under `src/llm/` and `src/evaluation/` are stubs with unimplemented method bodies.
