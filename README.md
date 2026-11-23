# Replaying post-answer model cascades

A cascade accepts a first answer or replaces it with a response from a fallback model. With multiple standalone answers per question, matching outputs by their draw index silently chooses a joint distribution. This project measures how that choice affects a comparison between two fixed policies.

[Manuscript](paper/main.pdf) · [Study protocol and pre-analysis amendments](PROTOCOL.md) · [Full numerical results](reports/results.json) · [Input attribution and licenses](data/NOTICE.md)

## Findings

* On 400 held-out MATH-500 questions, Nano-to-Super length escalation gained 0.164 accuracy points over an equally frequent random upgrade under independent pairing. A single relabeling shared by all questions always preserved the sign. Rematching 14 questions individually could erase the identity-match lead. The gate consumed at least 1,024 extra fallback tokens per request under every question-specific alignment.
* On held-out AIME questions, all 24 global permutations favored the recorded-token 14B-to-32B gate in accuracy. With returned answer length as the signal in a post hoc check, six favored random escalation. Signal choice matters even with the same grades and the same fallback.
* The 1.5B AIME archive assigned zero output tokens to 45 nonempty answers. The original recorded-field replay and the corrected measurement sensitivity are documented separately. The 1.5B input cannot support a streaming token-cost comparison. [Sensitivity results](reports/aime_characters.json)

These comparisons concern the archived draws. No provider latency, billing, or live serving improvement was measured.

## Reproduce the audit

The checked-in CSVs contain every input used by the reported analyses. Run the following from the project root. Python 3.12 and its standard library suffice. The research runs used an isolated Windows environment.

```powershell
py -3.12 -m venv .venv
.\Run-Study.ps1
```

The runner verifies the projected input hashes, tests the optimizers against exhaustive enumeration, recomputes all results, and checks the returned reports. On other platforms run `python -m unittest -v`, `python study.py`, `python repair_aime.py`, then `python verify_reports.py` in order. The execution manifest records the actual platform and interpreter. Numeric results use percentage points for correctness and the original fallback model's output-token counts per request for cost. Positive token differences mean the first policy spends more.

To regenerate the projections, obtain the two exact files linked in the data notice and run the following. Only Parquet projection requires PyArrow (version 23.0.0 was used). The main analysis reads CSV. The project excludes raw prompts and generated text.

```sh
python project_input.py aime /path/to/train-00000-of-00001.parquet
python project_input.py b300 /path/to/gtc2027-b300-oracle-v1.tar.gz
python repair_aime.py --project /path/to/train-00000-of-00001.parquet
```

To rebuild the manuscript tables and figure, use Matplotlib and Tectonic. From the project root run `python paper/display.py`, then `tectonic paper/main.tex`. Prose and captions are in the TeX source.

## Compare two policies on your own records

Provide a CSV with these headers and one row for every question and replicate position. Each question must have the same contiguous zero-based positions, at least two. A row describes the nominal identity pairing, whose labels can be reindexed. The gate and comparator give escalation probabilities, the two grades lie in [0,1], and the token column measures the fallback output length.

```csv
question,replicate,gate,comparator,first_grade,fallback_grade,fallback_tokens
example,0,1,0.5,0,1,200
example,1,0,0.5,1,0,80
```

```sh
python audit.py records.csv
python audit.py records.csv --quality-weight 100 --token-weight -0.001
```

The second command prices 1,000 extra fallback tokens at one accuracy point. Both policies see the same alignment. Output includes exact extremizing global permutations, the product replay, identity result, uniform-reindexing standard deviations, and the minimum number of question-specific reindexings that erase the identity sign. A null erasure count means every allowed reindexing preserves that sign. Regions condition on the archived grades and lengths. A stateless fallback must answer the original question under the same decoding configuration.
