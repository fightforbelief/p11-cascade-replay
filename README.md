# Does a language-model cascade win at the same spend?

A cascade pays for a first answer and may replace it with a fallback. Drawing from a standalone archive leaves the pairing between those answers unspecified. This project computes sharp bounds on a fixed gate's score minus the best affordable query-independent mixture of direct services. Each possible replay pays for its own selected outputs. The calculation returns witnesses and a dual lower bound. Known independent calls give a single empirical product replay.

[Manuscript (PDF)](paper/main.pdf) · [Original protocol](PROTOCOL.md) · [Spend comparison and amendments](FRONTIER_PROTOCOL.md) · [Data credits](data/NOTICE.md)

## Findings

* **AIME 14B to 32B.** On 24 held-out questions the length gate gained 4.19 accuracy points over equally frequent random escalation under independent replay. At equal output-token prices it lost 4.17 points to direct service. Even its most favorable question-specific alignment lost 1.04 points. Independent replay crosses zero when the fallback token price reaches 1.933 times the first-stage price. At four times the price it gains 2.15 points. The equal-price reversal recurred in 4,105 of 5,000 whole-question resamples and 170 of 200 refitted splits. The central 95% resampling range for the equal-spend gap was −13.54 to +4.17 points. [Sharp bounds](reports/spend_replay.json) · [Price boundaries](reports/price_threshold.json) · [Question resampling](reports/spend_sensitivity.json)
* **MATH-500 Nano to Super.** On 400 held-out questions the independent call-matched gain was 0.164 points (question-bootstrap range −0.17 to +0.50). The gate used at least 1,024 extra fallback output tokens per request under every question-specific alignment. Its equal-spend gain was 0.112 points under independent replay. Both alignment conventions admit a loss. [Call-matched results](reports/results.json) · [Question sensitivity](reports/question_sensitivity.json)
* **FrugalGPT.** Across all five published task tables, 100 nominal budgets gave 22 distinct test operating points within tasks. With the training-eligible service pool, 21 beat an equally affordable direct mixture, 13 by more than one score point. Direct weights selected on training data and frozen before testing dominated one of 100 labeled cascade points, matching its score at lower cost. [Results](reports/published_frontiers.json) · [Author tables and service lists](data/frugalgpt/NOTICE.md)
* **AutoMix.** Across five released tasks, three policy families and ten calibration seeds, the highest mean incremental benefit per cost selected a best equal-budget family on four tasks, including a complete tie on QuALITY. On NarrativeQA the choice changed, with a maximum loss of 0.141 score points. Paired seed-by-seed choices exhibited larger losses, up to 1.46 points. [Results](reports/automix_frontiers.json) · [Author results and implementation](data/automix/NOTICE.md)

The sharp bounds condition on the specific archive. The two external studies provide aggregate operating points. They cannot establish per-question sampling uncertainty. Output-token price scenarios are hypothetical, and no live latency or provider bills were measured. The archive's 1.5B AIME token field recorded zero for 45 nonempty answers, so that model is excluded from spend comparisons. [Telemetry sensitivity](reports/aime_characters.json)

## Reproduce the audit

The checked-in CSV and JSON files contain the numerical inputs for the reported analyses. Run the following from the project root. Python 3.12 and its standard library suffice. The research runs used an isolated Windows environment.

```powershell
py -3.12 -m venv .venv
.\Run-Study.ps1
```

The runner verifies inputs, runs eight unit tests, reproduces the analyses, and checks their outputs. The original and extended manifests record the Windows interpreter, platform and hashes. To run on another platform, execute the same Python scripts in the order shown in the runner from the project root. The external audits read the released summary tables without refitting the original models. [Runner](Run-Study.ps1) · [Original execution manifest](reports/execution.json) · [Extended execution manifest](reports/frontier_execution.json)

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

## Reuse the equal-spend calculation

The function `certificate(points, oracle)` in [spend_replay.py](spend_replay.py) accepts direct (cost, score) points and a function returning a feasible cascade (cost, score) pair that minimizes or maximizes score minus price times cost. Every cascade point must afford at least the cheapest direct service. The result includes the lower and upper excess scores, a feasible mixture for the minimum, a maximum-attaining point, and the numerical primal–dual gap. Scores supplied in percentage points give output in percentage points. [Exhaustive small instances and usage examples](test_spend.py)
