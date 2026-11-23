# Sources and permissions

The numeric records are projections of two public archives. Each row retains a question identifier, model configuration label, replicate index, output tokens, and published binary grade. We removed solution text, reference answers and prompts, shortened model identifiers, sorted rows, and converted boolean grades to integers. Source judgments and lengths are preserved.

* **b300.csv** is adapted from Teng-Ruei Chen (2026), *When Strong Reasoners Join the Pool: run record and code (NVIDIA B300, MATH-500)*, version 23939046. The source run record is © 2026 Teng-Ruei Chen and licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The published archive MD5 is 5429f8eacd46347ae3a724543a16dc88. The score ledger SHA-256 is 224045b989cb7232caafc5ae2ef46abdc1659cc378378632d652bc9317940b1a. [Source record](https://doi.org/10.5281/zenodo.23939046)
* **aime.csv** is adapted from MathArena, *AIME 2025 outputs*, revision 16e93d3bb9b0a25ea9422893464d46b3c5ff0ae9. The source and this projection are licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The publisher's Parquet object SHA-256 is c7980ee5518384624cb690d08b03bfff5892d19141b7ca1de666a2e863b0e589. [Source release](https://huggingface.co/datasets/MathArena/aime_2025_outputs/tree/16e93d3bb9b0a25ea9422893464d46b3c5ff0ae9)

The archive-specific licenses apply separately. The model and question identifiers refer to the original providers. Each digest for the projected CSVs appears in design.json. The projection script reproduces both files byte for byte from the specified original releases.

The supplemental aime_lengths.csv contains Unicode character counts of the released answer strings for the same MathArena records and follows the same CC BY-NC-SA 4.0 terms. repair_aime.py records its SHA-256 and reproduces the projection.
