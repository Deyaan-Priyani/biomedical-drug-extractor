# Drug names from 100 research abstracts

A small Python NLP project that reads 100 public breast-cancer research abstracts,
extracts medication candidates, and prepares a 20-abstract human review sample.

## Included run

The included run processed **100 abstracts** and produced **170 candidate mentions**
covering **66 unique candidate strings**. Candidates occurred in 40 abstracts.
These counts include drugs and other chemicals. Six abstracts required overlapping
model windows.

The 20-abstract review now contains AI-assisted reference labels based on the
supplied abstract text. With the rule of counting explicitly named medicines or
investigational drug agents, the sample contains four drug names, all in one
abstract. The model extracted 13 candidate entries across the sample: four correct,
nine incorrect, and zero missed. This gives **30.8% precision** and **100% recall**
against these reference labels. The small number of positive examples and the
AI-assisted labels limit the conclusions. Review the label notes before treating
the results as a human-validated benchmark.

One real example, from [PMID 39682607](https://pubmed.ncbi.nlm.nih.gov/39682607/),
mentions treatment with epirubicin, cyclophosphamide, and paclitaxel. Those names
appear with their surrounding text in the candidate CSV.

## What happens

1. **Fetch.** Download PubMed-indexed abstracts through Europe PMC and keep their
   article IDs and source links. The default query covers publications from 2024.
2. **Extract.** Run an existing biomedical named entity recognition (NER) model.
   NER labels spans of text. This project keeps BioBERT's `CHEMICAL` labels as
   drug candidates for human review.
3. **Review.** Read 20 randomly sampled abstracts and write the correct drug names.
4. **Evaluate.** Compare those human labels with the model predictions to calculate
   precision and recall.

The model is pretrained. This project implements data collection, model inference,
exports, and evaluation. It does not train a new model.

## Run the project

Use Python 3.12. Open a terminal in this folder.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Keep downloaded model files inside the project.
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_DISABLE_XET=1

python drug_ner.py fetch
python drug_ner.py extract
```

The first extraction downloads a model of approximately 430 MB. Inference runs
locally on the CPU and requires no paid API key. Internet is needed for fetching
abstracts and the initial model download. Later runs reuse the saved data and cache.

## Read the results

| File | Contents |
| --- | --- |
| `data/abstracts.json` | The 100 abstracts, source URLs, query, and retrieval date |
| `results/drug_candidates.csv` | Every extracted mention, model score, character offsets, and nearby text |
| `results/drug_counts.csv` | Mention frequency for each candidate name |
| `results/predictions.json` | Unique candidate names per abstract |
| `results/run_summary.json` | Actual processing counts and model version |
| `results/review_20.csv` | A reproducible sample of 20 abstracts for manual review |
| `results/evaluation.json` | Precision, recall, and the correct, incorrect, and missed names |

Open `drug_candidates.csv` in Excel or another CSV viewer. The `context` column
shows where each candidate appeared. Model scores express model confidence;
they do not establish that the name is correct or that it is a drug.

Repeated mentions are kept in the mention CSV. Evaluation counts each unique
surface name once per abstract, ignoring capitalization and repeated whitespace.
Brand names, generic names, and abbreviations are evaluated as separate strings.

## Fill in the human review

Open `results/review_20.csv`. For each of the 20 rows:

The included copy has AI-assisted proposed labels and explanatory notes. You can
verify or correct those labels, then rerun the evaluation below.

1. Read the entire abstract, including the parts without predictions.
2. Fill `gold_drugs` with a JSON list of all explicit drug names in that abstract.
   Copy each name as it appears in the text.
3. Use `[]` if you reviewed the abstract and found no drug names. A blank cell
   means the abstract has not been reviewed.
4. Use `notes` for uncertain cases. Resolve those cases before scoring.

Example cell contents:

```json
["tamoxifen", "pembrolizumab"]
```

For consistent labels, count named medicines or named investigational drug agents.
Exclude drug classes, broad treatments such as "chemotherapy", genes, and unrelated
chemicals. If an abbreviation and a full name both occur, include both surface
forms. This simple evaluation measures name extraction rather than drug identity
normalization. Reading without first consulting the predictions reduces the chance
of overlooking names the model missed.

Save as CSV, keeping the existing columns, and run:

```sh
python drug_ner.py evaluate
```

This creates `results/evaluation.json` with correct, incorrect, and missed names
for every reviewed abstract.

- **Precision** = correct extracted names / all extracted names.
- **Recall** = correct extracted names / all human-labeled names.

Scores are calculated across all reviewed abstracts together (micro averaging).
Undefined ratios are recorded as `null`. The program requires every review row
to be labeled, including rows with no drugs. A 20-abstract sample gives a small
project estimate, not a general clinical performance claim.

Extraction preserves existing manual review labels on reruns. Evaluation checks
that the review still matches the saved abstracts and current predictions.

## Limitations and sources

BioBERT was fine-tuned on the BC5CDR-chemicals and BC4CHEMD corpora. Its CHEMICAL
label covers drugs and other chemicals. It can include irrelevant compounds or
miss drug names, so all outputs are candidates until reviewed. No drug dictionary
or clinical validation is applied. Long abstracts are read in overlapping windows
rather than silently cut off. Subword pieces are grouped into words before
assembling named entities.

- [Model card, alvaroalon2/biobert_chemical_ner](https://huggingface.co/alvaroalon2/biobert_chemical_ner)
- [Europe PMC API](https://europepmc.org/RestfulWebService)
- [Hugging Face token classification](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.TokenClassificationPipeline)

The script pins the model revision for reproducibility. Abstracts retain their
original publishers' rights. Check reuse terms before publicly redistributing
the saved text. Article IDs and source links are retained for attribution.

## Explain it in an interview

"I used Python to collect 100 research abstracts and applied a pretrained
biomedical NER model to extract drug and chemical names. I exported the predictions
with source links and compared 20 abstracts against AI-assisted reference labels.
The evaluation showed that the model also extracts non-drug chemicals."

Describe the reference labels as human-validated only after a person checks them.
Use the actual counts and scores from your outputs in any resume bullet.
