"""Extract medication candidates from public abstracts, then evaluate human labels."""

import argparse
import csv
import html
import json
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
MODEL = "alvaroalon2/biobert_chemical_ner"
REVISION = "b57181cea10a3943c35cf31bb375ff9217886730"
QUERY = (
    'TITLE_ABS:"breast cancer" AND HAS_ABSTRACT:Y AND SRC:MED '
    'AND FIRST_PDATE:[2024-01-01 TO 2024-12-31]'
)
API = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def normalize(name):
    """Match unique surface names, ignoring case and repeated whitespace."""
    return " ".join(name.casefold().split())


def fetch(args):
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    destination = DATA / "abstracts.json"
    if destination.exists():
        cached = read_json(destination)
        if cached["query"] != args.query or len(cached["articles"]) != args.count:
            raise ValueError("Cached data differs from this request. Use a new project copy.")
        print(f"Using {len(cached['articles'])} saved abstracts.")
        return

    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers["User-Agent"] = "student-drug-name-extractor/1.0"
    articles = []
    seen = set()
    cursor = "*"
    while len(articles) < args.count:
        response = session.get(API, params={
            "query": args.query,
            "format": "json",
            "resultType": "core",
            "pageSize": min(args.count, 100),
            "cursorMark": cursor,
        }, timeout=60)
        response.raise_for_status()
        payload = response.json()
        results = payload.get("resultList", {}).get("result", [])
        for article in results:
            pmid = article.get("pmid")
            abstract = html.unescape(re.sub(r"<[^>]+>", " ", article.get("abstractText", "")))
            abstract = " ".join(abstract.split())
            if not pmid or not abstract or pmid in seen:
                continue
            seen.add(pmid)
            articles.append({
                "pmid": pmid,
                "title": article["title"],
                "abstract": abstract,
                "publication_date": article.get("firstPublicationDate", ""),
                "source_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            })
            if len(articles) == args.count:
                break
        next_cursor = payload.get("nextCursorMark")
        if not results or not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor

    if len(articles) != args.count:
        raise ValueError(f"Only found {len(articles)} usable abstracts for this query.")
    write_json(destination, {
        "source": "Europe PMC REST API, PubMed-indexed articles (SRC:MED)",
        "api_url": API,
        "query": args.query,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "articles": articles,
    })
    print(f"Saved {len(articles)} abstracts to {destination}")


def extract(args):
    import torch
    import transformers
    from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

    articles = read_json(DATA / "abstracts.json")["articles"]
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForTokenClassification.from_pretrained(MODEL, revision=REVISION)
    recognizer = pipeline(
        "token-classification", model=model, tokenizer=tokenizer,
        aggregation_strategy="first", device=-1,
    )
    predictions = []
    mentions = []
    windows = 0
    for index, article in enumerate(articles, start=1):
        text = article["abstract"]
        # Overlapping windows cover text beyond the model's 512-token limit.
        entities = recognizer(text, stride=64)
        token_count = len(tokenizer(text, truncation=False, verbose=False)["input_ids"])
        windows += int(token_count > tokenizer.model_max_length)
        names = {}
        for entity in entities:
            if entity["entity_group"] != "CHEMICAL":
                continue
            start, end = int(entity["start"]), int(entity["end"])
            name = text[start:end]
            names.setdefault(normalize(name), name)
            mentions.append({
                "pmid": article["pmid"], "drug_candidate": name,
                "model_score": round(float(entity["score"]), 4),
                "start": start, "end": end,
                "context": text[max(0, start - 120):min(len(text), end + 120)],
                "source_url": article["source_url"],
            })
        predictions.append({"pmid": article["pmid"], "predicted_drugs": list(names.values())})
        if index % 10 == 0 or index == len(articles):
            print(f"Read {index}/{len(articles)} abstracts; found {len(mentions)} candidate mentions.", flush=True)

    write_csv(RESULTS / "drug_candidates.csv", mentions, [
        "pmid", "drug_candidate", "model_score", "start", "end", "context", "source_url",
    ])
    write_json(RESULTS / "predictions.json", predictions)
    frequency = Counter(normalize(row["drug_candidate"]) for row in mentions)
    write_csv(RESULTS / "drug_counts.csv", [
        {"drug_candidate": name, "mentions": count} for name, count in frequency.most_common()
    ], ["drug_candidate", "mentions"])

    sample = random.Random(42).sample(predictions, min(20, len(predictions)))
    articles_by_id = {article["pmid"]: article for article in articles}
    review = RESULTS / "review_20.csv"
    if not review.exists():
        write_csv(review, [{
            "pmid": row["pmid"],
            "title": articles_by_id[row["pmid"]]["title"],
            "abstract": articles_by_id[row["pmid"]]["abstract"],
            "predicted_drugs": json.dumps(row["predicted_drugs"], ensure_ascii=False),
            "gold_drugs": "",
            "notes": "",
        } for row in sample], ["pmid", "title", "abstract", "predicted_drugs", "gold_drugs", "notes"])
    else:
        print("Kept existing review_20.csv so your manual labels are preserved.")

    write_json(RESULTS / "run_summary.json", {
        "model": MODEL, "revision": REVISION,
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "abstracts_processed": len(articles),
        "abstracts_with_candidates": sum(bool(row["predicted_drugs"]) for row in predictions),
        "candidate_mentions": len(mentions), "unique_candidate_names": len(frequency),
        "abstracts_requiring_overlapping_windows": windows,
        "review_sample_size": len(sample), "review_seed": 42,
        "precision": None, "recall": None,
        "note": "CHEMICAL-label predictions are drug candidates and can include other chemicals. Accuracy requires human gold labels.",
    })
    print(f"Results saved in {RESULTS}. Fill gold_drugs in review_20.csv to evaluate.")


def parse_names(value, field, pmid):
    names = json.loads(value)
    if not isinstance(names, list) or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError(f"{pmid}: {field} must be a JSON array of nonempty strings.")
    return {normalize(name) for name in names}


def evaluate(args):
    articles = {article["pmid"]: article for article in read_json(DATA / "abstracts.json")["articles"]}
    predictions = {
        row["pmid"]: {normalize(name) for name in row["predicted_drugs"]}
        for row in read_json(RESULTS / "predictions.json")
    }
    with args.review.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("The review file is empty.")
    missing = [row["pmid"] for row in rows if not row.get("gold_drugs", "").strip()]
    if missing:
        raise ValueError(f"Finish {len(missing)} unlabeled abstracts first. Use [] for a reviewed abstract with no drugs.")

    details = []
    reviewed = set()
    true_positives = false_positives = false_negatives = 0
    for row in rows:
        pmid = row["pmid"]
        if pmid not in articles or pmid in reviewed:
            raise ValueError(f"Unknown or duplicate PMID in review: {pmid}")
        reviewed.add(pmid)
        if row["abstract"] != articles[pmid]["abstract"]:
            raise ValueError(f"{pmid}: review text differs from the saved abstract.")
        predicted = predictions[pmid]
        if parse_names(row["predicted_drugs"], "predicted_drugs", pmid) != predicted:
            raise ValueError(f"{pmid}: predictions were edited or belong to a different run.")
        gold = parse_names(row["gold_drugs"], "gold_drugs", pmid)
        if any(name not in normalize(row["abstract"]) for name in gold):
            raise ValueError(f"{pmid}: copy each gold name exactly as it appears in the abstract.")
        true_positives += len(predicted & gold)
        false_positives += len(predicted - gold)
        false_negatives += len(gold - predicted)
        details.append({
            "pmid": pmid, "correct": sorted(predicted & gold),
            "incorrect": sorted(predicted - gold), "missed": sorted(gold - predicted),
        })
    predicted_total = true_positives + false_positives
    gold_total = true_positives + false_negatives
    precision = true_positives / predicted_total if predicted_total else None
    recall = true_positives / gold_total if gold_total else None
    write_json(RESULTS / "evaluation.json", {
        "unit": "Unique drug surface names per abstract, case-insensitive exact matching",
        "review_file": args.review.name,
        "abstracts_reviewed": len(rows), "true_positives": true_positives,
        "false_positives": false_positives, "false_negatives": false_negatives,
        "precision": precision, "recall": recall, "details": details,
    })
    summary = read_json(RESULTS / "run_summary.json")
    summary.update({
        "precision": precision, "recall": recall,
        "abstracts_reviewed": len(rows),
        "evaluation_file": "evaluation.json",
        "note": "Scores compare chemical-name predictions with reference drug labels. Review the CSV notes for annotation provenance.",
    })
    write_json(RESULTS / "run_summary.json", summary)
    print(f"Evaluated {len(rows)} reviewed abstracts.")
    print(f"Precision: {precision:.1%}" if precision is not None else "Precision: undefined (no predictions)")
    print(f"Recall: {recall:.1%}" if recall is not None else "Recall: undefined (no gold drug names)")


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Use a positive integer.")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("fetch", help="Save public research abstracts.")
    download.add_argument("--count", type=positive_integer, default=100)
    download.add_argument("--query", default=QUERY)
    download.set_defaults(action=fetch)
    extraction = commands.add_parser("extract", help="Find drug/chemical candidates with pretrained BioBERT.")
    extraction.set_defaults(action=extract)
    evaluation = commands.add_parser("evaluate", help="Score completed human review labels.")
    evaluation.add_argument("--review", type=Path, default=RESULTS / "review_20.csv")
    evaluation.set_defaults(action=evaluate)
    args = parser.parse_args()
    try:
        args.action(args)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
