"""Project the publisher records to answer-free replay inputs. Operates on known releases."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile

B300_MD5 = "5429f8eacd46347ae3a724543a16dc88"
AIME_SHA = "c7980ee5518384624cb690d08b03bfff5892d19141b7ca1de666a2e863b0e589"
AIME_MODELS = {f"deepseek/deepseek_distill_{size}":size for size in ["1.5b","14b","32b"]}
FIELDS = ["question","model","replicate","tokens","correct"]


def verify(path, algorithm, expected):
    with path.open("rb") as fh:
        actual=hashlib.file_digest(fh,algorithm).hexdigest()
    if actual!=expected:
        raise ValueError(f"publisher hash mismatch at {path}")


def write_csv(path, rows):
    rows.sort()
    with path.open("w",newline="",encoding="utf-8") as fh:
        writer=csv.writer(fh,lineterminator="\n")
        writer.writerow(FIELDS)
        writer.writerows(rows)


def aime(path):
    import pyarrow.parquet as pq
    verify(path,"sha256",AIME_SHA)
    columns=["problem_idx","model_config","idx_answer","output_tokens","correct"]
    rows=[]
    for batch in pq.ParquetFile(path).iter_batches(columns=columns):
        for x in batch.to_pylist():
            model=AIME_MODELS.get(x["model_config"])
            if model:
                if type(x["correct"]) is not bool:
                    raise ValueError("nonbinary AIME grade")
                rows.append((str(x["problem_idx"]),model,x["idx_answer"],x["output_tokens"],int(x["correct"])))
    return rows


def b300(path):
    verify(path,"md5",B300_MD5)
    member="gtc2027-b300-oracle-v1/b300_runs/score3_20261028/scores.jsonl"
    with tarfile.open(path,"r|gz") as archive:
        for info in archive:
            if info.name==member:
                buf=archive.extractfile(info).read()
                break
        else:
            raise ValueError("score ledger missing")
    if hashlib.sha256(buf).hexdigest()!="224045b989cb7232caafc5ae2ef46abdc1659cc378378632d652bc9317940b1a":
        raise ValueError("score ledger digest mismatch")
    rows=[]
    for line in io.BytesIO(buf):
        x=json.loads(line)
        if x["namespace"]!="formal" or x["score"] not in (0,1):
            raise ValueError("unexpected score/namespace")
        rows.append((x["query_id"],x["attempt_id"],x["draw_index"],x["token_count"],int(x["score"])))
    return rows


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source",choices=("aime","b300"))
    parser.add_argument("archive",type=Path)
    args=parser.parse_args()
    rows=(aime if args.source=="aime" else b300)(args.archive)
    dest=Path("data")/(args.source+".csv")
    dest.parent.mkdir(exist_ok=True)
    write_csv(dest,rows)
    print(len(rows),dest,hashlib.sha256(dest.read_bytes()).hexdigest())
