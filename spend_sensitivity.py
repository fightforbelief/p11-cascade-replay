"""Question-level sensitivity of the observed AIME spend reversal."""
import json
import random
from math import ceil
from pathlib import Path
from statistics import fmean
from question_sensitivity import N_BOOT, N_SPLIT, central
from spend_replay import direct_value, efficient
from study import DATA, MODELS, load_pair


def question_point(pair, cutoff):
    first, fallback = pair
    small_score = 100 * fmean(y for _, y in first)
    large_score = 100 * fmean(y for _, y in fallback)
    small_cost = fmean(t for t, _ in first)
    large_cost = fmean(t for t, _ in fallback)
    rate = fmean(t > cutoff for t, _ in first)
    kept = 100 * fmean(y if t <= cutoff else 0 for t, y in first)
    return small_score, large_score, small_cost, large_cost, rate, kept + rate * large_score, rate * large_cost


def compare(rows):
    a, b, s, l, rate, gate, paid = map(fmean, zip(*rows))
    gaps = [gate - (a + rate * (b - a))]
    for rho in (1, 4):
        frontier = efficient([(s, a), (rho*l, b)])
        gaps.append(gate - direct_value(frontier, s + rho*paid))
    return gaps


def summary(rows):
    return {
        "draws": len(rows),
        "call_gap_pp": central([x for x, _, _ in rows]),
        "spend_1_gap_pp": central([x for _, x, _ in rows]),
        "spend_4_gap_pp": central([x for _, _, x in rows]),
        "positive_calls_negative_spend_1": sum(a > 1e-9 and b < -1e-9 for a, b, _ in rows),
        "negative_spend_1_positive_spend_4": sum(b < -1e-9 and c > 1e-9 for _, b, c in rows),
    }


def main():
    spec = MODELS["aime-medium"]
    data, excluded = load_pair(DATA/spec["file"], spec["models"], spec["replicates"])
    assert not excluded
    baseline = json.loads(Path("reports/results.json").read_text())["aime-medium"]
    spend = json.loads(Path("reports/spend_replay.json").read_text())["aime-medium"]
    qs = sorted(data)
    original = [question_point(data[q], baseline["cut"]) for q in qs if q not in baseline["calibration"]]
    point = compare(original)
    expected = [baseline["evaluation"]["comparisons"]["random"]["accuracy_pp"]["product"]]
    expected.extend(spend[r]["product_gap_pp"] for r in ("1", "4"))
    assert max(abs(x-y) for x, y in zip(point, expected)) < 1e-8
    rng = random.Random(20261123)
    boot = [compare(rng.choices(original, k=len(original))) for _ in range(N_BOOT)]
    rng = random.Random(20261124)
    splits = []
    for _ in range(N_SPLIT):
        cal = set(rng.sample(qs, len(baseline["calibration"])))
        lengths = sorted(t for q in cal for t, _ in data[q][0])
        cut = lengths[ceil(.75 * len(lengths)) - 1]
        splits.append(compare([question_point(data[q], cut) for q in qs if q not in cal]))
    report = {"point_pp": point, "bootstrap": summary(boot), "refit_splits": summary(splits)}
    Path("reports/spend_sensitivity.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
