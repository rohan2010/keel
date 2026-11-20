"""Chapter 4 P5: noise bands (flips between identical runs), storage, lookup, and use in compare."""

import glob
import json
import os
from pathlib import Path

import pytest
from test_compare import make

from keel.evals import noise, runs
from keel.evals.compare import Arm, compare_arms, load_arm

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "results" / "runs"
# the research repo's experiments folder, if you have it (optional): KEEL_RESEARCH_RUNS=<path>
RESEARCH_REPO = Path(os.environ["KEEL_RESEARCH_RUNS"]) if os.environ.get("KEEL_RESEARCH_RUNS") else None


def test_band_formula_matches_the_research_reading():
    assert noise.band_of(28, 120) == pytest.approx(0.0882, abs=1e-4)      # "28 of 120 flip (+-9 points)"


def test_measure_counts_flips_and_refuses_different_configs(tmp_path):
    a = load_arm(make(tmp_path, "a", [True] * 8 + [False] * 12))
    b = load_arm(make(tmp_path, "b", [False, False, False] + [True] * 5 + [True, True] + [False] * 10))
    m = noise.measure(a, b)
    assert (m["n"], m["flips"], m["flip_rate"]) == (20, 5, 0.25) and m["net"] == pytest.approx(-0.05)
    with pytest.raises(ValueError, match="batch_size"):
        noise.measure(a, load_arm(make(tmp_path, "c", [True] * 20, batch=4)))


def test_record_lookup_and_pooling(tmp_path):
    base = tmp_path / "noise"
    a = load_arm(make(tmp_path, "a", [True] * 10 + [False] * 10))
    b = load_arm(make(tmp_path, "b", [False] * 4 + [True] * 6 + [False] * 10))
    c = load_arm(make(tmp_path, "c", [True] * 10 + [True] * 4 + [False] * 6))
    noise.record(a, b, base=base)
    noise.record(a, b, base=base)                                             # same pair twice: stored once
    assert noise.lookup(a.header, base)["measurements"] == 1
    noise.record(a, c, base=base)
    got = noise.lookup(a.header, base)
    assert (got["measurements"], got["flips"], got["n"]) == (2, 8, 40)
    assert got["band"] == pytest.approx(noise.band_of(4, 20))               # pooled = per-pair band at 4 flips
    assert noise.lookup(load_arm(make(tmp_path, "z", [True] * 20, batch=4)).header, base) is None


def test_compare_flags_a_delta_inside_the_band(tmp_path):
    a = load_arm(make(tmp_path, "a", [True] * 10 + [False] * 10))
    b = load_arm(make(tmp_path, "b", [True] * 11 + [False] * 9))
    band = {"band": 0.2, "flips": 10, "n": 20}
    c = compare_arms(a, b, noise_band=band)
    assert any("inside the noise band" in f for f in c.flags)
    assert not any("unknown" in f for f in c.flags)


@pytest.mark.skipif(not RESEARCH.is_dir(), reason="research snapshot not present")
def test_research_repeat_pair_on_todays_scoring():
    """m6dev A vs A2 (two runs of one config, rng mode aside): 26 of 120 flip on scorer v2."""
    d = {t: str(next(p for p in runs.find_runs(RESEARCH) if t in p.name)) for t in ("m6dev_A_files", "m6dev_A2_files")}
    with pytest.raises(ValueError, match="rng"):
        noise.measure(load_arm(d["m6dev_A_files"], fs=False), load_arm(d["m6dev_A2_files"], fs=False))
    m = noise.measure(load_arm(d["m6dev_A_files"], fs=False), load_arm(d["m6dev_A2_files"], fs=False), ("rng",))
    assert (m["flips"], m["n"]) == (26, 120)


@pytest.mark.skipif(RESEARCH_REPO is None or not RESEARCH_REPO.is_dir(),
                    reason="needs the research runs: set KEEL_RESEARCH_RUNS")
def test_reproduces_28_of_120_with_the_original_scoring():
    """results-summary finding 3: "28 of 120 items flip (+-9 points)" (first scorings), 29 after the 11-09 rescore."""
    def arm(tag, which):
        d = glob.glob(str(RESEARCH_REPO / f"{tag}*"))[0]
        f = sorted(glob.glob(f"{d}/outputs_scored_*.jsonl"))[which]
        h = runs.header(Path(RESEARCH) / Path(d).name) if (RESEARCH / Path(d).name).exists() else runs.header(d)
        return Arm(Path(d).name, [h], [runs._agent(json.loads(l), "dev") for l in open(f) if l.strip()])
    m = noise.measure(arm("20261108-152243", 0), arm("20261109-002623", 0), ("rng",))
    assert (m["flips"], m["n"]) == (28, 120) and round(100 * m["band"]) == 9
    assert noise.measure(arm("20261108-152243", -1), arm("20261109-002623", -1), ("rng",))["flips"] == 29
