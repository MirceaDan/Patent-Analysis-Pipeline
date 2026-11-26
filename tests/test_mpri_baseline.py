"""Tests for figure-description integration, label handling, features, splits and the baseline.
All data here is SYNTHETIC and generated in-test; it is never mixed with the real dataset."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("integ", ROOT / "scripts" / "integrate_figure_descriptions.py")
integ = importlib.util.module_from_spec(spec)
sys.modules["integ"] = integ
spec.loader.exec_module(integ)

from mpri_baseline import data as D, metrics, model, splits  # noqa: E402
from mpri_baseline.groups import related_family_groups  # noqa: E402


# ---------------------------------------------------------------- integration
def fams(n=3):
    return pd.DataFrame({"Family Number": [str(i) for i in range(1, n + 1)],
                         "is_annotated": ["True"] * n,
                         "F_MONITOR_REPRESENT": ["Y", "U", "N"][:n],
                         "label_M": ["Y", "U", "N"][:n]})


def figs(rows):
    cols = ["family_number", "publication_number", "kind_code", "figure_id", "figure_description",
            "processing_status", "publication_role", "source_document_id", "source_document_url",
            "image_source_url", "image_sha256", "vlm_model", "vlm_model_version", "prompt_version",
            "vlm_config_hash", "processed_at_utc"]
    base = dict.fromkeys(cols, "")
    return pd.DataFrame([{**base, "vlm_model": "m", "prompt_version": "v1", **r} for r in rows], columns=cols)


def res(rows):
    return pd.DataFrame(rows, columns=["family_number", "resolution_status"])


PUBS = pd.DataFrame({"family_number": ["1", "1", "2", "3"], "publication_number": ["US1", "EP1", "CN2", "US3"]})


def test_one_row_per_family_with_multi_figure_multi_publication_and_deterministic_order():
    f = figs([dict(family_number="1", publication_number="US1", figure_id="D00010", figure_description="ten",
                   processing_status="described", publication_role="master"),
              dict(family_number="1", publication_number="US1", figure_id="D00002", figure_description="two",
                   processing_status="described", publication_role="master"),
              dict(family_number="1", publication_number="EP1", figure_id="D00001", figure_description="ep one",
                   processing_status="described", publication_role="fallback")])
    r = res([("1", "resolved_with_figures"), ("1", "resolved_with_figures")])
    a = integ.build_enrichment(fams(), PUBS, f, r)
    b = integ.build_enrichment(fams(), PUBS, f.sample(frac=1, random_state=3), r)
    assert len(a) == 3 and a["Family Number"].is_unique
    assert a.loc[0, "image_descriptions"] == "two\nten\nep one"  # master first, natural figure order
    assert a.loc[0, "image_descriptions"] == b.loc[0, "image_descriptions"]
    prov = json.loads(a.loc[0, "image_description_provenance"])
    assert [p["figure_id"] for p in prov] == ["D00002", "D00010", "D00001"]
    assert a.loc[0, "n_image_descriptions"] == "3"


def test_missing_descriptions_kept_empty_and_status_distinguishes_causes():
    f = figs([dict(family_number="1", publication_number="US1", figure_id="D1", processing_status="vlm_failed"),
              dict(family_number="3", publication_number="US3", figure_id="D1", processing_status="skipped_limit")])
    r = res([("1", "resolved_with_figures"), ("2", "resolved_no_figures"), ("3", "resolved_with_figures")])
    a = integ.build_enrichment(fams(), PUBS, f, r).set_index("Family Number")
    assert a["figure_enrichment_status"].to_dict() == {
        "1": "vlm_failed", "2": "no_figures_listed_by_source", "3": "figures_discovered_not_processed"}
    assert (a["image_descriptions"] == "").all() and (a["n_image_descriptions"] == "0").all()
    assert integ.build_enrichment(fams(), PUBS, figs([]), res([])).iloc[0]["figure_enrichment_status"] == "not_enriched"


@pytest.mark.parametrize("rs,fs,exp", [
    (["document_not_found"], [], "document_not_found"),
    (["document_inaccessible"], [], "document_inaccessible"),
    (["resolved_with_figures"], ["image_inaccessible"], "image_inaccessible"),
    (["resolved_with_figures"], ["described", "vlm_failed"], "described")])
def test_family_status_rules(rs, fs, exp):
    assert integ.family_status(rs, fs) == exp


def test_labels_and_original_columns_preserved_and_unknown_stays_unknown():
    f = fams()
    a = integ.build_enrichment(f, PUBS, figs([]), res([]))
    for c in f.columns:
        assert (a[c] == f[c]).all()
    assert a.loc[1, "label_M"] == "U"


def test_failed_description_text_not_leaked_into_image_descriptions():
    f = figs([dict(family_number="1", publication_number="US1", figure_id="D1", figure_description="bad",
                   processing_status="vlm_failed")])
    assert integ.build_enrichment(fams(), PUBS, f, res([("1", "resolved_with_figures")])).loc[0, "image_descriptions"] == ""


# ---------------------------------------------------------------- labels / features
def frame(n=60, seed=0, desc_every=4):
    rng = np.random.default_rng(seed)
    words = {"M": "sensor monitor display", "P": "predict model simulate", "R": "recommend decision rank",
             "I": "actuator control valve"}
    rows = []
    for i in range(n):
        lab = {k: ("Y" if rng.random() < 0.4 else "N") for k in "MPRI"}
        text = " ".join(words[k] for k in "MPRI" if lab[k] == "Y") + f" noise{i % 7} common"
        rows.append({"Family Number": str(1000 + i), "Title": f"Title {i}", "Abstract": text,
                     "image_descriptions": f"figure shows {text}" if i % desc_every == 0 else "",
                     "n_image_descriptions": "1" if i % desc_every == 0 else "0",
                     **dict(zip(D.LABEL_COLUMNS, [lab[k] for k in "MPRI"]))})
    return pd.DataFrame(rows)


def test_label_matrix_masks_unknown_and_never_maps_u_to_n():
    df = frame(4)
    df.loc[0, D.LABEL_COLUMNS[1]] = "U"
    df.loc[1, D.LABEL_COLUMNS[2]] = ""
    y, known = D.label_matrix(df)
    assert not known[0, 1] and not known[1, 2] and known.sum() == 14
    with pytest.raises(ValueError):
        bad = df.copy()
        bad.loc[2, D.LABEL_COLUMNS[0]] = "maybe"
        D.label_matrix(bad)


def test_eligibility_drops_all_unknown_but_keeps_rows_without_descriptions():
    df = frame(6)
    df.loc[0, D.LABEL_COLUMNS] = ["U"] * 4
    df.loc[1, "Abstract"] = " "
    el = D.eligibility(df)
    assert not el.loc[0, "eligible_A"] and not el.loc[1, "eligible_A"]
    assert el.loc[2:, "eligible_A"].all()
    assert (el.loc[2:, "has_image_description"] == (df.loc[2:, "n_image_descriptions"] == "1")).all()
    assert not el.loc[~el["has_image_description"], "eligible_BC"].any()


def test_text_features_text_only_vs_with_images_and_empty_description():
    df = frame(4)
    t0, t1 = D.build_text(df, False), D.build_text(df, True)
    assert all("figure shows" not in t for t in t0)
    assert "figure shows" in t1[0] and t1[1] == t0[1]  # empty description adds nothing
    with pytest.raises(ValueError):
        D.assert_no_forbidden_features(["Title", "adm_ASSIGNEE_HIDDEN"])
    with pytest.raises(ValueError):
        D.assert_no_forbidden_features(["Title", "Claims"])


# ---------------------------------------------------------------- splits
def test_splits_deterministic_grouped_and_seed_dependent():
    df = frame(100)
    g = {f: str(i // 2) for i, f in enumerate(df["Family Number"])}  # pairs share a group
    a = splits.make_assignments(df, g, [1, 2])
    b = splits.make_assignments(df, g, [1, 2])
    pd.testing.assert_frame_equal(a, b)
    one = a[a["repeat"] == 0].set_index("family")["split"]
    for i in range(0, 100, 2):
        assert one[str(1000 + i)] == one[str(1001 + i)]
    assert set(one) == {"train", "val", "test"}
    assert not (a[a.repeat == 0]["split"].values == a[a.repeat == 1]["split"].values).all()


def test_related_family_groups_link_shared_priority_and_citation_only_when_asked():
    df = frame(3)
    for c in ["Priority Numbers", "Application Number", "Backward Citations", "Forward Citations",
              "Probable Patent Assignee"]:
        df[c] = ""
    df["Priority Numbers"] = ["P1; P2", "P2", "P9"]
    df["Title"] = ["alpha beta", "gamma delta", "epsilon zeta"]
    df["Abstract"] = ["one two", "three four", "five six"]
    df["Backward Citations"] = ["", "", "US0 A1"]
    pubs = pd.DataFrame({"family_number": ["1000", "1001", "1002"], "publication_number": ["US0", "US1", "US5"]})
    g = related_family_groups(df, pubs)
    assert g["1000"] == g["1001"] and g["1002"] == "1002"
    g2 = related_family_groups(df, pubs, use_citations=True)
    assert g["1002"] != g["1000"] and g2["1002"] == g2["1000"]  # citation link only in strict mode


# ---------------------------------------------------------------- model / metrics
def test_unknown_targets_do_not_influence_fit_and_vectorizer_uses_train_only():
    df = frame(80)
    tr, va = df.iloc[:50], df.iloc[50:]
    y, k = D.label_matrix(tr)
    yv, kv = D.label_matrix(va)
    k = k.copy()
    k[:5, 0] = False
    y1, y2 = y.copy(), y.copy()
    y2[:5, 0] = 1 - y2[:5, 0]  # flip the masked entries
    t_tr, t_va = D.build_text(tr, False), D.build_text(va, False) + ["onlyvalword onlyvalword"]
    yv2 = np.vstack([yv, np.zeros((1, 4))])
    kv2 = np.vstack([kv, np.ones((1, 4), bool)])
    m1, _ = model.fit(t_tr, y1, k, t_va, yv2, kv2)
    m2, _ = model.fit(t_tr, y2, k, t_va, yv2, kv2)
    np.testing.assert_allclose(m1.predict_proba(t_va), m2.predict_proba(t_va))
    assert "onlyvalword" not in m1.vectorizer.vocabulary_


def test_training_is_reproducible_and_probabilities_stable():
    df = frame(80)
    tr, va = df.iloc[:50], df.iloc[50:]
    y, k = D.label_matrix(tr)
    yv, kv = D.label_matrix(va)
    ps = [model.fit(D.build_text(tr, True), y, k, D.build_text(va, True), yv, kv, seed=3)[0]
          .predict_proba(D.build_text(va, True)) for _ in range(2)]
    np.testing.assert_array_equal(ps[0], ps[1])
    assert np.isfinite(ps[0]).all() and ((ps[0] >= 0) & (ps[0] <= 1)).all()
    assert ps[0][:, 0].mean() > 0 and ps[0].shape == (30, 4)


def test_single_class_label_falls_back_to_prior_and_empty_val_metric_uses_default_c():
    df = frame(30)
    y, k = D.label_matrix(df)
    y[:, 3] = 0
    m, log = model.fit(D.build_text(df, False), y, k, D.build_text(df.iloc[:2], False), y[:2], k[:2])
    assert m.heads[3] == 0.0
    assert m.predict_proba(["x"])[0, 3] == 0.0


def test_metrics_exclude_unknown_and_report_nan_when_undefined():
    y = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [1, 1, 0, 0]], float)
    known = np.array([[1, 1, 1, 1], [1, 1, 1, 0], [0, 1, 1, 1]], bool)
    p = np.array([[.9, .1, .2, .8], [.2, .9, .1, .5], [.1, .8, .3, .1]])
    pl = metrics.per_label(y, known, p)
    assert pl["M"]["n_known"] == 2 and pl["M"]["n_unknown_excluded"] == 1
    assert pl["R"]["support_pos"] == 0 and np.isnan(pl["R"]["average_precision"])
    assert pl["M"]["f1"] == 1.0
    agg = metrics.aggregate(y, known, p)
    assert np.isfinite(agg["micro_f1"]) and agg["hamming_loss_known_only"] <= 1
    assert metrics.aggregate(y, np.zeros_like(known), p)["hamming_loss_known_only"] == 0.0


def test_end_to_end_run_on_synthetic_dataset(tmp_path):
    from mpri_baseline import run_experiments as R
    df = frame(100)
    for c in ["is_annotated", "figure_enrichment_status", "Priority Numbers", "Application Number",
              "Backward Citations", "Forward Citations", "Probable Patent Assignee", *D.LABEL_ALIASES]:
        df[c] = ""
    df["is_annotated"] = "True"
    ds = tmp_path / "synthetic_enriched.csv"
    df.to_csv(ds, index=False)
    out = tmp_path / "run"
    args = type("A", (), dict(out=str(out), dataset=str(ds), seed=1, repeats=2, group_mode="default"))
    res = R.run(args)
    assert {"A", "B", "C"} <= set(res["pooled"])
    assert json.loads((out / "config.json").read_text())["n_eligible_A"] == 100
    assert (out / "splits.csv").exists() and list((out / "checkpoints").glob("*.joblib"))
