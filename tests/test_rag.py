"""The retrieval system, hermetic: small texts written here, no downloads, no model server."""
from __future__ import annotations

import json

import numpy as np
import pytest

from vxrag import rst
from vxrag.chunking import chunk_units
from vxrag.evaluation.labels import build_queries
from vxrag.evaluation.metrics import mrr_at_k, ndcg_at_k, paired_bootstrap, recall_at_k, unlabelled_share
from vxrag.evaluation.runner import run, select
from vxrag.ingest import ingest
from vxrag.lexical import BM25, tokenize_ident, tokenize_plain
from vxrag.retrieval import Index
from vxrag.store import CorpusStore, Unit

PAGE = """.. _calib-pots:

Potentiometers
**************

Introduction
============

The potentiometers are used for homing. The conversion is
``position = offset + scale * voltage``. See
:ref:`configuration generators<config-generators>` for the IO files.

.. figure:: /images/psm.png
   :width: 300

   A caption that is dropped

.. note::

   Hold the arm in the zero position before you start.

Procedure
=========

Run the script:

.. code-block:: bash

   dvrk_calibrate_potentiometers.py -t offsets -a PSM1 \\
       -c sawRobotIO1394-PSM1-12345.json

Then restart the console.
"""

GENERATORS = """.. _config-generators:

Configuration generators
************************

The generator turns a ``.cal`` file into the IO configuration used by the controller,
for example ``sawRobotIO1394-PSM1-12345.json``. It has to be run once for every arm.
"""


# ------------------------------------------------------------------------------ reading the sources

def test_rst_sections_labels_code_and_references():
    secs = rst.parse(PAGE)
    assert [(s.title, s.level) for s in secs] == [("Potentiometers", 1), ("Introduction", 2), ("Procedure", 2)]
    assert secs[0].labels == ["calib-pots"] and secs[1].path == ["Potentiometers", "Introduction"]
    intro, proc = secs[1], secs[2]
    assert "`position = offset + scale * voltage`" in intro.text       # literal kept, markup gone
    assert "caption" not in intro.text and "Hold the arm in the zero position" in intro.text
    assert intro.refs[0].target == "config-generators" and intro.refs[0].text == "configuration generators"
    assert "See configuration generators for the IO files." in intro.refs[0].sentence
    assert len(proc.code) == 1 and "sawRobotIO1394-PSM1-12345.json" in proc.code[0]
    assert proc.text.count("```") == 2 and "Then restart the console." in proc.text


@pytest.fixture()
def corpus(tmp_path, monkeypatch):
    """A tiny corpus in the layout the fetch scripts produce."""
    monkeypatch.setenv("VXRAG_DATA_DIR", str(tmp_path))
    docs = tmp_path / "sources" / "docs" / "pages"
    (docs / "calibration" / "Classic").mkdir(parents=True)
    (docs / "configuration").mkdir(parents=True)
    (docs / "calibration" / "Classic" / "potentiometers.rst").write_text(PAGE, encoding="utf-8")
    (docs / "configuration" / "generators.rst").write_text(GENERATORS, encoding="utf-8")
    issues = tmp_path / "sources" / "issues"
    issues.mkdir(parents=True)
    issues.joinpath("repo.json").write_text(json.dumps([
        {"repo": "org/repo", "number": 7, "title": "Firewire nodes not found", "state": "closed",
         "created_at": "2020-01-01T00:00:00Z", "closed_at": "2020-01-02T00:00:00Z", "updated_at": "2020-01-02T00:00:00Z",
         "labels": ["bug"], "author_association": "NONE", "body": "The boards are not detected after power on.",
         "comments": [{"id": 1, "created_at": "2020-01-02T00:00:00Z", "author_association": "CONTRIBUTOR",
                       "by_opener": False, "body": "Reload the kernel modules: rmmod firewire_ohci then modprobe."}]},
        {"repo": "org/repo", "number": 8, "title": "Add a meta package", "state": "open",
         "created_at": "2021-01-01T00:00:00Z", "updated_at": "2021-01-01T00:00:00Z", "labels": [],
         "author_association": "CONTRIBUTOR", "body": "", "comments": []}]), encoding="utf-8")
    (tmp_path / "sources" / "changelog").mkdir(parents=True)
    (tmp_path / "sources" / "changelog" / "CHANGELOG.md").write_text(
        "Change log\n==========\n\n2.4.0 (2026-01-07)\n==================\n\nStarting with 2.4.0 use ROS 2.\n\n"
        "* API changes:\n  * Top configuration level was console.json, now system.json\n"
        "* Bug fixes:\n  * Fixed homing\n\n2.3.0 (2024-08-30)\n==================\n\n* New features:\n  * Si support\n",
        encoding="utf-8")
    schemas = tmp_path / "dvrk" / "share" / "schemas"
    schemas.mkdir(parents=True)
    schemas.joinpath("dvrk-psm.schema.json").write_text(json.dumps({
        "title": "dVRK PSM", "description": "PSM configuration.", "required": ["kinematic"],
        "properties": {"kinematic": {"type": "string", "description": "Kinematic file."},
                       "tool_detection": {"type": "string", "enum": ["AUTOMATIC", "MANUAL", "FIXED"],
                                          "description": "How the instrument is detected."},
                       "arms": {"type": "array", "items": {"properties": {"name": {"type": "string"}}}}}}),
        encoding="utf-8")
    return CorpusStore(tmp_path / "rag.sqlite")


def test_ingestion_reads_every_source_and_is_idempotent(corpus):
    first = ingest(corpus)
    assert first["units"] == {"changelog": 4, "doc": 4, "schema": 5, "thread": 3}
    assert first["validation"]["references_to_unknown_labels"] == 0 and first["problems"] == []
    again = ingest(corpus)
    assert all(v["written"] == {"added": 0, "changed": 0, "removed": 0, "unchanged": v["units"]}
               for v in again["sources"].values())
    assert again["corpus_hash"] == first["corpus_hash"]
    units = {u.unit_id: u for u in corpus.units()}
    pots = units["doc:calibration/Classic/potentiometers#introduction"]
    assert pots.generation == "Classic" and pots.url.endswith("potentiometers.html#introduction")
    assert units["doc:configuration/generators#configuration-generators"].generation == "both"
    assert units["thread:repo/7#a"].authority == "maintainer" and "thread:repo/8#a" not in units
    assert units["changelog:2.4.0#api-changes"].version == "2.4.0"
    assert "`AUTOMATIC`" in units["schema:dvrk-psm#tool_detection"].text
    assert "schema:dvrk-psm#arms[].name" in units


def test_a_page_removed_upstream_is_removed_from_the_store(corpus, tmp_path):
    ingest(corpus)
    (tmp_path / "sources" / "docs" / "pages" / "configuration" / "generators.rst").unlink()
    report = ingest(corpus)
    assert report["sources"]["doc"]["written"]["removed"] == 1
    assert report["validation"]["references_to_unknown_labels"] == 1       # and the broken link is reported
    assert not any(u.doc_id == "doc:configuration/generators" for u in corpus.units())


# ------------------------------------------------------------------------------ chunking and lexical

def unit(uid, text, source="doc", title="T", path=None, **kw):
    return Unit(unit_id=uid, doc_id=uid.split("#")[0], source=source, kind="section", ord=kw.pop("ord", 0),
                title=title, path=path or [title], text=text, **kw)


def test_chunking_never_cuts_a_code_block_and_keeps_units_apart():
    code = "```\n" + "\n".join(f"line {i} of the listing" for i in range(120)) + "\n```"
    u = unit("doc:a#x", "Intro paragraph.\n\n" + code + "\n\nClosing paragraph.", title="X", path=["A", "X"])
    v = unit("doc:a#y", "Another section.", title="Y", path=["A", "Y"], ord=1)
    for strategy in ("section", "section_ctx"):
        chunks = chunk_units([u, v], strategy, max_chars=400)
        assert all(c.text.count("```") % 2 == 0 for c in chunks)
        assert {c.unit_id for c in chunks} == {"doc:a#x", "doc:a#y"}
    assert chunk_units([u, v], "section_ctx")[0].index_text.startswith("A > X\n")
    assert chunk_units([u, v], "section")[0].index_text.startswith("X\n")
    fixed = chunk_units([u, v], "fixed", window=300, overlap=50)
    assert len(fixed) > 5 and all(len(c.text) <= 300 for c in fixed)


def test_identifier_tokenizer_keeps_identifiers_whole():
    text = "Use /PSM1/measured_cp and sawRobotIO1394-PSM1-12345.json"
    assert "measured_cp" not in tokenize_plain(text) and "psm1/measured_cp" in tokenize_ident(text)
    assert {"saw", "robot", "io", "1394"} <= set(tokenize_ident(text))
    assert "sawrobotio1394-psm1-12345.json" in tokenize_ident(text)


def test_bm25_ranks_the_unit_that_has_the_rare_identifier_first():
    texts = ["the measured position of the arm is published", "use measured_cp to read the pose",
             "position and cp are common words in this page about position and cp and measured values"]
    assert int(np.argmax(BM25(texts, "ident").scores("measured_cp"))) == 1
    assert BM25(texts, "plain").scores("zzz").sum() == 0


def test_search_returns_units_not_chunks_and_respects_filters():
    long_text = "\n\n".join(["potentiometer offsets on the classic arm " * 12] * 8)       # several chunks
    units = [unit("doc:c/Classic/p#a", long_text, generation="Classic"),
             unit("doc:c/Si/p#a", "potentiometer offsets on the si arm", generation="Si"),
             unit("thread:r/1#q", "potentiometer offsets question", source="thread")]
    ix = Index(units, "section")
    assert len(ix.chunks) > 3
    ids = [h.unit_id for h in ix.search("potentiometer offsets", "bm25:ident")]
    assert sorted(ids) == sorted(u.unit_id for u in units)                # one result per unit
    assert "doc:c/Si/p#a" not in [h.unit_id for h in ix.search("potentiometer offsets", "bm25:ident", generation="Classic")]
    assert [h.unit_id for h in ix.search("potentiometer", "bm25:ident", sources={"thread"})] == ["thread:r/1#q"]
    assert "thread:r/1#q" not in [h.unit_id for h in ix.search("potentiometer", "bm25:ident", exclude_units={"thread:r/1#q"})]
    assert ix.search("nothing matches this", "bm25:plain") == []


def test_reciprocal_rank_fusion_of_two_lexical_strategies():
    units = [unit("doc:a#1", "alpha beta gamma"), unit("doc:a#2", "alpha_beta delta"), unit("doc:a#3", "gamma delta")]
    hits = Index(units, "section").search("alpha_beta gamma", "rrf:bm25:plain+bm25:ident")
    assert len(hits) == 3 and all(h.parts for h in hits)
    assert hits[0].score >= hits[1].score >= hits[2].score and [h.rank for h in hits] == [1, 2, 3]


# ------------------------------------------------------------------------------ evaluation

def test_metrics_against_hand_computation():
    rel = {"a": 2, "b": 1, "c": 2}
    ranked = ["x", "a", "b", "y"]
    # DCG = 3/log2(3) + 1/log2(4); ideal = 3/1 + 3/log2(3) + 1/2
    assert ndcg_at_k(ranked, rel, 10) == pytest.approx((3 / np.log2(3) + 0.5) / (3 + 3 / np.log2(3) + 0.5))
    assert recall_at_k(ranked, rel, 10) == 0.5 and recall_at_k(ranked, rel, 1) == 0.0
    assert mrr_at_k(ranked, rel) == 0.5 and unlabelled_share(ranked, rel) == 0.5
    assert ndcg_at_k([], rel) == 0.0 and ndcg_at_k(["a"], {}) == 0.0


def test_paired_bootstrap_separates_a_real_difference_from_noise():
    better = paired_bootstrap([0.2] * 40, [0.5] * 40)
    assert better["low"] > 0 and better["wins"] == 40
    same = paired_bootstrap([0.1, 0.9] * 20, [0.9, 0.1] * 20)
    assert same["low"] < 0 < same["high"]


def test_labels_come_from_links_and_replies_and_exclude_the_source(corpus):
    ingest(corpus)
    qs = {q.qid: q for q in build_queries(corpus.units())}
    xref = next(q for q in qs.values() if q.category == "xref")
    assert xref.relevant == {"doc:configuration/generators#configuration-generators": 2}
    assert xref.exclude == {"doc:calibration/Classic/potentiometers#introduction"}
    thread = qs["thread:repo/7"]
    assert thread.relevant == {"thread:repo/7#a": 2} and thread.exclude == {"thread:repo/7#q"}
    assert "thread:repo/8" not in qs                                   # open, no replies: not a query
    assert all(q.split in ("dev", "test") for q in qs.values())


def test_runs_respect_exclusions_and_selection_keeps_the_baseline_without_evidence(corpus):
    ingest(corpus)
    units = corpus.units()
    queries = build_queries(units)
    for q in queries:
        q.split = "dev"
    a = run(Index(units, "fixed"), queries, "bm25:plain", "dev", latency_probe=2)
    b = run(Index(units, "section_ctx"), queries, "bm25:ident", "dev", latency_probe=2)
    assert a["gates"] == {"excluded_units_returned": 0, "exclusion_respected": True, "split_isolation": True}
    assert a["queries"] == len(queries) and set(a["by_category"]) == {"xref", "thread", "identifier"}
    chosen = select([a, b], "bm25:plain @ fixed")
    # A handful of queries cannot give an interval above zero, so the simpler baseline stays.
    assert chosen["kept_baseline"] or chosen["decisions"][0]["credible_improvement"]
    assert chosen["decisions"][0]["candidate"] == "bm25:ident @ section_ctx"
