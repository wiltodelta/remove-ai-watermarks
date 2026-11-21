"""Paired accounting counts refusals and binds reviews and scores to exact text."""

import json

import pytest

from scripts import benchmark_text_apple as apple
from scripts import benchmark_text_kgw_receipts as kgw_receipts
from scripts import benchmark_text_kgw_sources as kgw_sources
from scripts import benchmark_text_local_repair as local_repair
from scripts import benchmark_text_pairs as bench
from scripts import benchmark_text_threefry_receipts as threefry_receipts


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.fixture
def inputs(tmp_path):
    arms = ["strict", "repair", "copy"]
    pair = {
        "id": "pair-1",
        "detector": "kgw-config-1",
        "marked": {
            "source": "The marked source.\r\n",
            "outputs": dict(zip(arms, [None, "Faithful paraphrase.", "The marked source.\r\n"], strict=True)),
        },
        "clean": {
            "source": "Clean control.",
            "outputs": dict(zip(arms, ["False positive.", "Faithful paraphrase.", None], strict=True)),
        },
    }
    manifest = write(tmp_path / "input.json", {"arms": arms, "detectors": {"kgw-config-1": 120}, "pairs": [pair]})
    observations = [
        {"detector": "kgw-config-1", "sha256": bench.text_digest(value), "detected": positive, "positions": 140}
        for value, positive in (
            ("The marked source.\r\n", True),
            ("Clean control.", False),
            ("Faithful paraphrase.", False),
            ("False positive.", True),
        )
    ]
    detections = write(tmp_path / "detections.json", observations)
    run = tmp_path / "run"
    bench.bind(manifest, run)
    packet = json.loads((run / "blind-pairs.json").read_bytes())
    assert len(packet) == 1
    assert set(packet[0]) == {"id", "source", "candidate"}
    assert packet[0]["source"] == pair["marked"]["source"]
    reviews = write(tmp_path / "reviews.json", {packet[0]["id"]: {"faithful": True, "reason": "Same facts."}})
    bench.seal(run, reviews)
    return manifest, run, reviews, detections


def test_paired_comparison_counts_every_attempt(inputs):
    report = bench.score(*inputs, baseline="strict")
    groups = {(row["side"], row["arm"]): row for row in report["groups"]}
    assert groups["marked", "strict"]["refused"] == 1
    assert groups["marked", "copy"]["unchanged"] == 1
    assert groups["marked", "repair"]["joint"] == 1
    assert groups["clean", "strict"]["detector_positive"] == 1
    assert "joint" not in groups["clean", "repair"]
    assert all(row["n"] == 1 for row in groups.values())
    paired = next(row for row in report["paired"] if row["arm"] == "repair")
    assert (paired["wins"], paired["losses"], paired["n"], paired["p_value"]) == (1, 0, 1, 1.0)


def test_symmetric_comparison_reviews_changed_clean_and_derives_retention_for_every_arm(inputs):
    manifest, _, reviews, detections = inputs
    data = json.loads(manifest.read_bytes())
    data["comparison_protocol"] = "symmetric-v1"
    marked = data["pairs"][0]["marked"]
    clean = data["pairs"][0]["clean"]
    shared = "The locked cabinet stays closed."
    marked["source"] = f"Original opening. {shared}\n\n```\nSame code.\n```"
    outputs = {
        "strict": f"First paraphrase. {shared}\n\n```\nSame code.\n```",
        "repair": f"Second paraphrase. {shared}\n\n```\nSame code.\n```",
        "copy": marked["source"],
    }
    marked["outputs"] = outputs
    marked["retained_prose"] = {"strict": [shared, "Same code."]}
    write(manifest, data)
    run = manifest.parent / "symmetric"
    bench.bind(manifest, run)
    packet = json.loads((run / "blind-pairs.json").read_bytes())
    assert len(packet) == 4
    assert all(row.keys() == {"id", "source", "candidate"} for row in packet)
    write(
        reviews,
        {
            row["id"]: {"faithful": row["candidate"] != "False positive.", "reason": "Fidelity review."}
            for row in packet
            if row["source"] != clean["source"]
        },
    )
    with pytest.raises(ValueError, match="label each blind pair"):
        bench.seal(run, reviews)
    write(
        reviews,
        {
            row["id"]: {"faithful": row["candidate"] != "False positive.", "reason": "Fidelity review."}
            for row in packet
        },
    )
    bench.seal(run, reviews)
    observations = json.loads(detections.read_bytes())
    observations[0]["sha256"] = bench.text_digest(marked["source"])
    observations += [{**observations[2], "sha256": bench.text_digest(outputs[arm])} for arm in ("strict", "repair")]
    write(detections, observations)
    with pytest.raises(ValueError, match="Missing 1 detector receipt"):
        bench.score(manifest, run, reviews, detections, baseline="strict")
    observations += [{**observations[2], "sha256": bench.text_digest(shared), "positions": 5}]
    write(detections, observations)
    result = bench.score(manifest, run, reviews, detections, baseline="strict")
    assert result["comparison_protocol"] == "symmetric-v1"
    groups = {(row["side"], row["arm"]): row for row in result["groups"]}
    for arm in ("strict", "repair"):
        marked_group = groups["marked", arm]
        assert marked_group["partial_unverified"] == marked_group["faithful"] == 1
        assert marked_group["joint"] == 0
    assert groups["clean", "strict"]["unfaithful"] == groups["clean", "strict"]["reviewed"] == 1
    assert groups["clean", "repair"]["faithful"] == groups["clean", "repair"]["reviewed"] == 1
    assert groups["clean", "copy"]["reviewed"] == 0
    assert all("joint" not in row for row in result["results"] if row["side"] == "clean")


def test_baseline_can_be_added_to_one_arm_without_relaxing_comparison(inputs):
    manifest, *_ = inputs
    data = json.loads(manifest.read_bytes())
    data["arms"] = ["strict"]
    for pair in data["pairs"]:
        for side in bench.SIDES:
            pair[side]["outputs"] = {"strict": pair[side]["outputs"]["strict"]}
    write(manifest, data)
    arms, _, trials, _ = bench._manifest(manifest, minimum_arms=1)
    assert arms == ["strict"]
    assert len(trials) == 2
    with pytest.raises(ValueError, match="requires at least 2"):
        bench.bind(manifest, manifest.parent / "single-arm")


def test_two_sealed_reviews_require_agreement_and_report_disputes(inputs):
    manifest, _, reviews, detections = inputs
    data = json.loads(manifest.read_bytes())
    data["comparison_protocol"] = "symmetric-v1"
    write(manifest, data)
    run = manifest.parent / "dual-review"
    bench.bind(manifest, run)
    packet = json.loads((run / "blind-pairs.json").read_bytes())
    assert len(packet) == 3
    first = {
        row["id"]: {"faithful": row["candidate"] != "False positive.", "reason": "First explanation."} for row in packet
    }
    second = {row["id"]: {"faithful": False, "reason": "Second explanation."} for row in packet}
    write(reviews, first)
    second_path = write(run / "second.json", second)
    with pytest.raises(ValueError, match="distinct"):
        bench.seal(run, reviews, second_reviews=reviews)
    assert not (run / "review-lock.json").exists()
    bench.seal(run, reviews, second_reviews=second_path)
    with pytest.raises(ValueError, match="changed since sealing"):
        bench.score(manifest, run, reviews, detections, baseline="strict")
    report = bench.score(manifest, run, reviews, detections, baseline="strict", second_reviews=second_path)
    groups = {(row["side"], row["arm"]): row for row in report["groups"]}
    assert report["reviewers"] == 2
    assert groups["marked", "repair"]["joint"] == 0
    assert groups["marked", "repair"]["review_disagreements"] == 1
    assert groups["clean", "repair"]["review_disagreements"] == 1
    assert groups["clean", "repair"]["unfaithful"] == 0
    assert groups["clean", "strict"]["unfaithful"] == 1
    second_path.write_bytes(second_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="changed since sealing"):
        bench.score(manifest, run, reviews, detections, baseline="strict", second_reviews=second_path)


def test_apple_merge_keeps_exact_text_and_refusals(inputs):
    manifest, *_ = inputs
    data = json.loads(manifest.read_bytes())
    marked, clean = (data["pairs"][0][side] for side in bench.SIDES)
    marked["source"] = "Original opening. The valve remains open."
    candidate = "New opening. The valve remains open.\u2028A final line."
    cases = [{"id": bench.text_digest(t["source"]), "source": t["source"]} for t in (marked, clean)]
    rows = [{"id": cases[0]["id"], "candidate": candidate}, {"id": cases[1]["id"], "candidate": "  "}]
    receipt = manifest.parent / "responses.jsonl"
    receipt.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="generation order"):
        apple._merge(data, "apple", list(reversed(cases)), receipt)
    assert "apple" not in data["arms"]
    attempts = apple._merge(data, "apple", cases, receipt)
    assert marked["outputs"]["apple"] == candidate
    assert marked["retained_prose"]["apple"] == ["The valve remains open."]
    assert clean["outputs"]["apple"] is None
    assert attempts[1]["error"] == "Empty model response."


def test_underscored_output_does_not_count_as_clear(inputs):
    *_, detections = inputs
    observations = json.loads(detections.read_bytes())
    observations[2]["positions"] = 119
    write(detections, observations)
    report = bench.score(*inputs, baseline="strict")
    marked = next(row for row in report["groups"] if row["side"] == "marked" and row["arm"] == "repair")
    assert marked["n"] == marked["returned"] == marked["insufficient_positions"] == 1
    assert marked["joint"] == 0


@pytest.mark.parametrize(
    ("detected", "positions", "expected_joint"), [(True, 140, 0), (False, 119, 0), (False, 120, 1)]
)
def test_retained_prose_requires_its_own_negative_receipt(inputs, detected, positions, expected_joint):
    manifest, _, reviews, detections = inputs
    data = json.loads(manifest.read_bytes())
    marked = data["pairs"][0]["marked"]
    marked["source"] += "Shared sentence."
    marked["outputs"]["copy"] = marked["source"]
    marked["outputs"]["repair"] += " Shared sentence."
    marked["retained_prose"] = {"repair": ["Shared sentence."]}
    write(manifest, data)
    run = manifest.parent / "with-retention"
    bench.bind(manifest, run)
    ident = json.loads((run / "blind-pairs.json").read_bytes())[0]["id"]
    write(reviews, {ident: {"faithful": True, "reason": "Same facts."}})
    bench.seal(run, reviews)
    observations = json.loads(detections.read_bytes())
    observations[0]["sha256"] = bench.text_digest(marked["source"])
    observations.append({**observations[2], "sha256": bench.text_digest(marked["outputs"]["repair"])})
    write(detections, observations)
    with pytest.raises(ValueError, match="Missing 1 detector receipt"):
        bench.score(manifest, run, reviews, detections, baseline="strict")
    observations.append(
        {
            "detector": "kgw-config-1",
            "sha256": bench.text_digest("Shared sentence."),
            "detected": detected,
            "positions": positions,
        }
    )
    write(detections, observations)
    report = bench.score(manifest, run, reviews, detections, baseline="strict")
    group = next(row for row in report["groups"] if row["side"] == "marked" and row["arm"] == "repair")
    assert group["clear"] == group["faithful"] == group["partial"] == 1
    assert group["partial_unverified"] == 1 - expected_joint
    assert group["joint"] == expected_joint


@pytest.mark.parametrize("breakage", ["review", "text", "receipt", "control"])
def test_score_fails_before_counting_invalid_evidence(inputs, breakage):
    manifest, _, reviews, detections = inputs
    if breakage == "review":
        reviews.write_bytes(reviews.read_bytes() + b"\n")
    elif breakage == "text":
        manifest.write_bytes(manifest.read_bytes().replace(b"Faithful paraphrase", b"Changed output"))
    else:
        observations = json.loads(detections.read_bytes())
        if breakage == "receipt":
            observations[2]["sha256"] = bench.text_digest("Different output.")
        else:
            observations[1]["detected"] = True
        write(detections, observations)
    with pytest.raises(ValueError, match=r"changed since|detector receipt|unqualified clean source"):
        bench.score(*inputs, baseline="strict")


@pytest.mark.parametrize("change_retained", [True, False])
def test_local_comparison_declares_retention_and_shares_one_residual_request(change_retained):
    first_sentence = "The cabinet will remain locked until Friday."
    original = first_sentence + " Staff will record the seal number in the log."
    restored = first_sentence + " Staff will enter the seal number in the log."
    changed = "Until Friday, the cabinet will stay locked."
    calls = []

    def initial(prose):
        calls.append(prose)
        return (restored,)

    def residual(prose):
        calls.append(prose)
        return (changed if change_retained else first_sentence,)

    result, trace = local_repair._rewrite(original, initial, residual)
    assert calls == [(original,), (first_sentence,)]
    assert trace["calls"] == 2
    assert result["retained_prose"] == {"restore": [first_sentence]}
    assert result["outputs"]["restore"] == restored
    expected = changed + " Staff will enter the seal number in the log." if change_retained else None
    assert result["outputs"]["document_residual"] == result["outputs"]["strict_residual"] == expected


def test_local_comparison_rejects_non_loopback_before_opening_inputs(tmp_path):
    with pytest.raises(ValueError, match="literal loopback"):
        local_repair.generate(tmp_path / "missing", tmp_path / "out", url="http://192.0.2.1/v1", model="test")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    ("change", "message"),
    [("review", "Reviews have changed since sealing"), ("reference", "Pinned file digest differs")],
)
def test_kgw_receipts_fail_closed_before_loading_the_reference(inputs, tmp_path, change, message):
    manifest, run, reviews, _ = inputs
    reference = tmp_path / "reference"
    reference.mkdir()
    for name in kgw_receipts.REFERENCE_FILES:
        (reference / name).write_bytes(b"approved content")
    config = write(
        tmp_path / "kgw.json",
        {
            "detector": "kgw-config-1",
            "key": 42,
            "gamma": 0.25,
            "delta": 2,
            "z_threshold": 4.0,
            "reference_sha256": {name: bench.text_digest("approved content") for name in kgw_receipts.REFERENCE_FILES},
        },
    )
    if change == "review":
        reviews.write_bytes(reviews.read_bytes() + b"\n")
    else:
        (reference / "watermark_processor.py").write_bytes(b"modified content")
    output = tmp_path / "kgw-receipts"
    with pytest.raises(ValueError, match=message):
        kgw_receipts.generate(manifest, run, reviews, config, reference, tmp_path / "missing-tokenizer", output)
    assert not output.exists()


def test_threefry_receipts_reject_a_different_tournament_profile():
    with pytest.raises(ValueError, match="exactly two tournament leaves"):
        threefry_receipts._settings(
            {
                "detector": "threefry",
                "alpha": 0.01,
                "eos_token_ids": [1],
                "scheme": {"key": 42, "ngram_len": 5, "depth": 30, "context_history_size": 1024, "num_leaves": 3},
            }
        )


def test_threefry_receipts_reject_changed_tokenizer_before_scoring(inputs, tmp_path, monkeypatch):
    manifest, run, reviews, _ = inputs
    reference = tmp_path / "reference"
    (reference / "synthid_text").mkdir(parents=True)
    for name in threefry_receipts.REFERENCE_FILES:
        (reference / name).write_bytes(b"approved reference")
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    for name in threefry_receipts.TOKENIZER_FILES:
        (tokenizer / name).write_bytes(b"approved tokenizer")
    (tokenizer / "tokenizer.json").write_bytes(b"changed tokenizer")
    config = write(
        tmp_path / "threefry.json",
        {
            "detector": "kgw-config-1",
            "alpha": 0.01,
            "eos_token_ids": [1],
            "scheme": {"key": 42, "ngram_len": 5, "depth": 30, "context_history_size": 1024, "num_leaves": 2},
            "reference_sha256": {
                name: bench.text_digest("approved reference") for name in threefry_receipts.REFERENCE_FILES
            },
            "tokenizer_sha256": {
                name: bench.text_digest("approved tokenizer") for name in threefry_receipts.TOKENIZER_FILES
            },
        },
    )
    output = tmp_path / "threefry-receipts"
    monkeypatch.setattr(
        threefry_receipts,
        "_reference_types",
        lambda *_: pytest.fail("A changed tokenizer must be rejected before scoring."),
    )
    with pytest.raises(ValueError, match=r"Pinned file digest differs for tokenizer\.json"):
        threefry_receipts.generate(manifest, run, reviews, config, reference, tokenizer, output)
    assert not output.exists()


def test_kgw_source_generation_requires_a_distinct_wrong_key():
    with pytest.raises(ValueError, match="wrong-key control must use a different"):
        kgw_sources._check_plan(
            {
                "minimum_positions": 120,
                "batch_size": 1,
                "seed": 1,
                "top_k": 48,
                "max_new_tokens": 480,
                "key": 42,
                "wrong_key": 42,
                "temperature": 0.8,
                "prompts": ["Synthetic source prompt."],
            }
        )
