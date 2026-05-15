import json

from picochat.sanity import PreH100SanityConfig, run_preh100_sanity


def test_run_preh100_sanity_writes_reports(tmp_path):
    report = run_preh100_sanity(PreH100SanityConfig(
        out_dir=str(tmp_path / "sanity"),
        precision="float32",
    ))

    assert report["status"] == "passed"
    assert {check["name"] for check in report["checks"]} == {
        "precision_backward",
        "kv_cache_equivalence",
        "resume_fingerprint_guard",
        "sharded_loader",
        "hf_export",
        "torch_compile",
    }
    assert any(check["status"] == "skip" for check in report["checks"])
    report_path = tmp_path / "sanity" / "preh100_sanity.json"
    markdown_path = tmp_path / "sanity" / "preh100_sanity.md"
    assert report_path.exists()
    assert markdown_path.exists()
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["status"] == "passed"
    assert "Picochat Pre-H100 Sanity" in markdown_path.read_text(encoding="utf-8")
