"""In-memory cache bounds: HF engine LRU and run-job eviction."""

import picochat.web as web


class _FakeProc:
    def __init__(self, alive):
        self._alive = alive

    def poll(self):
        return None if self._alive else 0


def test_register_run_job_evicts_oldest_finished_keeps_running(monkeypatch):
    web._RUN_JOBS.clear()
    monkeypatch.setattr(web, "_MAX_RUN_JOBS", 5)

    web._register_run_job("run", {"id": "run", "process": _FakeProc(True), "started_at": 1000})
    for i in range(10):
        web._register_run_job(f"f{i}", {"id": f"f{i}", "process": _FakeProc(False), "started_at": i})

    assert len(web._RUN_JOBS) == 5            # capped
    assert "run" in web._RUN_JOBS             # running job never evicted
    assert "f0" not in web._RUN_JOBS          # oldest finished evicted first
    assert "f9" in web._RUN_JOBS              # newest finished retained
    web._RUN_JOBS.clear()


def test_hf_engine_cache_is_bounded_lru(monkeypatch):
    web._HF_ENGINES.clear()
    monkeypatch.setattr(web, "_MAX_HF_ENGINES", 2)

    class Stub:
        def __init__(self, *, model_path, device="cpu"):
            self.model_path = model_path

    monkeypatch.setattr("picochat.hf_infer.HFGenerator", Stub)

    web._get_hf_engine("dir/a")
    web._get_hf_engine("dir/b")
    assert set(web._HF_ENGINES) == {"dir/a", "dir/b"}

    web._get_hf_engine("dir/c")               # evicts least-recently-used (a)
    assert set(web._HF_ENGINES) == {"dir/b", "dir/c"}

    web._get_hf_engine("dir/b")               # touch b -> most-recently-used
    web._get_hf_engine("dir/d")               # evicts c, not b
    assert set(web._HF_ENGINES) == {"dir/b", "dir/d"}

    # cache hit returns the same instance without reloading
    first = web._get_hf_engine("dir/b")
    second = web._get_hf_engine("dir/b")
    assert first is second
    web._HF_ENGINES.clear()
