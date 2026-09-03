import json

from verl.experimental.mopd_systems.timeline import emit_timeline_event


def test_emit_timeline_event_writes_jsonl_when_enabled(tmp_path, monkeypatch):
    timeline = tmp_path / "timeline.jsonl"
    monkeypatch.setenv("VERL_MOPD_TIMELINE_JSONL", str(timeline))

    class Scalar:
        def item(self):
            return "math"

    emit_timeline_event("teacher_topk_start", routing_key=Scalar(), shape=(1, 2, 3))

    record = json.loads(timeline.read_text(encoding="utf-8"))
    assert record["event"] == "teacher_topk_start"
    assert record["routing_key"] == "math"
    assert record["shape"] == [1, 2, 3]
    assert "time_ns" in record


def test_emit_timeline_event_is_noop_when_disabled(tmp_path, monkeypatch):
    monkeypatch.delenv("VERL_MOPD_TIMELINE_JSONL", raising=False)
    emit_timeline_event("student_rollout_start", output_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []
