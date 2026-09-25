import json
from pathlib import Path

import pytest

from offline.label import (
    guard_submit,
    parse_output_line,
    projected_cost,
    request_line,
    targets,
)


def test_request_line_shape() -> None:
    line = request_line("q1", "gpt-6-luna", "Q?")
    assert line["custom_id"] == "q1"
    assert line["url"] == "/v1/chat/completions"
    assert line["body"]["model"] == "gpt-6-luna"
    assert line["body"]["reasoning_effort"] == "none"
    assert line["body"]["messages"][-1] == {"role": "user", "content": "Q?"}


def test_parse_ok_line() -> None:
    body = {
        "choices": [{"message": {"content": "Answer: B"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    line = {"custom_id": "q1", "response": {"status_code": 200, "body": body}}
    assert parse_output_line(line) == {
        "id": "q1", "text": "Answer: B", "prompt_tokens": 10, "completion_tokens": 5,
    }


def test_parse_failed_line_is_empty_answer() -> None:
    line = {"custom_id": "q1", "response": {"status_code": 500, "body": {}}, "error": {}}
    assert parse_output_line(line)["text"] == ""


def test_targets_send_only_test_split_to_expensive() -> None:
    qs = [{"split": "train"}] * 7 + [{"split": "test"}] * 3
    t = targets(qs)
    assert len(t["gpt-6-luna"]) == 10
    assert len(t["gpt-6-sol"]) == 3


def test_projected_cost_applies_batch_discount() -> None:
    got = projected_cost({"gpt-6-luna": 0.001, "gpt-6-sol": 0.01}, {"gpt-6-luna": 1000, "gpt-6-sol": 300})
    assert got == pytest.approx(0.5 * (1.0 + 3.0))


def test_guard_refuses_without_pilot(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        guard_submit(tmp_path / "pilot.json", tmp_path / "batches.json")


def test_guard_refuses_over_budget(tmp_path: Path) -> None:
    (tmp_path / "pilot.json").write_text(json.dumps({"projected_batch_usd": 5.01}))
    with pytest.raises(SystemExit):
        guard_submit(tmp_path / "pilot.json", tmp_path / "batches.json")


def test_guard_refuses_double_submit(tmp_path: Path) -> None:
    (tmp_path / "pilot.json").write_text(json.dumps({"projected_batch_usd": 1.0}))
    (tmp_path / "batches.json").write_text("{}")
    with pytest.raises(SystemExit):
        guard_submit(tmp_path / "pilot.json", tmp_path / "batches.json")


def test_guard_allows_first_submit_in_budget(tmp_path: Path) -> None:
    (tmp_path / "pilot.json").write_text(json.dumps({"projected_batch_usd": 1.0}))
    guard_submit(tmp_path / "pilot.json", tmp_path / "batches.json")
