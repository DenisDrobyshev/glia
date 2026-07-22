from __future__ import annotations

from glia.trajectory import (
    ApprovalRequested,
    ApprovalResolved,
    Compacted,
    ModelCall,
    ModelDelta,
    ModelResponse,
    RunFinished,
    RunStarted,
    ToolCalled,
    ToolReturned,
    Trajectory,
)
from glia.types import Usage, assistant, user


def test_every_event_to_dict_carries_kind_and_payload():
    events = [
        RunStarted(prompt="p"),
        ModelCall(step=1, message_count=2, tool_count=3),
        ModelResponse(step=1, text="t", stop_reason="end_turn", tool_uses=("a",), usage=Usage(1, 2)),
        ModelDelta(step=1, text="chunk"),
        ToolCalled(step=1, tool_use_id="t1", name="f", arguments={"x": 1}),
        ToolReturned(step=1, tool_use_id="t1", name="f", content="r", is_error=False, duration_s=0.1),
        ApprovalRequested(step=1, tool_use_id="t1", name="f", arguments={"x": 1}),
        ApprovalResolved(step=1, tool_use_id="t1", name="f", allowed=False, reason="no"),
        Compacted(freed_messages=3, summary="s"),
        RunFinished(step=2, output="done", stop_reason="end_turn"),
    ]
    for event in events:
        d = event.to_dict()
        assert d["kind"] == event.kind
        assert "at" in d
    # spot-check a couple of payloads
    assert events[2].to_dict()["usage"]["input_tokens"] == 1
    assert events[7].to_dict()["allowed"] is False


def test_trajectory_usage_accumulates():
    traj = Trajectory.new()
    traj.add_usage(Usage(1, 1))
    traj.add_usage(Usage(2, 3))
    assert traj.usage.input_tokens == 3 and traj.usage.output_tokens == 4


def test_last_assistant_text_and_events_of():
    traj = Trajectory.new()
    assert traj.last_assistant_text() == ""  # none yet
    traj.add_message(user("q"))
    traj.add_message(assistant("first"))
    traj.add_message(assistant("second"))
    assert traj.last_assistant_text() == "second"


def test_events_survive_serialisation_as_loaded_records():
    traj = Trajectory.new(system="s")
    traj.record(ModelCall(step=1, message_count=1, tool_count=0))
    traj.record(RunFinished(step=1, output="ok", stop_reason="end_turn"))

    reloaded = Trajectory.from_dict(traj.to_dict())
    assert reloaded.system == "s"
    kinds = [e.kind for e in reloaded.events]
    assert kinds == ["model_call", "run_finished"]
    # A reloaded event still round-trips its recorded payload.
    assert reloaded.events[1].to_dict()["output"] == "ok"
    assert reloaded.events_of("model_call")  # filtering still works
