from __future__ import annotations

import pytest
from conftest import run

from glia.guardrails import no_secrets, require_pattern
from glia.memory import SummarizingCompactor, TrimmingCompactor, _keeps_message_pairs_intact, _render
from glia.providers import EchoLLM
from glia.tools import _stringify, _type_to_schema
from glia.trajectory import Trajectory
from glia.types import Message, Text, ToolResult, ToolUse, assistant, user

# --- memory / context engineering ---------------------------------------------


def test_cut_index_keeps_toolcall_and_results_together():
    # ... assistant(tool_use) then user(tool_result) must not be split.
    msgs = [
        user("q0"),
        Message("assistant", [ToolUse("t1", "f", {})]),
        Message("user", [ToolResult("t1", "r")]),
        assistant("a"),
    ]
    # keep_last=2 would cut at index 2 (splitting the pair); it backs up to 1.
    assert _keeps_message_pairs_intact(msgs, keep_last=2) == 1


def test_trimming_no_op_on_short_trajectory():
    traj = Trajectory.new()
    traj.add_message(user("only one"))
    compactor = TrimmingCompactor(max_messages=100, keep_last=20)
    assert compactor.should_compact(traj) is False
    assert run(compactor.compact(traj, EchoLLM())) == ""  # nothing to trim


def test_render_includes_tool_blocks():
    msgs = [
        Message("assistant", [ToolUse("t1", "search", {"q": "x"})]),
        Message("user", [ToolResult("t1", "found", is_error=True)]),
        assistant("done"),
    ]
    rendered = _render(msgs)
    assert "calls search" in rendered
    assert "tool (error): found" in rendered


def test_summarizing_compactor_no_op_when_nothing_old():
    traj = Trajectory.new()
    traj.add_message(user("hi"))
    compactor = SummarizingCompactor(max_messages=100, keep_last=20)
    assert run(compactor.compact(traj, EchoLLM(["unused"]))) == ""


# --- tool schema edge cases ---------------------------------------------------


def test_schema_list_and_dict():
    assert _type_to_schema(list[int]) == {"type": "array", "items": {"type": "integer"}}
    assert _type_to_schema(dict[str, int]) == {"type": "object"}


def test_schema_genuine_union_is_unconstrained():
    assert _type_to_schema(int | str) == {}  # not Optional — a real union


def test_schema_unknown_type_is_unconstrained():
    class Custom:
        pass

    assert _type_to_schema(Custom) == {}


def test_stringify_variants():
    assert _stringify("plain") == "plain"
    assert _stringify(None) == ""
    assert _stringify(5) == "5"
    assert _stringify({"a": 1}) == '{"a": 1}'

    class NotJson:
        def __repr__(self):
            return "<obj>"

    assert _stringify(NotJson()) == "<obj>"


# --- guardrails ---------------------------------------------------------------


def test_require_pattern():
    guard = require_pattern(r"\bhello\b")
    guard("well hello there")  # passes
    from glia.errors import GuardrailTripped

    with pytest.raises(GuardrailTripped):
        guard("goodbye")


def test_no_secrets_allows_clean_text():
    no_secrets()("nothing sensitive at all")  # does not raise


def test_no_secrets_flags_aws_key():
    from glia.errors import GuardrailTripped

    with pytest.raises(GuardrailTripped):
        no_secrets()("key AKIAIOSFODNN7EXAMPLE here")


def test_text_block_only_render():
    # A message whose only content is text renders as 'role: text'.
    assert _render([Message("user", [Text("plain")])]) == "user: plain"
