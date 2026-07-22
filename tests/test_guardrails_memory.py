from __future__ import annotations

from conftest import run

from glia import Agent, SummarizingCompactor, TrimmingCompactor
from glia.errors import GuardrailTripped
from glia.guardrails import block_pattern, max_length, no_secrets
from glia.providers import EchoLLM
from glia.trajectory import Trajectory
from glia.types import assistant, user


def test_input_guardrail_blocks_before_model_call():
    llm = EchoLLM(["should never run"])
    agent = Agent(llm, input_guardrails=[max_length(5)])
    import pytest

    with pytest.raises(GuardrailTripped):
        run(agent.run("this is far too long"))
    assert llm.calls == []  # model was never called


def test_output_guardrail_blocks_answer():
    agent = Agent(EchoLLM(["here is a forbidden word"]), output_guardrails=[block_pattern("forbidden")])
    import pytest

    with pytest.raises(GuardrailTripped):
        run(agent.run("hi"))


def test_no_secrets_guardrail():
    guard = no_secrets()
    import pytest

    with pytest.raises(GuardrailTripped):
        guard("my key is sk-ant-abcdefghij0123456789")
    guard("nothing sensitive here")  # passes


def _long_trajectory(n: int) -> Trajectory:
    traj = Trajectory.new(system="s")
    for i in range(n):
        traj.add_message(user(f"q{i}"))
        traj.add_message(assistant(f"a{i}"))
    return traj


def test_trimming_compactor_drops_oldest():
    traj = _long_trajectory(30)  # 60 messages
    compactor = TrimmingCompactor(max_messages=40, keep_last=20)
    assert compactor.should_compact(traj) is True
    summary = run(compactor.compact(traj, EchoLLM()))
    assert "trimmed" in summary
    assert len(traj.messages) == 20
    # Most recent turns are kept verbatim.
    assert traj.messages[-1].text() == "a29"


def test_summarizing_compactor_replaces_old_turns_with_a_note():
    traj = _long_trajectory(30)
    compactor = SummarizingCompactor(max_messages=40, keep_last=10)
    summary = run(compactor.compact(traj, EchoLLM(["SUMMARY OF EARLIER TURNS"])))
    assert summary == "SUMMARY OF EARLIER TURNS"
    # First message is now the compacted note; recent turns remain.
    assert "compacted" in traj.messages[0].text().lower()
    assert len(traj.messages) == 11  # 1 note + keep_last
