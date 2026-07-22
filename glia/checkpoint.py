"""Durable execution: a run is just a JSON document you can save and reload.

Because a :class:`~glia.trajectory.Trajectory` is plain serialisable data,
"checkpoint and resume" needs no special machinery — write it to disk, and
later pass it back to ``agent.run(trajectory=...)`` to pick up exactly where you
left off. The :func:`checkpointer` helper wires that into the event stream so a
crash mid-run loses at most one step.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from .agent import Hook
from .trajectory import Event, Trajectory

# Events after which state has meaningfully advanced and is worth persisting.
_DEFAULT_TRIGGERS = ("model_response", "tool_returned", "compacted", "run_finished")


def dumps(trajectory: Trajectory, *, indent: int | None = 2) -> str:
    return json.dumps(trajectory.to_dict(), indent=indent, ensure_ascii=False)


def loads(data: str) -> Trajectory:
    return Trajectory.from_dict(json.loads(data))


def save(trajectory: Trajectory, path: str | Path) -> None:
    Path(path).write_text(dumps(trajectory), encoding="utf-8")


def load(path: str | Path) -> Trajectory:
    return loads(Path(path).read_text(encoding="utf-8"))


def checkpointer(
    trajectory: Trajectory,
    path: str | Path,
    *,
    on: Iterable[str] = _DEFAULT_TRIGGERS,
) -> Hook:
    """Return a hook that snapshots ``trajectory`` to ``path`` after each
    triggering event. Bind it to the same trajectory you pass to ``run``::

        traj = Trajectory.new()
        agent = Agent(llm, hooks=[checkpointer(traj, "run.json")])
        await agent.run(prompt, trajectory=traj)   # resumable from run.json
    """
    triggers = set(on)
    destination = Path(path)

    def hook(event: Event) -> None:
        if event.kind in triggers:
            save(trajectory, destination)

    return hook
