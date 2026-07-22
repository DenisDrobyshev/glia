from __future__ import annotations

from conftest import run

from glia.approval import (
    ApprovalRequest,
    Decision,
    allow_only,
    approve_all,
    deny,
    deny_all,
    resolve,
)

REQ = ApprovalRequest(step=1, tool_use_id="t1", name="danger", arguments={"x": 1})


def test_resolve_normalises_bool_true():
    assert run(resolve(lambda r: True, REQ)) == Decision(True, "")


def test_resolve_normalises_bool_false():
    d = run(resolve(lambda r: False, REQ))
    assert d.allow is False and d.reason == "denied by policy"


def test_resolve_passes_through_decision():
    d = run(resolve(lambda r: Decision(False, "nope"), REQ))
    assert d == Decision(False, "nope")


def test_resolve_awaits_async_policy():
    async def policy(request):
        return Decision(True, "async ok")

    assert run(resolve(policy, REQ)) == Decision(True, "async ok")


def test_approve_all_and_deny_all():
    assert approve_all(REQ).allow is True
    assert deny_all(REQ).allow is False


def test_allow_only():
    policy = allow_only("safe")
    assert policy(ApprovalRequest(1, "i", "safe", {})).allow is True
    d = policy(ApprovalRequest(1, "i", "danger", {}))
    assert d.allow is False and "allowlist" in d.reason


def test_deny():
    policy = deny("danger")
    assert policy(ApprovalRequest(1, "i", "danger", {})).allow is False
    assert policy(ApprovalRequest(1, "i", "other", {})).allow is True
