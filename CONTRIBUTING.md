# Contributing to glia

Thanks for your interest. glia has one guiding principle, and contributions are
judged against it:

> **No hidden control flow.** A new feature must be inspectable — if you can't
> see it happen in the event stream or read it in one file, it doesn't fit.

## Setup

```bash
git clone https://github.com/DenisDrobyshev/glia
cd glia
pip install -e ".[anthropic,dev]"
```

## Checks (all must pass)

```bash
python -m pytest        # tests — run fully offline, no API key
python -m ruff check .  # lint
python -m mypy glia     # types (best-effort)
```

The whole test suite runs against the offline `EchoLLM`, so it's fast and
deterministic. New behaviour should come with a test that uses `EchoLLM` — if a
feature can't be tested offline, that's usually a design smell worth discussing
first.

## Guidelines

- **Keep the core dependency-free.** New runtime deps go behind an optional
  extra (like `[anthropic]`), imported lazily.
- **Prefer a primitive over a feature.** A small composable piece (a new
  `Compactor`, a new guardrail, a new provider) beats a bespoke flag on `Agent`.
- **Match the surrounding style.** Readable code with comments that explain
  _why_, not _what_. The codebase is meant to be read.
- **New provider?** Implement the `LLM` protocol in ~40 lines; see
  `glia/providers/echo.py` for the minimal shape.

## Opening a PR

1. Branch from `master`.
2. Add/adjust tests.
3. Make sure `pytest` and `ruff` are green.
4. Describe the change and how it stays inside the glass box.
