"""Structured output: get a typed object back, not a string to parse.

Run: python examples/03_structured_output.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from _common import make_llm

from glia import generate_structured
from glia.providers import call


@dataclass
class Contact:
    name: str
    email: str
    wants_demo: bool


async def main() -> None:
    llm = make_llm([call("respond", {"name": "Ada Lovelace", "email": "ada@analytical.engine", "wants_demo": True})])

    contact = await generate_structured(
        llm,
        "Extract the contact: Ada Lovelace (ada@analytical.engine) asked for a demo.",
        Contact,
    )
    print(type(contact).__name__, "->", contact)
    assert isinstance(contact, Contact)
    print("wants_demo is a real bool:", contact.wants_demo is True)


if __name__ == "__main__":
    asyncio.run(main())
