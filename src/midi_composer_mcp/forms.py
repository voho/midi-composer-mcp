"""Shared helper for resolving a 'form' into an ordered list of labels.

A form expresses an order of repeatable parts — used both for song structure
(intro/verse/chorus...) and for motif grammars (A/B/C...).
"""

from __future__ import annotations

import re


def resolve_form(form, known=None) -> list[str]:
    """Turn a form into an ordered list of labels.

    Accepts a list (["intro", "verse"] or ["A", "B", "A"]), a space/comma
    separated string ("intro verse chorus" / "A B A C"), or a bare letter form
    ("AABA" -> ["A", "A", "B", "A"]). A single word is one label ("verse" stays
    "verse"); it is split into letters only when it is all uppercase letters or
    when every letter is one of the `known` labels (e.g. motif names "abab").
    """
    if isinstance(form, (list, tuple)):
        labels = [str(x).strip() for x in form if str(x).strip()]
    elif isinstance(form, str):
        text = form.strip()
        known = set(known or ())
        if re.search(r"[\s,]", text):
            labels = [t for t in re.split(r"[,\s]+", text) if t]
        elif text in known:
            labels = [text]
        elif re.fullmatch(r"[A-Z]+", text) or (known and all(ch in known for ch in text)):
            labels = list(text)  # "AABA" -> A A B A
        else:
            labels = [text] if text else []
    else:
        raise ValueError(
            "form must be a list of labels or a string like 'verse chorus' or 'AABA'"
        )
    if not labels:
        raise ValueError("form is empty")
    return labels
