"""The circle of fifths: key relationships, signatures, and related keys.

A deterministic reference the LLM uses to choose keys, find closely related
keys for modulations or contrasting sections (bridge, middle eight), and build
fifth-motion progressions. Roots move by perfect fifths around the circle.
"""

from __future__ import annotations

from .notes import parse_notes

# Clockwise from C (each step up a perfect fifth). Primary spelling, key-signature
# accidental count (positive = sharps, negative = flats), the accidentals in
# order, the relative minor, and any enharmonic-equivalent major key.
_CIRCLE = [
    {"major": "C", "fifths": 0, "accidentals": [], "relative_minor": "A", "enharmonic": None},
    {"major": "G", "fifths": 1, "accidentals": ["F#"], "relative_minor": "E", "enharmonic": None},
    {"major": "D", "fifths": 2, "accidentals": ["F#", "C#"], "relative_minor": "B", "enharmonic": None},
    {"major": "A", "fifths": 3, "accidentals": ["F#", "C#", "G#"], "relative_minor": "F#", "enharmonic": None},
    {"major": "E", "fifths": 4, "accidentals": ["F#", "C#", "G#", "D#"], "relative_minor": "C#", "enharmonic": None},
    {"major": "B", "fifths": 5, "accidentals": ["F#", "C#", "G#", "D#", "A#"], "relative_minor": "G#", "enharmonic": "Cb"},
    {"major": "F#", "fifths": 6, "accidentals": ["F#", "C#", "G#", "D#", "A#", "E#"], "relative_minor": "D#", "enharmonic": "Gb"},
    {"major": "Db", "fifths": -5, "accidentals": ["Bb", "Eb", "Ab", "Db", "Gb"], "relative_minor": "Bb", "enharmonic": "C#"},
    {"major": "Ab", "fifths": -4, "accidentals": ["Bb", "Eb", "Ab", "Db"], "relative_minor": "F", "enharmonic": None},
    {"major": "Eb", "fifths": -3, "accidentals": ["Bb", "Eb", "Ab"], "relative_minor": "C", "enharmonic": None},
    {"major": "Bb", "fifths": -2, "accidentals": ["Bb", "Eb"], "relative_minor": "G", "enharmonic": None},
    {"major": "F", "fifths": -1, "accidentals": ["Bb"], "relative_minor": "D", "enharmonic": None},
]

_PC_TO_POSITION = {parse_notes(e["major"])[0].pitch_class: i for i, e in enumerate(_CIRCLE)}


def _entry(position: int) -> dict:
    return _CIRCLE[position % 12]


def circle_of_fifths(root: str | None = None) -> dict:
    """Return the circle of fifths; with a `root`, focus on that key and its neighbours.

    Without `root`: the twelve positions, each with its major key, relative
    minor, key-signature (sharp/flat count and the accidentals), and any
    enharmonic spelling. With `root` (a key name like 'C', 'Bb', 'F#'): adds a
    `focus` with the dominant (clockwise, +1 fifth) and subdominant
    (counterclockwise, -1 fifth) keys, the relative and parallel minors, and the
    closely related keys (those within one accidental plus their relative
    minors) — the natural targets for a modulation or a contrasting section.
    """
    result = {
        "circle": _CIRCLE,
        "order_clockwise": [e["major"] for e in _CIRCLE],
        "note": "Move clockwise = up a perfect 5th (sharper); counter-clockwise = down a 5th (flatter).",
    }
    if root is None:
        return result

    root_note = parse_notes(root)[0]
    pos = _PC_TO_POSITION.get(root_note.pitch_class)
    if pos is None:
        raise ValueError(f"{root!r} is not a standard key centre on the circle of fifths")
    here = _entry(pos)
    dominant = _entry(pos + 1)
    subdominant = _entry(pos - 1)
    closely_related = [
        {"key": here["relative_minor"] + " minor", "relation": "relative minor (vi)"},
        {"key": dominant["major"] + " major", "relation": "dominant (V)"},
        {"key": dominant["relative_minor"] + " minor", "relation": "iii (relative minor of V)"},
        {"key": subdominant["major"] + " major", "relation": "subdominant (IV)"},
        {"key": subdominant["relative_minor"] + " minor", "relation": "ii (relative minor of IV)"},
    ]
    result["focus"] = {
        "key": here["major"] + " major",
        "fifths": here["fifths"],
        "accidentals": here["accidentals"],
        "relative_minor": here["relative_minor"] + " minor",
        "parallel_minor": here["major"] + " minor",
        "dominant": dominant["major"],
        "subdominant": subdominant["major"],
        "closely_related_keys": closely_related,
    }
    return result
