"""The circle of fifths: key relationships, signatures, and related keys.

A deterministic reference the LLM uses to choose keys, find closely related
keys for modulations or contrasting sections (bridge, middle eight), and build
fifth-motion progressions. Roots move by perfect fifths around the circle.
"""

from __future__ import annotations

from .notes import Note, parse_notes, spell_pitch_class, transpose

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

_SHARP_ORDER = ["F#", "C#", "G#", "D#", "A#", "E#", "B#"]
_FLAT_ORDER = ["Bb", "Eb", "Ab", "Db", "Gb", "Cb", "Fb"]
_LETTER_FIFTHS = {"F": -1, "C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5}


def _fifths(note: Note) -> int:
    """Position of a spelled major key on the line of fifths (sharps +, flats -)."""
    return _LETTER_FIFTHS[note.letter] + 7 * note.accidental


def _signature(fifths: int) -> list[str]:
    return _SHARP_ORDER[:fifths] if fifths >= 0 else _FLAT_ORDER[:-fifths]


def _practical(note: Note, offset: int = 0) -> Note:
    """A key spelling that needs at most 7 accidentals (D# major -> Eb major).

    `offset` shifts the signature count (-3 for a minor key: its signature is its
    relative major's).
    """
    f = _fifths(note) + offset
    if -7 <= f <= 7:
        return note
    return spell_pitch_class(note.pitch_class, prefer_flats=f > 7)


def circle_of_fifths(root: str | None = None) -> dict:
    """Return the circle of fifths; with a `root`, focus on that key and its neighbours.

    Without `root`: the twelve positions, each with its major key, relative
    minor, key-signature (sharp/flat count and the accidentals), and any
    enharmonic spelling. With `root` (a key name like 'C', 'Bb', 'F#', 'Gb'):
    adds a `focus` spelled from that root — Gb stays Gb major with six flats,
    and F#'s dominant is C#, not Db — with the dominant (clockwise, +1 fifth)
    and subdominant (counterclockwise, -1 fifth) keys, the relative and parallel
    minors, and the closely related keys (those within one accidental plus their
    relative minors) — the natural targets for a modulation or a contrasting
    section. A theoretical key (more than 7 accidentals, e.g. D# major or Db
    minor) is given by its practical enharmonic (Eb major, C# minor).
    """
    result = {
        "circle": _CIRCLE,
        "order_clockwise": [e["major"] for e in _CIRCLE],
        "note": "Move clockwise = up a perfect 5th (sharper); counter-clockwise = down a 5th (flatter).",
    }
    if root is None:
        return result

    key = parse_notes(root)[0].without_octave()
    asked = key.name
    key = _practical(key)
    dominant = _practical(transpose(key, 7, 4))
    subdominant = _practical(transpose(key, 5, 3))

    def relative_minor(major: Note) -> Note:
        return transpose(major, 9, 5)  # a major 6th up = a minor 3rd down, spelled on its letter

    parallel = _practical(key, -3)
    closely_related = [
        {"key": relative_minor(key).name + " minor", "relation": "relative minor (vi)"},
        {"key": dominant.name + " major", "relation": "dominant (V)"},
        {"key": relative_minor(dominant).name + " minor", "relation": "iii (relative minor of V)"},
        {"key": subdominant.name + " major", "relation": "subdominant (IV)"},
        {"key": relative_minor(subdominant).name + " minor", "relation": "ii (relative minor of IV)"},
    ]
    fifths = _fifths(key)
    result["focus"] = {
        "key": key.name + " major",
        "fifths": fifths,
        "accidentals": _signature(fifths),
        "relative_minor": relative_minor(key).name + " minor",
        "parallel_minor": parallel.name + " minor",
        "dominant": dominant.name,
        "subdominant": subdominant.name,
        "closely_related_keys": closely_related,
    }
    if key.name != asked:
        result["focus"]["respelled_from"] = asked
    return result
