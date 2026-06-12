"""Scale database: scale types with intervals, generation and matching."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .notes import Note, parse_notes, spell_pitch_class, spelling_for_pcs, transpose

MAJOR_DEGREES = (0, 2, 4, 5, 7, 9, 11)

# Default degree label for each chromatic interval (used for non-heptatonic
# scales; heptatonic scales get sequential degree numbers 1-7).
_CHROMATIC_LABELS = {
    0: "1", 1: "b2", 2: "2", 3: "b3", 4: "3", 5: "4",
    6: "b5", 7: "5", 8: "b6", 9: "6", 10: "b7", 11: "7",
}


@dataclass(frozen=True)
class ScaleType:
    name: str
    intervals: tuple[int, ...]
    aliases: tuple[str, ...] = ()
    matchable: bool = True  # chromatic matches everything, so it is excluded


SCALES: dict[str, ScaleType] = {
    s.name: s
    for s in [
        ScaleType("major", (0, 2, 4, 5, 7, 9, 11), ("ionian", "maj")),
        ScaleType("natural minor", (0, 2, 3, 5, 7, 8, 10), ("minor", "min", "aeolian")),
        ScaleType("harmonic minor", (0, 2, 3, 5, 7, 8, 11)),
        ScaleType("melodic minor", (0, 2, 3, 5, 7, 9, 11), ("jazz minor", "melodic minor ascending")),
        ScaleType("dorian", (0, 2, 3, 5, 7, 9, 10)),
        ScaleType("phrygian", (0, 1, 3, 5, 7, 8, 10)),
        ScaleType("lydian", (0, 2, 4, 6, 7, 9, 11)),
        ScaleType("mixolydian", (0, 2, 4, 5, 7, 9, 10), ("dominant scale",)),
        ScaleType("locrian", (0, 1, 3, 5, 6, 8, 10)),
        ScaleType("major pentatonic", (0, 2, 4, 7, 9), ("pentatonic", "pentatonic major")),
        ScaleType("minor pentatonic", (0, 3, 5, 7, 10), ("pentatonic minor",)),
        ScaleType("blues", (0, 3, 5, 6, 7, 10), ("minor blues", "blues minor")),
        ScaleType("major blues", (0, 2, 3, 4, 7, 9), ("blues major",)),
        ScaleType("whole tone", (0, 2, 4, 6, 8, 10), ("wholetone",)),
        ScaleType("diminished whole-half", (0, 2, 3, 5, 6, 8, 9, 11), ("diminished", "octatonic", "whole-half diminished")),
        ScaleType("diminished half-whole", (0, 1, 3, 4, 6, 7, 9, 10), ("dominant diminished", "half-whole diminished")),
        ScaleType("altered", (0, 1, 3, 4, 6, 8, 10), ("super locrian", "altered dominant")),
        ScaleType("phrygian dominant", (0, 1, 4, 5, 7, 8, 10), ("spanish phrygian", "freygish")),
        ScaleType("double harmonic", (0, 1, 4, 5, 7, 8, 11), ("byzantine", "arabic")),
        ScaleType("hungarian minor", (0, 2, 3, 6, 7, 8, 11), ("gypsy minor",)),
        ScaleType("harmonic major", (0, 2, 4, 5, 7, 8, 11)),
        ScaleType("bebop dominant", (0, 2, 4, 5, 7, 9, 10, 11), ("bebop",)),
        ScaleType("chromatic", tuple(range(12)), (), False),
    ]
}


def _normalize(name: str) -> str:
    return re.sub(r"[\s_\-]+", "", name.strip().lower())


_SCALE_LOOKUP: dict[str, ScaleType] = {}
for _scale in SCALES.values():
    for _key in (_scale.name, *_scale.aliases):
        _SCALE_LOOKUP[_normalize(_key)] = _scale


def resolve_scale_type(scale_type: str) -> ScaleType:
    if not isinstance(scale_type, str):
        raise ValueError(f"Scale type must be a string, got {type(scale_type).__name__}")
    found = _SCALE_LOOKUP.get(_normalize(scale_type))
    if found is None:
        raise ValueError(
            f"Unknown scale type: {scale_type!r}. Known types: "
            + ", ".join(sorted(SCALES))
        )
    return found


def degree_labels(intervals: tuple[int, ...]) -> list[str]:
    """Degree labels such as ['1', '2', 'b3', ...] for a scale's intervals."""
    if len(intervals) == 7:
        labels = []
        for i, semis in enumerate(intervals):
            diff = semis - MAJOR_DEGREES[i]
            prefix = "#" * diff if diff > 0 else "b" * -diff
            labels.append(f"{prefix}{i + 1}")
        return labels
    return [_CHROMATIC_LABELS.get(s % 12, str(s)) for s in intervals]


def _label_digit(label: str) -> int:
    return int(label.lstrip("#b"))


def scale_notes(scale: ScaleType, root: Note) -> list[Note]:
    """Spelled notes of `scale` from `root`, including the octave root on top.

    Spelling follows the degree labels (a b3 lands on the third letter), so
    F major yields Bb rather than A#. If the root carries an octave, every
    note does too.
    """
    labels = degree_labels(scale.intervals)
    notes = [
        transpose(root, semis, _label_digit(label) - 1)
        for semis, label in zip(scale.intervals, labels)
    ]
    top_octave = None if root.octave is None else root.octave + 1
    if top_octave is not None and root.midi + 12 > 127:
        raise ValueError(f"Scale from {root.name} exceeds the MIDI range 0-127")
    notes.append(Note(root.letter, root.accidental, top_octave))
    return notes


def scale_info(scale_type: str, root: str | None = None) -> dict:
    """Describe a scale type; with a root, also generate its notes."""
    scale = resolve_scale_type(scale_type)
    result: dict = {
        "scale_type": scale.name,
        "aliases": list(scale.aliases),
        "intervals": list(scale.intervals),
        "degrees": degree_labels(scale.intervals),
        "note_count": len(scale.intervals),
    }
    if root is not None:
        root_note = parse_notes(root)[0]
        notes = scale_notes(scale, root_note)
        result["root"] = root_note.name
        result["name"] = f"{root_note.pitch_class_name} {scale.name}"
        result["notes"] = [n.name for n in notes]
        if root_note.octave is not None:
            result["midi"] = [n.midi for n in notes]
    return result


def list_scales() -> dict:
    return {
        "count": len(SCALES),
        "scales": [
            {
                "scale_type": s.name,
                "aliases": list(s.aliases),
                "intervals": list(s.intervals),
                "degrees": degree_labels(s.intervals),
                "note_count": len(s.intervals),
            }
            for s in SCALES.values()
        ],
    }


def match_scales(notes: str | list, exact_only: bool = False, limit: int = 20) -> dict:
    """Find scales containing all the given notes (octaves are ignored).

    A match is "exact" when the input covers every note of the scale.
    Results are sorted: exact matches first, then smaller (tighter) scales,
    then scales rooted on the first input note.
    """
    parsed = parse_notes(notes)
    pcs = {n.pitch_class for n in parsed}
    spelling = spelling_for_pcs(parsed)
    first_pc = parsed[0].pitch_class

    matches = []
    for scale in SCALES.values():
        if not scale.matchable:
            continue
        for root_pc in range(12):
            scale_pcs = {(root_pc + i) % 12 for i in scale.intervals}
            if not pcs <= scale_pcs:
                continue
            exact = pcs == scale_pcs
            if exact_only and not exact:
                continue
            root = spelling.get(root_pc) or spell_pitch_class(root_pc)
            generated = scale_notes(scale, root)
            matches.append(
                {
                    "match": "exact" if exact else "contains",
                    "root": root.name,
                    "scale_type": scale.name,
                    "name": f"{root.name} {scale.name}",
                    "notes": [n.name for n in generated],
                    "added_notes": sorted(
                        {n.name for n in generated[:-1] if n.pitch_class not in pcs}
                    ),
                    "_sort": (
                        not exact,
                        len(scale.intervals),
                        root.pitch_class != first_pc,
                        scale.name,
                        root.name,
                    ),
                }
            )

    matches.sort(key=lambda m: m.pop("_sort"))
    return {
        "input_notes": [n.name for n in parsed],
        "match_count": len(matches),
        "matches": matches[: max(1, limit)],
    }
