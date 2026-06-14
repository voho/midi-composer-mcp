"""Diatonic chords of a scale and degree-sequence resolution.

These are the atomic building blocks for chord progressions: the LLM decides
*which* degrees to use (the creative part); these functions only report what
chords live on each scale degree and resolve a chosen degree sequence into
concrete chords.
"""

from __future__ import annotations

import re

from .chords import ChordType, identify_chord_quality
from .notes import Note, parse_notes, transpose
from .scales import MAJOR_DEGREES, degree_labels, resolve_scale_type, scale_notes

_ROMAN_BASE = ("I", "II", "III", "IV", "V", "VI", "VII")
_ROMAN_VALUES = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7}

_DEGREE_NAMES = ("tonic", "supertonic", "mediant", "subdominant", "dominant", "submediant")
_HARMONIC_FUNCTIONS = ("tonic", "subdominant", "tonic", "subdominant", "dominant", "tonic", "dominant")

# How a chord quality is written after a roman numeral.
_ROMAN_QUALITY = {
    "major": "", "minor": "", "diminished": "°", "augmented": "+",
    "dominant 7": "7", "major 7": "Δ7", "minor 7": "7",
    "half-diminished": "ø7", "diminished 7": "°7",
    "minor major 7": "Δ7", "augmented major 7": "+Δ7",
}


def _roman_numeral(index: int, intervals: tuple[int, ...], quality: ChordType | None,
                   relative_pcs: frozenset[int]) -> str:
    diff = intervals[index] - MAJOR_DEGREES[index]
    prefix = "#" * diff if diff > 0 else "b" * -diff
    numeral = _ROMAN_BASE[index]
    minor_third = 3 in relative_pcs
    if minor_third:
        numeral = numeral.lower()
    suffix = _ROMAN_QUALITY.get(quality.name, "?") if quality else "?"
    return f"{prefix}{numeral}{suffix}"


def _stacked_chord(scale_intervals: tuple[int, ...], labels: list[str],
                   index: int, tone_count: int, root: Note) -> dict:
    """Build the chord on scale degree `index` by stacking scale thirds."""
    n = len(scale_intervals)
    tone_indices = [index + 2 * k for k in range(tone_count)]
    semitones = [scale_intervals[j % n] + 12 * (j // n) for j in tone_indices]
    base = semitones[0]

    tones = [
        transpose(root, semis, int(labels[j % n].lstrip("#b")) - 1 + 7 * (j // n))
        for semis, j in zip(semitones, tone_indices)
    ]
    relative_pcs = frozenset((s - base) % 12 for s in semitones)
    quality = identify_chord_quality(relative_pcs)
    chord_root = tones[0]

    entry: dict = {
        "degree": index + 1,
        "root": chord_root.name,
        "chord_type": quality.name if quality else "unknown",
        "symbol": f"{chord_root.pitch_class_name}{quality.symbol}" if quality else f"{chord_root.pitch_class_name}?",
        "notes": [t.name for t in tones],
        "intervals_from_chord_root": [s - base for s in semitones],
    }
    if chord_root.octave is not None:
        entry["midi"] = [t.midi for t in tones]
    if n == 7:
        entry["roman"] = _roman_numeral(index, scale_intervals, quality, relative_pcs)
        entry["degree_name"] = (
            _DEGREE_NAMES[index]
            if index < 6
            else ("leading tone" if scale_intervals[6] == 11 else "subtonic")
        )
        entry["harmonic_function"] = _HARMONIC_FUNCTIONS[index]
    return entry


def diatonic_chords(root: str, scale_type: str, sevenths: bool = False) -> dict:
    """The chord that lives on each degree of a scale (triads or sevenths).

    For 7-note scales each chord also gets a roman numeral, its classical
    degree name and harmonic function (tonic / subdominant / dominant).
    """
    scale = resolve_scale_type(scale_type)
    root_note = parse_notes(root)[0]
    labels = degree_labels(scale.intervals)
    degrees = scale_notes(scale, root_note)[:-1]
    tone_count = 4 if sevenths else 3

    chords = [
        _stacked_chord(scale.intervals, labels, i, tone_count, root_note)
        for i in range(len(scale.intervals))
    ]
    return {
        "root": root_note.name,
        "scale_type": scale.name,
        "scale_notes": [n.name for n in degrees],
        "sevenths": sevenths,
        "chords": chords,
    }


def _parse_degree(token: int | str, count: int) -> int:
    """Parse a scale degree given as 1-based int, '5', 'V', 'vi' or 'vii°'."""
    if isinstance(token, bool):
        raise ValueError(f"Invalid scale degree: {token!r}")
    if isinstance(token, int):
        degree = token
    elif isinstance(token, str):
        text = token.strip()
        if re.match(r"^[#b]", text):
            raise ValueError(
                f"Chromatic alterations are not supported in degree sequences: {token!r}."
                f" Degrees are positions in the chosen scale (1-{count})."
            )
        m = re.match(r"^(\d+|[ivIV]+)[°ø+Δ7oO]*$", text)
        core = m.group(1) if m else ""
        if core.isdigit():
            degree = int(core)
        else:
            degree = _ROMAN_VALUES.get(core.lower(), 0)
            if degree == 0:
                raise ValueError(
                    f"Invalid scale degree: {token!r}. Use 1-{count} or roman numerals I-VII."
                )
    else:
        raise ValueError(f"Invalid scale degree: {token!r} (use an integer or a roman numeral)")
    if not 1 <= degree <= count:
        raise ValueError(f"Scale degree {degree} out of range 1-{count} for this scale")
    return degree


def degrees_to_chords(root: str, scale_type: str, degrees, sevenths: bool = False) -> dict:
    """Resolve a degree sequence (e.g. [1, 5, 6, 4] or 'I V vi IV') to chords.

    The caller chooses the sequence; this only maps each degree to the chord
    that the scale defines there.
    """
    scale = resolve_scale_type(scale_type)
    if isinstance(degrees, (int, str)):
        degrees = [t for t in re.split(r"[,\s\-]+", str(degrees).strip()) if t]
    if not isinstance(degrees, (list, tuple)) or not degrees:
        raise ValueError("degrees must be a non-empty list like [1, 5, 6, 4] or 'I V vi IV'")
    indices = [_parse_degree(t, len(scale.intervals)) for t in degrees]

    base = diatonic_chords(root, scale_type, sevenths)
    by_degree = {c["degree"]: c for c in base["chords"]}
    progression = [dict(by_degree[d]) for d in indices]
    return {
        "root": base["root"],
        "scale_type": base["scale_type"],
        "degrees": indices,
        "sevenths": sevenths,
        "chords": progression,
        "symbols": [c["symbol"] for c in progression],
    }
