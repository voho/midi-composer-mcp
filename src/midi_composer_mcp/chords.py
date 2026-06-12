"""Chord database: chord types with intervals, generation and matching."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .notes import Note, parse_note, parse_notes, spell_pitch_class, spelling_for_pcs, transpose


@dataclass(frozen=True)
class ChordType:
    name: str
    symbol: str  # canonical suffix appended to the root, "" for major
    degrees: tuple[tuple[str, int], ...]  # (degree label, semitones from root)
    aliases: tuple[str, ...] = ()

    @property
    def intervals(self) -> tuple[int, ...]:
        return tuple(s for _, s in self.degrees)

    @property
    def pitch_classes(self) -> frozenset[int]:
        return frozenset(s % 12 for s in self.intervals)


CHORDS: dict[str, ChordType] = {
    c.name: c
    for c in [
        ChordType("major", "", (("1", 0), ("3", 4), ("5", 7)), ("maj", "M")),
        ChordType("minor", "m", (("1", 0), ("b3", 3), ("5", 7)), ("min", "-")),
        ChordType("diminished", "dim", (("1", 0), ("b3", 3), ("b5", 6)), ("°", "o")),
        ChordType("augmented", "aug", (("1", 0), ("3", 4), ("#5", 8)), ("+",)),
        ChordType("power chord", "5", (("1", 0), ("5", 7)), ("power",)),
        ChordType("suspended 2", "sus2", (("1", 0), ("2", 2), ("5", 7))),
        ChordType("suspended 4", "sus4", (("1", 0), ("4", 5), ("5", 7)), ("sus",)),
        ChordType("major 6", "6", (("1", 0), ("3", 4), ("5", 7), ("6", 9)), ("maj6", "M6")),
        ChordType("minor 6", "m6", (("1", 0), ("b3", 3), ("5", 7), ("6", 9)), ("min6",)),
        ChordType("dominant 7", "7", (("1", 0), ("3", 4), ("5", 7), ("b7", 10)), ("dom7",)),
        ChordType("major 7", "maj7", (("1", 0), ("3", 4), ("5", 7), ("7", 11)), ("M7", "ma7", "Δ", "Δ7")),
        ChordType("minor 7", "m7", (("1", 0), ("b3", 3), ("5", 7), ("b7", 10)), ("min7", "-7")),
        ChordType("minor major 7", "mMaj7", (("1", 0), ("b3", 3), ("5", 7), ("7", 11)), ("mM7", "minmaj7", "m(maj7)")),
        ChordType("diminished 7", "dim7", (("1", 0), ("b3", 3), ("b5", 6), ("bb7", 9)), ("°7", "o7")),
        ChordType("half-diminished", "m7b5", (("1", 0), ("b3", 3), ("b5", 6), ("b7", 10)), ("ø", "ø7", "min7b5", "m7(b5)", "half-diminished 7")),
        ChordType("augmented 7", "7#5", (("1", 0), ("3", 4), ("#5", 8), ("b7", 10)), ("aug7", "+7")),
        ChordType("augmented major 7", "maj7#5", (("1", 0), ("3", 4), ("#5", 8), ("7", 11)), ("augmaj7", "+M7")),
        ChordType("dominant 7 flat 5", "7b5", (("1", 0), ("3", 4), ("b5", 6), ("b7", 10))),
        ChordType("dominant 7 sus 4", "7sus4", (("1", 0), ("4", 5), ("5", 7), ("b7", 10))),
        ChordType("add 9", "add9", (("1", 0), ("3", 4), ("5", 7), ("9", 14))),
        ChordType("minor add 9", "madd9", (("1", 0), ("b3", 3), ("5", 7), ("9", 14)), ("m(add9)",)),
        ChordType("six nine", "6/9", (("1", 0), ("3", 4), ("5", 7), ("6", 9), ("9", 14)), ("69", "6add9")),
        ChordType("dominant 9", "9", (("1", 0), ("3", 4), ("5", 7), ("b7", 10), ("9", 14))),
        ChordType("major 9", "maj9", (("1", 0), ("3", 4), ("5", 7), ("7", 11), ("9", 14)), ("M9",)),
        ChordType("minor 9", "m9", (("1", 0), ("b3", 3), ("5", 7), ("b7", 10), ("9", 14)), ("min9",)),
        ChordType("dominant 7 flat 9", "7b9", (("1", 0), ("3", 4), ("5", 7), ("b7", 10), ("b9", 13))),
        ChordType("dominant 7 sharp 9", "7#9", (("1", 0), ("3", 4), ("5", 7), ("b7", 10), ("#9", 15)), ("hendrix",)),
        ChordType("dominant 11", "11", (("1", 0), ("3", 4), ("5", 7), ("b7", 10), ("9", 14), ("11", 17))),
        ChordType("minor 11", "m11", (("1", 0), ("b3", 3), ("5", 7), ("b7", 10), ("9", 14), ("11", 17)), ("min11",)),
        ChordType("dominant 13", "13", (("1", 0), ("3", 4), ("5", 7), ("b7", 10), ("9", 14), ("13", 21))),
    ]
}


def _normalize(name: str) -> str:
    return re.sub(r"[\s_\-]+", "", name.strip().lower())


# Exact (case-sensitive) lookup of symbols/aliases, e.g. "m7" vs "M7".
_CHORD_EXACT: dict[str, ChordType] = {}
# Case-insensitive fallback, excluding keys that become ambiguous when folded.
_CHORD_FOLDED: dict[str, ChordType] = {}
_folded_collisions: set[str] = set()
for _chord in CHORDS.values():
    for _key in (_chord.symbol, _chord.name, *_chord.aliases):
        _CHORD_EXACT.setdefault(_key, _chord)
        _folded = _normalize(_key)
        if _folded in _CHORD_FOLDED and _CHORD_FOLDED[_folded] is not _chord:
            _folded_collisions.add(_folded)
        else:
            _CHORD_FOLDED.setdefault(_folded, _chord)
for _key in _folded_collisions:
    _CHORD_FOLDED.pop(_key, None)


def resolve_chord_type(chord_type: str) -> ChordType:
    if not isinstance(chord_type, str):
        raise ValueError(f"Chord type must be a string, got {type(chord_type).__name__}")
    text = chord_type.strip()
    found = _CHORD_EXACT.get(text) or _CHORD_FOLDED.get(_normalize(text))
    if found is None:
        known = ", ".join(f"{c.name} ({c.symbol or 'no suffix'})" for c in CHORDS.values())
        raise ValueError(f"Unknown chord type: {chord_type!r}. Known types: {known}")
    return found


def chord_notes(chord: ChordType, root: Note) -> list[Note]:
    """Spelled chord tones from `root` (octave-aware when the root has one)."""
    return [
        transpose(root, semis, int(label.lstrip("#b")) - 1)
        for label, semis in chord.degrees
    ]


_SYMBOL_RE = re.compile(r"^([A-Ga-g][#♯b♭]*)(.*)$")


def parse_chord_symbol(symbol: str) -> tuple[Note, ChordType, Note | None]:
    """Parse a chord symbol like ``"Cmaj7"``, ``"f#m"``, ``"C4m7"`` or ``"C/E"``.

    Returns (root, chord type, optional slash-bass note). An octave digit may
    follow the root when it cannot be confused with a chord suffix: ``"C4"``
    is C major rooted at octave 4, while ``"C7"`` stays a dominant 7 chord
    (write ``"C47"`` for a dominant 7 on C4).
    """
    if not isinstance(symbol, str):
        raise ValueError(f"Chord symbol must be a string, got {type(symbol).__name__}")
    m = _SYMBOL_RE.match(symbol.strip())
    if not m:
        raise ValueError(
            f"Invalid chord symbol: {symbol!r}. Expected a root note plus an optional"
            f" suffix, e.g. 'C', 'Am', 'F#m7', 'Bb7', 'C/E'."
        )
    root_text, rest = m.groups()
    rest = rest.strip()

    bass: Note | None = None

    def _try(suffix: str, octave: str | None) -> tuple[Note, ChordType] | None:
        chord = _CHORD_EXACT.get(suffix) or _CHORD_FOLDED.get(_normalize(suffix)) if suffix else CHORDS["major"]
        if chord is None:
            return None
        return parse_note(root_text + (octave or "")), chord

    # 1) whole remainder as a suffix ("C7" -> dominant 7, "C6/9" -> six nine)
    parsed = _try(rest, None)
    # 2) a leading digit as an octave, remainder as suffix ("C4", "C4m7", "C47")
    if parsed is None:
        m2 = re.match(r"^(-?\d)(.*)$", rest)
        if m2:
            parsed = _try(m2.group(2).strip(), m2.group(1))
    # 3) slash bass ("C/E", "Cm7/G", "C4maj7/G3")
    if parsed is None and "/" in rest:
        head, _, bass_text = rest.rpartition("/")
        try:
            bass = parse_note(bass_text)
        except ValueError:
            bass = None
        if bass is not None:
            inner_root, inner_chord, _ = parse_chord_symbol(root_text + head)
            return inner_root, inner_chord, bass
    if parsed is None:
        known = ", ".join(sorted({c.symbol for c in CHORDS.values() if c.symbol}))
        raise ValueError(
            f"Unknown chord suffix in {symbol!r}. Known suffixes: (none = major), {known}"
        )
    root, chord = parsed
    return root, chord, bass


def chord_info(chord_type: str, root: str | None = None) -> dict:
    """Describe a chord type; with a root, also generate its notes."""
    chord = resolve_chord_type(chord_type)
    result: dict = {
        "chord_type": chord.name,
        "symbol_suffix": chord.symbol,
        "aliases": list(chord.aliases),
        "intervals": list(chord.intervals),
        "degrees": [label for label, _ in chord.degrees],
        "note_count": len(chord.degrees),
    }
    if root is not None:
        root_note = parse_notes(root)[0]
        notes = chord_notes(chord, root_note)
        result["root"] = root_note.name
        result["symbol"] = f"{root_note.pitch_class_name}{chord.symbol}"
        result["name"] = f"{root_note.pitch_class_name} {chord.name}"
        result["notes"] = [n.name for n in notes]
        if root_note.octave is not None:
            result["midi"] = [n.midi for n in notes]
    return result


def list_chords() -> dict:
    return {
        "count": len(CHORDS),
        "chords": [
            {
                "chord_type": c.name,
                "symbol_suffix": c.symbol,
                "example": f"C{c.symbol}",
                "aliases": list(c.aliases),
                "intervals": list(c.intervals),
                "degrees": [label for label, _ in c.degrees],
                "note_count": len(c.degrees),
            }
            for c in CHORDS.values()
        ],
    }


def identify_chord_quality(relative_pcs: frozenset[int]) -> ChordType | None:
    """Find the chord type whose pitch-class set (root = 0) matches exactly."""
    for chord in CHORDS.values():
        if chord.pitch_classes == relative_pcs:
            return chord
    return None


_INVERSION_NAMES = ("root position", "first inversion", "second inversion",
                    "third inversion", "fourth inversion", "fifth inversion")


def match_chords(notes: str | list, include_partial: bool = True, limit: int = 20) -> dict:
    """Find chords matching the given notes (octaves are ignored).

    "exact": the chord has exactly the input's pitch classes (inversions are
    reported with slash notation, the first input note is taken as the bass).
    "partial": every input note belongs to the chord, which has more notes;
    the missing notes are listed.
    """
    parsed = parse_notes(notes)
    pcs = frozenset(n.pitch_class for n in parsed)
    spelling = spelling_for_pcs(parsed)
    bass = parsed[0]

    exact: list[dict] = []
    partial: list[dict] = []
    for chord in CHORDS.values():
        for root_pc in range(12):
            chord_pcs = frozenset((root_pc + s) % 12 for s in chord.intervals)
            is_exact = chord_pcs == pcs
            if not is_exact and not (include_partial and pcs < chord_pcs):
                continue
            root = spelling.get(root_pc) or spell_pitch_class(root_pc)
            tones = chord_notes(chord, root)
            entry = {
                "match": "exact" if is_exact else "partial",
                "root": root.name,
                "chord_type": chord.name,
                "symbol": f"{root.name}{chord.symbol}",
                "name": f"{root.name} {chord.name}",
                "notes": [n.name for n in tones],
            }
            if is_exact:
                if bass.pitch_class != root_pc:
                    position = next(
                        i for i, t in enumerate(tones) if t.pitch_class == bass.pitch_class
                    )
                    entry["bass"] = bass.pitch_class_name
                    entry["symbol"] += f"/{bass.pitch_class_name}"
                    if position < len(_INVERSION_NAMES):
                        entry["inversion"] = _INVERSION_NAMES[position]
                entry["_sort"] = (bass.pitch_class != root_pc, len(chord.degrees), chord.name, root.name)
                exact.append(entry)
            else:
                missing = [t.name for t in tones if t.pitch_class not in pcs]
                entry["missing_notes"] = missing
                entry["_sort"] = (len(missing), len(chord.degrees), chord.name, root.name)
                partial.append(entry)

    exact.sort(key=lambda e: e.pop("_sort"))
    partial.sort(key=lambda e: e.pop("_sort"))
    matches = exact + partial
    return {
        "input_notes": [n.name for n in parsed],
        "exact_count": len(exact),
        "partial_count": len(partial),
        "matches": matches[: max(1, limit)],
    }
