"""Harmony-layer rules: intervals, analysis, voice-leading and reharmonization.

Classic, fully deterministic music-theory operations. Each is a mechanical
transform or lookup the LLM can chain — name an interval, analyze a
progression into Roman numerals, voice chords smoothly, or reharmonize with
secondary dominants, tritone subs and negative harmony.
"""

from __future__ import annotations

from .chords import (
    CHORDS,
    chord_notes,
    match_chords,
    parse_chord_symbol,
    resolve_chord_type,
)
from .diatonic import _HARMONIC_FUNCTIONS, _ROMAN_QUALITY
from .midi_io import _parse_chord_list, voice_chord
from .notes import LETTERS, Note, note_from_midi, parse_note, parse_notes, spell_pitch_class, transpose
from .scales import resolve_scale_type

# ---------------------------------------------------------------- intervals

_MAJOR_REF = {1: 0, 2: 2, 3: 4, 4: 5, 5: 7, 6: 9, 7: 11, 8: 12}
_PERFECT_NUMBERS = {1, 4, 5, 8}
_ORDINAL = {1: "unison", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
            6: "sixth", 7: "seventh", 8: "octave"}


def interval_between(note_a: str, note_b: str) -> dict:
    """Name the interval from `note_a` up to `note_b` (e.g. C -> Eb = minor third).

    Returns the semitone distance, the interval number and quality (perfect/
    major/minor/augmented/diminished), and the short label (P5, m3, M7...).
    With octaves it gives the signed semitone distance too. Deterministic.
    """
    a = parse_note(note_a)
    b = parse_note(note_b)
    letter_steps = (LETTERS.index(b.letter) - LETTERS.index(a.letter)) % 7
    number = letter_steps + 1
    semitone = (b.pitch_class - a.pitch_class) % 12
    diff = semitone - _MAJOR_REF[number]
    if diff <= -6:
        diff += 12
    elif diff >= 7:
        diff -= 12

    if number in _PERFECT_NUMBERS:
        quality = {0: "perfect", 1: "augmented", -1: "diminished", 2: "doubly augmented", -2: "doubly diminished"}.get(diff)
        letter = {"perfect": "P", "augmented": "A", "diminished": "d", "doubly augmented": "AA", "doubly diminished": "dd"}.get(quality, "?")
    else:
        quality = {0: "major", -1: "minor", 1: "augmented", -2: "diminished", 2: "doubly augmented", -3: "doubly diminished"}.get(diff)
        letter = {"major": "M", "minor": "m", "augmented": "A", "diminished": "d", "doubly augmented": "AA", "doubly diminished": "dd"}.get(quality, "?")

    result = {
        "from": a.name,
        "to": b.name,
        "semitones": semitone,
        "number": number,
        "quality": quality,
        "name": f"{quality} {_ORDINAL[number]}" if quality else f"{semitone} semitones",
        "short": f"{letter}{number}",
    }
    if a.octave is not None and b.octave is not None:
        result["signed_semitones"] = b.midi - a.midi
    return result


# ----------------------------------------------------------- chord analysis

_REL_TO_ROMAN = {
    0: ("I", ""), 1: ("II", "b"), 2: ("II", ""), 3: ("III", "b"), 4: ("III", ""),
    5: ("IV", ""), 6: ("IV", "#"), 7: ("V", ""), 8: ("VI", "b"), 9: ("VI", ""),
    10: ("VII", "b"), 11: ("VII", ""),
}


def _root_and_quality(item):
    """Resolve a chord (symbol or note array) to (root Note, ChordType)."""
    if isinstance(item, str):
        root, chord, _bass = parse_chord_symbol(item)
        return root, chord
    res = match_chords(item, include_partial=False, limit=1)
    if not res["matches"]:
        raise ValueError(f"Could not identify a chord from notes {item!r}")
    m = res["matches"][0]
    return parse_note(m["root"]), resolve_chord_type(m["chord_type"])


def analyze_progression(chords, root: str, scale_type: str = "major") -> dict:
    """Analyze a chord progression into Roman numerals relative to a key.

    The reverse of degrees_to_chords: given chords (symbols like ['C','Am','F','G']
    or note arrays) and a key, label each with its Roman numeral, scale degree,
    whether it is diatonic to the key, and (in 7-note keys) its harmonic function
    (tonic/subdominant/dominant). Chromatic chords get an accidental-prefixed
    numeral and are flagged. Deterministic.
    """
    scale = resolve_scale_type(scale_type)
    tonic = parse_notes(root)[0]
    intervals = scale.intervals
    items = chords
    if isinstance(items, str):
        items = [t for t in items.replace(",", " ").split() if t]
    if not isinstance(items, (list, tuple)) or not items:
        raise ValueError("chords must be a non-empty list of chord symbols or note arrays")

    out = []
    for item in items:
        croot, ctype = _root_and_quality(item)
        rel = (croot.pitch_class - tonic.pitch_class) % 12
        base, accidental = _REL_TO_ROMAN[rel]
        minor_third = 3 in ctype.pitch_classes and 4 not in ctype.pitch_classes
        numeral = base.lower() if minor_third else base
        suffix = _ROMAN_QUALITY.get(ctype.name, ctype.symbol)
        in_key = rel in intervals
        entry = {
            "symbol": item if isinstance(item, str) else f"{croot.pitch_class_name}{ctype.symbol}",
            "root": croot.pitch_class_name,
            "chord_type": ctype.name,
            "roman": f"{accidental}{numeral}{suffix}",
            "in_key": in_key,
        }
        if in_key and len(intervals) == 7:
            idx = intervals.index(rel)
            entry["degree"] = idx + 1
            entry["function"] = _HARMONIC_FUNCTIONS[idx]
        elif not in_key:
            entry["note"] = "chromatic / borrowed"
        out.append(entry)
    return {"key": f"{tonic.pitch_class_name} {scale.name}", "chords": out}


# ----------------------------------------------------------- voice leading

def _voicing_cost(prev_midis: list[int], cand_midis: list[int]) -> int:
    """Symmetric nearest-note distance — rewards common tones and small moves."""
    cost = sum(min(abs(c - p) for p in prev_midis) for c in cand_midis)
    cost += sum(min(abs(p - c) for c in cand_midis) for p in prev_midis)
    return cost


def voice_leading(chords, octave: int = 4) -> dict:
    """Voice a chord progression smoothly: each chord picks the inversion/register nearest the last.

    Minimizes voice movement between consecutive chords (keeping common tones in
    place), the way a pianist or arranger comps. Returns a voiced note list (with
    octaves) and MIDI numbers per chord — feed the `voicings` straight into
    chords_to_midi or an arrange/song chords... track (as note arrays) for
    natural-sounding pads. Deterministic.
    """
    parsed = _parse_chord_list(chords)
    if not 0 <= octave <= 9:
        raise ValueError(f"octave must be between 0 and 9, got {octave}")

    voicings = []
    prev_midis: list[int] | None = None
    for ch in parsed:
        tones = [t.without_octave() for t in ch["tones"]]
        n = len(tones)
        if prev_midis is None:
            voiced = voice_chord(tones, octave)
        else:
            best = None
            best_cost = None
            for shift in (-1, 0, 1):
                for rot in range(n):
                    rotated = tones[rot:] + tones[:rot]
                    cand = voice_chord(rotated, octave + shift)
                    midis = [v.midi for v in cand]
                    if not all(0 <= m <= 127 for m in midis):
                        continue
                    cost = _voicing_cost(prev_midis, midis)
                    # tie-break: prefer no octave shift, then lower rotation
                    key = (cost, abs(shift), rot)
                    if best_cost is None or key < best_cost:
                        best_cost, best = key, cand
            voiced = best if best is not None else voice_chord(tones, octave)
        voicings.append({
            "symbol": ch["symbol"] or " ".join(t.pitch_class_name for t in voiced),
            "notes": [v.name for v in voiced],
            "midi": [v.midi for v in voiced],
        })
        prev_midis = [v.midi for v in voiced]
    return {"voicings": voicings, "chords": [v["notes"] for v in voicings]}


# --------------------------------------------------------- reharmonization

def secondary_dominant(target: str, chord_type: str = "7") -> dict:
    """The secondary dominant (V/x): the dominant chord a 5th above a target chord.

    e.g. secondary_dominant('Dm') -> A7 (the V7 of ii in C major). Pass a chord
    symbol or a root note; `chord_type` defaults to a dominant 7. Deterministic.
    """
    root, _chord, _bass = parse_chord_symbol(target) if any(c.isalpha() for c in target[1:]) else (parse_notes(target)[0], None, None)
    dom_root = transpose(root.without_octave(), 7)
    chord = resolve_chord_type(chord_type)
    tones = chord_notes(chord, dom_root)
    return {
        "of": f"{root.pitch_class_name}",
        "symbol": f"{dom_root.pitch_class_name}{chord.symbol}",
        "root": dom_root.pitch_class_name,
        "chord_type": chord.name,
        "notes": [t.name for t in tones],
    }


def tritone_substitute(symbol: str) -> dict:
    """The tritone substitution: a dominant chord a tritone away (shared guide tones).

    e.g. tritone_substitute('G7') -> Db7. Classic for dominants resolving down a
    half step (G7->C becomes Db7->C). Deterministic.
    """
    root, chord, _bass = parse_chord_symbol(symbol)
    sub_root = spell_pitch_class((root.pitch_class + 6) % 12, prefer_flats=True)  # subs read as flats
    tones = chord_notes(chord, sub_root)
    return {
        "original": f"{root.pitch_class_name}{chord.symbol}",
        "symbol": f"{sub_root.pitch_class_name}{chord.symbol}",
        "root": sub_root.pitch_class_name,
        "chord_type": chord.name,
        "notes": [t.name for t in tones],
    }


def negative_harmony(notes, tonic: str) -> dict:
    """Reflect notes through the negative-harmony axis of a key (major <-> minor).

    Ernst Levy's negative harmony mirrors each pitch around the axis between the
    tonic and its fifth, so a major chord becomes its minor "shadow" and a
    progression keeps its function while flipping colour. Mirror formula relative
    to the tonic: rel -> (7 - rel) mod 12. Works on a melody or a chord's notes;
    octaves are placed near the originals. Deterministic.
    """
    tonic_note = parse_notes(tonic)[0]
    t_pc = tonic_note.pitch_class
    out = []
    flats = tonic_note.accidental <= 0  # minor-shadow pitches read more naturally as flats
    for n in parse_notes(notes):
        mirror_pc = (7 - (n.pitch_class - t_pc)) % 12
        if n.octave is None:
            out.append(spell_pitch_class(mirror_pc, prefer_flats=flats).name)
        else:
            # choose the octave putting the mirrored pitch nearest the original
            target = min(
                (mirror_pc + 12 * k for k in range(11)),
                key=lambda m: abs(m - n.midi),
            )
            out.append(note_from_midi(target, prefer_flats=flats).name)
    return {"tonic": tonic_note.pitch_class_name, "notes": out}
