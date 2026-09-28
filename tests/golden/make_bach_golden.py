"""Regenerate tests/golden/bach_chorale_voicing.json (the bach_chorale_voicing regression corpus).

The golden file pins bach_chorale_voicing's output, byte for byte (a SHA-256 of
its JSON), over the README progressions and a wider corpus in every key, with
and without a melody. It was generated on main *before* masters.py's parallel,
direct-motion and overlap tests were refactored into shared MIDI-level
predicates (so check_voice_leading and the chorale engine cannot disagree), and
tests/test_analysis.py asserts that nothing changed.

Only regenerate it when a change to the chorale rules is intended:

    python tests/golden/make_bach_golden.py
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from midi_composer_mcp.chant import guido_vowel_melody  # noqa: E402
from midi_composer_mcp.harmony import harmonize_melody  # noqa: E402
from midi_composer_mcp.masters import bach_chorale_voicing  # noqa: E402
from midi_composer_mcp.midi_io import _parse_chord_list, _with_octave  # noqa: E402
from midi_composer_mcp.notes import LETTERS, parse_note, transpose  # noqa: E402

GOLDEN = os.path.join(HERE, "bach_chorale_voicing.json")

# The tonic spellings of tests/test_invariants.py, and their relative minors.
TONICS = ["C", "G", "D", "A", "E", "B", "F#", "C#", "F", "Bb", "Eb", "Ab", "Db", "Gb"]
MINOR_TONICS = ["A", "E", "B", "F#", "C#", "G#", "D#", "A#", "D", "G", "C", "F", "Bb", "Eb"]

# (progression written in the home key, home tonic, scale type, extra variants)
MAJOR = [
    # README examples (transposed to C where the README used another key)
    "C G Am F", "Dm7 G7 Cmaj7", "C A7 Dm G7 C", "Dm7 Eb7 Abmaj7 B7 Emaj7 G7 Cmaj7",
    "C Am F G7 C", "Fmaj7 Cmaj7 Dm7 Em7", "C C Cadd4 Am7 C6",
    # tests/test_masters.py and docs/book
    "C Am F G7 C Dm G C", "C G/B Am C/G F G C", "C G Am", "C G/B C", "C Bdim7 G",
    # cadences, sequences, borrowed and applied chords, inversions
    "C F G7 C", "C F C", "C F G C", "C Am Dm G7 C", "C F Bdim Em Am Dm G C", "C G Am Em F C F G",
    "C Bb F C", "C Fm C", "C D7 G", "C C#dim7 Dm G7 C", "C E7 Am C7 F G7 C", "C F/A G/B C",
    "C Dm/F G7 C", "Csus4 C F G7 C", "C Caug F", "C F G7 Am", "C Am Dm G", "C Em/B Am G F C/E Dm7 G7 C",
    "Cmaj7 Am7 Dm7 G7 Cmaj7", "C G7/D C/E", "C G/D C", "C Em C", "C Dm7 Em7 Fmaj7 G7 C",
    "C D9 G", "C13 G7 C",
]
MINOR = [
    ("Am E7 Am", "natural minor"), ("Am E7 Am", "harmonic minor"), ("Am Dm E Am", "harmonic minor"),
    ("Am Dm E7 Am", "natural minor"), ("Am G F E", "natural minor"), ("Am Bm7b5 E7 Am", "harmonic minor"),
    ("Am Bb/D E7 Am", "harmonic minor"), ("AmMaj7 Dm7 E7 AmMaj7", "harmonic minor"),
    ("Am F C G", "natural minor"), ("Am7 Dm7 Fmaj7 Cmaj7", "natural minor"), ("Am F G C", "natural minor"),
    ("Am E/G# Am/G D/F# F E", "melodic minor"), ("Am Dm/F E Am", "harmonic minor"), ("Am F Dm E", "harmonic minor"),
]
DORIAN = ["Dm G Dm", "Dm G Am Dm", "Dm C G Dm", "Dm Em F G Dm"]
ARRAYS = [[["B", "D", "F", "G"], "C"], [["D", "F", "Ab", "B"], "C"], [["E", "G", "C"], ["F", "A", "C"], "G7", "C"]]
TUNE = (["C", "G", "Am", "G", "C", "Em", "C"], ["E4", "D4", "C4", "D4", "E4", "E4", "E4"])  # test_masters
NO_ROOT = ["C Am F G7 C", "Am E7 Am", "C7b5 F", "C G/B Am C/G F G C", "Am/G D", "Am7/G D", "Dm7 G7 Cmaj7"]

_SYMBOL = re.compile(r"^([A-G][#b]*)(.*?)(?:/([A-G][#b]*))?$")


def _shift(note: str, home: str, tonic: str) -> str:
    """Transpose one note name from the key of `home` to the key of `tonic`, spelled by letters."""
    h, t = parse_note(home), parse_note(tonic)
    semis = (t.pitch_class - h.pitch_class) % 12
    steps = (LETTERS.index(t.letter) - LETTERS.index(h.letter)) % 7
    n = parse_note(note)
    moved = transpose(n.without_octave(), semis, steps)
    if n.octave is None:
        return moved.name
    target = n.midi + semis if semis <= 6 else n.midi + semis - 12  # stay near the written register
    return _with_octave(moved, target).name


def _shift_chord(item, home, tonic):
    if isinstance(item, list):
        return [_shift(n, home, tonic) for n in item]
    root, quality, bass = _SYMBOL.match(item).groups()
    out = _shift(root, home, tonic) + quality
    return out + ("/" + _shift(bass, home, tonic) if bass else "")


def _shift_prog(prog, home, tonic):
    items = prog.split() if isinstance(prog, str) else prog
    return [_shift_chord(c, home, tonic) for c in items]


def _auto_melody(chords, leap: bool = False) -> list[str]:
    """A soprano line of chord tones: the first chord's third (or root) nearest B-flat 4, then
    each chord's tone nearest the previous note (lower on a tie). With `leap`, the nearest tone
    at least a minor third away, kept within C4-G5 (so the direct-motion, overlap and
    leading-tone rules are exercised too). Deterministic."""
    out, prev = [], None
    for ch in _parse_chord_list(chords):
        tones = [t.without_octave() for t in ch["tones"]]
        if prev is None:
            first = tones[1] if len(tones) > 1 else tones[0]
            prev = min((first.pitch_class + 12 * k for k in range(11)), key=lambda m: (abs(m - 70), m))
            out.append(_with_octave(first, prev).name)
            continue
        best = None
        for t in tones:
            for m in (prev - ((prev - t.pitch_class) % 12), prev + ((t.pitch_class - prev) % 12),
                      prev - ((prev - t.pitch_class) % 12) - 12, prev + ((t.pitch_class - prev) % 12) + 12):
                if leap and (abs(m - prev) < 3 or not 60 <= m <= 79):
                    continue
                key = (abs(m - prev), m)
                if best is None or key < best[0]:
                    best = (key, t, m)
        _, t, prev = best
        out.append(_with_octave(t, prev).name)
    return out


def corpus() -> list[dict]:
    cases = []

    def add(chords, root, scale_type="major", melody=None):
        cases.append({"chords": chords, "root": root, "scale_type": scale_type, "melody": melody})

    for prog in MAJOR:
        for tonic in TONICS:
            chords = _shift_prog(prog, "C", tonic)
            add(chords, tonic)
            add(chords, tonic, melody=_auto_melody(chords))
            add(chords, tonic, melody=_auto_melody(chords, leap=True))
    for prog, scale in MINOR:
        for tonic in MINOR_TONICS:
            chords = _shift_prog(prog, "A", tonic)
            add(chords, tonic, scale)
            add(chords, tonic, scale, melody=_auto_melody(chords))
            add(chords, tonic, scale, melody=_auto_melody(chords, leap=True))
    for prog in DORIAN:
        for tonic in TONICS:
            home_tonic = _shift("D", "C", tonic)
            chords = _shift_prog(prog, "D", home_tonic)
            add(chords, home_tonic, "dorian")
            add(chords, home_tonic, "dorian", melody=_auto_melody(chords))
    for prog in ARRAYS:
        for tonic in TONICS:
            chords = _shift_prog(prog, "C", tonic)
            add(chords, tonic)
    for tonic in TONICS:
        chords, tune = TUNE
        add(_shift_prog(chords, "C", tonic), tonic, melody=[_shift(n, "C", tonic) for n in tune])
    for prog in NO_ROOT:
        add(prog.split(), None)
    # the README's Guido -> harmonize_melody -> chorale chain
    chant = guido_vowel_melody("Ut queant laxis resonare fibris", "Dorian")["notes"]
    prog = harmonize_melody(chant, root="D", scale_type="dorian", in_scale=True)["progression"]
    add(prog, "D", "dorian", melody=chant)
    add(prog, "D", "dorian")
    return cases


def canonical(result: dict) -> str:
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"))


def run_case(args: dict) -> dict:
    try:
        result = bach_chorale_voicing(**args)
    except ValueError as e:
        return {"args": args, "error": str(e)}
    return {
        "args": args,
        "sha256": hashlib.sha256(canonical(result).encode("utf-8")).hexdigest(),
        "key": result["key"],
        "voices": result["voices"],
        "rule_breaks": result["rule_breaks"],
    }


def build() -> list[dict]:
    return [run_case(args) for args in corpus()]


def dump(cases: list[dict]) -> str:
    return "[\n" + ",\n".join(json.dumps(c, ensure_ascii=False, separators=(",", ":")) for c in cases) + "\n]\n"


if __name__ == "__main__":
    cases = build()
    with open(GOLDEN, "w", encoding="utf-8") as fh:
        fh.write(dump(cases))
    errors = sum(1 for c in cases if "error" in c)
    print(f"wrote {len(cases)} cases ({errors} raise ValueError) to {GOLDEN}")
