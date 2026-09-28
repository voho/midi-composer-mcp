"""20th-century and modern techniques: twelve-tone rows, pitch-class sets, minimalist processes.

- Schoenberg's twelve-tone method: the row, its inversion, retrograde and
  retrograde inversion in all twelve transpositions (the 12 x 12 matrix).
- Pitch-class set theory (Forte, Rahn): normal form, prime form (Rahn's
  algorithm), interval vector.
- Philip Glass's additive and subtractive process.
- Steve Reich's phasing (Piano Phase, Clapping Music): one pattern against a copy
  of itself that shifts by one step at a time.

(Messiaen's modes of limited transposition live in the scale database.)
Every function is deterministic.
"""

from __future__ import annotations

import re

from .midi_io import assign_octaves
from .notes import Note, parse_notes, spell_pitch_class, spelling_for_pcs

REST = "."
MAX_OUTPUT = 20000  # notes: the processes grow quadratically with the figure, so cap the result


def _tokens(value) -> list:
    if isinstance(value, str):
        return [t for t in re.split(r"[\s,]+", value.strip()) if t]
    if isinstance(value, (list, tuple)):
        return list(value)
    raise ValueError(f"expected notes as a string or a list, got {value!r}")


def _pcs_from(row) -> tuple[list[int], dict[int, Note]]:
    """Pitch classes from notes, or from numbers 0-11 (ints or digit strings: '0 4 7')."""
    tokens = _tokens(row)
    if tokens and all((isinstance(x, int) and not isinstance(x, bool))
                      or (isinstance(x, str) and re.fullmatch(r"-?\d+", x.strip())) for x in tokens):
        numbers = [int(x) for x in tokens]
        if any(not 0 <= x <= 11 for x in numbers):
            raise ValueError(f"pitch-class numbers must be 0-11 (C = 0), got {numbers}")
        return numbers, {}
    parsed = parse_notes(row)
    return [n.pitch_class for n in parsed], spelling_for_pcs(parsed)


def _check_octave(octave) -> None:
    if not isinstance(octave, int) or isinstance(octave, bool) or not 0 <= octave <= 8:
        raise ValueError(f"octave must be an integer between 0 and 8, got {octave!r}")


# ------------------------------------------------------------ twelve-tone


def twelve_tone_matrix(row) -> dict:
    """Schoenberg's twelve-tone matrix: all 48 forms of a tone row.

    `row` is the 12 pitch classes in order, as notes ('C C# D# ...') or numbers
    0-11 (C = 0), as a list or a string ('0 11 3 4 ...'). Rows of the matrix read left to right are the prime forms P,
    right to left the retrogrades R; columns top to bottom are the inversions I,
    bottom to top the retrograde inversions RI. Forms are labelled by the pitch
    class they start on (P0 starts on C, I7 on G, ...; R and RI are named after
    the P/I form they reverse). Returns the matrix as note names and every form
    by label, ready to use as melodies or chords. Deterministic.
    """
    pcs, spelling = _pcs_from(row)
    if len(pcs) != 12 or len(set(pcs)) != 12:
        raise ValueError("a twelve-tone row needs all 12 pitch classes exactly once")
    prefer_flats = any(n.accidental < 0 for n in spelling.values())

    def name(pc):
        return (spelling.get(pc) or spell_pitch_class(pc, prefer_flats)).name

    first = pcs[0]
    inv = [(2 * first - p) % 12 for p in pcs]  # the inversion starting on the same note
    matrix = [[(p + (i - first)) % 12 for p in pcs] for i in inv]
    forms = {}
    for r in matrix:
        forms[f"P{r[0]}"] = [name(p) for p in r]
        forms[f"R{r[0]}"] = [name(p) for p in reversed(r)]
    for c in range(12):
        col = [matrix[r][c] for r in range(12)]
        forms[f"I{col[0]}"] = [name(p) for p in col]
        forms[f"RI{col[0]}"] = [name(p) for p in reversed(col)]
    return {
        "row": [name(p) for p in pcs],
        "row_numbers": pcs,
        "matrix": [[name(p) for p in r] for r in matrix],
        "matrix_numbers": matrix,
        "forms": forms,
    }


# ------------------------------------------------------------ pitch-class sets

def _normal_form(pcs: list[int]) -> list[int]:
    """Rahn's normal form: the most compact rotation, packed to the left."""
    s = sorted(set(pcs))
    n = len(s)
    rotations = [s[i:] + [p + 12 for p in s[:i]] for i in range(n)]

    def key(rot):
        return [rot[-1] - rot[0]] + [rot[k] - rot[0] for k in range(n - 2, 0, -1)] + [rot[0]]

    best = min(rotations, key=key)
    return [p % 12 for p in best]


def _prime(pcs: list[int]) -> list[int]:
    nf = _normal_form(pcs)
    a = [(p - nf[0]) % 12 for p in nf]
    inv_nf = _normal_form([(-p) % 12 for p in pcs])
    b = [(p - inv_nf[0]) % 12 for p in inv_nf]

    def key(x):
        return [x[-1]] + x[-2:0:-1]

    return min(a, b, key=key)


def pitch_class_set(notes) -> dict:
    """Pitch-class set analysis (Forte / Rahn): normal form, prime form, interval vector.

    `notes` are notes or numbers 0-11 (a list, or a string such as '0 4 7');
    octaves and repetitions are ignored. The prime form names the set class
    (all transpositions and inversions of a set share it: every major and minor
    triad is [0,3,7]). It is computed with Rahn's algorithm, as in Straus's
    textbook; Forte's original table packs six classes differently (5-20, 6-Z29,
    6-31, 7-Z18, 7-20, 8-26). The interval-class
    vector counts the intervals between all pairs (ic1..ic6); the complement is
    the rest of the chromatic. Also reports transpositional and inversional
    symmetry (how many T/TI operations map the set onto itself), which is why
    the whole-tone scale or the diminished seventh sound 'rootless'.
    """
    found, spelling = _pcs_from(notes)
    pcs = sorted(set(found))
    if not pcs:
        raise ValueError("no pitch classes given")
    vector = [0] * 6
    for i in range(len(pcs)):
        for j in range(i + 1, len(pcs)):
            ic = min((pcs[j] - pcs[i]) % 12, (pcs[i] - pcs[j]) % 12)
            vector[ic - 1] += 1
    nf = _normal_form(pcs)
    t_sym = sum(1 for t in range(12) if {(p + t) % 12 for p in pcs} == set(pcs))
    i_sym = sum(1 for t in range(12) if {(t - p) % 12 for p in pcs} == set(pcs))

    def name(pc):
        return (spelling.get(pc) or spell_pitch_class(pc)).name

    return {
        "pitch_classes": pcs,
        "notes": [name(p) for p in pcs],
        "cardinality": len(pcs),
        "normal_form": nf,
        "normal_form_notes": [name(p) for p in nf],
        "prime_form": _prime(pcs),
        "interval_vector": vector,
        "complement": [p for p in range(12) if p not in pcs],
        "transpositional_symmetry": t_sym,
        "inversional_symmetry": i_sym,
    }


# ------------------------------------------------------------ minimalist processes

def _placed(notes, octave: int) -> list[str]:
    """Fix the figure's register once, so every repeat is the same pitches."""
    return [n.name for n in assign_octaves(parse_notes(notes), octave, "nearest")]


def additive_process(notes, mode: str = "additive", repeats: int = 1, octave: int = 4) -> dict:
    """Philip Glass's additive (or subtractive) process over a figure.

    Additive: play the first two notes, then the first three, and so on until
    the whole figure sounds (1 2, 1 2 3, 1 2 3 4, ...) — the process of 'Music
    in Fifths' and 'Einstein on the Beach'. Subtractive runs it backwards;
    'both' goes up and back down. Each stage can be repeated `repeats` times.
    Notes without an octave are placed once, from `octave`, so every stage
    repeats the same pitches. Returns the full note sequence and the stage
    boundaries (feed the notes to a notes track; stage lengths make the rhythm
    lurch in Glass's way). The output is capped at 20000 notes.
    """
    if mode not in ("additive", "subtractive", "both"):
        raise ValueError("mode must be 'additive', 'subtractive' or 'both'")
    if not isinstance(repeats, int) or isinstance(repeats, bool) or not 1 <= repeats <= 16:
        raise ValueError("repeats must be an integer between 1 and 16")
    _check_octave(octave)
    fig = _placed(notes, octave)
    if len(fig) < 2:
        raise ValueError("the figure needs at least 2 notes")
    sizes = list(range(2, len(fig) + 1))
    if mode == "subtractive":
        sizes = sizes[::-1]
    elif mode == "both":
        sizes = sizes + sizes[-2::-1]
    if sum(sizes) * repeats > MAX_OUTPUT:
        raise ValueError(f"that process would produce {sum(sizes) * repeats} notes; keep it under {MAX_OUTPUT} "
                         "(a shorter figure or fewer repeats)")
    seq, stages = [], []
    for size in sizes:
        for _ in range(repeats):
            stages.append({"start": len(seq), "length": size})
            seq.extend(fig[:size])
    return {"figure": fig, "mode": mode, "notes": seq, "stages": stages}


def phase_shift(notes, repeats_per_stage: int = 2, step_beats: float = 0.25, octave: int = 4) -> dict:
    """Steve Reich's phasing: a pattern against a copy of itself that slips ahead one step per stage.

    Voice 1 repeats the pattern; voice 2 plays it too, but after every
    `repeats_per_stage` cycles it moves one step ahead, until after
    len(pattern) stages the voices are back in unison. Reich's Clapping Music
    works this way (abrupt shifts); Piano Phase drifts gradually, and this
    shows its locked positions. The pattern may contain rests ('.'), which
    rotate with it: 'C C C . C C . C . C C .' is the Clapping Music rhythm.
    Notes without an octave are placed once, from `octave`, so both voices
    play the same pitches. Returns both voices (notes, plus a rhythm string
    when there are rests) and a two-track `render_hint` for arrange_to_midi.
    Each voice is capped at 20000 steps. Deterministic.
    """
    # one token per step, however the caller grouped them ('C E', ['C E', 'G'], ['C', '.', 'E'])
    tokens = [t for item in _tokens(notes) for t in re.split(r"[,\s]+", str(item).strip()) if t]
    if not isinstance(repeats_per_stage, int) or isinstance(repeats_per_stage, bool) or not 1 <= repeats_per_stage <= 16:
        raise ValueError("repeats_per_stage must be an integer between 1 and 16")
    if not isinstance(step_beats, (int, float)) or isinstance(step_beats, bool) or not 0.0625 <= step_beats <= 16:
        raise ValueError(f"step_beats must be a number between 0.0625 and 16, got {step_beats!r}")
    _check_octave(octave)
    if len(tokens) < 2:
        raise ValueError("the pattern needs at least 2 steps")
    if len(tokens) * (len(tokens) + 1) * repeats_per_stage > MAX_OUTPUT:
        raise ValueError(f"a {len(tokens)}-step pattern with {repeats_per_stage} repeats per stage is too long; "
                         f"keep len * (len + 1) * repeats under {MAX_OUTPUT}")
    sounding = [t for t in tokens if t != REST]
    if not sounding:
        raise ValueError("the pattern needs at least one note")
    placed = iter(_placed(sounding, octave))
    pat = [t if t == REST else next(placed) for t in tokens]
    has_rests = len(sounding) < len(pat)
    steps_1, steps_2 = [], []
    for stage in range(len(pat) + 1):
        k = stage % len(pat)
        shifted = pat[k:] + pat[:k]
        for _ in range(repeats_per_stage):
            steps_1.extend(pat)
            steps_2.extend(shifted)

    def voice(steps, name):
        track = {"type": "notes", "name": name, "notes": [t for t in steps if t != REST],
                 "step_beats": step_beats, "program": 0}
        if has_rests:
            track["rhythm"] = "".join(REST if t == REST else "o" for t in steps)
        return track

    t1, t2 = voice(steps_1, "voice 1"), voice(steps_2, "voice 2")
    result = {
        "pattern": pat,
        "stages": len(pat) + 1,
        "voice_1": t1["notes"],
        "voice_2": t2["notes"],
        "render_hint": {"tracks": [t1, t2]},
    }
    if has_rests:
        result["rhythm_1"], result["rhythm_2"] = t1["rhythm"], t2["rhythm"]
    return result
