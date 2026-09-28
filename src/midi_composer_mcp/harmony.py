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
from .diatonic import _HARMONIC_FUNCTIONS, _ROMAN_BASE, roman_suffix
from .midi_io import _parse_chord_list, _with_octave, voice_chord
from .notes import (
    LETTER_PCS, LETTERS, Note, best_spelling, parse_note, parse_notes, spell_pitch_class,
    spelling_for_pcs, transpose,
)
from .scales import MAJOR_DEGREES, resolve_scale_type, scale_notes

# ---------------------------------------------------------------- intervals

_MAJOR_REF = {1: 0, 2: 2, 3: 4, 4: 5, 5: 7, 6: 9, 7: 11}
_PERFECT_NUMBERS = {1, 4, 5}
_ORDINAL = {1: "unison", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
            6: "sixth", 7: "seventh", 8: "octave", 9: "ninth", 10: "tenth",
            11: "eleventh", 12: "twelfth", 13: "thirteenth", 14: "fourteenth",
            15: "fifteenth"}


def interval_between(note_a: str, note_b: str) -> dict:
    """Name the interval from `note_a` up to `note_b` (e.g. C -> Eb = minor third).

    Returns the semitone distance, the interval number and quality (perfect/
    major/minor/augmented/diminished), and the short label (P5, m3, M7...).
    Octave-less notes are pitch classes, so the interval is the simple one
    upward (1-7). When both notes carry octaves the real interval is named,
    including compound ones (C4 -> C5 = perfect octave P8, C4 -> E5 = major
    tenth M10) and descending ones (`direction`), plus the signed semitone
    distance. Deterministic.
    """
    a = parse_note(note_a)
    b = parse_note(note_b)
    direction = None
    if a.octave is not None and b.octave is not None:
        steps = (LETTERS.index(b.letter) + 7 * b.octave) - (LETTERS.index(a.letter) + 7 * a.octave)
        low, high = a, b
        if steps < 0 or (steps == 0 and b.midi < a.midi):
            low, high, steps = b, a, -steps
            direction = "descending"
        else:
            direction = "ascending" if b.midi != a.midi or steps else "unison"
        semitones = high.midi - low.midi
        number = steps + 1
        simple = (number - 1) % 7 + 1
        diff = semitones - 12 * ((number - 1) // 7) - _MAJOR_REF[simple]
    else:
        # pitch classes: the interval upward from a to b, sized from the letters so the
        # semitone count always agrees with the name (C->B# = A7 = 12, C->Cb = d8 = 11)
        number = (LETTERS.index(b.letter) - LETTERS.index(a.letter)) % 7 + 1
        semitones = (LETTER_PCS[b.letter] - LETTER_PCS[a.letter]) % 12 + b.accidental - a.accidental
        if number == 1 and semitones < 0:
            number, semitones = 8, semitones + 12  # C up to Cb is a diminished octave
        simple = (number - 1) % 7 + 1
        diff = semitones - 12 * ((number - 1) // 7) - _MAJOR_REF[simple]

    if simple in _PERFECT_NUMBERS:
        quality = {0: "perfect", 1: "augmented", -1: "diminished", 2: "doubly augmented", -2: "doubly diminished"}.get(diff)
        letter = {"perfect": "P", "augmented": "A", "diminished": "d", "doubly augmented": "AA", "doubly diminished": "dd"}.get(quality, "?")
    else:
        quality = {0: "major", -1: "minor", 1: "augmented", -2: "diminished", 2: "doubly augmented", -3: "doubly diminished"}.get(diff)
        letter = {"major": "M", "minor": "m", "augmented": "A", "diminished": "d", "doubly augmented": "AA", "doubly diminished": "dd"}.get(quality, "?")

    suffix = "th" if 10 <= number % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    ordinal = _ORDINAL.get(number, f"{number}{suffix}")
    result = {
        "from": a.name,
        "to": b.name,
        "semitones": semitones,
        "number": number,
        "quality": quality,
        "name": f"{quality} {ordinal}" if quality else f"{semitones} semitones",
        "short": f"{letter}{number}",
    }
    if direction is not None:
        result["signed_semitones"] = b.midi - a.midi
        result["direction"] = direction
        result["compound"] = number > 8
    return result


# ----------------------------------------------------------- chord analysis

# Fallback (spelling-blind) numerals, used only when the letter-based spelling
# would need more than a double accidental.
_REL_TO_ROMAN = {
    0: ("I", ""), 1: ("II", "b"), 2: ("II", ""), 3: ("III", "b"), 4: ("III", ""),
    5: ("IV", ""), 6: ("IV", "#"), 7: ("V", ""), 8: ("VI", "b"), 9: ("VI", ""),
    10: ("VII", "b"), 11: ("VII", ""),
}


def _root_and_quality(item):
    """Resolve a chord (symbol or note array) to (root Note, ChordType, bass Note or None)."""
    if isinstance(item, str):
        return parse_chord_symbol(item)
    res = match_chords(item, include_partial=False, limit=1)
    if not res["matches"]:  # a cluster that is no chord type: analyze it by its first note
        return parse_notes(item)[0], None, None
    m = res["matches"][0]
    bass = parse_note(m["bass"]) if "bass" in m else None
    return parse_note(m["root"]), resolve_chord_type(m["chord_type"]), bass


def _numeral_base(croot: Note, tonic: Note) -> tuple[str, str]:
    """(numeral, accidental prefix) of a chord root, read from its spelling.

    The numeral comes from the letter distance to the tonic and the accidental
    from how far the root sits from the major-scale degree on that letter, so
    Gb in C is bV and F# is #IV (matching diatonic_chords' numerals).
    """
    steps = (LETTERS.index(croot.letter) - LETTERS.index(tonic.letter)) % 7
    rel = (croot.pitch_class - tonic.pitch_class) % 12
    offset = (rel - MAJOR_DEGREES[steps] + 6) % 12 - 6
    if abs(offset) <= 2:
        return _ROMAN_BASE[steps], "#" * offset if offset > 0 else "b" * -offset
    base, accidental = _REL_TO_ROMAN[rel]
    return base, accidental


def analyze_progression(chords, root: str, scale_type: str = "major") -> dict:
    """Analyze a chord progression into Roman numerals relative to a key.

    The reverse of degrees_to_chords: given chords (symbols like ['C','Am','F','G']
    or note arrays) and a key, label each with its Roman numeral, scale degree,
    whether it is diatonic to the key, and (in 7-note keys) its harmonic function
    (tonic/subdominant/dominant). A chord is `in_key` only when **every** chord
    tone (and a slash bass) belongs to the scale — E7 in A natural minor is
    flagged because of its G#, and its `non_scale_notes` are listed. Numerals
    follow the chord's spelling (Gb in C = bV, F# = #IV). Deterministic.
    """
    scale = resolve_scale_type(scale_type)
    tonic = parse_notes(root)[0].without_octave()
    intervals = scale.intervals
    scale_pcs = {(tonic.pitch_class + i) % 12 for i in intervals}
    items = chords
    if isinstance(items, str):
        items = [t for t in items.replace(",", " ").split() if t]
    if not isinstance(items, (list, tuple)) or not items:
        raise ValueError("chords must be a non-empty list of chord symbols or note arrays")

    out = []
    for item in items:
        croot, ctype, bass = _root_and_quality(item)
        croot = croot.without_octave()
        rel = (croot.pitch_class - tonic.pitch_class) % 12
        if rel in intervals and len(intervals) == 7:
            # a scale degree keeps the scale's own numeral (Db in C# major is I, like diatonic_chords)
            idx = intervals.index(rel)
            offset = intervals[idx] - MAJOR_DEGREES[idx]
            base, accidental = _ROMAN_BASE[idx], "#" * offset if offset > 0 else "b" * -offset
        else:  # a chromatic root: its spelling decides between enharmonic numerals (bV vs #IV)
            base, accidental = _numeral_base(croot, tonic)
        if ctype is None:
            tones = [n.without_octave() for n in parse_notes(item)]
            rel_pcs = {(t.pitch_class - croot.pitch_class) % 12 for t in tones}
        else:
            tones = chord_notes(ctype, croot)
            rel_pcs = ctype.pitch_classes
        minor_third = 3 in rel_pcs and 4 not in rel_pcs
        numeral = base.lower() if minor_third else base
        suffix = roman_suffix(ctype, minor_third)
        if bass is not None:
            tones = [bass.without_octave(), *tones]
        outside = []
        for t in tones:
            if t.pitch_class not in scale_pcs and t.pitch_class_name not in outside:
                outside.append(t.pitch_class_name)
        in_key = not outside
        entry = {
            "symbol": item if isinstance(item, str) else (
                f"{croot.pitch_class_name}{ctype.symbol}" if ctype else " ".join(t.name for t in tones)),
            "root": croot.pitch_class_name,
            "chord_type": ctype.name if ctype else "unknown",
            "roman": f"{accidental}{numeral}{suffix}",
            "in_key": in_key,
        }
        if rel in intervals:
            idx = intervals.index(rel)
            entry["degree"] = idx + 1
            if in_key and len(intervals) == 7:
                entry["function"] = _HARMONIC_FUNCTIONS[idx]
        if not in_key:
            entry["non_scale_notes"] = outside
            entry["note"] = "chromatic / borrowed"
        out.append(entry)
    return {"key": f"{tonic.pitch_class_name} {scale.name}", "chords": out}


# ----------------------------------------------------------- voice leading

def _voicing_cost(prev_midis: list[int], cand_midis: list[int]) -> int:
    """Total voice movement between two voicings.

    With equal voice counts each voice moves to exactly one voice (the optimal
    one-to-one assignment on a line is sorted order), so a voice that must leap
    counts in full. Otherwise fall back to a symmetric nearest-note distance.
    """
    if len(prev_midis) == len(cand_midis):
        return sum(abs(p - c) for p, c in zip(sorted(prev_midis), sorted(cand_midis)))
    cost = sum(min(abs(c - p) for p in prev_midis) for c in cand_midis)
    cost += sum(min(abs(p - c) for c in cand_midis) for p in prev_midis)
    return cost


def voice_leading(chords, octave: int = 4) -> dict:
    """Voice a chord progression smoothly: each chord picks the inversion/register nearest the last.

    Minimizes voice movement between consecutive chords (keeping common tones in
    place), the way a pianist or arranger comps. A slash chord keeps its bass
    note at the bottom ('C/E' -> E below the voiced C triad). Returns a voiced
    note list (with octaves) and MIDI numbers per chord — feed the `voicings`
    straight into chords_to_midi or an arrange/song chords track (as note
    arrays) for natural-sounding pads. Deterministic.
    """
    parsed = _parse_chord_list(chords)
    if not isinstance(octave, int) or isinstance(octave, bool) or not 0 <= octave <= 9:
        raise ValueError(f"octave must be an integer between 0 and 9, got {octave!r}")

    voicings = []
    prev_midis: list[int] | None = None
    for ch in parsed:
        tones = [t.without_octave() for t in ch["tones"]]
        bass = ch["bass"].without_octave() if ch["bass"] is not None else None
        n = len(tones)
        if prev_midis is None:
            voiced = voice_chord(tones, octave, bass)
        else:
            best = None
            best_cost = None
            for shift in (-1, 0, 1):
                for rot in range(n):
                    rotated = tones[rot:] + tones[:rot]
                    try:  # a candidate register may fall outside MIDI 0-127: skip it
                        cand = voice_chord(rotated, octave + shift, bass)
                    except ValueError:
                        continue
                    midis = [v.midi for v in cand]
                    cost = _voicing_cost(prev_midis, midis)
                    held = len(set(prev_midis) & set(midis))
                    # tie-breaks: keep more common tones in place, no octave shift, lower rotation
                    key = (cost, -held, abs(shift), rot)
                    if best_cost is None or key < best_cost:
                        best_cost, best = key, cand
            voiced = best if best is not None else voice_chord(tones, octave, bass)
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
    if not isinstance(target, str):
        raise ValueError(f"target must be a chord symbol or note name, got {target!r}")
    try:  # any chord symbol ('Dm', 'C6/9', 'G7#9', a bare 'F#') ...
        root = parse_chord_symbol(target)[0]
    except ValueError:  # ... or a concrete note such as 'D4'
        root = parse_notes(target)[0]
    dom_root = transpose(root.without_octave(), 7, 4)  # a perfect 5th, spelled on the 5th letter (A# -> E#)
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
    half step (G7->C becomes Db7->C). Takes a dominant-family chord (7, 9, 13,
    7b9, 7#11...; the quality is kept) or a plain major triad, read as the
    dominant 7 it stands for (G -> Db7). Other qualities have no guide-tone
    tritone to share and raise ValueError. Deterministic.
    """
    root, chord, _bass = parse_chord_symbol(symbol)
    if chord.name == "major":
        chord = CHORDS["dominant 7"]
    elif not {4, 10} <= chord.pitch_classes:
        raise ValueError(
            f"{symbol!r} is not a dominant chord: a tritone substitution swaps dominants that"
            f" share their 3rd and b7 (e.g. G7 -> Db7)"
        )
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
    progression keeps its function while flipping colour. Works in any key: with
    rel = the note's distance above the tonic, the mirror is tonic + (7 - rel)
    mod 12 — in C, C E G -> G Eb C; in D, D F# A -> A F D (D minor). Spelling
    mirrors the letters too (E -> Eb, not D#). Works on a melody or a chord's
    notes; octaves are placed near the originals. Deterministic.
    """
    tonic_note = parse_notes(tonic)[0].without_octave()
    t_pc = tonic_note.pitch_class
    t_letter = LETTERS.index(tonic_note.letter)
    flats = tonic_note.accidental <= 0  # fallback spelling: minor-shadow pitches read as flats
    out = []
    for n in parse_notes(notes):
        mirror_pc = (2 * t_pc + 7 - n.pitch_class) % 12
        # mirror the letter around the tonic..fifth letter axis, then fix the accidental
        letter = LETTERS[(2 * t_letter + 4 - LETTERS.index(n.letter)) % 7]
        accidental = (mirror_pc - LETTER_PCS[letter] + 6) % 12 - 6
        spelled = (Note(letter, accidental) if abs(accidental) <= 2
                   else spell_pitch_class(mirror_pc, prefer_flats=flats))
        if n.octave is None:
            out.append(spelled.name)
        else:
            # choose the octave putting the mirrored pitch nearest the original
            target = min(
                (m for m in (mirror_pc + 12 * k for k in range(11)) if m <= 127),
                key=lambda m: (abs(m - n.midi), m),
            )
            out.append(_with_octave(spelled, target).name)
    return {"tonic": tonic_note.pitch_class_name, "notes": out}


# ----------------------------------------------------------- harmonization

# A deterministic preference among chord qualities — when two candidates share
# the same number of notes with the previous chord and are the same size, the
# more common quality wins, so a melody is harmonized with familiar chords first.
_TYPE_PREF = {name: i for i, name in enumerate([
    "major", "minor", "dominant 7", "major 7", "minor 7", "major 6", "minor 6",
    "suspended 4", "suspended 2", "dominant 9", "major 9", "minor 9", "six nine",
    "add 9", "minor add 9", "diminished", "augmented", "half-diminished",
    "minor major 7", "diminished 7",
])}


def harmonize_melody(notes, root: str | None = None, scale_type: str = "major",
                     in_scale: bool = False, max_chord_notes: int = 4,
                     allow_repeats: bool = True, options: int = 4,
                     octave: int = 4, step_beats: float = 2.0) -> dict:
    """Harmonize a melody: pick a chord under each note that maximizes reuse of the previous chord's notes.

    For every melody note it searches the **whole chord database** (all chord
    types on all roots) for chords that *contain* that note, ranks them by how
    many notes they share with the previously chosen chord (most shared first —
    smoothest voice leading), then auto-picks the top one and runs voice_leading
    for the inversions. Each note also reports the top `options` ranked
    alternatives. Deterministic.

    Optional key: pass `root`/`scale_type` with `in_scale=True` to restrict
    candidates to chords whose notes all fit that scale (any chord type, not just
    the seven diatonic chords); otherwise every chord is fair game.
    `max_chord_notes` caps complexity (4 = triads and sevenths). `allow_repeats`
    lets the same chord underlie consecutive notes (true maximum reuse); set it
    false to force a new chord each note. Returns the melody, the chosen
    progression with voiced notes and per-note options, plus a `render_hint`
    (a harmony track + the melody on top) for arrange_to_midi.
    """
    mel = parse_notes(notes)
    if not isinstance(max_chord_notes, int) or isinstance(max_chord_notes, bool) or not 2 <= max_chord_notes <= 6:
        raise ValueError("max_chord_notes must be an integer between 2 and 6")
    if not isinstance(options, int) or isinstance(options, bool):
        raise ValueError(f"options must be an integer, got {options!r}")
    options = max(1, min(12, options))
    if not isinstance(step_beats, (int, float)) or isinstance(step_beats, bool) or not 0.25 <= step_beats <= 16:
        raise ValueError(f"step_beats must be a number between 0.25 and 16 (so the render_hint renders), got {step_beats!r}")
    if in_scale and root is None:
        raise ValueError("in_scale=true needs a key: pass root (and scale_type)")

    # Spelling: the melody's own spellings, overridden by the key's where one is given;
    # other roots are spelled to agree with those (Eb G Bb gives Eb, never D#).
    pc_names: dict[int, Note] = spelling_for_pcs(mel)
    scale_pcs = None
    key_name = None
    if root is not None:
        scale = resolve_scale_type(scale_type)
        root_note = parse_notes(root)[0].without_octave()
        key_notes = {n.pitch_class: n for n in scale_notes(scale, root_note)[:-1]}
        pc_names.update(key_notes)
        scale_pcs = frozenset(key_notes)
        key_name = f"{root_note.name} {scale.name}"

    universe = []  # (root_pc, ChordType, pitch_class_set, size)
    for ctype in CHORDS.values():
        size = len(ctype.intervals)
        if 2 <= size <= max_chord_notes:
            for pc in range(12):
                universe.append((pc, ctype, frozenset((pc + s) % 12 for s in ctype.intervals), size))

    def spell(pc, ctype):
        return best_spelling(pc, lambda r: chord_notes(ctype, r), pc_names)

    chords_out = []
    symbols = []
    prev_pcs = None
    for n in mel:
        mpc = n.pitch_class
        cands = [c for c in universe if mpc in c[2] and (not in_scale or c[2] <= scale_pcs)]
        no_in_scale_chord = False
        if not cands:  # no in-scale chord holds this note: fall back to any chord that does
            cands = [c for c in universe if mpc in c[2]]
            no_in_scale_chord = True

        def rank(c):
            pc, ct, pcs, size = c
            # primary: most shared notes with the previous chord (smoothest reuse);
            # then the melody note as a lower chord tone (root first), then a common
            # quality, then a simpler chord — so triads beat dyads and clusters.
            shared = len(prev_pcs & pcs) if prev_pcs is not None else 0
            mpos = [(pc + s) % 12 for s in ct.intervals].index(mpc)
            return (-shared, mpos, _TYPE_PREF.get(ct.name, 99), size, pc)

        cands.sort(key=rank)
        if not allow_repeats and prev_pcs is not None and len(cands) > 1:
            # a different chord means different notes: C6 after Am7 is the same chord
            cands = [c for c in cands if c[2] != prev_pcs] or cands

        def describe(c):
            pc, ct, pcs, size = c
            r = spell(pc, ct)
            return {
                "symbol": f"{r.pitch_class_name}{ct.symbol}",
                "chord_type": ct.name,
                "notes": [t.name for t in chord_notes(ct, r)],
                "shares_with_previous": (len(prev_pcs & pcs) if prev_pcs is not None else None),
            }

        best = cands[0]
        entry = describe(best)
        entry["melody_note"] = n.name
        entry["options"] = [describe(c) for c in cands[:options]]
        if scale_pcs is not None and mpc not in scale_pcs:
            entry["melody_out_of_scale"] = True
        if no_in_scale_chord:
            entry["no_in_scale_chord"] = True
        chords_out.append(entry)
        symbols.append(entry["symbol"])
        prev_pcs = best[2]

    vl = voice_leading(symbols, octave=octave)
    for entry, v in zip(chords_out, vl["voicings"]):
        entry["voiced"] = v["notes"]

    result = {
        "melody": [n.name for n in mel],
        "progression": symbols,
        "allow_repeats": allow_repeats,
        "chords": chords_out,
        "voicings": vl["chords"],
        "render_hint": {"tracks": [
            {"type": "chords", "name": "harmony", "chords": vl["chords"], "beats_per_chord": step_beats},
            {"type": "notes", "name": "melody", "notes": [n.name for n in mel], "step_beats": step_beats, "octave": 5},
        ]},
    }
    if key_name:
        result["key"] = key_name
        result["in_scale"] = in_scale
    return result
