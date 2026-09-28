"""Harmony-layer rules: intervals, analysis, voice-leading and reharmonization.

Classic, fully deterministic music-theory operations. Each is a mechanical
transform or lookup the LLM can chain — name an interval, analyze a
progression into Roman numerals (figures, applied chords, specials, mixture),
voice chords smoothly, rank what may come next by Piston's table, or
reharmonize with secondary dominants, tritone subs and negative harmony.
"""

from __future__ import annotations

import re

from .chords import (
    CHORDS,
    chord_notes,
    match_chords,
    parse_chord_symbol,
    resolve_chord_type,
)
from .diatonic import _HARMONIC_FUNCTIONS, _ROMAN_BASE, borrowed_sources, diatonic_chords, roman_suffix
from .diatonic import numeral_base as _numeral_base
from .midi_io import _parse_chord_list, _with_octave, voice_chord
from .notes import (
    LETTER_PCS, LETTERS, Note, best_spelling, parse_note, parse_notes, spell_pitch_class,
    spelling_for_pcs, transpose,
)
from .roman import _MINOR_MODES, applied_reading, figure_for_bass, read_chord, roman_to_chords, special_reading
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

def _root_and_quality(item):
    """Resolve a chord (symbol or note array) to (root Note, ChordType, bass Note or None)."""
    if isinstance(item, str):
        return parse_chord_symbol(item)
    res = match_chords(item, include_partial=False, limit=12)
    if not res["matches"]:  # a cluster that is no chord type: analyze it by its first note
        return parse_notes(item)[0], None, None
    # prefer the reading whose spelled notes are the ones written (D F Ab B is Bdim7/D, not Ddim7 with Cb)
    written = {n.without_octave().name for n in parse_notes(item)}
    m = next((m for m in res["matches"] if set(m["notes"]) == written), res["matches"][0])
    bass = parse_note(m["bass"]) if "bass" in m else None
    return parse_note(m["root"]), resolve_chord_type(m["chord_type"]), bass


# How the figure grammar writes a triad's or seventh chord's quality in front of
# its figure — roman_to_chords' normalized form ('V65', 'viiø43', 'IΔ42', 'vii°7').
_FIGURE_MARKS = {
    "major": "", "minor": "", "diminished": "°", "augmented": "+",
    "dominant 7": "", "minor 7": "", "major 7": "Δ", "minor major 7": "Δ",
    "diminished 7": "°", "half-diminished": "ø", "augmented 7": "+", "augmented major 7": "+Δ",
}
# The figure roman_to_chords gives each augmented sixth.
_AUG_SIXTH_FIGURES = {"It6": "6", "Fr43": "43", "Ger65": "65", "Sw43": "43"}


def _chord_items(chords) -> list:
    """The `chords` argument as a list (a string splits on commas and whitespace)."""
    items = chords
    if isinstance(items, str):
        items = [t for t in items.replace(",", " ").split() if t]
    if not isinstance(items, (list, tuple)) or not items:
        raise ValueError("chords must be a non-empty list of chord symbols or note arrays")
    for item in items:
        if not isinstance(item, (str, list, tuple)):
            raise ValueError(f"Invalid chord: {item!r} (use a symbol like 'G7/B' or a note array)")
    return list(items)


def _numeral(croot: Note, tonic: Note, intervals: tuple[int, ...]) -> tuple[str, str]:
    """(numeral, accidental) of a chord root as analyze_progression's `roman` writes it."""
    rel = (croot.pitch_class - tonic.pitch_class) % 12
    if rel in intervals and len(intervals) == 7:
        # a scale degree keeps the scale's own numeral (Db in C# major is I, like diatonic_chords)
        idx = intervals.index(rel)
        offset = intervals[idx] - MAJOR_DEGREES[idx]
        return _ROMAN_BASE[idx], "#" * offset if offset > 0 else "b" * -offset
    return _numeral_base(croot, tonic)  # a chromatic root: its spelling decides (bV vs #IV)


def _dialect_prefix(base: str, accidental: str, croot: Note, tonic: Note, scale,
                    minor_third: bool) -> str | None:
    """The accidental roman_to_chords needs in front of `base` to read this root back.

    Outside natural, harmonic and melodic minor it is the `roman` accidental. In
    those minor modes the textbook numerals are written: the scale's own degrees
    take no accidental (III, VI, VII) and degrees 6 and 7 follow roman_to_chords'
    reading by quality (music21 Minor67Default) — in A minor F is VI, G#dim is
    vii°, while Fm needs 'bvi'. None when no numeral can write the root: an
    accidental is measured from the parallel major and a bare numeral means the
    key's own degree, so a root on the parallel major's degree that the key
    lacks (C# in A minor, E in C dorian) has no spelling in that dialect.
    """
    n = _ROMAN_BASE.index(base) + 1
    rel = (croot.pitch_class - tonic.pitch_class) % 12
    plain = scale.intervals[n - 1]  # what a numeral without an accidental reads
    if scale.name in _MINOR_MODES and n in (6, 7):
        plain = (9 if minor_third else 8) if n == 6 else (11 if minor_third else 10)
        offset = (rel - MAJOR_DEGREES[n - 1] + 6) % 12 - 6
        prefix = "" if rel == plain else ("#" * offset if offset > 0 else "b" * -offset)
    else:
        prefix = "" if scale.name in _MINOR_MODES and rel == plain else accidental
    if prefix == "" and rel != plain:
        return None
    return prefix


def _bass_degree(bass: Note, tonic: Note, intervals: tuple[int, ...]) -> int | str | None:
    """The bass as a scale degree: 4, or a string with its accidental ('#4') off the scale."""
    rel = (bass.pitch_class - tonic.pitch_class) % 12
    if rel in intervals:
        return intervals.index(rel) + 1
    if len(intervals) != 7:
        return None
    steps = (LETTERS.index(bass.letter) - LETTERS.index(tonic.letter)) % 7
    offset = (rel - intervals[steps] + 6) % 12 - 6
    return ("#" * offset if offset > 0 else "b" * -offset) + str(steps + 1)


def analyze_progression(chords, root: str, scale_type: str = "major") -> dict:
    """Analyze a chord progression into Roman numerals relative to a key.

    The reverse of degrees_to_chords and roman_to_chords: given chords (symbols
    like ['C','Am','F','G'] or note arrays) and a key, label each with its Roman
    numeral, scale degree, whether it is diatonic to the key, and (in 7-note
    keys) its harmonic function (tonic/subdominant/dominant). A chord is
    `in_key` only when **every** chord tone (and a slash bass) belongs to the
    scale — E7 in A natural minor is flagged because of its G#, and its
    `non_scale_notes` are listed (a note array keeps its written spelling).
    Numerals follow the chord's spelling (Gb in C = bV, F# = #IV). A note
    array's bass is its lowest note when every note has an octave, otherwise its
    first note (bach_chorale_voicing's convention).

    Each chord also gets:
    - figure: the bass as a chord member (music21's shorthand): '' / '6' / '64'
      for triads, '7' / '65' / '43' / '42' for seventh chords. Other chord types
      are '' in root position; inverted ones, a sus or non-chord bass (Csus4/F)
      and unknown clusters get null plus bass_degree (4, or '#4' off the scale).
    - roman_figured: the numeral as roman_to_chords reads it back — 'I6', 'V65',
      'viiø43', 'V65/ii', 'N6', 'Ger65', 'Cad64'; in minor keys the textbook
      numerals (A minor: F = VI, G = VII, G#dim = vii°, Fm = bvi).
    - applied (only for chords not in the key): 'V<suffix>/x' for a major or
      dominant-family chord a spelled P5 above, 'vii°/x' / 'vii°7/x' / 'viiø7/x'
      for a dim / dim7 / m7b5 chord a spelled m2 below, a diatonic major or
      minor triad x other than the tonic (A7 in C = V7/ii; Kostka & Payne). In
      minor keys V and the raised-7th vii° chords get function_note
      'harmonic-minor dominant' instead.
    - special: 'Neapolitan' (a major triad on the lowered 2nd: N6, bII),
      'It6' / 'Fr43' / 'Ger65' / 'Sw43' (a written note array with b6 in the bass
      and #4 an augmented sixth above it; chord_type null) or 'Cad64' (a tonic
      6/4 followed by a major-third chord on ^5: function stays 'tonic',
      function_note 'dominant (cadential 6/4)', per Aldwell & Schachter). The
      augmented-sixth sound written as a symbol ('Ab7') keeps its numeral and
      gets enharmonic_to 'Ger65'.
    - borrowed_from (every other out-of-key chord): the parallel modes on the
      tonic whose notes hold every chord tone and the bass, nearest first — Fm
      in C: C harmonic major, C harmonic minor, C natural minor, C phrygian, C
      locrian (the rule roman_to_chords and chord_palette use; mixture per
      Aldwell & Schachter, modal interchange per Berklee/Nettles & Graf).
    Figures, applied chords and specials are ported from music21's
    romanNumeralFromChord (BSD-3). In 7-note keys roman_figured round-trips
    through roman_to_chords (IV/x, multi-level V/V/V and bII7/x chords read back
    as one applied level or as a chromatic chord); it is null for a root that
    dialect cannot write — one on the parallel major's degree that the key
    lacks, like C#m in A minor. e.g. ['C','A7','Dm','G7/B','C'] in C -> roman
    I VI7 ii V7 I, roman_figured I V7/ii ii V65 I. Deterministic.
    """
    scale = resolve_scale_type(scale_type)
    tonic = parse_notes(root)[0].without_octave()
    intervals = scale.intervals
    heptatonic = len(intervals) == 7
    scale_pcs = {(tonic.pitch_class + i) % 12 for i in intervals}
    items = _chord_items(chords)
    readings = [read_chord(item) for item in items]

    out = []
    for i, (item, r) in enumerate(zip(items, readings)):
        written = r["written"]
        if r["root"] is not None:
            croot, ctype = r["root"], r["chord_type"]
        else:  # a cluster that is no chord type: analyze it by its first written note
            croot, ctype = r["notes"][0], None
        bass = r["bass"]
        rel = (croot.pitch_class - tonic.pitch_class) % 12
        base, accidental = _numeral(croot, tonic, intervals)
        if ctype is None:
            tones = r["notes"]
            rel_pcs = {(t.pitch_class - croot.pitch_class) % 12 for t in tones}
        else:
            tones = chord_notes(ctype, croot)
            if written:  # chord-tone order, but the spelling the caller wrote (F#, not Gb)
                spelled: dict[int, Note] = {}
                for n in r["notes"]:
                    spelled.setdefault(n.pitch_class, n)
                tones = [spelled.get(t.pitch_class, t) for t in tones]
            rel_pcs = ctype.pitch_classes
        minor_third = 3 in rel_pcs and 4 not in rel_pcs
        numeral = base.lower() if minor_third else base
        suffix = roman_suffix(ctype, minor_third)
        outside = []
        for t in (bass, *tones):
            if t.pitch_class not in scale_pcs and t.pitch_class_name not in outside:
                outside.append(t.pitch_class_name)
        in_key = not outside
        if isinstance(item, str):
            symbol = item
        elif ctype is not None:
            symbol = f"{croot.pitch_class_name}{ctype.symbol}"
        else:
            symbol = " ".join(n.without_octave().name for n in parse_notes(list(item)))
        entry = {
            "symbol": symbol,
            "root": croot.pitch_class_name,
            "chord_type": ctype.name if ctype else "unknown",
            "roman": f"{accidental}{numeral}{suffix}",
            "in_key": in_key,
        }
        if rel in intervals:
            idx = intervals.index(rel)
            entry["degree"] = idx + 1
            if in_key and heptatonic:
                entry["function"] = _HARMONIC_FUNCTIONS[idx]
        if not in_key:
            entry["non_scale_notes"] = outside
            entry["note"] = "chromatic / borrowed"

        # --- figures, applied chords, specials, borrowing (the shared roman.py rules)
        figure = figure_for_bass(ctype, croot, bass) if ctype is not None else None
        prefix = _dialect_prefix(base, accidental, croot, tonic, scale, minor_third) if heptatonic else accidental
        if prefix is None:  # roman_to_chords cannot write this root in this key
            roman_figured = None
        elif figure is not None and ctype.name in _FIGURE_MARKS:
            roman_figured = f"{prefix}{numeral}{_FIGURE_MARKS[ctype.name]}{figure}"
        else:  # root position of another type, or a bass the figure grammar cannot write
            roman_figured = f"{prefix}{numeral}{suffix}"
        applied = applied_reading(r, tonic, scale.name) if heptatonic else None
        special = None
        if heptatonic:
            special = special_reading(r, tonic, scale.name, readings[i + 1] if i + 1 < len(readings) else None)
        special_name = special.get("special") if special else None
        if special_name is not None:
            entry["special"] = special_name
            if special_name in _AUG_SIXTH_FIGURES:
                entry["chord_type"] = None
                figure = _AUG_SIXTH_FIGURES[special_name]
                roman_figured = special["roman_figured"]
            elif special_name == "Cad64":
                # 'Cad64' reads back as the key's own tonic triad; a borrowed one keeps its numeral
                if in_key and ctype.name == ("major" if intervals[2] == 4 else "minor"):
                    roman_figured = special["roman_figured"]
            else:
                roman_figured = special["roman_figured"]
            if "function_note" in special:
                entry["function_note"] = special["function_note"]
        elif special is not None and "enharmonic_to" in special:
            entry["enharmonic_to"] = special["enharmonic_to"]
        if special_name is None and applied is not None:
            if applied.get("applied"):
                entry["applied"] = applied["applied"]
                if figure is not None and ctype.name in _FIGURE_MARKS:
                    head = ("V" if applied["type"] == "dominant" else "vii") + _FIGURE_MARKS[ctype.name] + figure
                else:
                    head = applied["numeral"]
                roman_figured = f"{head}/{applied['target']}"
            else:
                entry["function_note"] = applied["function_note"]
        entry["figure"] = figure
        entry["roman_figured"] = roman_figured
        if figure is None:
            entry["bass_degree"] = _bass_degree(bass, tonic, intervals)
        if not in_key and special_name is None and applied is None:
            entry["borrowed_from"] = borrowed_sources([bass, *tones], tonic, scale.name)
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


# ------------------------------------------------------------- next chords

# Walter Piston, Harmony (1941), ch. 3, "Table of Usual Root Progressions", in its
# original-text wording: Piston writes every degree as a capital numeral and gives
# one table for major and minor. Each row is (degree, usually, sometimes, less
# often) — "VI is followed by II or V, sometimes III or IV, less often I". Mark
# DeVoto's revised table (5th edition, 1987) differs; this is the 1941 one. It is
# a table of rules, not corpus statistics: which progression to use is the caller's.
PISTON_1941 = (
    ("I", ("IV", "V"), ("VI",), ("II", "III")),
    ("II", ("V",), ("IV", "VI"), ("I", "III")),
    ("III", ("VI",), ("IV",), ("I", "II", "V")),
    ("IV", ("V",), ("I", "II"), ("III", "VI")),
    ("V", ("I",), ("IV", "VI"), ("II", "III")),
    ("VI", ("II", "V"), ("III", "IV"), ("I",)),
    ("VII", ("III",), ("I",), ()),
)
PISTON_TABLE = "Piston 1941, ch. 3"

_TIER_NAMES = ("resolution", "usual", "sometimes", "less often", "applied", "mixture", "unlisted")
_PISTON_WORDS = {1: "usually", 2: "sometimes", 3: "less often"}

# Mode mixture in a major key, from the parallel minor (Aldwell & Schachter; Kostka
# & Payne list vii°7 too). With sevenths the seventh forms replace ii° and iv.
_MIXTURE = ("ii°", "bIII", "iv", "bVI")
_MIXTURE_SEVENTHS = ("iiø7", "bIII", "iv7", "bVI", "vii°7")


def _or_list(numerals) -> str:
    """List numerals the way Piston's table does: 'II or V', 'I, II or V'."""
    items = list(numerals)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " or " + items[-1]


def piston_sentence(row: tuple) -> str:
    """One row of PISTON_1941 as Piston's sentence ('VII is followed by III, sometimes I.')."""
    text = f"{row[0]} is followed by {_or_list(row[1])}"
    if row[2]:
        text += f", sometimes {_or_list(row[2])}"
    if row[3]:
        text += f", less often {_or_list(row[3])}"
    return text + "."


def _piston_rule(row: tuple, tier: int) -> str:
    return f"Piston 1941: {row[0]} is {_PISTON_WORDS[tier]} followed by {_or_list(row[tier])}"


def _degree_of(root: Note, tonic: Note, intervals: tuple[int, ...]) -> int:
    """The numeral degree (1-7) analyze_progression gives a root (bVI -> 6, #iv° -> 4)."""
    return _ROMAN_BASE.index(_numeral(root, tonic, intervals)[0]) + 1


def next_chords(chords, root: str, scale_type: str = "major", sevenths: bool = False,
                include_chromatic: bool = True, sort: str = "rule", octave: int = 4,
                limit: int = 16) -> dict:
    """Rank the chords that may follow the LAST chord of `chords`, by a fixed rule table.

    Never by probability: the ranking is Walter Piston's "Table of Usual Root
    Progressions" (Harmony, 1941, ch. 3; the same table for major and minor —
    Mark DeVoto's 1987 revision differs), a table of rules, not corpus
    statistics, so the choice stays the caller's. The last chord is read the way
    analyze_progression reads it; its numeral degree picks the table row (bVI ->
    VI, #iv° -> IV: a chromatic last chord uses the row of its root letter, and
    `last.note` says so). The key must have 7 notes. Tiers:
    0 resolution — an applied chord's target first (A7 -> Dm in C);
    1 usual, 2 sometimes, 3 less often — the diatonic chords (triads, or
      sevenths with `sevenths`) on the degrees the row lists; in natural minor
      the harmonic-minor V and vii° (V7, vii°7) are added beside v and VII;
    4 applied — V7/x for each diatonic major or minor target x (not I) in the
      row's usual then sometimes lists, in the row's order (Kostka & Payne);
    5 mixture — major keys only (empty in minor and modal keys): Aldwell &
      Schachter's parallel-minor chords ii°, bIII, iv, bVI (with sevenths iiø7,
      bIII, iv7, bVI and vii°7) plus bVII (bVII7), which is the pop/jazz
      modal-interchange convention (Berklee), not A&S; ordered by the table
      tier of their degree;
    6 unlisted — a mixture chord whose degree the row does not list (I -> bVII).
    Each candidate: symbol, token (roman_to_chords' dialect; it round-trips),
    roman (analyze_progression's), notes, tier, tier_name, rule (the table row
    it comes from), common_tones with the last chord, root_motion (Schoenberg's
    classes, as schoenberg_progressions: 'ascending (strong)', 'descending',
    'superstrong'), movement (total semitones from the last chord's voicing to
    the candidate's best inversion — voice_leading's own search over the whole
    input, at `octave`) and in_key. sort='rule' orders by tier (within a tier:
    most common tones, then degree, then token; tier 4 keeps the row's order;
    tier 5 goes by the degree's table tier first); sort='movement' orders by
    (movement, tier, token), smoothest first (Scaler's Minimize Movement idea).
    A chord listed twice keeps its best entry; `limit` (1-64) cuts the list.
    Returns {key, last {symbol, roman, degree}, table, candidates, symbols,
    tokens}: feed a token to roman_to_chords or a symbol straight to
    voice_leading / chords_to_midi, and call again to grow a progression.
    e.g. next_chords(['C','Am'], 'C') -> Dm G (usual) F Em (sometimes) C (less
    often), A7 D7 B7 C7 (applied), Ddim Fm Eb (mixture), Ab Bb (unlisted).
    Deterministic.
    """
    if sort not in ("rule", "movement"):
        raise ValueError(f"sort must be 'rule' or 'movement', got {sort!r}")
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 64:
        raise ValueError(f"limit must be an integer between 1 and 64, got {limit!r}")
    for name, flag in (("sevenths", sevenths), ("include_chromatic", include_chromatic)):
        if not isinstance(flag, bool):
            raise ValueError(f"{name} must be true or false, got {flag!r}")
    if not isinstance(octave, int) or isinstance(octave, bool) or not 0 <= octave <= 9:
        raise ValueError(f"octave must be an integer between 0 and 9, got {octave!r}")
    scale = resolve_scale_type(scale_type)
    intervals = scale.intervals
    if len(intervals) != 7:
        raise ValueError(f"next_chords needs a 7-note key (Piston's table is for major and minor);"
                         f" {scale.name!r} has {len(intervals)} notes")
    tonic = parse_notes(root)[0].without_octave()
    key = tonic.name
    items = _chord_items(chords)

    analysis = analyze_progression(items, key, scale.name)
    last = analysis["chords"][-1]
    last_reading = read_chord(items[-1])
    last_root = parse_note(last["root"])
    degree = _degree_of(last_root, tonic, intervals)
    row = PISTON_1941[degree - 1]
    row_tiers = {_ROMAN_BASE.index(n) + 1: tier for tier in (1, 2, 3) for n in row[tier]}
    last_pcs = {n.pitch_class for n in last_reading["notes"]} | {last_reading["bass"].pitch_class}

    raw: list[dict] = []
    diatonic = diatonic_chords(key, scale.name, sevenths)["chords"]
    applied = applied_reading(last_reading, tonic, scale.name)
    if applied is not None and applied.get("applied"):
        raw.append({"symbol": diatonic[applied["target_degree"] - 1]["symbol"], "tier": 0,
                    "rule": "applied chord resolves to its target"})
    for deg, tier in row_tiers.items():
        raw.append({"symbol": diatonic[deg - 1]["symbol"], "tier": tier, "rule": _piston_rule(row, tier)})
    if scale.name == "natural minor":  # the minor mode's own dominant, with the raised leading tone
        raised = diatonic_chords(key, "harmonic minor", sevenths)["chords"]
        for deg in (5, 7):
            if deg in row_tiers:
                raw.append({"symbol": raised[deg - 1]["symbol"], "tier": row_tiers[deg],
                            "rule": f"{_piston_rule(row, row_tiers[deg])}; raised leading tone"
                                    f" (minor-mode dominant)"})
    if include_chromatic:
        triads = diatonic_chords(key, scale.name)["chords"]
        order = 0
        for tier in (1, 2):
            for numeral in row[tier]:
                x = triads[_ROMAN_BASE.index(numeral)]
                if x["degree"] == 1 or x["chord_type"] not in ("major", "minor"):
                    continue
                target = re.match(r"^[#b]*[IViv]+", x["roman"]).group(0)  # as applied_reading names it
                token = f"V7/{target}"
                raw.append({"symbol": roman_to_chords([token], key, scale.name)["symbols"][0], "tier": 4,
                            "token": token, "order": order,
                            "rule": f"applied V7 of {target} (Kostka & Payne); {_piston_rule(row, tier)}"})
                order += 1
        if scale.name == "major":
            tokens = list(_MIXTURE_SEVENTHS if sevenths else _MIXTURE) + ["bVII7" if sevenths else "bVII"]
            for token, ch in zip(tokens, roman_to_chords(tokens, key, scale.name)["chords"]):
                deg = _degree_of(parse_note(ch["root"]), tonic, intervals)
                if token.startswith("bVII"):
                    source = "bVII by the pop/jazz modal-interchange convention (Berklee: Nettles & Graf 1997)"
                elif token.startswith("vii"):
                    source = "mixture vii°7 from the parallel minor (Kostka & Payne; Aldwell & Schachter)"
                else:
                    source = "mixture from the parallel minor (Aldwell & Schachter)"
                ptier = row_tiers.get(deg)
                if ptier is None:
                    raw.append({"symbol": ch["symbol"], "tier": 6, "token": token,
                                "rule": f"{source}; Piston 1941 lists no {_ROMAN_BASE[deg - 1]} after {row[0]}"})
                else:
                    raw.append({"symbol": ch["symbol"], "tier": 5, "token": token, "ptier": ptier,
                                "rule": f"{source}; {_piston_rule(row, ptier)}"})

    from .masters import schoenberg_progressions  # masters imports this module at load time

    motions: dict[int, str] = {}
    candidates = []
    for c in raw:
        info = analyze_progression([c["symbol"]], key, scale.name)["chords"][0]
        reading = read_chord(c["symbol"])
        croot = parse_note(info["root"])
        pcs = {n.pitch_class for n in reading["notes"]} | {reading["bass"].pitch_class}
        if croot.pitch_class not in motions:
            steps = schoenberg_progressions([last_root.name, croot.name])["progressions"]
            motions[croot.pitch_class] = steps[0]["class"]
        voiced = voice_leading(items + [c["symbol"]], octave)["voicings"]
        candidates.append({
            "symbol": c["symbol"],
            "token": c.get("token", info["roman_figured"]),
            "roman": info["roman"],
            "notes": [n.pitch_class_name for n in reading["notes"]],
            "tier": c["tier"],
            "tier_name": _TIER_NAMES[c["tier"]],
            "rule": c["rule"],
            "common_tones": len(pcs & last_pcs),
            "root_motion": motions[croot.pitch_class],
            "movement": _voicing_cost(voiced[-2]["midi"], voiced[-1]["midi"]),
            "in_key": info["in_key"],
            "_key": (croot.pitch_class, frozenset(pcs)),
            "_degree": _degree_of(croot, tonic, intervals),
            "_order": c.get("order", 0),
            "_ptier": c.get("ptier", 0),
        })

    def rule_order(c):
        if c["tier"] == 4:
            return (4, c["_order"])
        if c["tier"] == 5:
            return (5, c["_ptier"], -c["common_tones"], c["_degree"], c["token"])
        return (c["tier"], -c["common_tones"], c["_degree"], c["token"])

    ranked, seen = [], set()
    for c in sorted(candidates, key=rule_order):
        if c["_key"] not in seen:  # the same chord twice keeps its best entry
            seen.add(c["_key"])
            ranked.append(c)
    if sort == "movement":
        ranked.sort(key=lambda c: (c["movement"], c["tier"], c["token"]))
    ranked = [{k: v for k, v in c.items() if not k.startswith("_")} for c in ranked[:limit]]

    last_out = {"symbol": last["symbol"], "roman": last["roman"], "degree": degree}
    if not last["in_key"]:
        last_out["note"] = "row chosen by root letter"
    return {
        "key": analysis["key"],
        "last": last_out,
        "table": PISTON_TABLE,
        "candidates": ranked,
        "symbols": [c["symbol"] for c in ranked],
        "tokens": [c["token"] for c in ranked],
    }
