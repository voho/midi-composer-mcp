"""Accompaniment layer: chord-relative figures and bass lines over a progression.

chord_pattern performs a figure (Alberti bass, broken chords, a scale run) over
every chord of a progression; bass_line writes a bass part (root, root-fifth,
root-octave, chromatic approach, walking, pedal) from the same chords. Both are
fixed rules — no chance, no taste — and both return a ready notes track plus a
``render_hint`` for arrange_to_midi / arrange_song.

Chords are read the way every other tool reads them: symbols through
midi_io._parse_chord_list and roman.read_chord (a note array's bass is its
lowest MIDI note when every note carries an octave, otherwise its first note;
its root is the exact match_chords reading over that bass, or none for a set
that is no table chord).

The mechanism of chord_pattern follows the Scaler 3 User Guide (Motions: a
chord-relative pattern re-mapped onto every chord; Play Mode Chord vs Scale;
Perform Mode Retrigger vs Follow). Its presets are public-domain keyboard
idioms: the Alberti bass (Domenico Alberti, 1730s) and the murky bass
(18th-century broken octaves). bass_line's root, root-fifth, root-octave and
approach patterns are standard bass idioms; its walking rule (chord tones on
the beats, a chromatic approach into the next chord on the last beat) is a
simplified codification of walking-bass pedagogy (e.g. Ed Friedland, Building
Walking Bass Lines, Hal Leonard 1995). A pedal point is the common-practice
held (or repeated) bass under changing harmony.

This module imports harmony, midi_io and roman at load time; none of them
imports it, so there is no cycle.
"""

from __future__ import annotations

import re

from .chords import chord_notes
from .generate import RHYTHM_REST, RHYTHM_STRONG, RHYTHM_WEAK, parse_rhythm
from .harmony import voice_leading
from .midi_io import _check_midi, _check_range, _parse_chord_list, _whole_steps, _with_octave, voice_chord
from .notes import LETTERS, Note, parse_note, transpose
from .roman import read_chord
from .scales import resolve_scale_type, scale_notes

# ------------------------------------------------------------------ shared


def _chord_items(chords) -> list:
    """Split the `chords` argument into items the way midi_io._parse_chord_list does."""
    if isinstance(chords, str):
        return [t for t in re.split(r"[,\s]+", chords.strip()) if t]
    if isinstance(chords, tuple):
        return list(chords)
    return chords


def _choice(value, options: tuple[str, ...], label: str) -> str:
    if not isinstance(value, str) or value.strip().lower() not in options:
        raise ValueError(f"{label} must be one of {options}, got {value!r}")
    return value.strip().lower()


def _check_bool(value, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be true or false, got {value!r}")
    return value


def _chord_beats(beats_per_chord, count: int) -> tuple[list[float], bool]:
    """One beat length per chord: (lengths, given_as_list)."""
    if isinstance(beats_per_chord, (list, tuple)):
        if len(beats_per_chord) != count:
            raise ValueError(
                f"beats_per_chord has {len(beats_per_chord)} entries but there are {count} chords;"
                f" give one number, or exactly one beat length per chord"
            )
        return [float(_check_range(f"beats_per_chord[{i}]", b, 0.25, 64))
                for i, b in enumerate(beats_per_chord)], True
    return [float(_check_range("beats_per_chord", beats_per_chord, 0.25, 64))] * count, False


def _chord_steps(beats: list[float], step_beats: float) -> list[int]:
    steps = []
    for i, b in enumerate(beats):
        s = _whole_steps(b, step_beats)
        if s is None:
            raise ValueError(
                f"chord {i + 1} lasts {b:g} beats, which is not a whole number of"
                f" {step_beats:g}-beat steps; change beats_per_chord or step_beats"
            )
        steps.append(s)
    return steps


def _symbol(parsed: dict, notes: list[Note]) -> str:
    return parsed["symbol"] or " ".join(n.pitch_class_name for n in notes)


# ------------------------------------------------------------ chord_pattern

PATTERN_PRESETS: dict[str, str] = {
    "alberti": "^1 3 2 3",   # Domenico Alberti's broken-chord bass: low, high, middle, high
    "up": "^1 2 3 4",
    "down": "^4 3 2 1",
    "updown": "^1 2 3 4 3 2",
    "murky": "^1, 1",        # murky bass: broken octaves
}
PATTERN_MODES = ("chord", "scale")
PATTERN_PHASES = ("restart", "continue")
_STEP_RE = re.compile(r"^(\^?)(\d+)(['’,]*)$")


def _parse_pattern(pattern) -> tuple[str, list[tuple[int, bool, int] | None], str | None]:
    """A preset name or space-separated steps -> (steps text, [(index, accent, octaves) | None], preset)."""
    if isinstance(pattern, (list, tuple)):
        if not pattern or not all(isinstance(t, str) for t in pattern):
            raise ValueError("pattern must be a preset name or steps like '^1 3 2 3' (a string)")
        pattern = " ".join(pattern)
    if not isinstance(pattern, str) or not pattern.strip():
        raise ValueError(
            f"pattern must be a preset ({', '.join(PATTERN_PRESETS)}) or space-separated steps"
            f" like '^1 3 2 3', got {pattern!r}"
        )
    preset = pattern.strip().lower()
    text = PATTERN_PRESETS.get(preset)
    if text is None:
        preset = None
        text = " ".join(pattern.split())
    steps: list[tuple[int, bool, int] | None] = []
    for token in text.split():
        if token == RHYTHM_REST:
            steps.append(None)
            continue
        m = _STEP_RE.match(token)
        if not m:
            raise ValueError(
                f"Invalid pattern step {token!r}: a step is '.' (rest) or a chord/scale index"
                f" such as '1', '^3' (accented), \"1'\" (an octave up) or '1,' (an octave down);"
                f" presets: {', '.join(PATTERN_PRESETS)}"
            )
        accent, digits, marks = m.groups()
        index = int(digits)
        if index < 1:
            raise ValueError(f"Invalid pattern step {token!r}: indices are 1-based (1 = the bottom note)")
        if index > 99:
            raise ValueError(f"Invalid pattern step {token!r}: indices run 1-99")
        octaves = sum(1 for c in marks if c in "'’") - marks.count(",")
        steps.append((index, bool(accent), octaves))
    if all(s is None for s in steps):
        raise ValueError("pattern has no note steps (only rests); give at least one index")
    return text, steps, preset


def _scale_base(voicing: list[Note], root: Note | None) -> Note:
    """The lowest voicing tone of the chord root's pitch class, else the lowest tone."""
    if root is not None:
        for n in voicing:
            if n.pitch_class == root.pitch_class:
                return n
    return voicing[0]


def chord_pattern(chords, pattern="alberti", beats_per_chord=4.0, step_beats: float = 0.5,
                  mode: str = "chord", phase: str = "restart", root: str | None = None,
                  scale_type: str = "major", octave: int = 4, smooth: bool = True,
                  sustain: bool = False) -> dict:
    """Perform a chord-relative figure over a whole progression -> one notes track.

    PATTERN: a preset or space-separated steps. A step is '.' (rest; with
    `sustain` a hold of the previous note — only within its chord: a sustained
    pattern that would hold a note over a rest opening the next chord is a
    ValueError) or [^]INDEX['|,]*: INDEX is 1-based, '^' accents the step ('O' in
    the rhythm, else 'o'), each ' raises it an octave and each , lowers it one
    (ABC-notation marks). Presets: alberti '^1 3 2 3', up '^1 2 3 4', down
    '^4 3 2 1', updown '^1 2 3 4 3 2', murky "^1, 1" (broken octaves).

    REFERENCE VOICING: smooth=True voices the chords with voice_leading(chords,
    octave); smooth=False with midi_io.voice_chord (root position, slash bass
    lowest), keeping note arrays that carry octaves exactly as written (so a
    voice_chords voicing can drive the pattern). Tones t[0..n-1] run bottom to top.

    MODES (Scaler's Play Mode): 'chord' maps INDEX k to t[(k-1) % n], raised
    12*((k-1)//n) semitones (index 4 of a triad is its bottom note an octave up).
    'scale' (needs `root`/`scale_type`) maps k to the (k-1)-th scale note above
    the base (base = step 1), spelled as the scale spells it; the base is the
    lowest voicing tone of the chord root's pitch class, or the lowest tone when
    the voicing has none (a rootless voicing that names no chord). A chord whose
    base is not in the scale falls back to chord mode (per_chord 'fallback').

    TIMING: each chord lasts `beats_per_chord` (one number, or a list with one
    per chord); beats / step_beats must be a whole number of steps. phase
    'restart' (Scaler Retrigger) restarts the pattern on every chord, 'continue'
    (Follow) runs it on across chord changes. Every pitch must be MIDI 0-127.

    Returns {pattern, mode, phase, steps_per_chord, total_beats, track (notes
    with octaves, one per onset; rhythm one character per step), per_chord
    [{symbol, voicing, notes, fallback?}], render_hint (a chords track of the
    voicings + the pattern track, for arrange_to_midi)}.
    e.g. chord_pattern(['C','Am','F','G'], 'alberti', beats_per_chord=2) ->
    C4 G4 E4 G4 | C4 A4 E4 A4 | C4 A4 F4 A4 | D4 B4 G4 B4, rhythm 'Oooo' per chord.
    """
    items = _chord_items(chords)
    parsed = _parse_chord_list(items)
    text, steps, preset = _parse_pattern(pattern)
    mode = _choice(mode, PATTERN_MODES, "mode")
    phase = _choice(phase, PATTERN_PHASES, "phase")
    step = float(_check_range("step_beats", step_beats, 0.0625, 16))
    beats, beats_listed = _chord_beats(beats_per_chord, len(parsed))
    per_chord_steps = _chord_steps(beats, step)
    _check_range("octave", octave, 0, 9, integer=True)
    _check_bool(smooth, "smooth")
    _check_bool(sustain, "sustain")

    scale = resolve_scale_type(scale_type)
    key_notes = key_root = None
    if root is not None:
        if not isinstance(root, str):
            raise ValueError(f"root must be a note name like 'C' or 'F#', got {root!r}")
        key_root = parse_note(root).without_octave()
        key_notes = scale_notes(scale, key_root)[:-1]
    elif mode == "scale":
        raise ValueError("mode='scale' needs a key: pass root (and scale_type), e.g. root='C'")

    if smooth:
        voicings = [[parse_note(n) for n in v["notes"]] for v in voice_leading(items, octave)["voicings"]]
    else:
        voicings = [voice_chord(p["tones"], octave, p["bass"]) for p in parsed]
    voicings = [sorted(v, key=lambda n: n.midi) for v in voicings]  # bottom to top
    if mode == "scale":
        roots = [read_chord(item)["root"] for item in items]
        key_pcs = [n.pitch_class for n in key_notes]

    notes: list[Note] = []
    rhythm: list[str] = []
    per_chord: list[dict] = []
    held_from: int | None = None  # the chord (0-based) whose note a '.' would currently hold
    g = 0  # global step index, for phase='continue'
    for i, (p, voiced, s) in enumerate(zip(parsed, voicings, per_chord_steps)):
        entry = {"symbol": _symbol(p, voiced), "voicing": [n.name for n in voiced], "notes": []}
        base = None
        if mode == "scale":
            base = _scale_base(voiced, roots[i])
            if base.pitch_class not in key_pcs:
                base = None
                entry["fallback"] = "chord"
        for j in range(s):
            spec = steps[(g if phase == "continue" else j) % len(steps)]
            g += 1
            if spec is None:
                if sustain and held_from is not None and held_from != i:
                    raise ValueError(
                        f"sustain would hold the last note of chord {held_from + 1}"
                        f" ({per_chord[held_from]['symbol']}) over the rest that opens chord {i + 1}"
                        f" ({entry['symbol']}), so it would sound under the wrong chord; make every"
                        f" chord open on an index step (start the pattern on one, or change its"
                        f" length or the phase), or set sustain=false"
                    )
                rhythm.append(RHYTHM_REST)
                continue
            index, accent, shift = spec
            if base is None:  # chord mode (or a fallback chord)
                wrap, pos = divmod(index - 1, len(voiced))
                tone = voiced[pos]
                note = Note(tone.letter, tone.accidental, tone.octave + wrap + shift)
            else:
                start = key_pcs.index(base.pitch_class)
                wrap, pos = divmod(start + index - 1, len(key_notes))
                semis = scale.intervals[pos] + 12 * wrap - scale.intervals[start]
                spelled = _with_octave(key_notes[pos], base.midi + semis)
                note = Note(spelled.letter, spelled.accidental, spelled.octave + shift)
            if not 0 <= note.midi <= 127:
                raise ValueError(
                    f"pattern step {index} over chord {i + 1} ({entry['symbol']}) gives {note.name},"
                    f" outside the MIDI range 0-127; lower the octave or the index"
                )
            notes.append(note)
            held_from = i
            entry["notes"].append(note.name)
            rhythm.append(RHYTHM_STRONG if accent else RHYTHM_WEAK)
        per_chord.append(entry)
    if not notes:
        raise ValueError("the pattern produces no notes over these chords; give it at least one index step")
    pattern_rhythm = parse_rhythm("".join(rhythm))

    total = sum(beats)
    track = {"type": "notes", "name": "pattern", "notes": [n.name for n in notes],
             "rhythm": pattern_rhythm, "step_beats": step, "sustain": sustain}
    chords_track = {"type": "chords", "name": "chords", "chords": [e["voicing"] for e in per_chord]}
    if beats_listed:
        chords_track["durations"] = beats
    else:
        chords_track["beats_per_chord"] = beats[0]
    result = {
        "pattern": text,
        "mode": mode,
        "phase": phase,
        "steps_per_chord": per_chord_steps,
        "total_beats": total,
        "track": track,
        "per_chord": per_chord,
        "render_hint": {"tracks": [chords_track, track]},
    }
    if preset:
        result["preset"] = preset
    if mode == "scale":
        result["key"] = f"{key_root.name} {scale.name}"
    return result


# ---------------------------------------------------------------- bass_line

BASS_STYLES = ("root", "root_fifth", "root_octave", "approach", "walking", "pedal")
BASS_ENDINGS = ("loop", "root")
BASS_LOW, BASS_HIGH = 28, 55  # E1-G3: open E to the 12th fret of the G string
BASS_PROGRAM = 33             # General MIDI Electric Bass (finger), 0-based
_SUSTAINED = {"root": True, "approach": True, "pedal": True,
              "root_fifth": False, "root_octave": False, "walking": False}


def _fold(note: Note) -> Note:
    """Move a note by octaves into E1-G3 (MIDI 28-55), keeping its spelling."""
    midi = note.midi
    while midi < BASS_LOW:
        midi += 12
    while midi > BASS_HIGH:
        midi -= 12
    return _with_octave(note, midi)


def _nearest(tone: Note, ref: int) -> Note:
    """`tone` at the pitch nearest MIDI `ref` (a tritone tie goes lower), folded."""
    up = (tone.pitch_class - ref) % 12
    midi = ref + up if up < 6 else ref + up - 12
    return _fold(_with_octave(tone, midi))


def _above(tone: Note, prev: Note) -> int:
    return prev.midi + ((tone.pitch_class - prev.pitch_class) % 12 or 12)


def _below(tone: Note, prev: Note) -> int:
    return prev.midi - ((prev.pitch_class - tone.pitch_class) % 12 or 12)


def _approach_sides(target: Note, prev: Note) -> tuple[Note, Note]:
    """(preferred, other) chromatic neighbours of `target`: from below (target-1 on
    the letter below: G -> F#, C -> B, Eb -> D) when the target is above `prev`,
    else from above (target+1 on the letter above: C -> Db)."""
    low = transpose(target, -1, -1)
    high = transpose(target, 1, 1)
    return (low, high) if target.midi > prev.midi else (high, low)


def _approach(target: Note, prev: Note) -> Note:
    """A chromatic approach into `target` (already placed nearest `prev` and folded).

    The preferred side of _approach_sides; if that pitch repeats the previous
    note, the other side is used. The result is folded into E1-G3; the next
    chord's bass is then placed nearest this note, so the approach always
    resolves by a semitone.
    """
    first, second = _approach_sides(target, prev)
    return _fold(second if first.midi == prev.midi else first)


def _seam_approach(target: Note, prev: Note) -> Note:
    """The loop's closing approach into `target`, the line's fixed first note.

    Unlike _approach the target cannot be re-placed near the approach, so the
    approach is never folded: the preferred side of _approach_sides, else the
    other side when the preferred pitch repeats `prev` or leaves E1-G3. When
    neither side is free (target E1 after F1, or G3 after F#3) the in-range
    neighbour is kept even though it repeats `prev`: it still resolves into the
    first note by a semitone.
    """
    first, second = _approach_sides(target, prev)

    def free(n: Note) -> bool:
        return BASS_LOW <= n.midi <= BASS_HIGH and n.midi != prev.midi

    if free(first):
        return first
    if free(second):
        return second
    return first if BASS_LOW <= first.midi <= BASS_HIGH else second


def _approach_to(prev: Note, nxt: Note | None, seam: Note | None) -> Note:
    """The approach after `prev`: into the placed loop start `seam`, else into `nxt`
    placed nearest `prev`."""
    if seam is not None:
        return _seam_approach(seam, prev)
    return _approach(_nearest(nxt, prev.midi), prev)


def _bass_reading(item, parsed: dict) -> dict:
    """{bass, root, ring (chord tones in chord order, root first), fifth, symbol} of one chord."""
    r = read_chord(item)
    ctype = r["chord_type"]
    if ctype is not None:
        root = r["root"]
        ring = chord_notes(ctype, root)
        fifth = next((t for (label, _), t in zip(ctype.degrees, ring) if label.lstrip("#b") == "5"), root)
    else:  # a set that is no table chord: read it upward from its bass
        root = r["bass"]
        others = [n for n in r["notes"] if n != root]
        ring = [root] + sorted(others, key=lambda n: (n.pitch_class - root.pitch_class) % 12)
        fifth = next((t for t in ring[1:]
                      if (LETTERS.index(t.letter) - LETTERS.index(root.letter)) % 7 == 4
                      and (t.pitch_class - root.pitch_class) % 12 in (6, 7, 8)), root)
    symbol = parsed["symbol"] or " ".join(n.name for n in r["notes"])
    return {"bass": r["bass"], "root": root, "ring": ring, "fifth": fifth, "symbol": symbol}


def _default_rhythm(style: str, s: int) -> str:
    if style in ("root", "pedal") or (style == "approach" and s < 2):
        return RHYTHM_STRONG + RHYTHM_REST * (s - 1)
    if style == "approach":
        return RHYTHM_STRONG + RHYTHM_REST * (s - 2) + RHYTHM_WEAK
    if style == "root_fifth":
        return ("O.o." * s)[:s]
    if style == "root_octave":
        return ("Oo" * s)[:s]
    return RHYTHM_STRONG + RHYTHM_WEAK * (s - 1)  # walking


class _MiddleTones:
    """Walking-bass middle tones of one chord.

    Ascending: the chord tones above the root in chord order (3rd, 5th, 7th —
    or the octave for a triad — 3rd ...), each nearest above the previous note;
    a tone repeating the previous note's pitch class is skipped (so C/E walks
    E G C, not E E'). Once a tone would pass G3 the rest descend: from the chord
    tone nearest below the previous note, down the chord (5th, 3rd, root ...).
    """

    def __init__(self, ring: list[Note]):
        self.ring = ring
        self.cycle = ring[1:] + ([ring[0]] if len(ring) <= 3 else [])
        self.index = 0
        self.down: int | None = None

    def next(self, prev: Note) -> Note:
        if self.down is None:
            for _ in range(len(self.cycle) - 1):
                if self.cycle[self.index % len(self.cycle)].pitch_class != prev.pitch_class:
                    break
                self.index += 1
            tone = self.cycle[self.index % len(self.cycle)]
            midi = _above(tone, prev)
            if midi <= BASS_HIGH:
                self.index += 1
                return _with_octave(tone, midi)
            desc = self.ring[::-1]
            self.down = min(range(len(desc)), key=lambda j: prev.midi - _below(desc[j], prev))
        desc = self.ring[::-1]
        tone = desc[self.down % len(desc)]
        self.down += 1
        return _fold(_with_octave(tone, _below(tone, prev)))


def bass_line(chords, style: str = "root", beats_per_chord=4.0, step_beats: float = 1.0,
              rhythm: str | None = None, octave: int = 2, pedal: str | None = None,
              ending: str = "loop", sustain: bool | None = None) -> dict:
    """Write a bass track from chords by fixed rules -> one notes track (GM finger bass).

    BASS NOTE: the slash bass, else the root; for a note array its lowest note
    (its first note when the notes carry no octaves). REGISTER: every note is
    folded by octaves into E1-G3 (MIDI 28-55), the practical walking register
    from open E to the 12th fret of the G string. PLACEMENT: the first note is
    the first bass in `octave`; each later chord's bass goes to the pitch nearest
    the previous sounding note (a tritone tie goes lower) — for root_fifth and
    root_octave nearest the previous chord's bass, since their second note only
    decorates it. Approach targets are placed nearest the previous note.

    STEPS: each chord lasts `beats_per_chord` (a number or one per chord);
    beats / step_beats must be whole. `rhythm` (O/o/.) must be one chord span
    long; it repeats on every chord and its onsets take the style's notes in order.

    STYLES (default rhythm per chord of s steps):
    - root: the bass on every onset; 'O' + '.'*(s-1), held.
    - root_fifth: bass, then the chord's own fifth (P5/b5/#5, else the root)
      nearest below it, alternating — the country/polka two-beat; 'O.o.' tiled.
      Over a chord whose bass already is its fifth (a 6/4: C/G, G7/D) the
      alternate is the root instead; an alternate that folds onto the bass's
      own pitch is played an octave above it.
    - root_octave: bass and the same note an octave up, alternating; 'Oo' tiled.
      A bass from Ab2 up (whose octave would pass G3) is played an octave lower,
      so the octave always sounds.
    - approach: the bass, then on the last onset a chromatic approach into the
      next chord's bass — from below when it lies above (G -> F#, Eb -> D), else
      from above (C -> Db); if that pitch repeats the previous note, from the
      other side. 'O' + '.'*(s-2) + 'o', held.
    - walking: onset 1 the bass, middle onsets chord tones above the root in
      chord order (3rd, 5th, 7th or octave ...) each nearest above — descending
      (5th, 3rd, root ...) once a tone would pass G3 — and the last onset the
      approach into the next chord; 'O' + 'o'*(s-1). A simplified codification of
      walking-bass pedagogy (Friedland, Building Walking Bass Lines, 1995).
    - pedal: every onset the `pedal` note (default the first bass), whatever the
      chord — a pedal point.
    ENDING: 'loop' approaches the first chord from the last, aiming at the line's
    actual first note (so a repeated section connects by a semitone; that
    approach is not folded, and takes the other side when its pitch would repeat
    the previous note or leave E1-G3 — if neither side is free, as for E1 after
    F1, the repeated neighbour is kept); 'root' gives the last chord no approach
    (walking takes its next middle tone, approach its bass).

    SUSTAIN: `sustain` None takes the style's default — root, approach and pedal
    hold each note over the rests after it, the others play each note for one
    step; True/False overrides it. A hold never crosses a chord change: as a
    notes track holds a note over every following '.', a held style whose
    `rhythm` opens with '.' is played detached (sustain false) over two or more
    chords, and sustain=True with such a rhythm is a ValueError. Pedal is exempt:
    its note is the same under every chord.

    Returns {style, ending, steps_per_chord, total_beats, track {type 'notes',
    name 'bass', notes with octaves, rhythm, step_beats, sustain, program 33},
    per_chord [{symbol, notes}], render_hint {tracks: [the bass track]}}.
    e.g. bass_line(['C','Am','F','G'], 'walking') -> C2 E2 G2 G#2 | A2 C3 E3 Gb3 |
    F3 C3 A2 Ab2 | G2 B2 D3 Db2 (Db2 leads back into the first C2);
    bass_line(['C','G/B','Am','F']) -> C2 B1 A1 F1.
    """
    items = _chord_items(chords)
    parsed = _parse_chord_list(items)
    style = _choice(style, BASS_STYLES, "style")
    ending = _choice(ending, BASS_ENDINGS, "ending")
    step = float(_check_range("step_beats", step_beats, 0.0625, 16))
    beats, _listed = _chord_beats(beats_per_chord, len(parsed))
    per_chord_steps = _chord_steps(beats, step)
    _check_range("octave", octave, 0, 4, integer=True)
    if sustain is not None:
        _check_bool(sustain, "sustain")
    if pedal is not None and style != "pedal":
        raise ValueError(f"pedal is only used with style='pedal' (style is {style!r})")
    if rhythm is not None:
        pattern = parse_rhythm(rhythm)
        bad = sorted({s for s in per_chord_steps if s != len(pattern)})
        if bad:
            raise ValueError(
                f"rhythm has {len(pattern)} steps but a chord lasts {bad[0]} steps"
                f" (beats_per_chord / step_beats); give one chord span, e.g."
                f" {_default_rhythm(style, bad[0])!r}"
            )
        if all(c == RHYTHM_REST for c in pattern):
            raise ValueError("rhythm has no onsets ('O' or 'o'); the bass would be silent")
        rhythms = [pattern] * len(parsed)
    else:
        rhythms = [_default_rhythm(style, s) for s in per_chord_steps]
    # a hold would carry each chord's last note over the rest that opens the next chord
    crosses = len(parsed) > 1 and style != "pedal" and rhythms[0][0] == RHYTHM_REST
    if sustain is None:
        held = _SUSTAINED[style] and not crosses
    elif sustain and crosses:
        raise ValueError(
            f"sustain would hold each chord's last bass note over the rest that opens the next"
            f" chord (rhythm {rhythms[0]!r} starts with '.'), so it would sound under the wrong"
            f" chord; start the rhythm with an onset, or leave sustain unset (or false) to play"
            f" the line detached"
        )
    else:
        held = sustain
    readings = [_bass_reading(item, p) for item, p in zip(items, parsed)]

    if style == "pedal":
        note = parse_note(pedal) if pedal is not None else readings[0]["bass"]
        if note.octave is None:
            note = Note(note.letter, note.accidental, octave)
        pedal_note = _fold(_check_midi(note))

    out: list[Note] = []
    per_chord: list[dict] = []
    prev: Note | None = None      # the previous sounding note
    anchor: Note | None = None    # where the next chord's bass is placed from
    first: Note | None = None     # the first chord's placed bass: where a loop starts again
    for i, (reading, pat) in enumerate(zip(readings, rhythms)):
        m = sum(1 for c in pat if c != RHYTHM_REST)
        last = i == len(readings) - 1
        if style == "pedal":
            seq = [pedal_note] * m
        else:
            b = reading["bass"]
            bass = _fold(Note(b.letter, b.accidental, octave)) if anchor is None else _nearest(b, anchor.midi)
            if style == "root_octave" and bass.midi + 12 > BASS_HIGH:
                bass = _with_octave(bass, bass.midi - 12)   # its octave above would pass G3
            if first is None:
                first = bass
            seq = [bass]
            # the last onset approaches the next chord's bass (placed nearest the note before
            # it) or, closing a loop, the line's fixed first note; ending='root': nothing
            nxt = None if last else readings[i + 1]["bass"]
            seam = first if last and ending == "loop" else None
            approaches = nxt is not None or seam is not None
            if style == "root":
                seq *= m
            elif style == "root_fifth":
                alt = reading["fifth"]
                if alt.pitch_class == bass.pitch_class:   # a 6/4 chord: the bass is the fifth
                    alt = reading["root"]
                fifth = _fold(_with_octave(alt, _below(alt, bass)))
                if fifth.midi == bass.midi:               # folded back onto the bass: an octave above
                    fifth = _with_octave(alt, bass.midi + 12)
                seq = [bass if k % 2 == 0 else fifth for k in range(m)]
            elif style == "root_octave":
                high = _with_octave(bass, bass.midi + 12)
                seq = [bass if k % 2 == 0 else high for k in range(m)]
            elif style == "approach":
                seq *= m
                if m >= 2 and approaches:
                    seq[-1] = _approach_to(bass, nxt, seam)
            else:  # walking
                middle = _MiddleTones(reading["ring"])
                count = m - 2 if approaches else m - 1
                for _ in range(max(0, count)):
                    seq.append(middle.next(seq[-1]))
                if m >= 2 and approaches:
                    seq.append(_approach_to(seq[-1], nxt, seam))
            anchor = bass if style in ("root_fifth", "root_octave") else seq[-1]
        if seq:
            prev = seq[-1]
        out.extend(seq)
        per_chord.append({"symbol": reading["symbol"], "notes": [n.name for n in seq]})
    if prev is None:
        raise ValueError("the bass line has no notes")

    track = {"type": "notes", "name": "bass", "notes": [n.name for n in out],
             "rhythm": parse_rhythm("".join(rhythms)), "step_beats": step,
             "sustain": held, "program": BASS_PROGRAM}
    return {
        "style": style,
        "ending": ending,
        "steps_per_chord": per_chord_steps,
        "total_beats": sum(beats),
        "track": track,
        "per_chord": per_chord,
        "render_hint": {"tracks": [track]},
    }
