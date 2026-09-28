"""Deterministic MIDI rendering: note sequences, rhythms, chords, drums and
full multi-track arrangements to .mid.

These renderers add no musical decisions of their own — they write exactly
the notes, rhythm, chords and drum hits they are given. Output files land in
``MIDI_COMPOSER_OUTPUT_DIR`` (default ``./midi_output``) and are also
returned base64-encoded.

Chords can be comped: per-chord ``durations`` (variable harmonic rhythm), an
O/o/. ``rhythm`` of strikes (with ``sustain``), and a guitar-style ``strum``.
Every renderer and track can be swung: ``swing`` is the DAW/MPC swing ratio
(0.5 straight, 2/3 triplet swing, at most 0.75) applied to pairs of
``swing_unit``-beat steps on the absolute timeline — on-beats never move. A
part that ends mid-pair reports its swung end, so a file always lasts exactly
as long as reported. The defaults (no durations, no rhythm, no strum, swing
0.5) write exactly the files these renderers always wrote.
"""

from __future__ import annotations

import base64
import datetime
import math
import os
import re
import tempfile
import unicodedata

from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo

from .chords import chord_notes, parse_chord_symbol
from .generate import RHYTHM_REST, RHYTHM_STRONG, RHYTHM_WEAK, parse_rhythm
from .notes import LETTER_PCS, Note, parse_notes

TICKS_PER_BEAT = 480
DEFAULT_OUTPUT_DIR = "midi_output"
OCTAVE_POLICIES = ("nearest", "ascending")
DRUM_CHANNEL = 9  # General MIDI percussion channel (channel 10, 0-indexed)
SWING_UNITS = (0.25, 0.5, 1.0)  # swung step: sixteenths, eighths or quarters (in beats)
STRUM_DIRECTIONS = ("down", "up", "alternate")
_EPS = 1e-9

# General MIDI percussion key map (note numbers on the drum channel).
GM_DRUMS: dict[str, int] = {
    "kick": 36, "bass_drum": 36, "acoustic_bass_drum": 35, "kick2": 35,
    "snare": 38, "acoustic_snare": 38, "electric_snare": 40,
    "side_stick": 37, "rimshot": 37, "clap": 39, "hand_clap": 39,
    "closed_hat": 42, "hat": 42, "hihat": 42, "closed_hihat": 42,
    "pedal_hat": 44, "open_hat": 46, "open_hihat": 46,
    "low_tom": 45, "mid_tom": 47, "high_tom": 50, "floor_tom": 43,
    "crash": 49, "crash2": 57, "ride": 51, "ride_bell": 53, "splash": 55, "china": 52,
    "tambourine": 54, "cowbell": 56, "vibraslap": 58, "clave": 75, "woodblock": 76,
    "shaker": 82, "maracas": 70, "cabasa": 69, "triangle": 81,
    "conga": 63, "bongo": 60, "timbale": 65, "agogo": 67, "guiro": 73, "whistle": 71,
}


# ---------------------------------------------------------------- validation

def _check_range(name: str, value, low, high, integer: bool = False):
    ok = isinstance(value, int) and not isinstance(value, bool) if integer \
        else isinstance(value, (int, float)) and not isinstance(value, bool)
    if not ok or not low <= value <= high:
        kind = "an integer" if integer else "a number"
        raise ValueError(f"{name} must be {kind} between {low} and {high}, got {value!r}")
    return value


def _check_swing(ratio, unit, where: str = "") -> tuple[float, float]:
    """Validate a swing ratio (0.5-0.75) and swing unit (0.25, 0.5 or 1.0 beats)."""
    if not isinstance(ratio, (int, float)) or isinstance(ratio, bool) or not 0.5 <= ratio <= 0.75:
        raise ValueError(
            f"{where}swing must be a ratio between 0.5 (straight) and 0.75"
            f" (2/3 = triplet swing), got {ratio!r}"
        )
    if not isinstance(unit, (int, float)) or isinstance(unit, bool) or unit not in SWING_UNITS:
        raise ValueError(
            f"{where}swing_unit must be 0.25 (swung sixteenths), 0.5 (swung eighths)"
            f" or 1.0 (swung quarters) beats, got {unit!r}"
        )
    return ratio, float(unit)


# --------------------------------------------------------------------- swing

def _swing_warp(t: float, ratio: float, unit: float) -> float:
    """Move one beat position onto the swung grid.

    The timeline is cut into windows of two `unit` steps, [2uk, 2uk + 2u). Inside a
    window, with x = t - 2uk: the on-step half is stretched (x' = 2r·x for x < u) and
    the off-step half squeezed (x' = 2ur + 2(1-r)(x-u)). Window edges never move —
    every beat for swung sixteenths/eighths, every other beat for swung quarters — so
    on-beats, bar lines and whole-bar lengths stay put; ratio 0.5 is the identity.
    """
    if ratio == 0.5:
        return t
    window = 2 * unit
    k = math.floor(t / window + _EPS)
    x = max(0.0, t - window * k)
    if x < unit:
        return window * k + 2 * ratio * x
    return window * k + 2 * unit * ratio + 2 * (1 - ratio) * (x - unit)


def _swing_events(events: list[dict], ratio: float, unit: float) -> list[dict]:
    """Warp every event's start and end (absolute beats) onto the swung grid."""
    if ratio == 0.5:
        return events
    swung = []
    for e in events:
        start = _swing_warp(e["start"], ratio, unit)
        if "program" in e:
            swung.append(dict(e, start=start))
            continue
        end = _swing_warp(e["start"] + e["duration"], ratio, unit)
        swung.append(dict(e, start=start, duration=end - start))
    return swung


def _swing_spans(resolved: list[dict], ratio: float, unit: float) -> list[dict]:
    """Warp the reported start_beat/duration_beats of resolved chords."""
    if ratio == 0.5:
        return resolved
    out = []
    for r in resolved:
        start = _swing_warp(r["start_beat"], ratio, unit)
        end = _swing_warp(r["start_beat"] + r["duration_beats"], ratio, unit)
        out.append(dict(r, start_beat=start, duration_beats=end - start))
    return out


def _with_octave(note: Note, midi: int) -> Note:
    octave = (midi - LETTER_PCS[note.letter] - note.accidental) // 12 - 1
    return Note(note.letter, note.accidental, octave)


def _check_midi(note: Note) -> Note:
    if not 0 <= note.midi <= 127:
        raise ValueError(f"Note {note.name} is outside the MIDI range 0-127 (C-1 to G9)")
    return note


def resolve_drum(name) -> tuple[int, str]:
    """Resolve a drum lane name ('kick', 'snare', ...) or note number to MIDI."""
    if isinstance(name, bool):
        raise ValueError(f"Invalid drum: {name!r}")
    if isinstance(name, int):
        if not 0 <= name <= 127:
            raise ValueError(f"Drum note number out of range 0-127: {name}")
        return name, str(name)
    if isinstance(name, str):
        key = name.strip().lower().replace(" ", "_").replace("-", "_")
        if key in GM_DRUMS:
            return GM_DRUMS[key], key
        if re.fullmatch(r"-?\d+", key):
            return resolve_drum(int(key))
        raise ValueError(
            f"Unknown drum {name!r}. Use a General MIDI note number (0-127) or a name like: "
            + ", ".join(sorted(set(GM_DRUMS)))
        )
    raise ValueError(f"Invalid drum: {name!r} (use a name or a note number)")


# ----------------------------------------------------- octave assignment

def assign_octaves(notes: list[Note], default_octave: int, policy: str) -> list[Note]:
    """Give every note a concrete octave.

    Notes that already carry an octave are kept. Others are placed relative
    to the previous note: ``nearest`` picks the closest octave (good for
    melodies), ``ascending`` never steps down (good for scale runs — the
    repeated root lands an octave up).
    """
    if policy not in OCTAVE_POLICIES:
        raise ValueError(f"octave_policy must be one of {OCTAVE_POLICIES}, got {policy!r}")
    # 'nearest' follows the melody, but stays within two octaves of its anchor
    # (the first note, or the latest note written with an octave), so a repeating
    # progression cannot drift out of range while ordinary lines are left alone.
    anchor: int | None = None
    placed: list[Note] = []
    prev: int | None = None
    for n in notes:
        if n.octave is not None:
            midi = n.midi
            result = n
            anchor = midi
        else:
            if prev is None:
                midi = Note(n.letter, n.accidental, default_octave).midi  # Cb at octave 4 is Cb4
                anchor = midi
            else:
                above = prev + ((n.pitch_class - prev) % 12)
                if policy == "ascending":
                    midi = above
                else:
                    below = above - 12
                    midi = above if above - prev <= prev - below else below
                    if abs(midi - anchor) > 24 or not 0 <= midi <= 127:
                        options = [m for m in (above, below) if 0 <= m <= 127] or [midi]
                        midi = min(options, key=lambda m: (abs(m - anchor), m))
            result = _with_octave(n, midi)
        placed.append(_check_midi(result))
        prev = midi
    return placed


def voice_chord(tones: list[Note], octave: int, bass: Note | None = None) -> list[Note]:
    """Voice a chord: explicit octaves are kept, the rest stack upwards.

    The first octave-less tone lands at `octave`; each following tone goes to
    the nearest pitch strictly above the previous one. A slash bass without
    an octave is placed strictly below the lowest chord tone; a slash bass with
    an octave ('C/E3') stays put and the chord is stacked above it.
    """
    voiced: list[Note] = []
    prev: int | None = None
    if bass is not None and bass.octave is not None and tones and tones[0].octave is None:
        prev = bass.midi  # stack the chord strictly above the given bass
    for t in tones:
        if t.octave is not None:
            midi = t.midi
            voiced.append(t)
        else:
            if prev is None:
                midi = Note(t.letter, t.accidental, octave).midi  # by letter: Cb at 4 is Cb4
            else:
                midi = prev + ((t.pitch_class - prev) % 12 or 12)
            voiced.append(_with_octave(t, midi))
        prev = midi
    if bass is not None:
        if bass.octave is not None:
            voiced.insert(0, bass)
        else:
            first = voiced[0].midi
            voiced.insert(0, _with_octave(bass, first - ((first - bass.pitch_class) % 12 or 12)))
    return [_check_midi(v) for v in voiced]


# ----------------------------------------------------------- event building

# An event is {"midi": int, "label": str, "start": float, "duration": float,
#              "velocity": int} with beat-based start/duration.

def _melody_events(notes: list[Note], rhythm: str | None, step_beats: float,
                   velocity: int, accent_velocity: int, sustain: bool,
                   start: float = 0.0) -> tuple[list[dict], float]:
    """Turn placed notes (+ optional rhythm pattern) into timed events."""
    events: list[dict] = []
    if rhythm is None:
        for i, note in enumerate(notes):
            events.append({"midi": note.midi, "label": note.name,
                           "start": start + i * step_beats,
                           "duration": step_beats, "velocity": velocity})
        return events, len(notes) * step_beats

    pattern = parse_rhythm(rhythm)
    index = 0
    for step, symbol in enumerate(pattern):
        if symbol == RHYTHM_REST:
            if sustain and events:
                events[-1]["duration"] += step_beats
            continue
        note = notes[index % len(notes)]
        index += 1
        events.append({
            "midi": note.midi,
            "label": note.name,
            "start": start + step * step_beats,
            "duration": step_beats,
            "velocity": accent_velocity if symbol == RHYTHM_STRONG else velocity,
        })
    return events, len(pattern) * step_beats


def _parse_chord_list(chords) -> list[dict]:
    """Normalize the `chords` argument into [{symbol, tones, bass}, ...].

    Accepts a string of symbols ("C Am F G"), or a list whose items are each
    either a chord symbol ("Am", "C4maj7", "C/E") or a list of note names
    (["C", "E", "G"] / ["C4", "E4", "G4"]).
    """
    if isinstance(chords, str):
        chords = [t for t in re.split(r"[,\s]+", chords.strip()) if t]
    if not isinstance(chords, (list, tuple)) or not chords:
        raise ValueError(
            "chords must be a non-empty list of chord symbols and/or note arrays,"
            " e.g. ['C', 'Am', 'F', 'G'] or [['C','E','G'], 'G7']"
        )
    parsed: list[dict] = []
    for item in chords:
        if isinstance(item, str):
            root, chord_type, bass = parse_chord_symbol(item)
            tones = chord_notes(chord_type, root)
            symbol = f"{root.pitch_class_name}{chord_type.symbol}"
            if bass is not None:
                symbol += f"/{bass.pitch_class_name}"
            parsed.append({"symbol": symbol, "tones": tones, "bass": bass})
        elif isinstance(item, (list, tuple)):
            tones = parse_notes(list(item))
            parsed.append({"symbol": None, "tones": tones, "bass": None})
        else:
            raise ValueError(f"Invalid chord entry: {item!r} (use a symbol string or a list of notes)")
    return parsed


def _check_durations(durations, count: int, where: str = "") -> list[float]:
    """Validate per-chord durations: one number (0.25-64 beats) per chord."""
    if not isinstance(durations, (list, tuple)) or not durations:
        raise ValueError(
            f"{where}durations must be a list of beat lengths, one per chord (e.g. [2, 2, 4]),"
            f" got {durations!r}"
        )
    if len(durations) != count:
        raise ValueError(
            f"{where}durations has {len(durations)} entries but there are {count} chords;"
            f" give exactly one duration per chord"
        )
    return [float(_check_range(f"{where}durations[{i}]", d, 0.25, 64)) for i, d in enumerate(durations)]


def _whole_steps(length: float, step: float) -> int | None:
    """How many `step`s fit in `length`, if that is a whole number (else None)."""
    q = length / step
    n = round(q)
    return n if n >= 1 and abs(q - n) < 1e-6 else None


def _comp_strikes(pattern: str, bounds: list[float], step: float, sustain: bool,
                  where: str = "") -> list[list[tuple]]:
    """Read a comping rhythm into strikes per chord: (onset, length, symbol, grid step).

    `bounds` are the chord boundaries relative to the track start ([0, d0, d0+d1, ...]).
    The pattern is either one chord long (repeated for every chord; all chords must be
    the same whole number of steps) or the whole track long. A strike never crosses a
    chord change; with `sustain`, rests extend the previous strike up to the chord end.
    """
    count = len(bounds) - 1
    lengths = [bounds[k + 1] - bounds[k] for k in range(count)]
    total = bounds[-1]
    equal = all(abs(d - lengths[0]) < 1e-6 for d in lengths)
    per_chord = _whole_steps(lengths[0], step) if equal else None
    whole = _whole_steps(total, step)

    onsets: list[tuple[int, float, int, str]] = []  # (chord, onset, grid step, symbol)
    if per_chord is not None and len(pattern) == per_chord:
        for k in range(count):
            for s, symbol in enumerate(pattern):
                onsets.append((k, bounds[k] + s * step, k * per_chord + s, symbol))
    elif whole is not None and len(pattern) == whole:
        k = 0
        for j, symbol in enumerate(pattern):
            t = j * step
            while k < count - 1 and t + _EPS >= bounds[k + 1]:
                k += 1
            onsets.append((k, t, j, symbol))
    else:
        if per_chord is not None:
            one = f"{per_chord} (one chord: {lengths[0]:g} beats at step_beats {step:g}, repeated per chord)"
        elif not equal:
            one = "a per-chord length (not possible: the chords have different durations)"
        else:
            one = f"a per-chord length (not possible: {lengths[0]:g} beats is not a whole number of {step:g}-beat steps)"
        if whole is not None:
            full = f"{whole} (the whole {total:g}-beat track at step_beats {step:g})"
        else:
            full = f"a whole-track length (not possible: {total:g} beats is not a whole number of {step:g}-beat steps)"
        raise ValueError(f"{where}chord rhythm has {len(pattern)} steps; it must have {one} or {full}")

    strikes: list[list[list]] = [[] for _ in range(count)]
    current: list | None = None
    current_chord = -1
    for k, onset, grid, symbol in onsets:
        if k != current_chord:
            current, current_chord = None, k  # a strike never carries over a chord change
        chord_end = bounds[k + 1]
        if symbol == RHYTHM_REST:
            if sustain and current is not None:
                current[1] = min(onset + step, chord_end) - current[0]
            continue
        current = [onset, min(onset + step, chord_end) - onset, symbol, grid]
        strikes[k].append(current)
    return [[tuple(s) for s in chord] for chord in strikes]


def _chord_events(parsed_chords: list[dict], beats_per_chord: float, octave: int,
                  arpeggiate: bool, velocity: int, start: float = 0.0,
                  durations=None, rhythm: str | None = None, step_beats: float = 0.5,
                  sustain: bool = False, accent_velocity: int = 100, strum: float = 0.0,
                  strum_direction: str = "down",
                  where: str = "") -> tuple[list[dict], list[dict], float]:
    """Voice and time a chord sequence.

    Each chord lasts `beats_per_chord`, or its entry in `durations`. Without a
    `rhythm` a chord is one block strike (or an arpeggio); with one, every O/o
    strikes the whole voicing (see _comp_strikes). With `strum`, voice i of a strike
    enters i*strum beats late (counted from the lowest note for 'down', the highest
    for 'up'; 'alternate' flips on odd grid steps, or odd chords without a rhythm),
    and all voices end together.
    """
    _check_range(f"{where}strum", strum, 0, 0.25)
    if not isinstance(strum_direction, str) or strum_direction.strip().lower() not in STRUM_DIRECTIONS:
        raise ValueError(f"{where}strum_direction must be one of {STRUM_DIRECTIONS}, got {strum_direction!r}")
    direction = strum_direction.strip().lower()
    _check_range(f"{where}step_beats", step_beats, 0.0625, 16)
    _check_range(f"{where}accent_velocity", accent_velocity, 1, 127, integer=True)
    if arpeggiate and rhythm is not None:
        raise ValueError(f"{where}a chord rhythm cannot be combined with arpeggiate; use one or the other")
    if arpeggiate and strum:
        raise ValueError(f"{where}strum cannot be combined with arpeggiate; use one or the other")

    count = len(parsed_chords)
    if durations is None:
        offsets = [i * beats_per_chord for i in range(count)]
        lengths = [beats_per_chord] * count
        total = count * beats_per_chord
    else:
        lengths = _check_durations(durations, count, where)
        offsets = [sum(lengths[:i]) for i in range(count)]
        total = sum(lengths)
    if rhythm is not None:
        strikes = _comp_strikes(parse_rhythm(rhythm), offsets + [total], step_beats, sustain, where)
    else:  # one block strike per chord, on its chord index for 'alternate'
        strikes = [[(offsets[i], lengths[i], RHYTHM_WEAK, i)] for i in range(count)]

    events: list[dict] = []
    resolved: list[dict] = []
    for i, chord in enumerate(parsed_chords):
        voiced = voice_chord(chord["tones"], octave, chord["bass"])
        chord_start = start + offsets[i]
        length = lengths[i]
        if arpeggiate:
            tone_beats = length / len(voiced)
            for k, note in enumerate(voiced):
                events.append({"midi": note.midi, "label": note.name,
                               "start": chord_start + k * tone_beats,
                               "duration": tone_beats, "velocity": velocity})
        elif rhythm is None and not strum:
            for note in voiced:
                events.append({"midi": note.midi, "label": note.name,
                               "start": chord_start,
                               "duration": length, "velocity": velocity})
        else:
            by_pitch = sorted(range(len(voiced)), key=lambda v: voiced[v].midi)
            for onset, strike_len, symbol, grid in strikes[i]:  # onset is track-relative
                if strum and (len(voiced) - 1) * strum >= strike_len - _EPS:
                    raise ValueError(
                        f"{where}strum {strum:g} is too wide for a {strike_len:g}-beat strike of"
                        f" {len(voiced)} notes (the last note would start at or after the strike's"
                        f" end); use a strum below {strike_len / max(1, len(voiced) - 1):g}"
                    )
                down = direction == "down" or (direction == "alternate" and grid % 2 == 0)
                order = by_pitch if down else by_pitch[::-1]
                delay = {v: rank * strum for rank, v in enumerate(order)}
                vel = accent_velocity if symbol == RHYTHM_STRONG else velocity
                for v, note in enumerate(voiced):
                    events.append({"midi": note.midi, "label": note.name,
                                   "start": start + onset + delay[v],
                                   "duration": strike_len - delay[v], "velocity": vel})
        resolved.append({
            "symbol": chord["symbol"] or " ".join(n.pitch_class_name for n in voiced),
            "notes": [n.name for n in voiced],
            "midi": [n.midi for n in voiced],
            "start_beat": chord_start,
            "duration_beats": length,
        })
    return events, resolved, total


def _drum_events(lanes, step_beats: float, velocity: int, accent_velocity: int,
                 start: float = 0.0) -> tuple[list[dict], list[dict], float]:
    """Build drum events from {lane name: rhythm pattern} on the drum channel."""
    if not isinstance(lanes, dict) or not lanes:
        raise ValueError(
            "drum `lanes` must be a non-empty mapping of drum name to rhythm pattern,"
            " e.g. {'kick': 'O...O...', 'snare': '..O...O.', 'hat': 'oooooooo'}"
        )
    events: list[dict] = []
    resolved: list[dict] = []
    max_steps = 0
    for name, pattern in lanes.items():
        midi, label = resolve_drum(name)
        pat = parse_rhythm(pattern)
        max_steps = max(max_steps, len(pat))
        hits = 0
        for step, symbol in enumerate(pat):
            if symbol == RHYTHM_REST:
                continue
            events.append({
                "midi": midi,
                "label": label,
                "start": start + step * step_beats,
                "duration": step_beats,
                "velocity": accent_velocity if symbol == RHYTHM_STRONG else velocity,
            })
            hits += 1
        resolved.append({"drum": label, "note": midi, "pattern": pat, "hits": hits})
    return events, resolved, max_steps * step_beats


# ------------------------------------------------------------- file writing

def _meta_text(text: str) -> str:
    """Make a track name storable in a MIDI meta event (Latin-1): 'Smyčce' -> 'Smycce'."""
    out = []
    for ch in text:
        try:
            ch.encode("latin-1")
            out.append(ch)
        except UnicodeEncodeError:
            base = "".join(c for c in unicodedata.normalize("NFKD", ch) if c.isascii())
            out.append(base or "?")
    return "".join(out)


def _events_to_track(events: list[dict], channel: int, program: int,
                     name: str | None = None) -> MidiTrack:
    """Timed note (and optional program-change) events -> one MIDI track.

    A note event is {"midi", "start", "duration", "velocity"}; a program-change
    event is {"start", "program"} (used when a stitched song part changes
    instrument between sections). At one tick: note-offs, then program changes,
    then note-ons.
    """
    timed: list[tuple[int, int, Message]] = []
    for e in events:
        on_tick = round(e["start"] * TICKS_PER_BEAT)
        if "program" in e:
            timed.append((on_tick, 1, Message("program_change", program=e["program"], channel=channel)))
            continue
        off_tick = max(on_tick + 1, round((e["start"] + e["duration"]) * TICKS_PER_BEAT))
        timed.append((on_tick, 2, Message("note_on", note=e["midi"], velocity=e["velocity"], channel=channel)))
        timed.append((off_tick, 0, Message("note_off", note=e["midi"], velocity=0, channel=channel)))
    timed.sort(key=lambda t: (t[0], t[1]))

    track = MidiTrack()
    if name:
        track.append(MetaMessage("track_name", name=_meta_text(name), time=0))
    track.append(Message("program_change", program=program, channel=channel, time=0))
    now = 0
    for tick, _, msg in timed:
        msg.time = tick - now
        now = tick
        track.append(msg)
    track.append(MetaMessage("end_of_track", time=0))
    return track


def _build_file(parts: list[dict], tempo: int, beats_per_bar: int = 4,
                total_beats: float = 0.0) -> MidiFile:
    mid = MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    conductor = MidiTrack()
    conductor.append(MetaMessage("set_tempo", tempo=bpm2tempo(tempo), time=0))
    conductor.append(MetaMessage("time_signature", numerator=beats_per_bar, denominator=4, time=0))
    # the conductor track lasts the full reported length, so trailing rests (a groove
    # ending in rests, a section's empty last bars) are part of the file
    conductor.append(MetaMessage("end_of_track", time=max(0, round(total_beats * TICKS_PER_BEAT))))
    mid.tracks.append(conductor)
    for part in parts:
        mid.tracks.append(_events_to_track(part["events"], part["channel"],
                                           part["program"], part.get("name")))
    return mid


def _safe_file_name(file_name: str | None, default_stem: str) -> str:
    if file_name:
        name = os.path.basename(file_name.strip())
    else:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        name = f"{default_stem}_{stamp}.mid"
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    if not name.lower().endswith((".mid", ".midi")):
        name += ".mid"
    return name


def _write_file(mid: MidiFile, file_name: str | None, output_dir: str | None,
                default_stem: str) -> dict:
    directory = output_dir or os.environ.get("MIDI_COMPOSER_OUTPUT_DIR") or DEFAULT_OUTPUT_DIR
    os.makedirs(directory, exist_ok=True)
    name = _safe_file_name(file_name, default_stem)
    path = os.path.abspath(os.path.join(directory, name))
    # write to a temporary file first, so a failed save never leaves a corrupt .mid behind
    fd, tmp = tempfile.mkstemp(suffix=".mid", dir=os.path.dirname(path))
    os.close(fd)
    umask = os.umask(0)
    os.umask(umask)
    os.chmod(tmp, 0o666 & ~umask)  # mkstemp creates 0600; give the file normal permissions
    try:
        mid.save(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    with open(path, "rb") as fh:
        data = fh.read()
    return {
        "file": path,
        "file_name": name,
        "size_bytes": len(data),
        "base64": base64.b64encode(data).decode("ascii"),
    }


def _serialize_events(events: list[dict]) -> list[dict]:
    return [
        {
            "note": e["label"],
            "midi": e["midi"],
            "start_beat": e["start"],
            "duration_beats": e["duration"],
            "velocity": e["velocity"],
        }
        for e in events
    ]


def _common_meta(tempo: int, total_beats: float) -> dict:
    return {
        "tempo": tempo,
        "ticks_per_beat": TICKS_PER_BEAT,
        "total_beats": total_beats,
        "duration_seconds": round(total_beats * 60 / tempo, 3),
    }


# ------------------------------------------------------------------ renders

def render_notes(notes, rhythm: str | None = None, step_beats: float = 0.5,
                 tempo: int = 120, octave: int = 4, octave_policy: str = "nearest",
                 velocity: int = 90, accent_velocity: int = 110, sustain: bool = False,
                 program: int = 0, swing: float = 0.5, swing_unit: float = 0.5,
                 file_name: str | None = None, output_dir: str | None = None) -> dict:
    """Render a note sequence (scale, arpeggio, melody) to a MIDI file."""
    parsed = parse_notes(notes)
    _check_range("tempo", tempo, 10, 400, integer=True)
    _check_range("step_beats", step_beats, 0.0625, 16)
    _check_range("octave", octave, -1, 9, integer=True)
    _check_range("velocity", velocity, 1, 127, integer=True)
    _check_range("accent_velocity", accent_velocity, 1, 127, integer=True)
    _check_range("program", program, 0, 127, integer=True)
    swing, swing_unit = _check_swing(swing, swing_unit)

    placed = assign_octaves(parsed, octave, octave_policy)
    events, total_beats = _melody_events(placed, rhythm, step_beats,
                                         velocity, accent_velocity, sustain)
    events = _swing_events(events, swing, swing_unit)
    total_beats = _swing_warp(total_beats, swing, swing_unit)
    mid = _build_file([{"events": events, "channel": 0, "program": program, "name": "notes"}], tempo,
                      total_beats=total_beats)
    result = _write_file(mid, file_name, output_dir, "notes")
    result.update(_common_meta(tempo, total_beats))
    result["note_count"] = len(events)
    result["events"] = _serialize_events(events)
    return result


def render_chords(chords, beats_per_chord: float = 4.0, tempo: int = 120,
                  octave: int = 4, arpeggiate: bool = False, velocity: int = 80,
                  program: int = 0, durations: list[float] | None = None,
                  rhythm: str | None = None, step_beats: float = 0.5, sustain: bool = False,
                  accent_velocity: int = 100, strum: float = 0.0, strum_direction: str = "down",
                  swing: float = 0.5, swing_unit: float = 0.5, file_name: str | None = None,
                  output_dir: str | None = None) -> dict:
    """Render a chord sequence to a MIDI file (block chords, comped strikes or arpeggios)."""
    parsed_chords = _parse_chord_list(chords)
    _check_range("tempo", tempo, 10, 400, integer=True)
    _check_range("beats_per_chord", beats_per_chord, 0.25, 64)
    _check_range("octave", octave, -1, 9, integer=True)
    _check_range("velocity", velocity, 1, 127, integer=True)
    _check_range("program", program, 0, 127, integer=True)
    swing, swing_unit = _check_swing(swing, swing_unit)

    events, resolved, total_beats = _chord_events(
        parsed_chords, beats_per_chord, octave, arpeggiate, velocity,
        durations=durations, rhythm=rhythm, step_beats=step_beats, sustain=sustain,
        accent_velocity=accent_velocity, strum=strum, strum_direction=strum_direction)
    events = _swing_events(events, swing, swing_unit)
    resolved = _swing_spans(resolved, swing, swing_unit)
    total_beats = _swing_warp(total_beats, swing, swing_unit)
    mid = _build_file([{"events": events, "channel": 0, "program": program, "name": "chords"}], tempo,
                      total_beats=total_beats)
    result = _write_file(mid, file_name, output_dir, "chords")
    result.update(_common_meta(tempo, total_beats))
    result["chord_count"] = len(resolved)
    result["chords"] = resolved
    return result


def render_drums(lanes, step_beats: float = 0.5, tempo: int = 120, velocity: int = 100,
                 accent_velocity: int = 120, swing: float = 0.5, swing_unit: float = 0.5,
                 file_name: str | None = None, output_dir: str | None = None) -> dict:
    """Render a drum pattern (named lanes of rhythm strings) to a MIDI file."""
    _check_range("tempo", tempo, 10, 400, integer=True)
    _check_range("step_beats", step_beats, 0.0625, 16)
    _check_range("velocity", velocity, 1, 127, integer=True)
    _check_range("accent_velocity", accent_velocity, 1, 127, integer=True)
    swing, swing_unit = _check_swing(swing, swing_unit)

    events, resolved, total_beats = _drum_events(lanes, step_beats, velocity, accent_velocity)
    events = _swing_events(events, swing, swing_unit)
    total_beats = _swing_warp(total_beats, swing, swing_unit)
    mid = _build_file([{"events": events, "channel": DRUM_CHANNEL, "program": 0, "name": "drums"}], tempo,
                      total_beats=total_beats)
    result = _write_file(mid, file_name, output_dir, "drums")
    result.update(_common_meta(tempo, total_beats))
    result["hit_count"] = len(events)
    result["lanes"] = resolved
    return result


def render_song(melody_notes, chords, melody_rhythm: str | None = None,
                step_beats: float = 0.5, beats_per_chord: float = 4.0,
                tempo: int = 120, melody_octave: int = 5, chord_octave: int = 4,
                octave_policy: str = "nearest", melody_velocity: int = 95,
                accent_velocity: int = 115, chord_velocity: int = 70,
                sustain: bool = False, arpeggiate_chords: bool = False,
                melody_program: int = 0, chord_program: int = 0,
                swing: float = 0.5, swing_unit: float = 0.5,
                file_name: str | None = None, output_dir: str | None = None) -> dict:
    """Render a melody track and a chord track into one two-track MIDI file."""
    parsed_melody = parse_notes(melody_notes)
    parsed_chords = _parse_chord_list(chords)
    _check_range("tempo", tempo, 10, 400, integer=True)
    _check_range("step_beats", step_beats, 0.0625, 16)
    _check_range("beats_per_chord", beats_per_chord, 0.25, 64)
    _check_range("melody_octave", melody_octave, -1, 9, integer=True)
    _check_range("chord_octave", chord_octave, -1, 9, integer=True)
    for name, value in (("melody_velocity", melody_velocity),
                        ("accent_velocity", accent_velocity),
                        ("chord_velocity", chord_velocity)):
        _check_range(name, value, 1, 127, integer=True)
    _check_range("melody_program", melody_program, 0, 127, integer=True)
    _check_range("chord_program", chord_program, 0, 127, integer=True)
    swing, swing_unit = _check_swing(swing, swing_unit)

    placed = assign_octaves(parsed_melody, melody_octave, octave_policy)
    melody_events, melody_beats = _melody_events(placed, melody_rhythm, step_beats,
                                                 melody_velocity, accent_velocity, sustain)
    chord_events, resolved, chord_beats = _chord_events(parsed_chords, beats_per_chord,
                                                        chord_octave, arpeggiate_chords,
                                                        chord_velocity)
    melody_events = _swing_events(melody_events, swing, swing_unit)
    chord_events = _swing_events(chord_events, swing, swing_unit)
    resolved = _swing_spans(resolved, swing, swing_unit)
    melody_beats = _swing_warp(melody_beats, swing, swing_unit)
    chord_beats = _swing_warp(chord_beats, swing, swing_unit)
    mid = _build_file(
        [
            {"events": melody_events, "channel": 0, "program": melody_program, "name": "melody"},
            {"events": chord_events, "channel": 1, "program": chord_program, "name": "chords"},
        ],
        tempo,
        total_beats=max(melody_beats, chord_beats),
    )
    result = _write_file(mid, file_name, output_dir, "song")
    result.update(_common_meta(tempo, max(melody_beats, chord_beats)))
    result["melody_beats"] = melody_beats
    result["chord_beats"] = chord_beats
    result["melody_events"] = _serialize_events(melody_events)
    result["chords"] = resolved
    return result


# ------------------------------------------------------ multi-track arrange

def _channel_allocator(reserved=()):
    """Yield channels 0,1,...,8,10,...,15, skipping the drum channel and any `reserved` ones."""
    for ch in range(16):
        if ch != DRUM_CHANNEL and ch not in reserved:
            yield ch
    raise ValueError(
        "Too many melodic tracks: MIDI has 15 melodic channels (channel 10 is reserved for"
        " drums). Merge some parts, or give tracks that share an instrument the same channel."
    )


def build_track_events(track: dict, index: int, step_beats: float,
                       beats_per_chord: float, base_start: float = 0.0,
                       swing: float = 0.5, swing_unit: float = 0.5,
                       apply_swing: bool = True) -> dict:
    """Build the timed events for one track (no channel assigned yet).

    Used by both render_arrangement (single timeline) and the song assembler
    (sections placed at successive offsets). `base_start` is added in front of
    the track's own `start_beat`. Returns the events plus enough metadata for a
    caller to assign a channel and summarize the track.

    The track's own `swing`/`swing_unit` override the renderer's. With
    `apply_swing` the events (and reported beats) are warped here, on the absolute
    timeline; the song assembler passes False, warps after placing each section,
    and reads the resolved (ratio, unit) back from the result's "swing".
    """
    if not isinstance(track, dict):
        raise ValueError(f"track {index} must be an object, got {type(track).__name__}")
    ttype = track.get("type")
    if ttype not in ("notes", "chords", "drums"):
        raise ValueError(
            f"track {index} has invalid type {ttype!r}; use 'notes', 'chords' or 'drums'"
        )
    name = track.get("name") or ttype
    if not isinstance(name, str):
        raise ValueError(f"track {index} name must be a string, got {name!r}")
    rel_start = track.get("start_beat", 0.0)
    _check_range(f"track {index} start_beat", rel_start, 0, 100000)
    start = base_start + rel_start
    program = track.get("program", 0)
    _check_range(f"track {index} program", program, 0, 127, integer=True)
    t_swing = track.get("swing")
    t_unit = track.get("swing_unit")
    ratio, unit = _check_swing(swing if t_swing is None else t_swing,
                               swing_unit if t_unit is None else t_unit, f"track {index} ")

    if ttype == "drums":
        velocity = track.get("velocity", 100)
        accent = track.get("accent_velocity", 120)
        t_step = track.get("step_beats", step_beats)
        _check_range(f"track {index} velocity", velocity, 1, 127, integer=True)
        _check_range(f"track {index} accent_velocity", accent, 1, 127, integer=True)
        _check_range(f"track {index} step_beats", t_step, 0.0625, 16)
        events, resolved, length = _drum_events(track.get("lanes"), t_step, velocity, accent, start)
        detail = {"lanes": resolved}
    elif ttype == "notes":
        velocity = track.get("velocity", 90)
        accent = track.get("accent_velocity", 110)
        t_step = track.get("step_beats", step_beats)
        octave = track.get("octave", 4)
        policy = track.get("octave_policy", "nearest")
        _check_range(f"track {index} velocity", velocity, 1, 127, integer=True)
        _check_range(f"track {index} accent_velocity", accent, 1, 127, integer=True)
        _check_range(f"track {index} step_beats", t_step, 0.0625, 16)
        _check_range(f"track {index} octave", octave, -1, 9, integer=True)
        placed = assign_octaves(parse_notes(track.get("notes")), octave, policy)
        events, length = _melody_events(placed, track.get("rhythm"), t_step, velocity,
                                        accent, track.get("sustain", False), start)
        detail = None  # serialized below, after any swing
    else:  # chords
        velocity = track.get("velocity", 80)
        octave = track.get("octave", 4)
        bpc = track.get("beats_per_chord", beats_per_chord)
        _check_range(f"track {index} velocity", velocity, 1, 127, integer=True)
        _check_range(f"track {index} octave", octave, -1, 9, integer=True)
        _check_range(f"track {index} beats_per_chord", bpc, 0.25, 64)
        events, resolved, length = _chord_events(
            _parse_chord_list(track.get("chords")), bpc, octave, track.get("arpeggiate", False),
            velocity, start, durations=track.get("durations"), rhythm=track.get("rhythm"),
            step_beats=track.get("step_beats", step_beats), sustain=track.get("sustain", False),
            accent_velocity=track.get("accent_velocity", 100), strum=track.get("strum", 0.0),
            strum_direction=track.get("strum_direction", "down"), where=f"track {index} ")
        detail = {"chords": resolved}

    rel_end = rel_start + length
    if apply_swing and ratio != 0.5:
        # warp on the absolute timeline; the track's reported start/length follow the grid
        events = _swing_events(events, ratio, unit)
        if ttype == "chords":
            detail = {"chords": _swing_spans(detail["chords"], ratio, unit)}
        end = _swing_warp(start + length, ratio, unit)
        start = _swing_warp(start, ratio, unit)
        length = end - start
        rel_end = end - base_start
    if detail is None:
        detail = {"events": _serialize_events(events)}

    return {
        "name": name, "type": ttype, "program": program,
        "is_drums": ttype == "drums", "events": events,
        "start": start, "rel_start": rel_start, "length": length,
        "rel_end": rel_end, "detail": detail, "swing": (ratio, unit),
    }


def _build_track(track: dict, index: int, step_beats: float, beats_per_chord: float,
                 channels, swing: float = 0.5, swing_unit: float = 0.5) -> dict:
    b = build_track_events(track, index, step_beats, beats_per_chord,
                           swing=swing, swing_unit=swing_unit)
    explicit_channel = track.get("channel")
    if b["is_drums"]:
        channel = DRUM_CHANNEL  # General MIDI percussion always lives on channel 10
    elif explicit_channel is None:
        channel = next(channels)
    else:
        channel = explicit_channel
        _check_range(f"track {index} channel", channel, 0, 15, integer=True)

    summary = {
        "name": b["name"],
        "type": b["type"],
        "channel": channel,
        "program": b["program"],
        "start_beat": b["start"],
        "length_beats": b["length"],
        "end_beat": b["start"] + b["length"],
        "event_count": len(b["events"]),
        **b["detail"],
    }
    return {"events": b["events"], "channel": channel, "program": b["program"],
            "name": b["name"], "end_beat": b["start"] + b["length"], "summary": summary}


def render_arrangement(tracks, tempo: int = 120, file_name: str | None = None,
                       output_dir: str | None = None, step_beats: float = 0.5,
                       beats_per_chord: float = 4.0, swing: float = 0.5,
                       swing_unit: float = 0.5) -> dict:
    """Render any number of named tracks into one multi-track MIDI file.

    `tracks` is a list of track objects, each `{"type": "notes"|"chords"|"drums", ...}`:

    - notes:  {"type": "notes", "notes": [...], "rhythm": "O.o.", "octave": 3,
               "program": 33, "octave_policy": "nearest", "sustain": false}
    - chords: {"type": "chords", "chords": ["Am","F","C","G"], "beats_per_chord": 4,
               "durations": [4, 2, 2], "rhythm": "O..o..o.", "step_beats": 0.25,
               "sustain": false, "accent_velocity": 100, "strum": 0.03,
               "strum_direction": "down", "octave": 4, "arpeggiate": false, "program": 0}
    - drums:  {"type": "drums", "lanes": {"kick": "O...", "snare": "..O.", "hat": "oooo"}}

    Shared per-track options: `name`, `velocity`, `start_beat` (beat offset),
    `step_beats`, `swing`/`swing_unit` (override the renderer's), `channel`
    (auto-assigned around explicitly chosen ones; drums are always on channel 10).
    """
    if not isinstance(tracks, (list, tuple)) or not tracks:
        raise ValueError("tracks must be a non-empty list of track objects")
    _check_range("tempo", tempo, 10, 400, integer=True)
    _check_range("step_beats", step_beats, 0.0625, 16)
    _check_range("beats_per_chord", beats_per_chord, 0.25, 64)
    swing, swing_unit = _check_swing(swing, swing_unit)

    # auto-assigned channels steer clear of the ones tracks claim explicitly
    reserved = {t.get("channel") for t in tracks
                if isinstance(t, dict) and t.get("type") != "drums"
                and isinstance(t.get("channel"), int) and not isinstance(t.get("channel"), bool)}
    channels = _channel_allocator(reserved)
    built = [_build_track(t, i, step_beats, beats_per_chord, channels, swing, swing_unit)
             for i, t in enumerate(tracks)]

    total_beats = max((b["end_beat"] for b in built), default=0.0)
    mid = _build_file(
        [{"events": b["events"], "channel": b["channel"],
          "program": b["program"], "name": b["name"]} for b in built],
        tempo,
        total_beats=total_beats,
    )
    result = _write_file(mid, file_name, output_dir, "arrangement")
    result.update(_common_meta(tempo, total_beats))
    result["track_count"] = len(built)
    result["tracks"] = [b["summary"] for b in built]
    return result
