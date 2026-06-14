"""Deterministic MIDI rendering: note sequences, rhythms, chords, drums and
full multi-track arrangements to .mid.

These renderers add no musical decisions of their own — they write exactly
the notes, rhythm, chords and drum hits they are given. Output files land in
``MIDI_COMPOSER_OUTPUT_DIR`` (default ``./midi_output``) and are also
returned base64-encoded.
"""

from __future__ import annotations

import base64
import datetime
import os
import re

from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo

from .chords import chord_notes, parse_chord_symbol
from .generate import RHYTHM_REST, RHYTHM_STRONG, parse_rhythm
from .notes import LETTER_PCS, Note, parse_notes

TICKS_PER_BEAT = 480
DEFAULT_OUTPUT_DIR = "midi_output"
OCTAVE_POLICIES = ("nearest", "ascending")
DRUM_CHANNEL = 9  # General MIDI percussion channel (channel 10, 0-indexed)

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
    placed: list[Note] = []
    prev: int | None = None
    for n in notes:
        if n.octave is not None:
            midi = n.midi
            result = n
        else:
            if prev is None:
                midi = n.pitch_class + (default_octave + 1) * 12
            else:
                above = prev + ((n.pitch_class - prev) % 12)
                if policy == "ascending":
                    midi = above
                else:
                    below = above - 12
                    midi = above if above - prev <= prev - below else below
            result = _with_octave(n, midi)
        placed.append(_check_midi(result))
        prev = midi
    return placed


def voice_chord(tones: list[Note], octave: int, bass: Note | None = None) -> list[Note]:
    """Voice a chord: explicit octaves are kept, the rest stack upwards.

    The first octave-less tone lands at `octave`; each following tone goes to
    the nearest pitch strictly above the previous one. A slash bass without
    an octave is placed strictly below the lowest chord tone.
    """
    voiced: list[Note] = []
    prev: int | None = None
    for t in tones:
        if t.octave is not None:
            midi = t.midi
            voiced.append(t)
        else:
            if prev is None:
                midi = t.pitch_class + (octave + 1) * 12
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


def _chord_events(parsed_chords: list[dict], beats_per_chord: float, octave: int,
                  arpeggiate: bool, velocity: int,
                  start: float = 0.0) -> tuple[list[dict], list[dict], float]:
    events: list[dict] = []
    resolved: list[dict] = []
    for i, chord in enumerate(parsed_chords):
        voiced = voice_chord(chord["tones"], octave, chord["bass"])
        chord_start = start + i * beats_per_chord
        if arpeggiate:
            tone_beats = beats_per_chord / len(voiced)
            for k, note in enumerate(voiced):
                events.append({"midi": note.midi, "label": note.name,
                               "start": chord_start + k * tone_beats,
                               "duration": tone_beats, "velocity": velocity})
        else:
            for note in voiced:
                events.append({"midi": note.midi, "label": note.name,
                               "start": chord_start,
                               "duration": beats_per_chord, "velocity": velocity})
        resolved.append({
            "symbol": chord["symbol"] or " ".join(n.pitch_class_name for n in voiced),
            "notes": [n.name for n in voiced],
            "midi": [n.midi for n in voiced],
            "start_beat": chord_start,
            "duration_beats": beats_per_chord,
        })
    return events, resolved, len(parsed_chords) * beats_per_chord


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

def _events_to_track(events: list[dict], channel: int, program: int,
                     name: str | None = None) -> MidiTrack:
    timed: list[tuple[int, int, Message]] = []
    for e in events:
        on_tick = round(e["start"] * TICKS_PER_BEAT)
        off_tick = max(on_tick + 1, round((e["start"] + e["duration"]) * TICKS_PER_BEAT))
        timed.append((on_tick, 1, Message("note_on", note=e["midi"], velocity=e["velocity"], channel=channel)))
        timed.append((off_tick, 0, Message("note_off", note=e["midi"], velocity=0, channel=channel)))
    timed.sort(key=lambda t: (t[0], t[1]))

    track = MidiTrack()
    if name:
        track.append(MetaMessage("track_name", name=name, time=0))
    track.append(Message("program_change", program=program, channel=channel, time=0))
    now = 0
    for tick, _, msg in timed:
        msg.time = tick - now
        now = tick
        track.append(msg)
    track.append(MetaMessage("end_of_track", time=0))
    return track


def _build_file(parts: list[dict], tempo: int) -> MidiFile:
    mid = MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    conductor = MidiTrack()
    conductor.append(MetaMessage("set_tempo", tempo=bpm2tempo(tempo), time=0))
    conductor.append(MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    conductor.append(MetaMessage("end_of_track", time=0))
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
    mid.save(path)
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
                 program: int = 0, file_name: str | None = None,
                 output_dir: str | None = None) -> dict:
    """Render a note sequence (scale, arpeggio, melody) to a MIDI file."""
    parsed = parse_notes(notes)
    _check_range("tempo", tempo, 10, 400, integer=True)
    _check_range("step_beats", step_beats, 0.0625, 16)
    _check_range("octave", octave, -1, 9, integer=True)
    _check_range("velocity", velocity, 1, 127, integer=True)
    _check_range("accent_velocity", accent_velocity, 1, 127, integer=True)
    _check_range("program", program, 0, 127, integer=True)

    placed = assign_octaves(parsed, octave, octave_policy)
    events, total_beats = _melody_events(placed, rhythm, step_beats,
                                         velocity, accent_velocity, sustain)
    mid = _build_file([{"events": events, "channel": 0, "program": program, "name": "notes"}], tempo)
    result = _write_file(mid, file_name, output_dir, "notes")
    result.update(_common_meta(tempo, total_beats))
    result["note_count"] = len(events)
    result["events"] = _serialize_events(events)
    return result


def render_chords(chords, beats_per_chord: float = 4.0, tempo: int = 120,
                  octave: int = 4, arpeggiate: bool = False, velocity: int = 80,
                  program: int = 0, file_name: str | None = None,
                  output_dir: str | None = None) -> dict:
    """Render a chord sequence to a MIDI file (block chords or arpeggios)."""
    parsed_chords = _parse_chord_list(chords)
    _check_range("tempo", tempo, 10, 400, integer=True)
    _check_range("beats_per_chord", beats_per_chord, 0.25, 64)
    _check_range("octave", octave, -1, 9, integer=True)
    _check_range("velocity", velocity, 1, 127, integer=True)
    _check_range("program", program, 0, 127, integer=True)

    events, resolved, total_beats = _chord_events(parsed_chords, beats_per_chord,
                                                  octave, arpeggiate, velocity)
    mid = _build_file([{"events": events, "channel": 0, "program": program, "name": "chords"}], tempo)
    result = _write_file(mid, file_name, output_dir, "chords")
    result.update(_common_meta(tempo, total_beats))
    result["chord_count"] = len(resolved)
    result["chords"] = resolved
    return result


def render_drums(lanes, step_beats: float = 0.5, tempo: int = 120, velocity: int = 100,
                 accent_velocity: int = 120, file_name: str | None = None,
                 output_dir: str | None = None) -> dict:
    """Render a drum pattern (named lanes of rhythm strings) to a MIDI file."""
    _check_range("tempo", tempo, 10, 400, integer=True)
    _check_range("step_beats", step_beats, 0.0625, 16)
    _check_range("velocity", velocity, 1, 127, integer=True)
    _check_range("accent_velocity", accent_velocity, 1, 127, integer=True)

    events, resolved, total_beats = _drum_events(lanes, step_beats, velocity, accent_velocity)
    mid = _build_file([{"events": events, "channel": DRUM_CHANNEL, "program": 0, "name": "drums"}], tempo)
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

    placed = assign_octaves(parsed_melody, melody_octave, octave_policy)
    melody_events, melody_beats = _melody_events(placed, melody_rhythm, step_beats,
                                                 melody_velocity, accent_velocity, sustain)
    chord_events, resolved, chord_beats = _chord_events(parsed_chords, beats_per_chord,
                                                        chord_octave, arpeggiate_chords,
                                                        chord_velocity)
    mid = _build_file(
        [
            {"events": melody_events, "channel": 0, "program": melody_program, "name": "melody"},
            {"events": chord_events, "channel": 1, "program": chord_program, "name": "chords"},
        ],
        tempo,
    )
    result = _write_file(mid, file_name, output_dir, "song")
    result.update(_common_meta(tempo, max(melody_beats, chord_beats)))
    result["melody_beats"] = melody_beats
    result["chord_beats"] = chord_beats
    result["melody_events"] = _serialize_events(melody_events)
    result["chords"] = resolved
    return result


# ------------------------------------------------------ multi-track arrange

def _channel_allocator():
    """Yield channels 0,1,...,8,10,...,15 (skipping the drum channel)."""
    for ch in range(16):
        if ch != DRUM_CHANNEL:
            yield ch


def _build_track(track: dict, index: int, step_beats: float, beats_per_chord: float,
                 channels) -> dict:
    if not isinstance(track, dict):
        raise ValueError(f"track {index} must be an object, got {type(track).__name__}")
    ttype = track.get("type")
    if ttype not in ("notes", "chords", "drums"):
        raise ValueError(
            f"track {index} has invalid type {ttype!r}; use 'notes', 'chords' or 'drums'"
        )
    name = track.get("name") or ttype
    start = track.get("start_beat", 0.0)
    _check_range(f"track {index} start_beat", start, 0, 100000)
    program = track.get("program", 0)
    _check_range(f"track {index} program", program, 0, 127, integer=True)

    explicit_channel = track.get("channel")

    if ttype == "drums":
        channel = DRUM_CHANNEL if explicit_channel is None else explicit_channel
        velocity = track.get("velocity", 100)
        accent = track.get("accent_velocity", 120)
        t_step = track.get("step_beats", step_beats)
        _check_range(f"track {index} velocity", velocity, 1, 127, integer=True)
        _check_range(f"track {index} accent_velocity", accent, 1, 127, integer=True)
        _check_range(f"track {index} step_beats", t_step, 0.0625, 16)
        events, resolved, length = _drum_events(track.get("lanes"), t_step, velocity, accent, start)
        detail = {"lanes": resolved}
    elif ttype == "notes":
        channel = next(channels) if explicit_channel is None else explicit_channel
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
        detail = {"events": _serialize_events(events)}
    else:  # chords
        channel = next(channels) if explicit_channel is None else explicit_channel
        velocity = track.get("velocity", 80)
        octave = track.get("octave", 4)
        bpc = track.get("beats_per_chord", beats_per_chord)
        _check_range(f"track {index} velocity", velocity, 1, 127, integer=True)
        _check_range(f"track {index} octave", octave, -1, 9, integer=True)
        _check_range(f"track {index} beats_per_chord", bpc, 0.25, 64)
        events, resolved, length = _chord_events(_parse_chord_list(track.get("chords")), bpc,
                                                 octave, track.get("arpeggiate", False),
                                                 velocity, start)
        detail = {"chords": resolved}

    if explicit_channel is not None:
        _check_range(f"track {index} channel", channel, 0, 15, integer=True)

    summary = {
        "name": name,
        "type": ttype,
        "channel": channel,
        "program": program,
        "start_beat": start,
        "length_beats": length,
        "end_beat": start + length,
        "event_count": len(events),
        **detail,
    }
    return {"events": events, "channel": channel, "program": program,
            "name": name, "end_beat": start + length, "summary": summary}


def render_arrangement(tracks, tempo: int = 120, file_name: str | None = None,
                       output_dir: str | None = None, step_beats: float = 0.5,
                       beats_per_chord: float = 4.0) -> dict:
    """Render any number of named tracks into one multi-track MIDI file.

    `tracks` is a list of track objects, each `{"type": "notes"|"chords"|"drums", ...}`:

    - notes:  {"type": "notes", "notes": [...], "rhythm": "O.o.", "octave": 3,
               "program": 33, "octave_policy": "nearest", "sustain": false}
    - chords: {"type": "chords", "chords": ["Am","F","C","G"], "beats_per_chord": 4,
               "octave": 4, "arpeggiate": false, "program": 0}
    - drums:  {"type": "drums", "lanes": {"kick": "O...", "snare": "..O.", "hat": "oooo"}}

    Shared per-track options: `name`, `velocity`, `start_beat` (beat offset),
    `step_beats`, `channel` (auto-assigned, drums forced to channel 10).
    """
    if not isinstance(tracks, (list, tuple)) or not tracks:
        raise ValueError("tracks must be a non-empty list of track objects")
    _check_range("tempo", tempo, 10, 400, integer=True)
    _check_range("step_beats", step_beats, 0.0625, 16)
    _check_range("beats_per_chord", beats_per_chord, 0.25, 64)

    channels = _channel_allocator()
    built = [_build_track(t, i, step_beats, beats_per_chord, channels)
             for i, t in enumerate(tracks)]

    mid = _build_file(
        [{"events": b["events"], "channel": b["channel"],
          "program": b["program"], "name": b["name"]} for b in built],
        tempo,
    )
    result = _write_file(mid, file_name, output_dir, "arrangement")
    total_beats = max((b["end_beat"] for b in built), default=0.0)
    result.update(_common_meta(tempo, total_beats))
    result["track_count"] = len(built)
    result["tracks"] = [b["summary"] for b in built]
    return result
