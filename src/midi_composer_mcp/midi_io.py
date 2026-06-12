"""Deterministic MIDI rendering: note sequences, rhythms and chords to .mid.

These renderers add no musical decisions of their own — they write exactly
the notes, rhythm and chords they are given. Output files land in
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

def _melody_events(notes: list[Note], rhythm: str | None, step_beats: float,
                   velocity: int, accent_velocity: int, sustain: bool) -> tuple[list[dict], float]:
    """Turn placed notes (+ optional rhythm pattern) into timed events."""
    events: list[dict] = []
    if rhythm is None:
        for i, note in enumerate(notes):
            events.append({"note": note, "start": i * step_beats,
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
            "note": note,
            "start": step * step_beats,
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
                  arpeggiate: bool, velocity: int) -> tuple[list[dict], list[dict], float]:
    events: list[dict] = []
    resolved: list[dict] = []
    for i, chord in enumerate(parsed_chords):
        voiced = voice_chord(chord["tones"], octave, chord["bass"])
        start = i * beats_per_chord
        if arpeggiate:
            tone_beats = beats_per_chord / len(voiced)
            for k, note in enumerate(voiced):
                events.append({"note": note, "start": start + k * tone_beats,
                               "duration": tone_beats, "velocity": velocity})
        else:
            for note in voiced:
                events.append({"note": note, "start": start,
                               "duration": beats_per_chord, "velocity": velocity})
        resolved.append({
            "symbol": chord["symbol"] or " ".join(n.pitch_class_name for n in voiced),
            "notes": [n.name for n in voiced],
            "midi": [n.midi for n in voiced],
            "start_beat": start,
            "duration_beats": beats_per_chord,
        })
    return events, resolved, len(parsed_chords) * beats_per_chord


# ------------------------------------------------------------- file writing

def _events_to_track(events: list[dict], channel: int, program: int,
                     name: str | None = None) -> MidiTrack:
    timed: list[tuple[int, int, Message]] = []
    for e in events:
        on_tick = round(e["start"] * TICKS_PER_BEAT)
        off_tick = max(on_tick + 1, round((e["start"] + e["duration"]) * TICKS_PER_BEAT))
        midi = e["note"].midi
        timed.append((on_tick, 1, Message("note_on", note=midi, velocity=e["velocity"], channel=channel)))
        timed.append((off_tick, 0, Message("note_off", note=midi, velocity=0, channel=channel)))
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
            "note": e["note"].name,
            "midi": e["note"].midi,
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
