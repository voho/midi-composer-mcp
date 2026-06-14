"""Drums, Euclidean rhythm, multi-track arrangement, and playable WAV audio."""

import struct
import wave
from collections import Counter

import pytest
from mido import MidiFile

from midi_composer_mcp.audio import render_midi_to_wav
from midi_composer_mcp.generate import euclidean_rhythm
from midi_composer_mcp.midi_io import (
    DRUM_CHANNEL,
    render_arrangement,
    render_drums,
    render_notes,
)


# ----------------------------------------------------------- euclidean rhythm

def test_euclidean_tresillo():
    result = euclidean_rhythm(3, 8)
    assert result["pattern"] == "O..o..o."  # 3-against-8 tresillo, gaps 3-3-2
    assert result["onset_positions"] == [0, 3, 6]


def test_euclidean_cinquillo_and_edges():
    assert euclidean_rhythm(5, 8)["pattern"].count("o") + euclidean_rhythm(5, 8)["pattern"].count("O") == 5
    assert euclidean_rhythm(4, 16)["onset_positions"] == [0, 4, 8, 12]  # even quarters
    assert euclidean_rhythm(0, 8)["pattern"] == "." * 8
    assert euclidean_rhythm(8, 8)["pattern"] == "O" + "o" * 7


def test_euclidean_rotation_and_validation():
    base = euclidean_rhythm(3, 8)["pattern"]
    rotated = euclidean_rhythm(3, 8, rotation=1)["pattern"]
    assert rotated != base
    assert set(rotated) <= {"O", "o", "."}
    with pytest.raises(ValueError):
        euclidean_rhythm(9, 8)
    with pytest.raises(ValueError):
        euclidean_rhythm(3, 0)


# -------------------------------------------------------------------- drums

def test_render_drums(tmp_path):
    result = render_drums(
        {"kick": "O...O...", "snare": "..O...O.", "hat": "oooooooo"},
        output_dir=str(tmp_path),
    )
    assert result["hit_count"] == 2 + 2 + 8
    mid = MidiFile(result["file"])
    note_ons = [m for t in mid.tracks for m in t if m.type == "note_on" and m.velocity > 0]
    assert all(m.channel == DRUM_CHANNEL for m in note_ons)
    notes = {m.note for m in note_ons}
    assert {36, 38, 42} <= notes  # kick, snare, closed hat GM notes


def test_render_drums_accepts_euclidean_and_numbers(tmp_path):
    result = render_drums(
        {"kick": euclidean_rhythm(3, 8)["pattern"], "56": "o.o.o.o."},
        output_dir=str(tmp_path),
    )
    notes = {l["note"] for l in result["lanes"]}
    assert 36 in notes and 56 in notes  # named kick + raw cowbell number


def test_render_drums_validation(tmp_path):
    with pytest.raises(ValueError, match="lanes"):
        render_drums({}, output_dir=str(tmp_path))
    with pytest.raises(ValueError, match="[Uu]nknown drum"):
        render_drums({"banjo": "O..."}, output_dir=str(tmp_path))


# ------------------------------------------------------------- arrangement

def test_arrange_multitrack(tmp_path):
    result = render_arrangement(
        [
            {"type": "chords", "name": "pad", "chords": ["Am", "F", "C", "G"],
             "beats_per_chord": 4, "octave": 4, "program": 48},
            {"type": "notes", "name": "bass", "notes": ["A", "F", "C", "G"],
             "octave": 2, "program": 33, "step_beats": 4.0},
            {"type": "notes", "name": "lead", "notes": ["A4", "C5", "E5", "D5"],
             "octave": 5, "program": 0},
            {"type": "drums", "name": "drums",
             "lanes": {"kick": "O...O...", "snare": "..O...O.", "hat": "oooooooo"}},
        ],
        tempo=100,
        output_dir=str(tmp_path),
    )
    assert result["track_count"] == 4
    mid = MidiFile(result["file"])
    assert len(mid.tracks) == 5  # conductor + 4
    # channels auto-assigned, drums on the GM percussion channel, no clashes
    channels = [t["channel"] for t in result["tracks"]]
    assert channels == [0, 1, 2, DRUM_CHANNEL]
    names = [m.name for t in mid.tracks for m in t if m.type == "track_name"]
    assert names == ["pad", "bass", "lead", "drums"]


def test_arrange_start_beat_offset(tmp_path):
    result = render_arrangement(
        [
            {"type": "drums", "lanes": {"kick": "O...O...O...O..."}},
            {"type": "notes", "name": "lead", "notes": ["C5", "E5"], "start_beat": 8.0},
        ],
        output_dir=str(tmp_path),
    )
    lead = next(t for t in result["tracks"] if t["name"] == "lead")
    assert lead["start_beat"] == 8.0
    assert lead["events"][0]["start_beat"] == 8.0


def test_arrange_explicit_channel(tmp_path):
    result = render_arrangement(
        [{"type": "notes", "notes": ["C4"], "channel": 5}],
        output_dir=str(tmp_path),
    )
    assert result["tracks"][0]["channel"] == 5


def test_arrange_validation(tmp_path):
    with pytest.raises(ValueError, match="non-empty list"):
        render_arrangement([], output_dir=str(tmp_path))
    with pytest.raises(ValueError, match="invalid type"):
        render_arrangement([{"type": "bongos", "notes": ["C"]}], output_dir=str(tmp_path))


# --------------------------------------------------------- playable WAV audio

def _read_wav(path):
    with wave.open(path, "rb") as w:
        params = (w.getnchannels(), w.getsampwidth(), w.getframerate())
        frames = w.readframes(w.getnframes())
    samples = struct.unpack("<%dh" % (len(frames) // 2), frames)
    return params, samples


def test_audio_preview_is_audible(tmp_path):
    midi = render_notes(["C4", "E4", "G4", "C5"], tempo=120, output_dir=str(tmp_path))
    audio = render_midi_to_wav(midi["file"])
    (channels, width, rate), samples = _read_wav(audio["file"])
    assert (channels, width, rate) == (1, 2, 44100)
    assert audio["note_count"] == 4
    assert max(abs(s) for s in samples) > 1000  # real, non-silent signal
    assert audio["duration_seconds"] > 0


def test_audio_preview_of_full_arrangement(tmp_path):
    midi = render_arrangement(
        [
            {"type": "chords", "chords": ["Am", "F"], "beats_per_chord": 2},
            {"type": "notes", "name": "bass", "notes": ["A", "F"], "octave": 2,
             "program": 33, "step_beats": 2.0},
            {"type": "drums", "lanes": {"kick": "O.o.", "snare": "..O.", "hat": "oooo"}},
        ],
        tempo=110,
        output_dir=str(tmp_path),
    )
    audio = render_midi_to_wav(midi["file"], sample_rate=22050)
    (channels, width, rate), samples = _read_wav(audio["file"])
    assert rate == 22050
    assert audio["drum_count"] > 0 and audio["note_count"] > 0
    assert max(abs(s) for s in samples) > 1000


def test_audio_missing_file():
    with pytest.raises(ValueError, match="not found"):
        render_midi_to_wav("/nonexistent/no.mid")


def test_generated_midi_has_no_hanging_notes(tmp_path):
    # every note_on must be matched by a note_off — a hung note is unplayable
    midi = render_arrangement(
        [
            {"type": "notes", "notes": ["C4", "E4", "G4"], "rhythm": "O.o.O.o."},
            {"type": "chords", "chords": ["C", "G"], "arpeggiate": True},
            {"type": "drums", "lanes": {"kick": "O...", "snare": "..O."}},
        ],
        output_dir=str(tmp_path),
    )
    mid = MidiFile(midi["file"])
    on, off = Counter(), Counter()
    for track in mid.tracks:
        for m in track:
            if m.type == "note_on" and m.velocity > 0:
                on[(m.channel, m.note)] += 1
            elif m.type == "note_off" or (m.type == "note_on" and m.velocity == 0):
                off[(m.channel, m.note)] += 1
    assert on == off and sum(on.values()) > 0
    assert mid.length > 0
