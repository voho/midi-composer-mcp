import base64
import os

import pytest
from mido import MidiFile

from midi_composer_mcp.diatonic import degrees_to_chords
from midi_composer_mcp.midi_io import assign_octaves, render_chords, render_notes, render_song, voice_chord
from midi_composer_mcp.notes import parse_notes
from midi_composer_mcp.scales import scale_info


def _note_ons(path):
    mid = MidiFile(path)
    return [m for track in mid.tracks for m in track if m.type == "note_on" and m.velocity > 0]


def test_assign_octaves_ascending_scale():
    notes = parse_notes(scale_info("major", "C")["notes"])
    placed = assign_octaves(notes, 4, "ascending")
    midis = [n.midi for n in placed]
    assert midis == [60, 62, 64, 65, 67, 69, 71, 72]  # ends on C5


def test_assign_octaves_nearest():
    placed = assign_octaves(parse_notes("C B C"), 4, "nearest")
    assert [n.midi for n in placed] == [60, 59, 60]  # B drops below, C returns
    explicit = assign_octaves(parse_notes("C4 B5 C3"), 4, "nearest")
    assert [n.midi for n in explicit] == [60, 83, 48]  # explicit octaves win


def test_voice_chord_stacks_upwards():
    voiced = voice_chord(parse_notes("C E G"), 4)
    assert [n.midi for n in voiced] == [60, 64, 67]
    # tones below the previous one get pushed up an octave
    voiced = voice_chord(parse_notes("G B D"), 3)
    assert [n.midi for n in voiced] == [55, 59, 62]


def test_voice_chord_with_bass():
    tones = parse_notes("C E G")
    bass = parse_notes("E")[0]
    voiced = voice_chord(tones, 4, bass)
    assert [n.midi for n in voiced] == [52, 60, 64, 67]  # E3 below C4


def test_render_scale_to_midi(tmp_path):
    notes = scale_info("major", "C")["notes"]
    result = render_notes(notes, octave_policy="ascending", output_dir=str(tmp_path))
    assert os.path.isfile(result["file"])
    assert result["note_count"] == 8
    assert result["total_beats"] == 4.0
    assert [e["midi"] for e in result["events"]] == [60, 62, 64, 65, 67, 69, 71, 72]
    ons = _note_ons(result["file"])
    assert [m.note for m in ons] == [60, 62, 64, 65, 67, 69, 71, 72]
    # base64 payload matches the file on disk
    with open(result["file"], "rb") as fh:
        assert base64.b64decode(result["base64"]) == fh.read()


def test_render_notes_with_rhythm(tmp_path):
    result = render_notes(
        ["C5", "D5", "E5"],
        rhythm="O.o.O..o",
        step_beats=0.5,
        output_dir=str(tmp_path),
    )
    # 4 sounding notes; notes wrap around the 3-note pool
    assert result["note_count"] == 4
    assert [e["note"] for e in result["events"]] == ["C5", "D5", "E5", "C5"]
    assert [e["start_beat"] for e in result["events"]] == [0.0, 1.0, 2.0, 3.5]
    velocities = [e["velocity"] for e in result["events"]]
    assert velocities[0] > velocities[1]  # O is accented, o is not
    assert result["total_beats"] == 4.0


def test_render_notes_sustain_extends(tmp_path):
    result = render_notes(["C4"], rhythm="O...", sustain=True, output_dir=str(tmp_path))
    assert result["events"][0]["duration_beats"] == 2.0  # 4 steps x 0.5 beats
    plain = render_notes(["C4"], rhythm="O...", sustain=False, output_dir=str(tmp_path))
    assert plain["events"][0]["duration_beats"] == 0.5


def test_render_chords(tmp_path):
    result = render_chords(["C", "Am", "F", "G7"], output_dir=str(tmp_path))
    assert result["chord_count"] == 4
    assert result["total_beats"] == 16.0
    assert result["chords"][0]["midi"] == [60, 64, 67]
    assert result["chords"][3]["notes"] == ["G4", "B4", "D5", "F5"]
    ons = _note_ons(result["file"])
    assert len(ons) == 3 + 3 + 3 + 4


def test_render_chords_accepts_note_arrays_and_slash(tmp_path):
    result = render_chords([["C4", "E4", "G4"], "C/E"], output_dir=str(tmp_path))
    assert result["chords"][0]["midi"] == [60, 64, 67]
    assert result["chords"][1]["midi"][0] < result["chords"][1]["midi"][1]
    assert result["chords"][1]["symbol"] == "C/E"


def test_render_chords_from_progression_output(tmp_path):
    progression = degrees_to_chords("A", "minor", "i VI III VII")
    result = render_chords(progression["symbols"], output_dir=str(tmp_path))
    assert result["chord_count"] == 4
    assert [c["symbol"] for c in result["chords"]] == ["Am", "F", "C", "G"]


def test_render_chords_arpeggiated(tmp_path):
    block = render_chords(["C"], output_dir=str(tmp_path))
    arp = render_chords(["C"], arpeggiate=True, output_dir=str(tmp_path))
    starts = [e["start_beat"] for e in arp["chords"]]
    assert len(_note_ons(arp["file"])) == len(_note_ons(block["file"])) == 3
    block_msgs = MidiFile(block["file"])
    # block chord: all note_ons at time 0 offsets within the track
    ons = [m for m in block_msgs.tracks[1] if m.type == "note_on" and m.velocity > 0]
    assert all(m.time == 0 for m in ons[1:])


def test_render_song_two_tracks(tmp_path):
    result = render_song(
        melody_notes=["E5", "G5", "A5", "C6"],
        chords=["Am", "F"],
        beats_per_chord=2.0,
        output_dir=str(tmp_path),
    )
    mid = MidiFile(result["file"])
    assert len(mid.tracks) == 3  # conductor + melody + chords
    assert result["melody_beats"] == 2.0
    assert result["chord_beats"] == 4.0
    names = [m.name for t in mid.tracks for m in t if m.type == "track_name"]
    assert names == ["melody", "chords"]


def test_render_song_with_rhythm(tmp_path):
    result = render_song(
        melody_notes=["A4", "C5", "E5"],
        melody_rhythm="O.oO" * 4,
        chords=["Am", "F", "C", "G"],
        step_beats=0.5,
        beats_per_chord=2.0,
        output_dir=str(tmp_path),
    )
    assert result["melody_beats"] == result["chord_beats"] == 8.0


def test_tempo_written(tmp_path):
    result = render_notes(["C"], tempo=90, output_dir=str(tmp_path))
    mid = MidiFile(result["file"])
    tempos = [m for t in mid.tracks for m in t if m.type == "set_tempo"]
    assert len(tempos) == 1
    assert round(60_000_000 / tempos[0].tempo) == 90


def test_file_name_sanitized(tmp_path):
    result = render_notes(["C"], file_name="../weird name!.mid", output_dir=str(tmp_path))
    assert os.path.dirname(result["file"]) == str(tmp_path)
    assert "/" not in result["file_name"]
    assert result["file_name"].endswith(".mid")


def test_out_of_range_note_rejected(tmp_path):
    with pytest.raises(ValueError, match="MIDI range"):
        render_notes(["G9", "A9"], output_dir=str(tmp_path))  # A9 = 129 > 127


def test_validation_errors(tmp_path):
    with pytest.raises(ValueError):
        render_notes(["C"], tempo=5000, output_dir=str(tmp_path))
    with pytest.raises(ValueError):
        render_notes(["C"], velocity=0, output_dir=str(tmp_path))
    with pytest.raises(ValueError):
        render_notes(["C"], octave_policy="sideways", output_dir=str(tmp_path))
    with pytest.raises(ValueError):
        render_chords([], output_dir=str(tmp_path))
    with pytest.raises(ValueError):
        render_notes(["C"], rhythm="O.x", output_dir=str(tmp_path))
