import pytest
from mido import MidiFile

from midi_composer_mcp.generate import groove, list_grooves
from midi_composer_mcp.structure import plan_sections, render_song_structure


# ------------------------------------------------------------------- grooves

def test_groove_presets():
    assert groove("four_on_floor")["pattern"] == "O...O...O...O..."
    assert groove("tresillo")["pattern"] == "O..o..o."
    assert set(groove("son_clave_32")["pattern"]) <= {"O", "o", "."}
    # name normalization
    assert groove("Four On Floor")["name"] == "four_on_floor"
    with pytest.raises(ValueError, match="[Uu]nknown groove"):
        groove("polka")


def test_list_grooves():
    g = list_grooves()
    assert g["count"] >= 10
    assert all(gr["description"] and gr["pattern"] for gr in g["grooves"])


# -------------------------------------------------------------- plan_sections

def test_plan_sections_uniform():
    plan = plan_sections("AABA", bars=8)
    assert plan["total_bars"] == 32
    starts = [s["start_bar"] for s in plan["sections"]]
    assert starts == [0, 8, 16, 24]
    assert plan["sections"][0]["length_beats"] == 32


def test_plan_sections_named_and_per_section_bars():
    plan = plan_sections("intro verse chorus verse chorus outro",
                         bars={"intro": 2, "outro": 4}, tempo=120)
    sec = {s["section"]: s for s in plan["sections"]}
    # named defaults kick in (verse=8, chorus=8) and explicit overrides apply
    assert plan["sections"][0]["bars"] == 2          # intro override
    assert plan["sections"][-1]["bars"] == 4         # outro override
    assert sec_count(plan, "verse") == 2
    assert "start_seconds" in plan["sections"][0]
    assert plan["total_seconds"] > 0


def sec_count(plan, name):
    return sum(1 for s in plan["sections"] if s["section"] == name)


def test_plan_letter_vs_words():
    assert [s["section"] for s in plan_sections("AABA")["sections"]] == ["A", "A", "B", "A"]
    assert [s["section"] for s in plan_sections(["verse", "chorus"])["sections"]] == ["verse", "chorus"]


# -------------------------------------------------------------- arrange_song

def _section(chords, bars=4):
    return {"bars": bars, "tracks": [
        {"type": "chords", "name": "keys", "chords": chords, "beats_per_chord": 4, "octave": 4},
        {"type": "notes", "name": "bass", "notes": [c[0] for c in chords], "step_beats": 4.0, "octave": 2, "program": 33},
    ]}


def test_arrange_song_sequences_sections(tmp_path):
    sections = {
        "verse": _section(["Am", "F", "C", "G"]),
        "chorus": {"bars": 4, "tracks": [
            {"type": "chords", "name": "keys", "chords": ["F", "G", "C", "Am"], "beats_per_chord": 4, "octave": 4},
            {"type": "notes", "name": "bass", "notes": ["F", "G", "C", "A"], "step_beats": 4.0, "octave": 2, "program": 33},
            {"type": "notes", "name": "lead", "notes": ["C5", "E5", "G5", "E5"], "octave": 5, "program": 80},
        ]},
    }
    song = render_song_structure(sections, form="verse chorus verse chorus",
                                 tempo=110, output_dir=str(tmp_path), file_name="song.mid")
    assert song["total_bars"] == 16
    assert song["section_count"] == 4
    assert [s["section"] for s in song["sections"]] == ["verse", "chorus", "verse", "chorus"]
    assert [s["start_bar"] for s in song["sections"]] == [0, 4, 8, 12]
    # same-named tracks are stitched into one MIDI track for the whole song
    names = {t["name"] for t in song["tracks"]}
    assert names == {"keys", "bass", "lead"}
    mid = MidiFile(song["file"])
    track_names = [m.name for t in mid.tracks for m in t if m.type == "track_name"]
    assert sorted(track_names) == ["bass", "keys", "lead"]


def test_arrange_song_lead_only_in_chorus_rests_elsewhere(tmp_path):
    sections = {
        "verse": _section(["Am", "F"]),
        "chorus": {"bars": 2, "tracks": [
            {"type": "chords", "name": "keys", "chords": ["F", "C"], "beats_per_chord": 4, "octave": 4},
            {"type": "notes", "name": "lead", "notes": ["C5", "E5"], "octave": 5},
        ]},
    }
    song = render_song_structure(sections, form="verse chorus", tempo=120, output_dir=str(tmp_path))
    lead = next(t for t in song["tracks"] if t["name"] == "lead")
    # lead exists (from chorus) but has no events during the verse
    assert lead["event_count"] == 2


def test_arrange_song_drums_share_channel(tmp_path):
    sections = {"a": {"bars": 1, "tracks": [
        {"type": "drums", "name": "drums", "lanes": {"kick": "O...O...O...O..."}},
    ]}}
    song = render_song_structure(sections, form="a a", output_dir=str(tmp_path))
    drums = next(t for t in song["tracks"] if t["name"] == "drums")
    assert drums["channel"] == 9 and drums["is_drums"]


def test_arrange_song_validation(tmp_path):
    with pytest.raises(ValueError, match="non-empty"):
        render_song_structure({}, output_dir=str(tmp_path))
    with pytest.raises(ValueError, match="not defined"):
        render_song_structure({"a": {"tracks": [{"type": "chords", "chords": ["C"]}]}},
                              form="a b", output_dir=str(tmp_path))
