"""chord_pattern and bass_line (the accompaniment layer).

Unit tests pin the spec examples (traced by hand); the sweeps check the
properties in all 14 tonic spellings, every style, chaining into
arrange_to_midi, and that bad input only ever raises ValueError.
"""

from __future__ import annotations

import pytest

from midi_composer_mcp.accompany import (
    BASS_HIGH,
    BASS_LOW,
    BASS_STYLES,
    PATTERN_PRESETS,
    bass_line,
    chord_pattern,
)
from midi_composer_mcp.chords import parse_chord_symbol
from midi_composer_mcp.diatonic import diatonic_chords
from midi_composer_mcp.harmony import voice_leading
from midi_composer_mcp.midi_io import render_arrangement
from midi_composer_mcp.notes import LETTERS, parse_note
from midi_composer_mcp.roman import read_chord, roman_to_chords
from midi_composer_mcp.scales import resolve_scale_type, scale_notes

TONICS = ["C", "G", "D", "A", "E", "B", "F#", "C#", "F", "Bb", "Eb", "Ab", "Db", "Gb"]


def midis(names):
    return [parse_note(n).midi for n in names]


def letter_degree(name: str, tonic: str) -> int:
    return (LETTERS.index(parse_note(name).letter) - LETTERS.index(parse_note(tonic).letter)) % 7


def rel_pcs(names, tonic: str) -> list[int]:
    t = parse_note(tonic).pitch_class
    return [(parse_note(n).pitch_class - t) % 12 for n in names]


# ================================================================ chord_pattern

def test_alberti_example_exactly():
    r = chord_pattern(["C", "Am", "F", "G"], "alberti", beats_per_chord=2, step_beats=0.5)
    assert r["track"]["notes"] == ["C4", "G4", "E4", "G4", "C4", "A4", "E4", "A4",
                                   "C4", "A4", "F4", "A4", "D4", "B4", "G4", "B4"]
    assert r["track"]["rhythm"] == "OoooOoooOoooOooo"
    assert r["preset"] == "alberti" and r["pattern"] == "^1 3 2 3"
    # the reference voicings are voice_leading's
    assert [e["voicing"] for e in r["per_chord"]] == voice_leading(["C", "Am", "F", "G"])["chords"]
    assert [e["notes"] for e in r["per_chord"]][1] == ["C4", "A4", "E4", "A4"]
    assert r["steps_per_chord"] == [4, 4, 4, 4] and r["total_beats"] == 8


def test_scale_mode_example_exactly():
    r = chord_pattern(["C", "Am", "F", "G"], "^1 2 3 5", beats_per_chord=2, step_beats=0.5,
                      mode="scale", root="C")
    assert [e["notes"] for e in r["per_chord"]] == [
        ["C4", "D4", "E4", "G4"], ["A4", "B4", "C5", "E5"], ["F4", "G4", "A4", "C5"], ["G4", "A4", "B4", "D5"]]
    assert r["track"]["rhythm"] == "Oooo" * 4
    assert r["key"] == "C major"
    assert not any("fallback" in e for e in r["per_chord"])


def test_continue_phase_wraps_the_pattern_across_chords():
    # a 3-step pattern over 2-step chords: 1 2 | 3 1 | 2 3
    r = chord_pattern(["C", "G", "Am"], "^1 2 3", beats_per_chord=1, step_beats=0.5, phase="continue",
                      smooth=False)
    assert [e["notes"] for e in r["per_chord"]] == [["C4", "E4"], ["D5", "G4"], ["C5", "E5"]]
    assert r["track"]["rhythm"] == "OooOoo"
    restart = chord_pattern(["C", "G", "Am"], "^1 2 3", beats_per_chord=1, step_beats=0.5, smooth=False)
    assert [e["notes"] for e in restart["per_chord"]] == [["C4", "E4"], ["G4", "B4"], ["A4", "C5"]]
    assert restart["track"]["rhythm"] == "OoOoOo"


def test_murky_bass_is_broken_octaves():
    r = chord_pattern(["C"], "murky")
    assert r["track"]["notes"] == ["C3", "C4"] * 4
    assert r["track"]["rhythm"] == "Oo" * 4


def test_presets_and_octave_marks():
    assert set(PATTERN_PRESETS) == {"alberti", "up", "down", "updown", "murky"}
    up = chord_pattern(["C"], "up", beats_per_chord=2)
    assert up["track"]["notes"] == ["C4", "E4", "G4", "C5"]      # index 4 of a triad: bottom + octave
    down = chord_pattern(["C"], "down", beats_per_chord=2)
    assert down["track"]["notes"] == ["C5", "G4", "E4", "C4"]
    ud = chord_pattern(["C"], "updown", beats_per_chord=3)
    assert ud["track"]["notes"] == ["C4", "E4", "G4", "C5", "G4", "E4"]
    marks = chord_pattern(["C"], "1' 3,, 2'' 1,'", beats_per_chord=2)
    assert marks["track"]["notes"] == ["C5", "G2", "E6", "C4"]  # ' up, , down, mixed marks cancel
    assert marks["track"]["rhythm"] == "oooo"                  # no '^': no accents


def test_rests_and_sustain():
    r = chord_pattern(["C", "F"], "^1 . 3 .", beats_per_chord=2, sustain=True)
    assert r["track"]["rhythm"] == "O.o.O.o."
    assert r["track"]["notes"] == ["C4", "G4", "C4", "A4"]
    assert r["track"]["sustain"] is True
    assert chord_pattern(["C"], "1")["track"]["sustain"] is False


def test_slash_chord_root_position_puts_the_bass_first():
    r = chord_pattern(["C/E"], "1 2 3 4", beats_per_chord=2, smooth=False)
    assert r["per_chord"][0]["voicing"] == ["E3", "C4", "E4", "G4"]
    assert r["track"]["notes"][0] == "E3"


def test_note_arrays_with_octaves_are_kept_as_written_when_not_smooth():
    # an explicit voicing (written out of order) drives the pattern bottom to top
    r = chord_pattern([["G4", "C4", "E5"]], "1 2 3 4", beats_per_chord=2, smooth=False)
    assert r["per_chord"][0]["voicing"] == ["C4", "G4", "E5"]
    assert r["track"]["notes"] == ["C4", "G4", "E5", "C5"]
    # smooth=True re-voices it with voice_leading
    s = chord_pattern([["G4", "C4", "E5"]], "1 2 3", beats_per_chord=1.5)
    assert s["per_chord"][0]["voicing"] == voice_leading([["G4", "C4", "E5"]])["chords"][0]


def test_rootless_voicings_in_scale_mode_start_on_the_lowest_tone():
    # Levine's rootless ii-V-I (A, B, A voicings), as voice_chords returns it
    rootless = [["F3", "A3", "C4", "E4"], ["F3", "A3", "B3", "E4"], ["E3", "G3", "B3", "D4"]]
    r = chord_pattern(rootless, "1 2 3 4", beats_per_chord=2, smooth=False, mode="scale", root="C")
    assert [e["notes"] for e in r["per_chord"]] == [
        ["F3", "G3", "A3", "B3"], ["F3", "G3", "A3", "B3"], ["E3", "F3", "G3", "A3"]]
    # F A B E names no chord: no root, so the base is simply the lowest tone
    assert read_chord(rootless[1])["root"] is None


def test_scale_mode_spells_as_the_scale_and_falls_back_off_scale():
    r = chord_pattern(["Db", "Bbm", "D"], "1 2 3 4", beats_per_chord=2, smooth=False, mode="scale", root="C#")
    assert r["per_chord"][0]["notes"] == ["C#4", "D#4", "E#4", "F#4"]  # Db is C# in C# major
    assert r["per_chord"][1]["notes"] == ["A#4", "B#4", "C#5", "D#5"]  # Bb is its 6th degree, A#
    assert not any("fallback" in e for e in r["per_chord"][:2])
    assert r["per_chord"][2]["fallback"] == "chord"                    # D is not in C# major
    assert r["per_chord"][2]["notes"] == ["D4", "F#4", "A4", "D5"]
    # the scale base is the chord root even when a slash bass sits below it
    s = chord_pattern(["C/E"], "1 2 3", beats_per_chord=1.5, smooth=False, mode="scale", root="C")
    assert s["track"]["notes"] == ["C4", "D4", "E4"]
    # a pentatonic ladder
    p = chord_pattern(["Am"], "1 2 3 4 5 6", beats_per_chord=3, mode="scale", root="A",
                      scale_type="minor pentatonic", smooth=False)
    assert p["track"]["notes"] == ["A4", "C5", "D5", "E5", "G5", "A5"]


def test_beats_list_uses_chords_track_durations(tmp_path):
    r = chord_pattern(["C", "G", "Am"], "alberti", beats_per_chord=[2, 2, 4], step_beats=0.5)
    assert r["steps_per_chord"] == [4, 4, 8] and r["total_beats"] == 8
    chords_track, pattern_track = r["render_hint"]["tracks"]
    assert chords_track["durations"] == [2.0, 2.0, 4.0] and "beats_per_chord" not in chords_track
    assert len(pattern_track["rhythm"]) * pattern_track["step_beats"] == 8
    out = render_arrangement(r["render_hint"]["tracks"], output_dir=str(tmp_path))
    assert {t["end_beat"] for t in out["tracks"]} == {8.0}


@pytest.mark.parametrize("kwargs,match", [
    ({"beats_per_chord": 1, "step_beats": 0.75}, "whole number"),
    ({"mode": "scale"}, "needs a key"),
    ({"pattern": "1 0 3"}, "1-based"),
    ({"pattern": "^"}, "Invalid pattern step"),
    ({"pattern": "1 x 3"}, "Invalid pattern step"),
    ({"pattern": "^ 1"}, "Invalid pattern step"),
    ({"pattern": ". . ."}, "no note steps"),
    ({"pattern": ""}, "preset"),
    ({"beats_per_chord": [2, 2, 2]}, "one beat length per chord"),
    ({"mode": "arpeggio"}, "mode must be one of"),
    ({"phase": "follow"}, "phase must be one of"),
    ({"octave": 9, "pattern": "1''"}, "MIDI range"),
    ({"smooth": "yes"}, "smooth"),
    ({"root": "H"}, "Invalid note"),
    ({"scale_type": "nonsense"}, "Unknown scale type"),
])
def test_chord_pattern_rejects_bad_input(kwargs, match):
    with pytest.raises(ValueError, match=match):
        chord_pattern(["C", "G"], **kwargs)


# ------------------------------------------------------------ pattern invariants

@pytest.mark.parametrize("mode,pattern", [("chord", "alberti"), ("chord", "^1 2 3 4 3 2 1 3"),
                                          ("scale", "^1 2 3 5 4 3 2 1")])
def test_pattern_over_diatonic_triads_is_the_same_in_every_key(mode, pattern):
    """Root-position voicings: the same intervals, degrees and letter degrees in all keys."""
    ref = None
    for tonic in TONICS:
        symbols = [c["symbol"] for c in diatonic_chords(tonic, "major")["chords"]]
        r = chord_pattern(symbols, pattern, beats_per_chord=4, step_beats=0.5, mode=mode,
                          root=tonic if mode == "scale" else None, smooth=False)
        spans = [e["notes"] for e in r["per_chord"]]
        shape = (
            [[b - a for a, b in zip(midis(s), midis(s)[1:])] for s in spans],   # intervals inside each chord
            [rel_pcs(s, tonic) for s in spans],                                   # scale degrees
            [[letter_degree(n, tonic) for n in s] for s in spans],                # spelled on the key's letters
            r["track"]["rhythm"],
        )
        ref = ref or shape
        assert shape == ref, tonic
        key = {n.name for n in scale_notes(resolve_scale_type("major"), parse_note(tonic))}
        assert all(parse_note(n).without_octave().name in key for n in r["track"]["notes"]), tonic


@pytest.mark.parametrize("tonic", TONICS)
def test_smooth_pattern_plays_only_chord_tones_and_chains(tonic, tmp_path):
    symbols = [c["symbol"] for c in diatonic_chords(tonic, "major")["chords"]]
    r = chord_pattern(symbols, "alberti", beats_per_chord=2, step_beats=0.5)
    for sym, entry in zip(symbols, r["per_chord"]):
        root, ctype, _ = parse_chord_symbol(sym)
        tones = {(root.pitch_class + s) % 12 for s in ctype.intervals}
        assert {parse_note(n).pitch_class for n in entry["notes"]} <= tones
        assert set(entry["notes"]) <= set(entry["voicing"])    # alberti uses indices 1-3 only
    track = r["track"]
    assert len(track["rhythm"]) * track["step_beats"] == r["total_beats"] == 14
    out = render_arrangement(r["render_hint"]["tracks"], output_dir=str(tmp_path), file_name="p.mid")
    assert {t["end_beat"] for t in out["tracks"]} == {14.0}


def test_pattern_chains_from_roman_numerals_and_progressions(tmp_path):
    symbols = roman_to_chords("I V6 vi IV", "G")["symbols"]
    r = chord_pattern(symbols, "up", mode="scale", root="G", beats_per_chord=2)
    assert r["per_chord"][1]["symbol"] == "D/F#"
    assert r["per_chord"][1]["notes"][0].startswith("D")    # the scale run starts on the root, not the bass
    render_arrangement(r["render_hint"]["tracks"], output_dir=str(tmp_path), file_name="r.mid")


# =================================================================== bass_line

def test_walking_example_exactly():
    r = bass_line(["C", "Am", "F", "G"], "walking")
    # the loop's closing approach aims at the line's real first note, C2 (not a C3 nearest D3)
    assert [e["notes"] for e in r["per_chord"]] == [
        ["C2", "E2", "G2", "G#2"], ["A2", "C3", "E3", "Gb3"], ["F3", "C3", "A2", "Ab2"], ["G2", "B2", "D3", "Db2"]]
    assert r["track"]["rhythm"] == "Oooo" * 4
    assert r["track"]["sustain"] is False and r["track"]["program"] == 33
    assert r["track"]["name"] == "bass" and r["track"]["step_beats"] == 1.0


def test_root_example_exactly():
    r = bass_line(["C", "G/B", "Am", "F"])
    assert r["track"]["notes"] == ["C2", "B1", "A1", "F1"]
    assert r["track"]["rhythm"] == "O..." * 4 and r["track"]["sustain"] is True
    assert [e["symbol"] for e in r["per_chord"]] == ["C", "G/B", "Am", "F"]


def test_walking_ending_root_takes_the_next_chord_tone():
    r = bass_line(["C", "Am", "F", "G"], "walking", ending="root")
    assert r["per_chord"][-1]["notes"] == ["G2", "B2", "D3", "G3"]   # G triad: 3rd, 5th, then the octave
    assert r["per_chord"][:3] == bass_line(["C", "Am", "F", "G"], "walking")["per_chord"][:3]
    assert bass_line(["C", "G7"], "walking", ending="root")["per_chord"][-1]["notes"] == ["G2", "B2", "D3", "F3"]


@pytest.mark.parametrize("chords,expected", [
    (["D", "G"], "F#2"),    # target above: from below, on the letter below (G -> F#)
    (["C", "Eb"], "D2"),    # Eb -> D
    (["D", "C"], "Db2"),    # target below: from above, on the letter above (C -> Db)
    (["Eb", "Gb"], "F2"),   # Gb from below -> F
    (["A", "C"], "B2"),     # C from below -> B
])
def test_approach_spelling(chords, expected):
    r = bass_line(chords, "approach", ending="root")
    assert r["per_chord"][0]["notes"] == [r["per_chord"][0]["notes"][0], expected]
    nxt = parse_note(r["per_chord"][1]["notes"][0])
    assert abs(parse_note(expected).midi - nxt.midi) == 1
    assert abs(LETTERS.index(parse_note(expected).letter) - LETTERS.index(nxt.letter)) in (1, 6)


def test_approach_switches_side_instead_of_repeating_the_previous_note():
    # F lies above E, so the approach would come from below — E2, the note just played: from above instead
    r = bass_line(["E", "F"], "approach", ending="root")
    assert r["per_chord"][0]["notes"] == ["E2", "Gb2"]
    assert r["per_chord"][1]["notes"] == ["F2", "F2"]       # ending='root': the last chord has no approach
    # the walking example's Am bar (E3 -> F3) does the same: Gb3 (pinned in test_walking_example_exactly)


def test_approach_folded_at_the_top_of_the_register_still_resolves_by_a_semitone():
    # D major I-vi: B2 D3 F#3, then G3 would be approached by F#3 (a repeat) or Ab3 (above G3):
    # Ab3 folds to Ab2 and the next bass is placed nearest it, on G2
    r = bass_line(["B", "G"], "walking", ending="root", octave=2)
    bars = [e["notes"] for e in r["per_chord"]]
    assert bars[0] == ["B2", "D#3", "F#3", "Ab2"] and bars[1][0] == "G2"


def test_every_style_stays_in_the_walking_register():
    assert bass_line(["E"], "root_fifth", octave=1)["track"]["notes"] == ["E1", "B1"]   # B0 folded up
    # G4 would pass G3: the pair is played an octave lower instead of collapsing to G3 G3
    assert bass_line(["G"], "root_octave", octave=3)["track"]["notes"] == ["G2", "G3"] * 2
    progressions = [["C", "Am", "F", "G"], ["E", "B7", "C#m7", "A"], ["G", "D/F#", "Em7", "Cmaj7", "Bb13"],
                    ["F#m7b5", "B7b9", "Em"], [["C", "Db", "D"], ["E3", "G3", "C4"]]]
    for chords in progressions:
        for style in BASS_STYLES:
            for octave in range(5):
                for ending in ("loop", "root"):
                    r = bass_line(chords, style, octave=octave, ending=ending, step_beats=0.5)
                    assert all(BASS_LOW <= m <= BASS_HIGH for m in midis(r["track"]["notes"])), (chords, style)


def test_root_fifth_and_root_octave_patterns():
    r = bass_line(["C", "Am", "F", "G"], "root_fifth")
    assert r["track"]["rhythm"] == "O.o." * 4 and r["track"]["sustain"] is False
    assert [e["notes"] for e in r["per_chord"]] == [["C2", "G1"], ["A1", "E1"], ["F1", "C2"], ["G1", "D2"]]
    r = bass_line(["C", "Am", "F", "G"], "root_octave")
    assert r["track"]["rhythm"] == "Oo" * 8
    # each bass is placed from the previous bass, so the octaves never collapse into repeats
    assert [e["notes"] for e in r["per_chord"]] == [["C2", "C3"] * 2, ["A1", "A2"] * 2, ["F1", "F2"] * 2,
                                                    ["G1", "G2"] * 2]
    # the chord's own fifth: b5 of a diminished chord, #5 of an augmented one
    assert bass_line(["Bdim"], "root_fifth", octave=2)["track"]["notes"] == ["B2", "F2"]
    assert bass_line(["Caug"], "root_fifth", octave=2)["track"]["notes"] == ["C2", "G#1"]


def test_root_octave_never_collapses_into_unisons():
    # a bass from Ab2 up would fold its octave back onto itself: the pair moves an octave down
    r = bass_line(["A", "D", "E", "A"], "root_octave")
    assert [e["notes"] for e in r["per_chord"]] == [["A1", "A2"] * 2, ["D2", "D3"] * 2, ["E2", "E3"] * 2,
                                                    ["A1", "A2"] * 2]
    assert [e["notes"] for e in bass_line(["G", "C"], "root_octave")["per_chord"]] == [
        ["G2", "G3"] * 2, ["C2", "C3"] * 2]
    assert [e["notes"] for e in bass_line(["E", "A", "B"], "root_octave")["per_chord"]] == [
        ["E2", "E3"] * 2, ["A1", "A2"] * 2, ["B1", "B2"] * 2]
    assert bass_line(["C"], "root_octave", octave=3)["track"]["notes"] == ["C2", "C3"] * 2
    assert bass_line(["E"], "root_octave", octave=0)["track"]["notes"] == ["E1", "E2"] * 2   # E0 folded up


def test_root_fifth_over_a_six_four_chord_alternates_with_the_root():
    # the bass already is the fifth (C/G, G7/D): alternate with the root, never with the bass itself
    r = bass_line(["C/G", "G7/D"], "root_fifth")
    assert [e["notes"] for e in r["per_chord"]] == [["G2", "C2"], ["D2", "G1"]]
    # a set with no fifth: the alternate is its root, which would fold onto the bass: an octave above
    assert bass_line([["C1", "D1"]], "root_fifth", octave=1)["track"]["notes"] == ["C2", "C3"]


def test_loop_ending_approaches_the_actual_first_note():
    # the loop replays C2, so the closing approach resolves into C2, not into a C3 nearest the last note
    assert bass_line(["C"], "walking")["track"]["notes"] == ["C2", "E2", "G2", "Db2"]
    assert [e["notes"] for e in bass_line(["E", "A", "B", "E"], "approach")["per_chord"]] == [
        ["E2", "G#2"], ["A2", "A#2"], ["B2", "D#3"], ["E3", "F2"]]
    # E1 after F1: D#1 is below the register and F1 repeats the previous note, so the repeat is kept
    assert [e["notes"] for e in bass_line(["Em7", "Fm7"], "approach", octave=1)["per_chord"]] == [
        ["E1", "Gb1"], ["F1", "F1"]]
    # G3 after F#3: Ab3 is above the register, so the walk repeats F#3 and resolves up into G3
    assert bass_line(["G", "B"], "walking", octave=3)["per_chord"][1]["notes"] == ["B2", "D#3", "F#3", "F#3"]
    # ending='root' still gives the last chord no approach
    assert bass_line(["E", "A", "B", "E"], "approach", ending="root")["per_chord"][-1]["notes"] == ["E3", "E3"]


def test_loop_seam_resolves_by_a_semitone_when_the_section_repeats(tmp_path):
    import mido
    from midi_composer_mcp.structure import render_song_structure

    b = bass_line(["C", "Am", "F", "G"], "walking")
    out = render_song_structure({"A": {"tracks": b["render_hint"]["tracks"]}}, form="A A",
                                output_dir=str(tmp_path), file_name="loop.mid")
    ons = [msg.note for track in mido.MidiFile(out["file"]).tracks for msg in track
           if msg.type == "note_on" and msg.velocity > 0]
    assert ons[:16] == ons[16:] == midis(b["track"]["notes"])
    assert abs(ons[15] - ons[16]) == 1                        # Db2 -> C2 across the repeat


def test_sustained_bass_never_holds_across_a_chord_change(tmp_path):
    def events(result):
        out = render_arrangement(result["render_hint"]["tracks"], output_dir=str(tmp_path))
        return [(e["note"], e["start_beat"], e["duration_beats"]) for e in out["tracks"][-1]["events"]]

    # a rhythm that rests on the downbeat: a held style is played detached over two or more chords
    r = bass_line(["C", "Db"], "root", rhythm="..O.")
    assert r["track"]["sustain"] is False
    assert events(r) == [("C2", 2.0, 1.0), ("Db2", 6.0, 1.0)]
    assert events(bass_line(["C", "G"], "approach", rhythm=".O.o")) == [
        ("C2", 1.0, 1.0), ("Ab1", 3.0, 1.0), ("G1", 5.0, 1.0), ("B1", 7.0, 1.0)]
    # the default rhythms open on the bass, so they stay held within each chord
    assert events(bass_line(["C", "F"], "approach")) == [("C2", 0.0, 3.0), ("E2", 3.0, 1.0),
                                                           ("F2", 4.0, 3.0), ("Db2", 7.0, 1.0)]
    # asking for a hold that would cross the change is an error; sustain=False plays any style detached
    with pytest.raises(ValueError, match="wrong chord"):
        bass_line(["C", "Db"], "root", rhythm="..O.", sustain=True)
    assert bass_line(["C", "F"], "root", sustain=False)["track"]["sustain"] is False
    assert bass_line(["C", "F"], "walking", sustain=True)["track"]["sustain"] is True
    # one chord (nothing to cross) and a pedal (the same note under every chord) keep their hold
    assert bass_line(["C"], "root", rhythm="..O.")["track"]["sustain"] is True
    assert bass_line(["C", "F"], "pedal", rhythm="..O.", sustain=True)["track"]["sustain"] is True
    with pytest.raises(ValueError, match="sustain must be true or false"):
        bass_line(["C", "F"], sustain="yes")


def test_sustained_pattern_refuses_to_hold_across_a_chord_change():
    with pytest.raises(ValueError, match="chord 1 \\(C\\) over the rest that opens chord 2 \\(B\\)"):
        chord_pattern(["C", "B"], ". 1 2 3", beats_per_chord=2, sustain=True, smooth=False)
    # continue: 1 2 | . 1 | 2 . — the second chord opens on the pattern's rest
    with pytest.raises(ValueError, match="opens chord 2"):
        chord_pattern(["C", "G", "Am"], "^1 2 .", beats_per_chord=1, step_beats=0.5, phase="continue",
                      sustain=True)
    # rests inside a chord hold within it; one chord has no change to cross; sustain off is untouched
    assert chord_pattern(["C", "F"], "^1 . 3 .", beats_per_chord=2, sustain=True)["track"]["sustain"] is True
    assert chord_pattern(["C"], ". 1 2 3", beats_per_chord=2, sustain=True)["track"]["notes"] == ["C4", "E4", "G4"]
    assert chord_pattern(["C", "B"], ". 1 2 3", beats_per_chord=2, smooth=False)["track"]["rhythm"] == ".ooo" * 2


def test_pedal_ignores_the_chords():
    r = bass_line(["C", "F", "G7", "Am"], "pedal")
    assert r["track"]["notes"] == ["C2"] * 4 and r["track"]["sustain"] is True
    assert bass_line(["C", "F", "G7"], "pedal", pedal="G")["track"]["notes"] == ["G2"] * 3
    assert bass_line(["C", "F"], "pedal", pedal="D4")["track"]["notes"] == ["D3"] * 2   # folded into E1-G3


def test_user_rhythm_is_applied_to_every_chord():
    r = bass_line(["C", "G"], "walking", beats_per_chord=4, step_beats=0.5, rhythm="O.o.o.o.")
    assert r["track"]["rhythm"] == "O.o.o.o." * 2
    assert len(r["track"]["notes"]) == 8
    r = bass_line(["C", "F"], "root_fifth", rhythm="Oo.o")
    assert [e["notes"] for e in r["per_chord"]] == [["C2", "G1", "C2"], ["F2", "C2", "F2"]]


def test_note_arrays_use_their_lowest_note_as_bass():
    pads = voice_leading(["C", "G", "Am", "F"])["chords"]          # C4 E4 G4 / B3 D4 G4 / C4 E4 A4 / C4 F4 A4
    assert bass_line(pads)["track"]["notes"] == ["C2", "B1", "C2", "C2"]  # the pads' own lowest notes
    assert bass_line([["G4", "E3", "C4"], ["C3", "F3", "A3"]])["track"]["notes"] == ["E2", "C2"]
    assert bass_line([["E", "G", "C"]])["track"]["notes"] == ["E2"]  # no octaves: the first note


def test_beats_list_gives_per_chord_default_rhythms(tmp_path):
    r = bass_line(["C", "G", "Am"], "walking", beats_per_chord=[2, 2, 4])
    assert r["steps_per_chord"] == [2, 2, 4] and r["track"]["rhythm"] == "OoOoOooo"
    out = render_arrangement(r["render_hint"]["tracks"], output_dir=str(tmp_path))
    assert out["tracks"][0]["end_beat"] == 8.0 and out["tracks"][0]["program"] == 33


@pytest.mark.parametrize("kwargs,match", [
    ({"rhythm": "O.o"}, "rhythm has 3 steps"),
    ({"rhythm": "...."}, "no onsets"),
    ({"style": "slap"}, "style must be one of"),
    ({"ending": "fade"}, "ending must be one of"),
    ({"step_beats": 1.5}, "whole number"),
    ({"style": "pedal", "pedal": "H2"}, "Invalid note"),
    ({"style": "pedal", "pedal": 7}, "Not a note name"),
    ({"pedal": "C"}, "only used with style='pedal'"),
    ({"octave": 7}, "octave"),
    ({"beats_per_chord": [4]}, "one beat length per chord"),
    ({"beats_per_chord": [4, 3], "rhythm": "O..."}, "rhythm has 4 steps"),
])
def test_bass_line_rejects_bad_input(kwargs, match):
    with pytest.raises(ValueError, match=match):
        bass_line(["C", "G"], **kwargs)


# ------------------------------------------------------------ bass invariants

@pytest.mark.parametrize("tonic", TONICS)
def test_walking_line_is_letter_spelled_in_every_key(tonic):
    """I-vi-IV-V: bars start on the bass, walk on chord tones as the chord spells them, and end on a
    semitone approach on the neighbouring letter. (The pitch classes of the middle tones are NOT the
    same in every key: whether a bar ascends or turns down depends on the absolute E1-G3 register.)"""
    symbols = roman_to_chords("I vi IV V", tonic)["symbols"]
    r = bass_line(symbols, "walking")
    assert r["track"]["rhythm"] == "Oooo" * 4
    bars = [e["notes"] for e in r["per_chord"]]
    for k, (sym, bar) in enumerate(zip(symbols, bars)):
        root, ctype, _ = parse_chord_symbol(sym)
        chord = {n.name for n in __import__("midi_composer_mcp.chords", fromlist=["x"]).chord_notes(ctype, root)}
        assert parse_note(bar[0]).without_octave().name == root.name
        assert all(parse_note(n).without_octave().name in chord for n in bar[1:3]), (tonic, bar)
        target = parse_chord_symbol(symbols[(k + 1) % 4])[0]
        app = parse_note(bar[3])
        assert (app.pitch_class - target.pitch_class) % 12 in (1, 11), (tonic, bar)
        assert (LETTERS.index(app.letter) - LETTERS.index(target.letter)) % 7 in (1, 6), (tonic, bar)
        nxt = bars[k + 1][0] if k < 3 else bars[0][0]   # the loop replays the first note
        assert abs(app.midi - parse_note(nxt).midi) == 1, (tonic, bar)   # it resolves by a semitone
    assert all(BASS_LOW <= m <= BASS_HIGH for m in midis(r["track"]["notes"]))


@pytest.mark.parametrize("style", BASS_STYLES)
def test_root_style_family_is_transposition_consistent(style):
    """Each bar's first note is the chord's bass, spelled as written, in every key."""
    for tonic in TONICS:
        symbols = roman_to_chords("I vi ii V", tonic)["symbols"]
        r = bass_line(symbols, style)
        for sym, entry in zip(symbols, r["per_chord"]):
            first = parse_note(entry["notes"][0]).without_octave().name
            expected = symbols[0] if style == "pedal" else sym
            assert first == parse_chord_symbol(expected)[0].name, (tonic, style)


_LOOP_PROGRESSIONS = ["I vi IV V", "I V vi IV", "ii7 V7 Imaj7", "I VI7 ii7 V7", "vi IV I V", "I"]


@pytest.mark.parametrize("tonic", TONICS)
def test_loop_seam_resolves_into_the_first_note_in_every_key(tonic):
    """ending='loop': the last note is a letter-neighbour semitone from the line's first note, in
    every key, register and progression. It repeats the note before it only when the other side
    is outside E1-G3 (E1 after F1, G3 after F#3)."""
    for numerals in _LOOP_PROGRESSIONS:
        symbols = roman_to_chords(numerals, tonic)["symbols"]
        for style in ("walking", "approach"):
            for octave in range(5):
                for step in (1.0, 0.5):
                    notes = [parse_note(n) for n in bass_line(symbols, style, octave=octave,
                                                              step_beats=step)["track"]["notes"]]
                    first, last, before = notes[0], notes[-1], notes[-2]
                    where = (tonic, numerals, style, octave, step)
                    assert abs(last.midi - first.midi) == 1, where
                    assert (LETTERS.index(last.letter) - LETTERS.index(first.letter)) % 7 in (1, 6), where
                    assert all(BASS_LOW <= n.midi <= BASS_HIGH for n in notes), where
                    if last.midi == before.midi:
                        other = 2 * first.midi - last.midi
                        assert not BASS_LOW <= other <= BASS_HIGH, where


@pytest.mark.parametrize("tonic", TONICS)
def test_root_octave_pairs_are_octaves_in_every_key(tonic):
    """Every root_octave pair is the bass and the same note exactly an octave up, inside E1-G3."""
    for numerals in ("I IV V I", "I vi IV V", "ii7 V7 Imaj7", "I V6 vi IV64"):
        symbols = roman_to_chords(numerals, tonic)["symbols"]
        for octave in range(5):
            r = bass_line(symbols, "root_octave", octave=octave)
            for sym, entry in zip(symbols, r["per_chord"]):
                low, high = midis(entry["notes"][0::2]), midis(entry["notes"][1::2])
                assert len(set(low)) == 1 and len(set(high)) == 1, (tonic, sym, octave, entry)
                assert high[0] - low[0] == 12, (tonic, sym, octave, entry)
                assert BASS_LOW <= low[0] and high[0] <= BASS_HIGH, (tonic, sym, octave, entry)
                assert {parse_note(n).name[:-1] for n in entry["notes"]} == {parse_chord_symbol(sym)[2].name
                                                                             if parse_chord_symbol(sym)[2]
                                                                             else parse_chord_symbol(sym)[0].name}


@pytest.mark.parametrize("tonic", TONICS)
def test_root_fifth_alternate_is_never_the_bass_itself_in_every_key(tonic):
    """The alternate is the chord's fifth — or its root over a 6/4 — and never the bass's own pitch."""
    for numerals in ("I64 V7 I", "IV64 I V65 I6", "I IV V I", "ii65 V43 I"):
        symbols = roman_to_chords(numerals, tonic)["symbols"]
        for octave in range(5):
            r = bass_line(symbols, "root_fifth", octave=octave)
            for sym, entry in zip(symbols, r["per_chord"]):
                root, ctype, slash = parse_chord_symbol(sym)
                bass, alt = parse_note(entry["notes"][0]), parse_note(entry["notes"][1])
                fifth = next(t for (label, _), t in zip(ctype.degrees,
                             __import__("midi_composer_mcp.chords", fromlist=["x"]).chord_notes(ctype, root))
                             if label.lstrip("#b") == "5")
                expected = root if fifth.pitch_class == bass.pitch_class else fifth
                assert alt.name[:-1] == expected.name, (tonic, sym, octave, entry)
                assert alt.midi != bass.midi and BASS_LOW <= alt.midi <= BASS_HIGH, (tonic, sym, octave, entry)


_RHYTHMS = ["O...", ".O..", "..O.", "...O", "O.o.", ".o.o", "Oo.."]


@pytest.mark.parametrize("tonic", TONICS)
def test_no_bass_note_sounds_across_a_chord_change(tonic, tmp_path):
    """Rendered, no bass event (bar the pedal's) starts under one chord and sounds into the next."""
    symbols = roman_to_chords("I vi ii V", tonic)["symbols"]
    for style in BASS_STYLES:
        for rhythm in _RHYTHMS:
            for sustain in (None, True, False):
                try:
                    r = bass_line(symbols, style, rhythm=rhythm, sustain=sustain)
                except ValueError as e:
                    assert sustain is True and rhythm[0] == "." and style != "pedal", e
                    continue
                out = render_arrangement(r["render_hint"]["tracks"], output_dir=str(tmp_path), file_name="b.mid")
                for e in out["tracks"][0]["events"]:
                    start, end = e["start_beat"], e["start_beat"] + e["duration_beats"]
                    if style != "pedal":
                        assert int(start // 4) == int((end - 1e-9) // 4), (tonic, style, rhythm, sustain, e)


@pytest.mark.parametrize("phase", ["restart", "continue"])
def test_no_sustained_pattern_note_sounds_across_a_chord_change(phase, tmp_path):
    patterns = ["^1 . 3 .", ". 1 2 3", "^1 2 .", "1 . .", ". . 1", "^1 3 2 3", "1 2 3 . 2"]
    for tonic in TONICS:
        symbols = roman_to_chords("I vi IV V", tonic)["symbols"]
        for pattern in patterns:
            for bpc in (1, 1.5, 2):
                try:
                    r = chord_pattern(symbols, pattern, beats_per_chord=bpc, phase=phase, sustain=True)
                except ValueError as e:
                    assert "wrong chord" in str(e) or "no notes" in str(e), e
                    continue
                out = render_arrangement(r["render_hint"]["tracks"], output_dir=str(tmp_path), file_name="p.mid")
                for e in out["tracks"][1]["events"]:
                    start, end = e["start_beat"], e["start_beat"] + e["duration_beats"]
                    assert int(start // bpc) == int((end - 1e-9) // bpc), (tonic, pattern, bpc, phase, e)


def test_bass_and_pattern_chain_into_one_arrangement(tmp_path):
    chords = ["C", "Am", "F", "G"]
    pattern = chord_pattern(chords, "alberti", beats_per_chord=4, step_beats=0.5)
    bass = bass_line(chords, "walking")
    tracks = pattern["render_hint"]["tracks"] + bass["render_hint"]["tracks"]
    out = render_arrangement(tracks, output_dir=str(tmp_path), file_name="a.mid")
    assert [t["name"] for t in out["tracks"]] == ["chords", "pattern", "bass"]
    assert {t["end_beat"] for t in out["tracks"]} == {16.0}


# ------------------------------------------------ only ValueErrors, never crashes

_BAD_VALUES = [None, 7, -1, 3.5, 2.0, True, "", "zz", [], [None], {}, {"a": 1}, ["C", 5], "C" * 5,
               ["C", "Db", "D"], [["C", "Db", "D"], "G"], "0 4 7", ["C", ".", "E"]]


@pytest.mark.parametrize("fn", [chord_pattern, bass_line])
def test_bad_input_only_raises_value_errors(fn):
    import inspect
    failures = []
    for pname in inspect.signature(fn).parameters:
        for bad in _BAD_VALUES:
            kwargs = {"chords": ["C", "G"], pname: bad}
            if pname == "pedal":
                kwargs["style"] = "pedal"
            try:
                fn(**kwargs)
            except ValueError:
                pass
            except Exception as e:  # noqa: BLE001 - the list of failures is the point
                failures.append(f"{fn.__name__}({pname}={bad!r}) raised {type(e).__name__}: {e}")
    assert not failures, "\n".join(failures)
