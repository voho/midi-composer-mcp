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
    assert [e["notes"] for e in r["per_chord"]] == [
        ["C2", "E2", "G2", "G#2"], ["A2", "C3", "E3", "Gb3"], ["F3", "C3", "A2", "Ab2"], ["G2", "B2", "D3", "Db3"]]
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
    assert bass_line(["G"], "root_octave", octave=3)["track"]["notes"] == ["G3", "G3"] * 2  # G4 folded down
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
        if k < 3:
            assert abs(app.midi - parse_note(bars[k + 1][0]).midi) == 1   # it resolves by a semitone
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
