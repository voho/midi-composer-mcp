"""Chords-track comping (durations, rhythm, sustain, strum) and swing on every renderer.

The first test is a byte-for-byte regression guard: a corpus of calls to the six
renderer tools with their original parameters must write exactly the same .mid
files as the code did before these options existed. The golden files in
``tests/golden/`` were generated from that original code; regenerate them only
on purpose, with ``python tests/test_rendering_ext.py --write-goldens``.
"""

from __future__ import annotations

import math
import os
import re
import sys

import pytest

if __name__ == "__main__":  # run as a script: render with this checkout's src, not an installed copy
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import mido  # noqa: E402

from midi_composer_mcp import server  # noqa: E402
from midi_composer_mcp.chords import CHORDS  # noqa: E402
from midi_composer_mcp.generate import GROOVES  # noqa: E402
from midi_composer_mcp.midi_io import (  # noqa: E402
    _chord_events,
    _parse_chord_list,
    _swing_events,
    _swing_warp,
    render_arrangement,
    render_chords,
    render_drums,
    render_notes,
    render_song,
)
from midi_composer_mcp.structure import render_song_structure  # noqa: E402

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")

_SECTIONS = {
    "intro": {"bars": 2, "tracks": [
        {"type": "chords", "name": "keys", "chords": ["Am7", "Fmaj7"], "program": 4, "velocity": 60},
        {"type": "drums", "name": "drums", "step_beats": 0.25,
         "lanes": {"hat": "o.o.o.o.o.o.o.o." * 2}},
    ]},
    "verse": {"bars": 4, "tracks": [
        {"type": "chords", "name": "keys", "chords": ["Am", "F", "C", "G"], "program": 4},
        {"type": "notes", "name": "bass", "notes": ["A2", "F2", "C3", "G2"], "step_beats": 4,
         "program": 33},
        {"type": "notes", "name": "lead", "notes": "E5 D5 C5 B4 C5 A4", "rhythm": "O.o.O..oo.O.o...",
         "sustain": True, "start_beat": 2, "program": 80},
        {"type": "drums", "name": "drums", "step_beats": 0.25,
         "lanes": {"kick": "O...O...O...O...", "snare": "....O.......O...", "hat": "oooooooooooooooo"}},
    ]},
    "chorus": {"bars": 4, "tracks": [
        {"type": "chords", "name": "keys", "chords": ["F", "G", "Em", "Am"], "program": 48,
         "arpeggiate": True},
        {"type": "notes", "name": "bass", "notes": ["F2", "G2", "E2", "A2"], "step_beats": 4,
         "program": 33, "channel": 5},
        {"type": "drums", "name": "drums", "step_beats": 0.25,
         "lanes": {"kick": "O..o..o.O..o..o.", "clap": "....O.......O...", "open_hat": "..o...o...o...o."}},
    ]},
}

# (golden name, server tool, keyword arguments) — only parameters the original tools had.
CORPUS = [
    ("notes_default", "notes_to_midi", {"notes": ["C", "D", "E", "F", "G", "A", "B", "C"]}),
    ("notes_rhythm_sustain", "notes_to_midi", {
        "notes": "C5 D5 E5", "rhythm": "O.o.O..o", "sustain": True, "tempo": 96,
        "velocity": 70, "accent_velocity": 120, "program": 24}),
    ("notes_ascending_third_steps", "notes_to_midi", {
        "notes": ["G", "A", "B", "C", "D", "E", "F#", "G"], "octave_policy": "ascending",
        "step_beats": 1 / 3, "octave": 3}),
    ("chords_default", "chords_to_midi", {"chords": ["C", "Am", "F", "G"]}),
    ("chords_mixed", "chords_to_midi", {
        "chords": ["Cmaj7", "C/E", ["C4", "E4", "G4"], "G7", "Bb/D3"], "beats_per_chord": 2,
        "octave": 3, "program": 48, "velocity": 90, "tempo": 100}),
    ("chords_arpeggiated", "chords_to_midi", {
        "chords": "Dm7 G7 Cmaj7", "beats_per_chord": 1.5, "arpeggiate": True}),
    ("drums_default", "drums_to_midi", {
        "lanes": {"kick": "O...O...", "snare": "..O...O.", "hat": "oooooooo"}}),
    ("drums_varied", "drums_to_midi", {
        "lanes": {"kick": "O..o..o.O..o..o.", "39": "....O.......O...", "shaker": "oo.o"},
        "step_beats": 0.25, "tempo": 128, "velocity": 70, "accent_velocity": 127}),
    ("song_default", "song_to_midi", {
        "melody_notes": ["E", "D", "C", "D", "E", "E", "E"], "chords": ["C", "G", "C"]}),
    ("song_varied", "song_to_midi", {
        "melody_notes": "A4 C5 E5 D5 C5 B4", "chords": ["Am", "Dm", "E7", "Am"],
        "melody_rhythm": "O.o.oo.O" * 4, "sustain": True, "arpeggiate_chords": True,
        "beats_per_chord": 4, "melody_program": 73, "chord_program": 0, "tempo": 88,
        "chord_velocity": 60}),
    ("arrange_default", "arrange_to_midi", {"tracks": [
        {"type": "chords", "chords": ["Am", "F", "C", "G"]},
        {"type": "notes", "notes": ["A2", "F2", "C3", "G2"], "step_beats": 4, "program": 33},
        {"type": "notes", "notes": "E5 D5 C5 B4", "rhythm": "O.o.O..o", "octave": 5},
        {"type": "drums", "lanes": {"kick": "O...O...", "snare": "..O...O.", "hat": "oooooooo"}},
    ]}),
    ("arrange_offsets", "arrange_to_midi", {
        "tracks": [
            {"type": "chords", "name": "pad", "chords": ["Cmaj7", ["D4", "F4", "A4", "C5"]],
             "arpeggiate": True, "program": 89, "channel": 3},
            {"type": "notes", "name": "arp", "notes": ["C", "E", "G", "B"], "step_beats": 0.25,
             "start_beat": 1.5, "octave_policy": "ascending", "velocity": 64},
            {"type": "drums", "name": "perc", "step_beats": 1 / 3, "start_beat": 2,
             "lanes": {"conga": "O.oo.o", "clave": "o..o.."}},
        ],
        "tempo": 140, "step_beats": 0.25, "beats_per_chord": 3}),
    ("arrange_song_sections", "arrange_song", {
        "sections": _SECTIONS, "form": "intro verse chorus verse chorus", "tempo": 104}),
    ("arrange_song_three_four", "arrange_song", {
        "sections": {
            "A": {"tracks": [{"type": "chords", "chords": ["G", "D/F#", "Em"]},
                             {"type": "notes", "notes": "B4 A4 G4 A4 B4 B4", "step_beats": 1.5}]},
            "B": {"bars": 2, "tracks": [{"type": "notes", "notes": ["C5", "B4", "A4"], "rhythm": "O.oo.."},
                                        {"type": "drums", "lanes": {"kick": "O....."}}]},
        },
        "form": "AABA", "beats_per_bar": 3, "step_beats": 0.25}),
]


def _render_corpus(out_dir) -> dict[str, bytes]:
    rendered = {}
    for name, tool, kwargs in CORPUS:
        result = getattr(server, tool)(**kwargs, file_name=f"{name}.mid", output_dir=str(out_dir))
        with open(result["file"], "rb") as fh:
            rendered[name] = fh.read()
    return rendered


@pytest.fixture(scope="module")
def rendered_corpus(tmp_path_factory):
    return _render_corpus(tmp_path_factory.mktemp("corpus"))


@pytest.mark.parametrize("name", [c[0] for c in CORPUS])
def test_default_renders_are_byte_identical_to_goldens(name, rendered_corpus):
    with open(os.path.join(GOLDEN_DIR, f"{name}.mid"), "rb") as fh:
        assert rendered_corpus[name] == fh.read(), f"{name}.mid differs from its golden file"


_EXPLICIT_DEFAULTS = {
    "chords_to_midi": {"durations": None, "rhythm": None, "step_beats": 0.5, "sustain": False,
                       "accent_velocity": 100, "strum": 0.0, "strum_direction": "down"},
}


@pytest.mark.parametrize("name,tool,kwargs", CORPUS)
def test_spelled_out_defaults_are_byte_identical_too(name, tool, kwargs, tmp_path):
    extra = dict(_EXPLICIT_DEFAULTS.get(tool, {}), swing=0.5, swing_unit=0.5)
    result = getattr(server, tool)(**kwargs, **extra, file_name="x.mid", output_dir=str(tmp_path))
    with open(result["file"], "rb") as fh, open(os.path.join(GOLDEN_DIR, f"{name}.mid"), "rb") as golden:
        assert fh.read() == golden.read()


# ------------------------------------------------------------------ helpers

TONICS = ["C", "G", "D", "A", "E", "B", "F#", "C#", "F", "Bb", "Eb", "Ab", "Db", "Gb"]


def _notes(path) -> list[tuple[float, float, int, int]]:
    """(on beat, off beat, MIDI note, velocity) for every note in the file, sorted."""
    mid = mido.MidiFile(path)
    found = []
    for track in mid.tracks:
        now, open_notes = 0, {}
        for m in track:
            now += m.time
            if m.type == "note_on" and m.velocity > 0:
                open_notes.setdefault((m.channel, m.note), []).append((now, m.velocity))
            elif m.type in ("note_off", "note_on"):
                on, vel = open_notes[(m.channel, m.note)].pop(0)
                found.append((on / mid.ticks_per_beat, now / mid.ticks_per_beat, m.note, vel))
    return sorted(found)


def _events(chords, **kw):
    """Timed events of a chord sequence, straight from the chord renderer."""
    events, resolved, total = _chord_events(_parse_chord_list(chords), kw.pop("beats_per_chord", 4.0),
                                            kw.pop("octave", 4), kw.pop("arpeggiate", False),
                                            kw.pop("velocity", 80), **kw)
    return events, resolved, total


def _onsets(events) -> list[tuple[float, int]]:
    return sorted({(round(e["start"], 9), e["velocity"]) for e in events})


def _lasts_as_reported(result) -> bool:
    """The file ends exactly at the reported total_beats (to the tick), and its length in
    seconds matches duration_seconds (which is rounded to milliseconds)."""
    mid = mido.MidiFile(result["file"])
    ticks = max(sum(m.time for m in track) for track in mid.tracks)
    return (ticks == round(result["total_beats"] * mid.ticks_per_beat)
            and abs(mid.length - result["duration_seconds"]) <= 0.0005 + 1e-9)


# ------------------------------------------------------------ chord rhythm

def test_rhythm_strikes_the_whole_voicing_with_accents(tmp_path):
    r = render_chords(["C", "G"], beats_per_chord=2, rhythm="O.o.", step_beats=0.5,
                      output_dir=str(tmp_path), file_name="c.mid")
    notes = _notes(r["file"])
    assert sorted({(on, vel) for on, _, _, vel in notes}) == [(0.0, 100), (1.0, 80), (2.0, 100), (3.0, 80)]
    assert [n[2] for n in notes if n[0] == 1.0] == [60, 64, 67]      # the whole C voicing
    assert [n[2] for n in notes if n[0] == 3.0] == [67, 71, 74]      # the whole G voicing
    assert all(off - on == 0.5 for on, off, _, _ in notes)           # one step each, no sustain
    assert r["total_beats"] == 4


def test_accent_velocity_is_configurable_and_o_uses_velocity():
    events, _, _ = _events(["C"], beats_per_chord=1, rhythm="Oo", velocity=40, accent_velocity=90)
    assert _onsets(events) == [(0.0, 90), (0.5, 40)]


def test_sustain_holds_only_within_a_chord():
    events, _, _ = _events(["C", "G"], beats_per_chord=2, rhythm="O...", sustain=True)
    assert {(e["start"], e["duration"]) for e in events} == {(0.0, 2.0), (2.0, 2.0)}
    # whole-track form: the rests after the C strike never spill into G, which has no onset
    events, _, _ = _events(["C", "G"], beats_per_chord=2, rhythm="O.......", sustain=True)
    assert {(e["start"], e["duration"]) for e in events} == {(0.0, 2.0)}
    assert len(events) == 3
    events, _, _ = _events(["C", "G"], beats_per_chord=2, rhythm="O.o.", sustain=True)
    assert {(e["start"], e["duration"]) for e in events} == {(0.0, 1.0), (1.0, 1.0), (2.0, 1.0), (3.0, 1.0)}


def test_durations_give_variable_harmonic_rhythm(tmp_path):
    events, resolved, total = _events(["C", "G", "Am"], durations=[2, 2, 4])
    assert [c["start_beat"] for c in resolved] == [0.0, 2.0, 4.0]
    assert [c["duration_beats"] for c in resolved] == [2.0, 2.0, 4.0]
    assert total == 8
    r = server.chords_to_midi(["C", "G", "Am", "F"], durations=[2, 2, 4, 8],
                              output_dir=str(tmp_path), file_name="d.mid")
    assert [(c["start_beat"], c["duration_beats"]) for c in r["chords"]] == [(0, 2), (2, 2), (4, 4), (8, 8)]
    assert r["total_beats"] == 16 and _lasts_as_reported(r)
    assert sorted({(on, off) for on, off, _, _ in _notes(r["file"])}) == [(0, 2), (2, 4), (4, 8), (8, 16)]


def test_durations_replace_beats_per_chord_for_arpeggios_too():
    events, _, total = _events(["C", "G"], durations=[3, 1], arpeggiate=True)
    assert [e["start"] for e in events] == [0.0, 1.0, 2.0, 3.0, 3 + 1 / 3, 3 + 2 / 3]
    assert total == 4


def test_whole_track_rhythm_across_mixed_durations():
    events, _, _ = _events(["C", "G", "Am"], durations=[2, 2, 4], rhythm="O...o...O.o.o.o.")
    by_start = {}
    for e in events:
        by_start.setdefault(e["start"], set()).add(e["label"])
    assert sorted(by_start) == [0.0, 2.0, 4.0, 5.0, 6.0, 7.0]
    assert by_start[0.0] == {"C4", "E4", "G4"} and by_start[2.0] == {"G4", "B4", "D5"}
    assert by_start[5.0] == {"A4", "C5", "E5"}
    with pytest.raises(ValueError, match="different durations"):
        _events(["C", "G", "Am"], durations=[2, 2, 4], rhythm="O.o.")


def test_whole_track_rhythm_is_clipped_at_each_chord_change():
    # chord boundary at 1.5 falls inside a 1-beat step: the strike at 1 stops at 1.5 and
    # the G is silent until its own first onset at 2
    events, _, _ = _events(["C", "G"], durations=[1.5, 2.5], rhythm="oooo", step_beats=1.0, sustain=True)
    spans = {}
    for e in events:
        spans.setdefault((e["start"], e["duration"]), set()).add(e["label"])
    assert spans == {(0.0, 1.0): {"C4", "E4", "G4"}, (1.0, 0.5): {"C4", "E4", "G4"},
                     (2.0, 1.0): {"G4", "B4", "D5"}, (3.0, 1.0): {"G4", "B4", "D5"}}


def test_rhythm_length_mismatch_names_both_lengths():
    with pytest.raises(ValueError) as err:
        _events(["C", "G"], beats_per_chord=2, rhythm="O.o")
    assert "4 (one chord" in str(err.value) and "8 (the whole" in str(err.value)


# ------------------------------------------------------------------- strum

def test_strum_offsets_share_a_common_end():
    events, _, _ = _events(["C"], beats_per_chord=2, strum=0.03)
    got = sorted((e["midi"], e["start"], e["start"] + e["duration"]) for e in events)
    assert [m for m, _, _ in got] == [60, 64, 67]
    assert [s for _, s, _ in got] == pytest.approx([0, 0.03, 0.06])
    assert [end for _, _, end in got] == pytest.approx([2, 2, 2])
    up, _, _ = _events(["C"], beats_per_chord=2, strum=0.03, strum_direction="up")
    assert [s for _, s in sorted((e["midi"], e["start"]) for e in up)] == pytest.approx([0.06, 0.03, 0])


def test_strum_counts_from_the_lowest_sounding_note():
    # an explicit voicing written top-down still strums low to high
    events, _, _ = _events([["G4", "E4", "C4"]], beats_per_chord=1, strum=0.05)
    assert sorted((e["start"], e["midi"]) for e in events) == [(0.0, 60), (0.05, 64), (0.1, 67)]


def test_alternate_strum_flips_on_odd_grid_steps():
    events, _, _ = _events(["C", "G"], beats_per_chord=1, rhythm="oo", strum=0.02,
                           strum_direction="alternate")
    firsts = {}
    for e in events:
        if round(e["start"] * 1000) % 500 == 0:   # the first voice of each strike, on the grid
            firsts[e["start"]] = e["midi"]
    # down from C4, up from G4 (the top of C), down from G4, up from D5 (the top of G)
    assert firsts == {0.0: 60, 0.5: 67, 1.0: 67, 1.5: 74}


def test_alternate_strum_without_rhythm_uses_chord_parity():
    events, _, _ = _events(["C", "G", "C"], beats_per_chord=1, strum=0.02, strum_direction="alternate")
    firsts = {e["start"]: e["midi"] for e in events if e["start"] in (0.0, 1.0, 2.0)}
    assert firsts == {0.0: 60, 1.0: 74, 2.0: 60}
    # wherever the track starts: the chord index, not the grid, sets the stroke
    events, _, _ = _events(["C", "G", "C"], beats_per_chord=1, strum=0.02, strum_direction="alternate", start=0.5)
    firsts = {e["start"]: e["midi"] for e in events if e["start"] in (0.5, 1.5, 2.5)}
    assert firsts == {0.5: 60, 1.5: 74, 2.5: 60}


def _strokes(result, grid: float) -> dict:
    """{strike beat: 'down' | 'up'} of a strummed chords file: which end of the chord enters first."""
    strikes = {}
    for on, _off, midi, _vel in _notes(result["file"]):
        strikes.setdefault(round(on / grid) * grid, []).append((on, midi))
    return {beat: "down" if min(v)[1] == min(m for _, m in v) else "up" for beat, v in sorted(strikes.items())}


def test_alternate_strum_follows_the_absolute_grid(tmp_path):
    def strokes(rhythm, start_beat, grid=0.5):
        r = server.arrange_to_midi([{"type": "chords", "chords": ["C"], "beats_per_chord": 2, "rhythm": rhythm,
                                     "step_beats": 0.5, "strum": 0.05, "strum_direction": "alternate",
                                     "start_beat": start_beat}], output_dir=str(tmp_path), file_name="a.mid")
        return _strokes(r, grid)
    # a track entering on the 'and' of 1 opens with an upstroke: down on the beats, up on the 'and's
    assert strokes("oooo", 0.5) == {0.5: "up", 1.0: "down", 1.5: "up", 2.0: "down"}
    # the same onsets written two ways strum the same way
    assert strokes("ooo.", 0.5) == strokes(".ooo", 0) == {0.5: "up", 1.0: "down", 1.5: "up"}
    # an even number of steps late: down on the beat again
    assert strokes("oooo", 1.0) == {1.0: "down", 1.5: "up", 2.0: "down", 2.5: "up"}
    # a start between grid steps takes the step it falls in, and the strokes still alternate
    assert strokes("oooo", 0.25, grid=0.25) == {0.25: "down", 0.75: "up", 1.25: "down", 1.75: "up"}


def test_alternate_strum_in_a_song_counts_from_the_section_start(tmp_path):
    # sections start on bar lines, so a repeated section strums alike even when a bar holds an
    # odd number of steps (quarter strums in 3/4)
    comp = {"type": "chords", "name": "g", "chords": ["C"], "beats_per_chord": 3, "rhythm": "ooo",
            "step_beats": 1, "strum": 0.05, "strum_direction": "alternate"}
    song = render_song_structure({"A": {"bars": 1, "tracks": [comp]}}, form="A A", beats_per_bar=3,
                                 output_dir=str(tmp_path), file_name="s.mid")
    assert _strokes(song, 1.0) == {0.0: "down", 1.0: "up", 2.0: "down", 3.0: "down", 4.0: "up", 5.0: "down"}


@pytest.mark.parametrize("sevenths", [False, True])
@pytest.mark.parametrize("tonic", TONICS)
def test_offset_comping_strums_on_the_absolute_grid_in_every_key(tonic, sevenths):
    from midi_composer_mcp.diatonic import degrees_to_chords
    kw = dict(beats_per_chord=2, rhythm="O..o..o.", step_beats=0.25, strum=0.03,
              strum_direction="alternate", sustain=True)
    symbols = degrees_to_chords(tonic, "major", [1, 6, 4, 5], sevenths)["symbols"]
    timing, directions = _comping_shape(symbols, **kw)
    late_timing, late = _comping_shape(symbols, start=0.25, **kw)
    # one sixteenth late: the same part, but every strike now sits on an odd step of the grid
    assert [(round(s, 6), round(e, 6), v) for s, e, v in late_timing] == \
        [(round(s + 0.25, 6), round(e + 0.25, 6), v) for s, e, v in timing]
    assert directions == ["down", "up", "down"] * 4 and late == ["up", "down", "up"] * 4
    assert _comping_shape(symbols, start=0.5, **kw)[1] == directions   # two steps late: the same strokes


# ------------------------------------------------------- strum at MIDI resolution

def _voices_stay_in_their_chords(result, strikes_per_chord: int = 1) -> bool:
    """Every voice of every chord starts inside that chord's span (to the tick), and the
    file ends exactly where reported: no strummed voice runs over a chord change."""
    ons = [round(on * 480) for on, *_ in _notes(result["file"])]
    for c in result["chords"]:
        lo = round(c["start_beat"] * 480)
        hi = round((c["start_beat"] + c["duration_beats"]) * 480)
        if sum(lo <= t < hi for t in ons) != len(c["midi"]) * strikes_per_chord:
            return False
    return _lasts_as_reported(result)


def _widest(err) -> float:
    """The strum a 'too wide' error suggests (from a ValueError or a pytest.raises result)."""
    return float(re.search(r"use a strum of at most ([\d.]+)", str(getattr(err, "value", err))).group(1))


def test_strum_at_the_width_limit_is_refused_at_midi_resolution(tmp_path):
    out = str(tmp_path)
    # 2 x 0.1249 < 0.25 beats, but the last voice rounds onto the chord change (tick 120)
    with pytest.raises(ValueError, match="too wide") as err:
        server.chords_to_midi(["C", "G"], beats_per_chord=0.25, strum=0.1249, output_dir=out)
    r = server.chords_to_midi(["C", "G"], beats_per_chord=0.25, strum=_widest(err), output_dir=out,
                              file_name="w.mid")
    assert _voices_stay_in_their_chords(r)
    # swing squeezes the off-beat strike: the check runs on the swung ticks
    kw = dict(beats_per_chord=1, rhythm="oo", step_beats=0.5, swing=0.75)
    with pytest.raises(ValueError, match="too wide") as err:
        server.chords_to_midi(["C", "G"], strum=0.2495, output_dir=out, **kw)
    r = server.chords_to_midi(["C", "G"], strum=_widest(err), output_dir=out, file_name="s.mid", **kw)
    assert _voices_stay_in_their_chords(r, strikes_per_chord=2)
    # and in a song, on the song's placed and swung timeline
    with pytest.raises(ValueError, match="too wide"):
        render_song_structure({"v": {"tracks": [dict(kw, type="chords", chords=["C", "G"], strum=0.2495)]}},
                              beats_per_bar=3, swing=0.75, swing_unit=0.5, output_dir=out)


@pytest.mark.parametrize("swing", [0.5, 2 / 3, 0.75])
@pytest.mark.parametrize("sevenths", [False, True])
@pytest.mark.parametrize("tonic", TONICS)
def test_strums_near_the_limit_render_or_refuse_in_every_key(tonic, sevenths, swing, tmp_path):
    from midi_composer_mcp.diatonic import degrees_to_chords
    symbols = degrees_to_chords(tonic, "major", [1, 6, 4, 5], sevenths)["symbols"]
    voices = 4 if sevenths else 3
    limit = 0.5 / (voices - 1)      # two 0.5-beat strikes per chord; the off-beat one swings short
    widest = None
    for gap in (0, 1e-5, 1e-4, 5e-4, 1e-3, 2e-3, 3e-3, 5e-3, 1e-2, 3e-2):
        strum = min(0.25, limit - gap)
        try:
            r = server.chords_to_midi(symbols, beats_per_chord=1, rhythm="oo", step_beats=0.5, strum=strum,
                                      swing=swing, output_dir=str(tmp_path), file_name="n.mid")
        except ValueError as err:
            assert "too wide" in str(err)
            widest = _widest(err)
            assert strum > widest       # the suggested strum is never one that is refused
            continue
        assert _voices_stay_in_their_chords(r, strikes_per_chord=2), strum
    assert widest is not None           # the gap-0 strum is always refused
    r = server.chords_to_midi(symbols, beats_per_chord=1, rhythm="oo", step_beats=0.5, strum=widest,
                              swing=swing, output_dir=str(tmp_path), file_name="w.mid")
    assert _voices_stay_in_their_chords(r, strikes_per_chord=2)


def test_tresillo_stab_example(tmp_path):
    r = server.arrange_to_midi([{"type": "chords", "chords": ["Am", "F", "C", "G"], "beats_per_chord": 2,
                                 "step_beats": 0.25, "rhythm": "O..o..o.", "strum": 0.03}],
                               output_dir=str(tmp_path), file_name="t.mid")
    strikes = {}
    for on, off, midi, vel in _notes(r["file"]):
        strikes.setdefault(round(on * 4) / 4, []).append((on, off, midi, vel))
    assert sorted(strikes) == [k + s for k in (0, 2, 4, 6) for s in (0, 0.75, 1.5)]   # tresillo per chord
    for grid, voices in strikes.items():
        voices.sort()
        assert [v[2] for v in voices] == sorted(v[2] for v in voices)            # low to high
        assert [v[0] - grid for v in voices] == pytest.approx([0, 0.03, 0.06], abs=1 / 480)
        assert {v[1] for v in voices} == {grid + 0.25}                              # common end
        assert {v[3] for v in voices} == {100 if grid % 2 == 0 else 80}            # O then o, o
    assert r["total_beats"] == 8 and _lasts_as_reported(r)


# ------------------------------------------------------------------- swing

def test_swing_warp_values():
    r, u = 2 / 3, 0.5
    assert _swing_warp(0.5, r, u) == pytest.approx(0.6667, abs=1e-4)
    assert _swing_warp(1.0, r, u) == 1.0
    assert _swing_warp(1.5, r, u) == pytest.approx(1 + 2 / 3)
    (e,) = _swing_events([{"midi": 60, "label": "C4", "start": 0.5, "duration": 0.5, "velocity": 90}], r, u)
    assert (e["start"], e["start"] + e["duration"]) == pytest.approx((0.6667, 1.0), abs=1e-4)


def test_swung_eighths_play_long_short(tmp_path):
    r = server.notes_to_midi(["C4", "D4", "E4", "F4"], swing=2 / 3, output_dir=str(tmp_path), file_name="s.mid")
    assert [e["start_beat"] for e in r["events"]] == pytest.approx([0, 2 / 3, 1, 1 + 2 / 3])
    assert [e["duration_beats"] for e in r["events"]] == pytest.approx([2 / 3, 1 / 3, 2 / 3, 1 / 3])
    assert r["total_beats"] == 2.0 and _lasts_as_reported(r)
    ons = [on for on, _, _, _ in _notes(r["file"])]
    assert ons == pytest.approx([0, 2 / 3, 1, 1 + 2 / 3], abs=1 / 480)


@pytest.mark.parametrize("unit", [0.25, 0.5, 1.0])
def test_straight_swing_is_the_identity(unit):
    for t in [0, 0.1, 0.25, 0.3, 0.5, 0.75, 1, 1.2, 1.5, 2.9, 7.75, 100.125]:
        assert _swing_warp(t, 0.5, unit) == t


@pytest.mark.parametrize("unit", [0.25, 0.5, 1.0])
@pytest.mark.parametrize("ratio", [0.5, 0.55, 0.6, 2 / 3, 0.7, 0.75])
def test_swing_keeps_window_edges_order_and_total_length(ratio, unit, tmp_path):
    edges = [2 * unit * k for k in range(9)]
    assert [_swing_warp(t, ratio, unit) for t in edges] == pytest.approx(edges)
    ts = [i / 48 for i in range(200)]
    warped = [_swing_warp(t, ratio, unit) for t in ts]
    assert warped == sorted(warped) and len(set(warped)) == len(warped)   # strictly monotone
    r = render_drums({"hat": "o" * 16, "kick": "O...O...O...O..."}, step_beats=0.25, swing=ratio,
                     swing_unit=unit, output_dir=str(tmp_path), file_name="d.mid")
    assert r["total_beats"] == 4 and _lasts_as_reported(r)


def test_swing_applies_to_notes_chords_and_drums_in_every_renderer(tmp_path):
    out = str(tmp_path)
    song = render_song(["C5"] * 8, ["C", "G"], beats_per_chord=2, arpeggiate_chords=True,
                       swing=2 / 3, output_dir=out, file_name="s.mid")
    assert [e["start_beat"] for e in song["melody_events"]][:2] == pytest.approx([0, 2 / 3])
    assert song["total_beats"] == 4
    chords = render_chords(["C", "G"], beats_per_chord=1, rhythm="oo", swing=2 / 3, output_dir=out,
                           file_name="c.mid")
    assert sorted({on for on, _, _, _ in _notes(chords["file"])}) == pytest.approx([0, 2 / 3, 1, 5 / 3], abs=1 / 480)
    arr = render_arrangement([{"type": "notes", "notes": ["C4"], "rhythm": "oo"},
                              {"type": "drums", "lanes": {"hat": "oo"}},
                              {"type": "chords", "chords": ["C"], "beats_per_chord": 1, "rhythm": "oo"}],
                             swing=0.75, output_dir=out, file_name="a.mid")
    assert sorted({on for on, _, _, _ in _notes(arr["file"])}) == pytest.approx([0, 0.75], abs=1 / 480)


def test_track_swing_overrides_the_renderer(tmp_path):
    r = render_arrangement([
        {"type": "notes", "name": "straight", "notes": ["C4", "D4"], "swing": 0.5},
        {"type": "notes", "name": "swung", "notes": ["E4", "F4"]},
        {"type": "notes", "name": "sixteenths", "notes": ["G4", "A4"], "step_beats": 0.25,
         "swing_unit": 0.25},
    ], swing=2 / 3, output_dir=str(tmp_path), file_name="o.mid")
    starts = {t["name"]: [e["start_beat"] for e in t["events"]] for t in r["tracks"]}
    assert starts["straight"] == [0.0, 0.5]
    assert starts["swung"] == pytest.approx([0, 2 / 3])
    assert starts["sixteenths"] == pytest.approx([0, 1 / 3])


def test_swing_reports_warped_spans(tmp_path):
    r = render_arrangement([
        {"type": "chords", "chords": ["C", "G"], "durations": [1.5, 2.5]},
        {"type": "notes", "notes": ["C5"], "start_beat": 0.5, "step_beats": 0.5},
    ], swing=2 / 3, output_dir=str(tmp_path), file_name="w.mid")
    chords, pickup = r["tracks"]
    spans = [x for c in chords["chords"] for x in (c["start_beat"], c["duration_beats"])]
    assert spans == pytest.approx([0, 5 / 3, 5 / 3, 4 - 5 / 3])   # the change at 1.5 swings to 1.667
    assert pickup["start_beat"] == pytest.approx(2 / 3) and pickup["end_beat"] == pytest.approx(1.0)
    assert pickup["events"][0]["start_beat"] == pytest.approx(2 / 3)
    assert r["total_beats"] == 4 and _lasts_as_reported(r)


def test_offset_tracks_swing_on_the_absolute_timeline(tmp_path):
    # a track starting on an off-beat lands on the swung off-beat, not on its own grid
    r = render_arrangement([{"type": "notes", "notes": ["C4", "D4"], "start_beat": 0.5}],
                           swing=2 / 3, output_dir=str(tmp_path), file_name="p.mid")
    assert [e["start_beat"] for e in r["tracks"][0]["events"]] == pytest.approx([2 / 3, 1.0])


def test_song_sections_swing_on_the_song_bar_grid(tmp_path):
    out = str(tmp_path)
    # 4/4, a section starting at bar 3 (beat 8): the off-beat eighths of the verse are swung
    # exactly as the same track placed at beat 8 of an arrangement (the windows tile the bar)
    verse = [{"type": "notes", "name": "lead", "notes": ["C4", "D4", "E4", "F4"]}]
    song = render_song_structure({"intro": {"bars": 2, "tracks": [{"type": "drums", "lanes": {"kick": "O"}}]},
                                  "verse": {"bars": 1, "tracks": verse}},
                                 form="intro verse", swing=2 / 3, output_dir=out, file_name="s.mid")
    assert song["sections"][1]["start_bar"] == 2 and song["sections"][1]["start_beat"] == 8
    lead = [on for on, _, midi, _ in _notes(song["file"]) if midi in (60, 62, 64, 65)]
    assert lead == pytest.approx([8, 8 + 2 / 3, 9, 9 + 2 / 3], abs=1 / 480)
    arr = render_arrangement([dict(verse[0], start_beat=8)], swing=2 / 3, output_dir=out, file_name="v.mid")
    assert [on for on, *_ in _notes(arr["file"])] == lead
    # 3/4 with swung quarters: the pairs count from every downbeat, so the section at
    # start_bar 3 (beat 9) swings exactly as it does at beat 0, and the bar's last quarter,
    # which has no partner, stays straight (the bar line at 12 never moves)
    part = {"type": "notes", "name": "p", "notes": ["C4", "D4", "E4"], "step_beats": 1}
    song = render_song_structure({"a": {"bars": 3, "tracks": [{"type": "drums", "lanes": {"kick": "O"}}]},
                                  "b": {"bars": 1, "tracks": [part]}},
                                 form="a b", beats_per_bar=3, swing=0.75, swing_unit=1.0,
                                 output_dir=out, file_name="t.mid")
    alone = render_song_structure({"b": {"bars": 1, "tracks": [part]}}, beats_per_bar=3, swing=0.75,
                                  swing_unit=1.0, output_dir=out, file_name="u.mid")
    in_song = [(on, off) for on, off, midi, _ in _notes(song["file"]) if midi != 36]
    in_alone = [(on + 9, off + 9) for on, off, _, _ in _notes(alone["file"])]
    assert in_song == in_alone == [(9.0, 10.5), (10.5, 11.0), (11.0, 12.0)]
    assert song["total_beats"] == 12 and alone["total_beats"] == 3
    assert _lasts_as_reported(song) and _lasts_as_reported(alone)


def test_arrange_song_passes_chord_comping_fields_through(tmp_path):
    out = str(tmp_path)
    keys = {"type": "chords", "name": "keys", "chords": ["Am", "F", "C"], "durations": [2, 1, 1],
            "rhythm": "O.o.oooo", "step_beats": 0.5, "sustain": True, "accent_velocity": 111,
            "strum": 0.02, "strum_direction": "alternate"}
    song = server.arrange_song({"intro": {"bars": 1, "tracks": [{"type": "drums", "lanes": {"kick": "O"}}]},
                                "verse": {"bars": 1, "tracks": [keys]}},
                               form="intro verse", swing=0.6, output_dir=out, file_name="s.mid")
    arr = server.arrange_to_midi([dict(keys, start_beat=4)], swing=0.6, output_dir=out, file_name="a.mid")
    in_song = [n for n in _notes(song["file"]) if n[2] != 36]
    assert in_song == _notes(arr["file"])
    assert any(vel == 111 for *_, vel in in_song)
    assert _lasts_as_reported(song) and _lasts_as_reported(arr)


def test_program_changes_follow_the_swung_grid(tmp_path):
    song = render_song_structure({
        "a": {"bars": 1, "tracks": [{"type": "notes", "name": "x", "notes": ["C4"], "step_beats": 3}]},
        "b": {"bars": 1, "tracks": [{"type": "notes", "name": "x", "notes": ["D4"], "step_beats": 3,
                                     "program": 40}]},
    }, form="a b", beats_per_bar=3, swing=0.75, swing_unit=1.0, output_dir=str(tmp_path), file_name="p.mid")
    mid = mido.MidiFile(song["file"])
    now, change_at, d4_at = 0, None, None
    for m in mid.tracks[1]:
        now += m.time
        if m.type == "program_change" and m.program == 40:
            change_at = now
        if m.type == "note_on" and m.note == 62:
            d4_at = now
    # at the section start: swing on the song's bar grid never moves a bar line
    assert change_at == d4_at == 3 * 480 == round(song["sections"][1]["start_beat"] * 480)


def test_odd_meter_swung_quarters_keep_every_bar_line(tmp_path):
    out = str(tmp_path)
    # 'A B A B' in 3/4 with swung quarters: a section, and its repeat, swings from its own
    # downbeat (long-short, then the unpaired last quarter straight); no note crosses a bar line
    song = server.arrange_song({
        "A": {"bars": 1, "tracks": [{"type": "notes", "name": "m", "notes": "C4 E4 G4", "step_beats": 1}]},
        "B": {"bars": 1, "tracks": [{"type": "notes", "name": "m", "notes": "D4 F4 A4", "step_beats": 1,
                                     "program": 40}]},
    }, form="A B A B", beats_per_bar=3, swing=0.75, swing_unit=1.0, output_dir=out, file_name="o.mid")
    assert [s["start_beat"] for s in song["sections"]] == [0, 3, 6, 9]
    a = [(0.0, 1.5, 60), (1.5, 2.0, 64), (2.0, 3.0, 67)]
    b = [(3.0, 4.5, 62), (4.5, 5.0, 65), (5.0, 6.0, 69)]
    assert [n[:3] for n in _notes(song["file"])] == a + b + [(on + 6, off + 6, m) for on, off, m in a + b]
    now, changes = 0, []
    for m in mido.MidiFile(song["file"]).tracks[1]:
        now += m.time
        if m.type == "program_change":
            changes.append((now, m.program))
    assert changes == [(0, 0), (3 * 480, 40), (6 * 480, 0), (9 * 480, 40)]   # on the section starts
    assert song["total_beats"] == 12 and _lasts_as_reported(song)
    # one 2-bar section at triplet swing: bar 2 plays as bar 1, from its own downbeat at 3
    two = server.arrange_song({"A": {"bars": 2, "tracks": [
        {"type": "notes", "notes": "C4 E4 G4 C5 G4 E4", "step_beats": 1}]}},
        beats_per_bar=3, swing=2 / 3, swing_unit=1.0, output_dir=out, file_name="t.mid")
    assert [on for on, *_ in _notes(two["file"])] == pytest.approx([0, 4 / 3, 2, 3, 3 + 4 / 3, 5], abs=1 / 480)
    assert two["total_beats"] == 6 and _lasts_as_reported(two)
    one = server.arrange_song({"A": {"bars": 1, "tracks": [
        {"type": "notes", "notes": "C4 E4 G4", "step_beats": 1}]}},
        beats_per_bar=3, swing=0.75, swing_unit=1.0, output_dir=out, file_name="one.mid")
    assert one["total_beats"] == 3 and one["total_bars"] == 1 and _lasts_as_reported(one)


@pytest.mark.parametrize("bar", range(1, 8))
@pytest.mark.parametrize("unit", [0.25, 0.5, 1.0])
@pytest.mark.parametrize("ratio", [0.55, 2 / 3, 0.75])
def test_bar_anchored_swing_warp(bar, unit, ratio):
    ts = [i / 48 for i in range(48 * 3 * bar + 1)]
    warped = [_swing_warp(t, ratio, unit, bar) for t in ts]
    assert warped == sorted(warped) and len(set(warped)) == len(warped)          # strictly monotone
    assert [_swing_warp(k * bar, ratio, unit, bar) for k in range(4)] == [0, bar, 2 * bar, 3 * bar]
    for t, w in zip(ts, warped):                                                 # every bar alike
        downbeat = bar * math.floor(t / bar + 1e-9)
        assert w - downbeat == pytest.approx(_swing_warp(t - downbeat, ratio, unit, bar), abs=1e-9)
    if (bar / (2 * unit)).is_integer():   # the windows tile the bar: exactly the absolute grid
        assert warped == [_swing_warp(t, ratio, unit) for t in ts]
    else:                                 # swung quarters in an odd meter: the last quarter is straight
        for t, w in zip(ts, warped):
            if t % bar >= bar - 1:
                assert w == t
        assert _swing_warp(1.0, ratio, unit, bar) == (2 * ratio if bar > 1 else 1.0)


@pytest.mark.parametrize("bpb", range(1, 8))
@pytest.mark.parametrize("unit", [0.25, 0.5, 1.0])
@pytest.mark.parametrize("ratio", [2 / 3, 0.75])
def test_song_swing_repeats_sections_alike_and_never_moves_a_bar_line(bpb, unit, ratio, tmp_path):
    # every step of a 2-bar section filled, in any meter, played three times ('A A A')
    per_bar = int(bpb / unit)
    section = {"bars": 2, "tracks": [
        {"type": "notes", "name": "n", "notes": ["C4", "D4", "E4"], "rhythm": "o" * (2 * per_bar),
         "step_beats": unit},
        {"type": "chords", "name": "c", "chords": ["C", "G"], "beats_per_chord": bpb, "rhythm": "o" * per_bar,
         "step_beats": unit, "strum": 0.01, "strum_direction": "alternate"},
        {"type": "drums", "name": "d", "lanes": {"hat": "o" * (2 * per_bar)}, "step_beats": unit},
    ]}
    song = render_song_structure({"A": section}, form="A A A", beats_per_bar=bpb, swing=ratio, swing_unit=unit,
                                 output_dir=str(tmp_path), file_name="s.mid")
    assert song["total_beats"] == 6 * bpb and _lasts_as_reported(song)
    tick_bar = bpb * 480
    notes = [(round(on * 480), round(off * 480), midi) for on, off, midi, _ in _notes(song["file"])]
    passes = [sorted((on - p * 2 * tick_bar, off - p * 2 * tick_bar, m) for on, off, m in notes
                     if p * 2 * tick_bar <= on < (p + 1) * 2 * tick_bar) for p in range(3)]
    assert passes[0] == passes[1] == passes[2] and len(passes[0]) * 3 == len(notes)
    bars = [sorted((on - k * tick_bar, off - k * tick_bar) for on, off, _ in notes
                   if k * tick_bar <= on < (k + 1) * tick_bar) for k in range(6)]
    assert all(b == bars[0] for b in bars)                   # every bar swings alike
    assert all(0 <= on < off <= tick_bar for on, off in bars[0])   # and no note crosses a bar line
    assert min(on for on, _ in bars[0]) == 0                 # the downbeat is on the bar line


# ---------------------------------------------------- rendering contracts

@pytest.mark.parametrize("ratio", [2 / 3, 0.75])
@pytest.mark.parametrize("unit", [0.25, 0.5, 1.0])
def test_swung_and_strummed_files_last_as_long_as_reported(ratio, unit, tmp_path):
    out = str(tmp_path)
    for name, (pattern, step, _desc) in GROOVES.items():
        r = render_drums({"clave": pattern}, step_beats=step, swing=ratio, swing_unit=unit,
                         output_dir=out, file_name="g.mid")
        assert _lasts_as_reported(r), name
    for length in range(1, 9):   # odd lengths end mid-window: the reported length follows the swing
        r = render_notes(["C4"] * length, swing=ratio, swing_unit=unit, output_dir=out, file_name="n.mid")
        assert _lasts_as_reported(r), length
    r = render_chords(["C", "Am7", "F/A", "G7"], beats_per_chord=1.5, rhythm="O.o", step_beats=0.5,
                      sustain=True, strum=0.05, strum_direction="alternate", swing=ratio, swing_unit=unit,
                      output_dir=out, file_name="c.mid")
    assert _lasts_as_reported(r)
    r = render_arrangement([
        {"type": "chords", "chords": ["C", "G"], "durations": [2.5, 1], "strum": 0.1, "start_beat": 0.25},
        {"type": "notes", "notes": ["E5", "D5", "C5"], "rhythm": "O.oo.", "start_beat": 1.5, "sustain": True},
        {"type": "drums", "lanes": {"hat": "ooooooo"}, "step_beats": 0.25, "swing": 0.5},
    ], swing=ratio, swing_unit=unit, output_dir=out, file_name="a.mid")
    assert _lasts_as_reported(r)
    for bpb in (3, 4, 5):
        r = render_song_structure({"v": {"tracks": [
            {"type": "notes", "notes": ["C4", "D4", "E4"], "step_beats": 1},
            {"type": "chords", "chords": ["C"], "beats_per_chord": bpb, "rhythm": "o" * (2 * bpb),
             "strum": 0.02}]}}, form="v v v", beats_per_bar=bpb, swing=ratio, swing_unit=unit,
            output_dir=out, file_name="s.mid")
        assert _lasts_as_reported(r), bpb


def _comping_shape(chords, **kw) -> tuple[list, list]:
    """Key-free shape of a comped part: every (start, end, velocity), and per strike
    (voices sharing an end) whether it was strummed from the bottom or the top."""
    events, _, _ = _events(chords, **kw)
    timing = sorted((round(e["start"], 9), round(e["start"] + e["duration"], 9), e["velocity"]) for e in events)
    strikes = {}
    for e in events:
        strikes.setdefault(round(e["start"] + e["duration"], 9), []).append(e)
    directions = []
    for _end, group in sorted(strikes.items()):
        midis = [e["midi"] for e in sorted(group, key=lambda e: e["start"])]
        assert midis in (sorted(midis), sorted(midis, reverse=True))
        directions.append("down" if midis == sorted(midis) else "up")
    return timing, directions


@pytest.mark.parametrize("sevenths", [False, True])
@pytest.mark.parametrize("tonic", TONICS)
def test_comping_and_strum_are_identical_in_every_key(tonic, sevenths):
    from midi_composer_mcp.diatonic import degrees_to_chords
    kw = dict(beats_per_chord=2, rhythm="O..o..o.", step_beats=0.25, strum=0.03,
              strum_direction="alternate", sustain=True)
    reference = _comping_shape(degrees_to_chords("C", "major", [1, 6, 4, 5], sevenths)["symbols"], **kw)
    shape = _comping_shape(degrees_to_chords(tonic, "major", [1, 6, 4, 5], sevenths)["symbols"], **kw)
    assert shape == reference
    assert reference[1] == ["down", "up", "down"] * 4     # grid steps 0, 3, 6 of each 8-step chord


@pytest.mark.parametrize("root", ["C", "F#", "Bb"])
def test_every_chord_type_strums_low_to_high_with_a_common_end(root):
    for ctype in CHORDS.values():
        symbol = f"{root}{ctype.symbol}"
        for direction in ("down", "up"):
            events, _, _ = _events([symbol], beats_per_chord=2, strum=0.02, strum_direction=direction)
            by_pitch = sorted(events, key=lambda e: e["midi"])
            if direction == "up":
                by_pitch.reverse()
            assert [e["start"] for e in by_pitch] == pytest.approx([0.02 * k for k in range(len(events))]), symbol
            assert {round(e["start"] + e["duration"], 9) for e in events} == {2.0}, symbol


# ------------------------------------------------------------------ errors

@pytest.mark.parametrize("call", [
    lambda o: render_chords(["C", "G"], beats_per_chord=2, rhythm="O.o", output_dir=o),
    lambda o: render_chords(["C", "G"], durations=[2], output_dir=o),
    lambda o: render_chords(["C", "G"], durations="2 2", output_dir=o),
    lambda o: render_chords(["C", "G"], durations=[2, 0.1], output_dir=o),
    lambda o: render_notes(["C4"], swing=0.8, output_dir=o),
    lambda o: render_notes(["C4"], swing=0.4, output_dir=o),
    lambda o: render_drums({"kick": "O"}, swing_unit=0.3, output_dir=o),
    lambda o: render_chords(["C"], strum=0.3, output_dir=o),
    lambda o: render_chords(["C"], beats_per_chord=1, rhythm="oooo", step_beats=0.25, strum=0.2, output_dir=o),
    lambda o: render_chords(["C"], rhythm="oooooooo", arpeggiate=True, output_dir=o),
    lambda o: render_chords(["C"], strum=0.05, arpeggiate=True, output_dir=o),
    lambda o: render_chords(["C"], strum=0.05, strum_direction="sideways", output_dir=o),
    lambda o: render_chords(["C"], accent_velocity=0, output_dir=o),
    lambda o: render_arrangement([{"type": "notes", "notes": ["C4"], "swing": 0.9}], output_dir=o),
    lambda o: render_arrangement([{"type": "drums", "lanes": {"kick": "O"}, "swing_unit": 2}], output_dir=o),
    lambda o: render_arrangement([{"type": "chords", "chords": ["C"], "rhythm": "O.x"}], output_dir=o),
    lambda o: render_song_structure({"v": {"tracks": [{"type": "chords", "chords": ["C"], "strum": 1}]}},
                                    output_dir=o),
    lambda o: render_song_structure({"v": {"tracks": [{"type": "notes", "notes": ["C4"]}]}}, swing=True,
                                    output_dir=o),
])
def test_bad_comping_and_swing_raise_value_errors(call, tmp_path):
    with pytest.raises(ValueError):
        call(str(tmp_path))


def test_strum_too_wide_message_is_actionable():
    with pytest.raises(ValueError, match="too wide"):
        _events(["C7"], beats_per_chord=0.5, rhythm="oo", step_beats=0.25, strum=0.1)


if __name__ == "__main__":
    if "--write-goldens" not in sys.argv:
        sys.exit("usage: python tests/test_rendering_ext.py --write-goldens")
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    for golden, data in _render_corpus(GOLDEN_DIR).items():
        print(f"wrote {golden}.mid ({len(data)} bytes)")
