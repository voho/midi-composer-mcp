"""Chords-track comping (durations, rhythm, sustain, strum) and swing on every renderer.

The first test is a byte-for-byte regression guard: a corpus of calls to the six
renderer tools with their original parameters must write exactly the same .mid
files as the code did before these options existed. The golden files in
``tests/golden/`` were generated from that original code; regenerate them only
on purpose, with ``python tests/test_rendering_ext.py --write-goldens``.
"""

from __future__ import annotations

import os
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


def test_song_sections_swing_on_the_global_grid(tmp_path):
    out = str(tmp_path)
    # 4/4, a section starting at bar 3 (beat 8): the off-beat eighths of the verse are swung
    verse = [{"type": "notes", "name": "lead", "notes": ["C4", "D4", "E4", "F4"]}]
    song = render_song_structure({"intro": {"bars": 2, "tracks": [{"type": "drums", "lanes": {"kick": "O"}}]},
                                  "verse": {"bars": 1, "tracks": verse}},
                                 form="intro verse", swing=2 / 3, output_dir=out, file_name="s.mid")
    assert song["sections"][1]["start_bar"] == 2 and song["sections"][1]["start_beat"] == 8
    lead = [on for on, _, midi, _ in _notes(song["file"]) if midi in (60, 62, 64, 65)]
    assert lead == pytest.approx([8, 8 + 2 / 3, 9, 9 + 2 / 3], abs=1 / 480)
    # 3/4 with swung quarters: the section at start_bar 3 (beat 9) sits mid-window, so a
    # section-local grid would be wrong; it must match the same track placed at beat 9
    part = {"type": "notes", "name": "p", "notes": ["C4", "D4", "E4"], "step_beats": 1}
    song = render_song_structure({"a": {"bars": 3, "tracks": [{"type": "drums", "lanes": {"kick": "O"}}]},
                                  "b": {"bars": 1, "tracks": [part]}},
                                 form="a b", beats_per_bar=3, swing=0.75, swing_unit=1.0,
                                 output_dir=out, file_name="t.mid")
    arr = render_arrangement([dict(part, start_beat=9)], swing=0.75, swing_unit=1.0,
                             output_dir=out, file_name="u.mid")
    in_song = [(on, off) for on, off, midi, _ in _notes(song["file"]) if midi != 36]
    in_arr = [(on, off) for on, off, _, _ in _notes(arr["file"])]
    assert in_song == in_arr == [(9.5, 10.0), (10.0, 11.5), (11.5, 12.0)]
    assert _lasts_as_reported(song) and _lasts_as_reported(arr)


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
    assert change_at == d4_at == round(3.5 * 480)


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


TONICS = ["C", "G", "D", "A", "E", "B", "F#", "C#", "F", "Bb", "Eb", "Ab", "Db", "Gb"]


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
