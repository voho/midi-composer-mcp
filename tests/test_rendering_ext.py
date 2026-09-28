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

from midi_composer_mcp import server  # noqa: E402

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


if __name__ == "__main__":
    if "--write-goldens" not in sys.argv:
        sys.exit("usage: python tests/test_rendering_ext.py --write-goldens")
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    for golden, data in _render_corpus(GOLDEN_DIR).items():
        print(f"wrote {golden}.mid ({len(data)} bytes)")
