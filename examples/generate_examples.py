"""Worked examples that exercise the whole toolset, end to end.

Run it to render the pieces as MIDI plus playable WAV previews:

    python examples/generate_examples.py [output_dir]

Each piece is composed only from the server's tools (here imported as plain
functions): scales and chords, the harmony rules (circle of fifths, voice
leading, analysis), the melody generators (degrees, motif grammar, melodic
walk, tintinnabuli), rhythm grooves, and the song-structure assembler. The
melodies are built from scale degrees (or snapped to the scale), so they are
diatonic to the harmony by construction.
"""

from __future__ import annotations

import os
import sys

from midi_composer_mcp.audio import render_midi_to_wav
from midi_composer_mcp.counterpoint import species_counterpoint
from midi_composer_mcp.harmony import voice_leading
from midi_composer_mcp.melody import (
    motif_grammar,
    notes_from_degrees,
    snap_to_scale,
    tintinnabuli_voice,
)
from midi_composer_mcp.midi_io import render_arrangement
from midi_composer_mcp.structure import render_song_structure


def _bars(pattern: str, n: int) -> str:
    return pattern * n


def arvo_part_tintinnabuli(out_dir: str) -> dict:
    """'Holy minimalism' a la Arvo Part: a stepwise M-voice shadowed by an A-minor
    tintinnabuli T-voice, over a tonic drone. Slow, sparse, bell-like."""
    # M-voice: stepwise descending gestures resolving to the tonic (degrees of A minor).
    m_degrees = motif_grammar(
        "ABAC",
        {"A": [5, 4, 3, 2], "B": {"vary": "A", "transpose": -1}, "C": [3, 2, 1, 1]},
        kind="degrees",
    )["degrees"]
    m_voice = notes_from_degrees("A4", "natural minor", m_degrees)["notes"]
    # T-voice: the nearest A-minor triad note below each melody note (Part's first position).
    t_voice = tintinnabuli_voice(m_voice, "Am", position="inferior", rank=1)["t_voice"]

    midi = render_arrangement(
        [
            {"type": "notes", "name": "M-voice", "notes": m_voice, "step_beats": 2.0,
             "octave": 4, "program": 49, "velocity": 80, "sustain": True},
            {"type": "notes", "name": "T-voice (bells)", "notes": t_voice, "step_beats": 2.0,
             "octave": 4, "program": 9, "velocity": 64, "sustain": True},
            {"type": "notes", "name": "drone", "notes": ["A2"] * (len(m_voice) // 2),
             "step_beats": 4.0, "octave": 2, "program": 48, "velocity": 44, "sustain": True},
        ],
        tempo=54, output_dir=out_dir, file_name="arvo_part_tintinnabuli.mid",
    )
    return midi


def whole_song(out_dir: str) -> dict:
    """A full song in C major with intro / verse / chorus / bridge / outro, using
    voice-led pads, degree-and-grammar melodies (diatonic by construction), a
    euclidean-ish bass and groove drums."""
    KICK = "O...O...O...O..."
    SNARE = "....O.......O..."
    HAT = "o.o.o.o.o.o.o.o."

    def drums(bars: int, name: str = "drums") -> dict:
        return {"type": "drums", "name": name, "step_beats": 0.25, "lanes": {
            "kick": _bars(KICK, bars), "snare": _bars(SNARE, bars), "hat": _bars(HAT, bars)}}

    def bassline(roots, name="bass") -> dict:
        return {"type": "notes", "name": name, "notes": roots, "step_beats": 4.0,
                "octave": 2, "program": 33, "velocity": 92}

    # Verse: I–V–vi–IV, voiced smoothly; melody from a motif grammar in scale degrees.
    verse_chords = voice_leading(["C", "G", "Am", "F"], octave=4)["chords"]
    verse_mel = notes_from_degrees(
        "C5", "major",
        motif_grammar("ABAC",
                      {"A": [5, 5, 6, 5], "B": {"vary": "A", "transpose": 1},
                       "C": {"vary": "A", "retrograde": True}}, kind="degrees")["degrees"],
    )["notes"]

    # Chorus: IV–V–I–vi, a higher hook (snapped to the key as a safety net).
    chorus_chords = voice_leading(["F", "G", "C", "Am"], octave=4)["chords"]
    chorus_mel = snap_to_scale(
        motif_grammar("ABAB",
                      {"A": "G5 A5 G5 E5", "B": "F5 G5 E5 C5"}, kind="notes")["notes"],
        "C", "major")["notes"]

    # Bridge: a darker vi-centred turn, ii–V into the relative minor feel.
    bridge_chords = voice_leading(["Dm", "G", "Em", "Am"], octave=4)["chords"]
    bridge_mel = notes_from_degrees("C5", "major", [2, 4, 6, 5, 4, 2, 7, 1] + [3, 2, 1, 7, 1, 2, 1, 1])["notes"]

    sections = {
        "intro": {"bars": 2, "tracks": [
            {"type": "chords", "name": "keys", "chords": voice_leading(["Am", "F"], octave=4)["chords"],
             "beats_per_chord": 4, "octave": 4, "program": 89, "arpeggiate": True, "velocity": 60},
        ]},
        "verse": {"bars": 4, "tracks": [
            {"type": "chords", "name": "keys", "chords": verse_chords, "beats_per_chord": 4, "program": 0, "velocity": 66},
            bassline(["C", "G", "A", "F"]),
            {"type": "notes", "name": "lead", "notes": verse_mel, "octave": 5, "program": 80, "velocity": 92},
            drums(4),
        ]},
        "chorus": {"bars": 4, "tracks": [
            {"type": "chords", "name": "keys", "chords": chorus_chords, "beats_per_chord": 4, "program": 0, "velocity": 74},
            bassline(["F", "G", "C", "A"]),
            {"type": "notes", "name": "lead", "notes": chorus_mel, "octave": 5, "program": 81, "velocity": 104},
            drums(4),
        ]},
        "bridge": {"bars": 4, "tracks": [
            {"type": "chords", "name": "keys", "chords": bridge_chords, "beats_per_chord": 4, "program": 48, "velocity": 64},
            bassline(["D", "G", "E", "A"]),
            {"type": "notes", "name": "lead", "notes": bridge_mel, "step_beats": 1.0, "octave": 5, "program": 80, "velocity": 88},
            drums(4),
        ]},
        "outro": {"bars": 2, "tracks": [
            {"type": "chords", "name": "keys", "chords": voice_leading(["F", "C"], octave=4)["chords"],
             "beats_per_chord": 4, "octave": 4, "program": 89, "arpeggiate": True, "velocity": 58},
            bassline(["F", "C"]),
        ]},
    }
    return render_song_structure(
        sections, form="intro verse chorus verse chorus bridge chorus outro",
        tempo=104, output_dir=out_dir, file_name="whole_song.mid",
    )


def third_species_counterpoint(out_dir: str) -> dict:
    """A third-species (4:1) counterpoint to a cantus firmus — the rules supplied
    by the `counterpoint` tool, rendered via its `render_hint`."""
    cp = species_counterpoint(
        ["C5", "D5", "E5", "F5", "E5", "D5", "C5"], "C", "major",
        species=3, position="above",
    )
    return render_arrangement(cp["render_hint"]["tracks"], tempo=80,
                              output_dir=out_dir, file_name="counterpoint_species3.mid")


def tintinnabuli_song(out_dir: str) -> dict:
    """'Compose with tintinnabuli rules over a few maj7 chords, two verses and a
    chorus' — a worked answer to the advanced prompt, in A minor / C major."""
    # Verse: an A-minor M-voice shadowed by its A-minor tintinnabuli T-voice,
    # over maj7/m7 pads (voice-led for smoothness).
    m_voice = notes_from_degrees(
        "A4", "natural minor",
        motif_grammar("ABAC", {"A": [1, 2, 3, 2], "B": {"vary": "A", "transpose": 1},
                               "C": [3, 2, 1, 1]}, kind="degrees")["degrees"],
    )["notes"]
    t_voice = tintinnabuli_voice(m_voice, "Am", position="inferior", rank=1)["t_voice"]
    verse_pads = voice_leading(["Am7", "Dm7", "Fmaj7", "Cmaj7"], octave=4)["chords"]
    chorus_pads = voice_leading(["Fmaj7", "Cmaj7", "Dm7", "Em7"], octave=4)["chords"]
    chorus_mel = notes_from_degrees("C5", "major", [5, 6, 8, 6, 5, 3, 2, 1])["notes"]

    sections = {
        "verse": {"bars": 4, "tracks": [
            {"type": "chords", "name": "pads", "chords": verse_pads, "beats_per_chord": 4, "program": 89, "velocity": 55},
            {"type": "notes", "name": "M-voice", "notes": m_voice, "step_beats": 2.0, "octave": 5, "program": 48, "sustain": True},
            {"type": "notes", "name": "T-voice", "notes": t_voice, "step_beats": 2.0, "octave": 4, "program": 9, "velocity": 60, "sustain": True},
        ]},
        "chorus": {"bars": 4, "tracks": [
            {"type": "chords", "name": "pads", "chords": chorus_pads, "beats_per_chord": 4, "program": 89, "velocity": 64},
            {"type": "notes", "name": "M-voice", "notes": chorus_mel, "step_beats": 2.0, "octave": 5, "program": 48, "sustain": True},
            {"type": "notes", "name": "bass", "notes": ["F", "C", "D", "E"], "step_beats": 4.0, "octave": 2, "program": 33},
        ]},
    }
    return render_song_structure(sections, form="verse verse chorus", tempo=72,
                                 output_dir=out_dir, file_name="tintinnabuli_song.mid")


def main() -> None:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "examples_output"
    os.makedirs(out_dir, exist_ok=True)
    for builder in (arvo_part_tintinnabuli, third_species_counterpoint, tintinnabuli_song, whole_song):
        midi = builder(out_dir)
        wav = render_midi_to_wav(midi["file"])
        print(f"{midi['file_name']:32s} {midi.get('total_bars', '?')!s:>4} bars  "
              f"{midi['duration_seconds']:6.1f}s  -> {wav['file_name']}")
    print(f"\nFiles written to {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
