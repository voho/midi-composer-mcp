"""The scale/chord databases are rich, described, and every entry generates cleanly."""

import pytest

from midi_composer_mcp.chords import CHORDS, chord_info, chord_notes, list_chords
from midi_composer_mcp.notes import parse_note
from midi_composer_mcp.scales import SCALES, list_scales, scale_info, scale_notes


def test_rich_scale_count():
    # common modes, jazz, symmetric, and exotic/world scales
    assert len(SCALES) >= 40
    names = set(SCALES)
    assert {"major", "dorian", "lydian", "harmonic minor", "melodic minor"} <= names
    assert {"whole tone", "augmented", "diminished whole-half", "altered"} <= names
    assert {"hirajoshi", "double harmonic", "hungarian minor", "persian", "in sen"} <= names


def test_rich_chord_count():
    assert len(CHORDS) >= 35
    names = set(CHORDS)
    assert {"major", "minor", "diminished", "augmented"} <= names
    assert {"dominant 7", "major 7", "minor 7", "half-diminished", "diminished 7"} <= names
    assert {"dominant 13", "major 13", "minor 11", "major 7 sharp 11", "six nine"} <= names


def test_every_scale_has_description():
    for scale in SCALES.values():
        assert scale.description, f"scale {scale.name} has no description"
        assert scale.description[0].isupper() and scale.description.endswith(".")
        assert 4 <= len(scale.description.split()) <= 45


def test_every_chord_has_description():
    for chord in CHORDS.values():
        assert chord.description, f"chord {chord.name} has no description"
        assert chord.description[0].isupper() and chord.description.endswith(".")
        assert 4 <= len(chord.description.split()) <= 45


def test_descriptions_surface_in_outputs():
    assert scale_info("dorian")["description"]
    assert scale_info("major", "C")["description"]
    assert chord_info("maj7")["description"]
    assert chord_info("m7", "D")["description"]
    assert all(s["description"] for s in list_scales()["scales"])
    assert all(c["description"] for c in list_chords()["chords"])


def test_every_scale_generates_in_range():
    # generation works for every root, spells distinct letters where expected,
    # and stays within the MIDI range when given a concrete octave
    for scale in SCALES.values():
        for root in ("C", "F#", "Bb", "E"):
            notes = scale_notes(scale, parse_note(root))
            assert len(notes) == len(scale.intervals) + 1
        midi = [n.midi for n in scale_notes(scale, parse_note("C4"))]
        assert all(0 <= m <= 127 for m in midi)
        assert midi == sorted(midi)  # ascending


def test_every_chord_generates_in_range():
    for chord in CHORDS.values():
        for root in ("C", "F#", "Bb", "E"):
            notes = chord_notes(chord, parse_note(root))
            assert len(notes) == len(chord.degrees)
        midi = [n.midi for n in chord_notes(chord, parse_note("C3"))]
        assert all(0 <= m <= 127 for m in midi)


def test_exotic_scales_spell_correctly():
    assert scale_info("hungarian minor", "A")["notes"] == ["A", "B", "C", "D#", "E", "F", "G#", "A"]
    assert scale_info("double harmonic", "C")["notes"] == ["C", "Db", "E", "F", "G", "Ab", "B", "C"]
    assert scale_info("phrygian dominant", "E")["notes"] == ["E", "F", "G#", "A", "B", "C", "D", "E"]
    assert scale_info("lydian dominant", "C")["notes"] == ["C", "D", "E", "F#", "G", "A", "Bb", "C"]
    assert scale_info("hirajoshi", "C")["notes"] == ["C", "D", "Eb", "G", "Ab", "C"]


def test_extended_chords_spell_correctly():
    assert chord_info("maj13", "C")["notes"] == ["C", "E", "G", "B", "D", "A"]
    assert chord_info("7#11", "C")["notes"] == ["C", "E", "G", "Bb", "F#"]
    assert chord_info("m6/9", "C")["notes"] == ["C", "Eb", "G", "A", "D"]
    assert chord_info("add4", "C")["notes"] == ["C", "E", "F", "G"]


def test_new_chord_symbols_parse():
    from midi_composer_mcp.chords import parse_chord_symbol
    for symbol, expected in [
        ("Cmaj13", "major 13"), ("Am11", "minor 11"), ("G7#11", "dominant 7 sharp 11"),
        ("Cm6/9", "minor six nine"), ("D7sus2", "dominant 7 sus 2"), ("Cadd4", "add 4"),
    ]:
        _, chord, _ = parse_chord_symbol(symbol)
        assert chord.name == expected, f"{symbol} -> {chord.name}"
