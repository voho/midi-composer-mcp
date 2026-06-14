import pytest

from midi_composer_mcp.notes import Note, note_from_midi, parse_note, parse_notes, transpose


def test_parse_basic_notes():
    assert parse_note("C").pitch_class == 0
    assert parse_note("c").name == "C"
    assert parse_note("F#").pitch_class == 6
    assert parse_note("Bb").pitch_class == 10
    assert parse_note("Ebb").pitch_class == 2
    assert parse_note("A♭").name == "Ab"
    assert parse_note("g♯").name == "G#"


def test_parse_octaves_and_midi():
    assert parse_note("C4").midi == 60  # middle C
    assert parse_note("A4").midi == 69
    assert parse_note("C-1").midi == 0
    assert parse_note("G9").midi == 127
    # octave digit follows the letter, as in standard notation
    assert parse_note("Cb4").midi == 59
    assert parse_note("B#3").midi == 60
    assert parse_note("C").midi is None


def test_parse_invalid_notes():
    for bad in ["H", "C###", "", "12", "do", "C#b#b#"]:
        with pytest.raises(ValueError):
            parse_note(bad)


def test_parse_notes_formats():
    for source in (["C", "E", "G"], "c e g", "C,E,G", "C, E,  G"):
        assert [n.name for n in parse_notes(source)] == ["C", "E", "G"]
    assert [n.name for n in parse_notes(["c e", "g4"])] == ["C", "E", "G4"]
    with pytest.raises(ValueError):
        parse_notes("")
    with pytest.raises(ValueError):
        parse_notes([1, 2])


def test_transpose_proper_spelling():
    # minor third above Bb is Db, not C#
    assert transpose(parse_note("Bb"), 3, 2).name == "Db"
    # major third above C is E
    assert transpose(parse_note("C"), 4, 2).name == "E"
    # diminished seventh above C is Bbb
    assert transpose(parse_note("C"), 9, 6).name == "Bbb"
    # ninth above C4 lands in the next octave
    assert transpose(parse_note("C4"), 14, 8).name == "D5"
    # augmented fifth above C is G#
    assert transpose(parse_note("C"), 8, 4).name == "G#"


def test_transpose_octave_carry():
    assert transpose(parse_note("A4"), 3, 2).name == "C5"  # octave number changes at C
    assert transpose(parse_note("B3"), 1, 0).name == "B#3"  # same letter keeps octave


def test_transpose_table_fallback():
    # without letter distance, falls back to sharp/flat tables
    assert transpose(parse_note("C"), 6, None).name == "F#"
    assert transpose(parse_note("F"), 6, None).name == "B"
    assert transpose(parse_note("Bb"), 6, None).name == "E"


def test_note_from_midi():
    assert note_from_midi(60).name == "C4"
    assert note_from_midi(61).name == "C#4"
    assert note_from_midi(61, prefer_flats=True).name == "Db4"
    with pytest.raises(ValueError):
        note_from_midi(128)


def test_note_names_roundtrip():
    for name in ["C", "F#", "Bb3", "C5", "Ebb", "G#7", "A-1"]:
        assert parse_note(name).name == name
    assert parse_note("c").name == "C"
    assert parse_note("eb3").name == "Eb3"
