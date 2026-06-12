import pytest

from midi_composer_mcp.chords import (
    CHORDS,
    chord_info,
    list_chords,
    match_chords,
    parse_chord_symbol,
    resolve_chord_type,
)


def test_resolve_chord_types():
    assert resolve_chord_type("m7").name == "minor 7"
    assert resolve_chord_type("M7").name == "major 7"  # case matters for m7/M7
    assert resolve_chord_type("maj7").name == "major 7"
    assert resolve_chord_type("MAJ7").name == "major 7"
    assert resolve_chord_type("minor").name == "minor"
    assert resolve_chord_type("°").name == "diminished"
    assert resolve_chord_type("half-diminished").name == "half-diminished"
    with pytest.raises(ValueError, match="Unknown chord type"):
        resolve_chord_type("supermassive")


def test_basic_chords():
    assert chord_info("maj", "C")["notes"] == ["C", "E", "G"]
    assert chord_info("min", "F")["notes"] == ["F", "Ab", "C"]
    assert chord_info("7", "G")["notes"] == ["G", "B", "D", "F"]
    assert chord_info("dim7", "C")["notes"] == ["C", "Eb", "Gb", "Bbb"]
    assert chord_info("aug", "C")["notes"] == ["C", "E", "G#"]
    assert chord_info("maj", "Bb")["notes"] == ["Bb", "D", "F"]
    assert chord_info("9", "C")["notes"] == ["C", "E", "G", "Bb", "D"]


def test_chord_symbols_in_output():
    info = chord_info("min", "F")
    assert info["symbol"] == "Fm"
    assert info["name"] == "F minor"
    assert chord_info("major", "C")["symbol"] == "C"


def test_octave_aware_chord():
    info = chord_info("maj7", "C4")
    assert info["notes"] == ["C4", "E4", "G4", "B4"]
    assert info["midi"] == [60, 64, 67, 71]
    # extensions cross the octave
    nine = chord_info("9", "C4")
    assert nine["notes"][-1] == "D5"
    assert nine["midi"] == [60, 64, 67, 70, 74]
    low = chord_info("m", "Eb3")
    assert low["midi"] == [51, 54, 58]


def test_chord_without_root_is_db_entry():
    info = chord_info("m7")
    assert info["intervals"] == [0, 3, 7, 10]
    assert info["degrees"] == ["1", "b3", "5", "b7"]
    assert "notes" not in info


def test_parse_chord_symbol():
    root, chord, bass = parse_chord_symbol("F#m7")
    assert (root.name, chord.name, bass) == ("F#", "minor 7", None)
    root, chord, _ = parse_chord_symbol("Bb7")
    assert (root.name, chord.name) == ("Bb", "dominant 7")  # suffix beats octave
    root, chord, _ = parse_chord_symbol("C4maj7")
    assert (root.name, chord.name) == ("C4", "major 7")
    root, chord, _ = parse_chord_symbol("C47")
    assert (root.name, chord.name) == ("C4", "dominant 7")
    root, chord, _ = parse_chord_symbol("C6/9")
    assert chord.name == "six nine"
    root, chord, bass = parse_chord_symbol("C/E")
    assert (root.name, chord.name, bass.name) == ("C", "major", "E")
    root, chord, bass = parse_chord_symbol("Cm7/G")
    assert (root.name, chord.name, bass.name) == ("C", "minor 7", "G")
    with pytest.raises(ValueError):
        parse_chord_symbol("Czzz")


def test_match_exact():
    result = match_chords("c e g")
    top = result["matches"][0]
    assert top["match"] == "exact"
    assert top["symbol"] == "C"
    assert top["chord_type"] == "major"
    assert "inversion" not in top


def test_match_inversion():
    result = match_chords(["E", "G", "C"])
    top = result["matches"][0]
    assert top["symbol"] == "C/E"
    assert top["inversion"] == "first inversion"
    assert top["bass"] == "E"


def test_match_seventh():
    result = match_chords(["C", "E", "G", "B"])
    assert result["matches"][0]["symbol"] == "Cmaj7"


def test_match_enharmonic_equivalents_c6_am7():
    result = match_chords(["C", "E", "G", "A"])
    exact = [m for m in result["matches"] if m["match"] == "exact"]
    symbols = {m["symbol"] for m in exact}
    assert "C6" in symbols
    assert any(s.startswith("Am7") for s in symbols)
    # bass note C -> root-position C6 sorts first
    assert exact[0]["symbol"] == "C6"


def test_match_partial():
    result = match_chords(["C", "E"], include_partial=True)
    partial = [m for m in result["matches"] if m["match"] == "partial"]
    assert partial
    c_major = next(m for m in partial if m["symbol"] == "C")
    assert c_major["missing_notes"] == ["G"]


def test_match_octaves_ignored():
    a = match_chords(["C3", "E4", "G5"])
    b = match_chords(["C", "E", "G"])
    assert [m["symbol"] for m in a["matches"]] == [m["symbol"] for m in b["matches"]]


def test_list_chords():
    listing = list_chords()
    assert listing["count"] == len(CHORDS)
    examples = {c["example"] for c in listing["chords"]}
    assert {"C", "Cm", "C7", "Cmaj7", "Cdim"} <= examples
