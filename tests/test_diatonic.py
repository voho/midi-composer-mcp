import pytest

from midi_composer_mcp.diatonic import degrees_to_chords, diatonic_chords


def test_c_major_triads():
    result = diatonic_chords("C", "major")
    symbols = [c["symbol"] for c in result["chords"]]
    romans = [c["roman"] for c in result["chords"]]
    assert symbols == ["C", "Dm", "Em", "F", "G", "Am", "Bdim"]
    assert romans == ["I", "ii", "iii", "IV", "V", "vi", "vii°"]
    assert [c["harmonic_function"] for c in result["chords"]] == [
        "tonic", "subdominant", "tonic", "subdominant", "dominant", "tonic", "dominant",
    ]
    assert result["chords"][6]["degree_name"] == "leading tone"


def test_c_major_sevenths():
    result = diatonic_chords("C", "major", sevenths=True)
    symbols = [c["symbol"] for c in result["chords"]]
    romans = [c["roman"] for c in result["chords"]]
    assert symbols == ["Cmaj7", "Dm7", "Em7", "Fmaj7", "G7", "Am7", "Bm7b5"]
    assert romans == ["IΔ7", "ii7", "iii7", "IVΔ7", "V7", "vi7", "viiø7"]


def test_a_natural_minor_triads():
    result = diatonic_chords("A", "natural minor")
    symbols = [c["symbol"] for c in result["chords"]]
    romans = [c["roman"] for c in result["chords"]]
    assert symbols == ["Am", "Bdim", "C", "Dm", "Em", "F", "G"]
    assert romans == ["i", "ii°", "bIII", "iv", "v", "bVI", "bVII"]
    assert result["chords"][6]["degree_name"] == "subtonic"


def test_harmonic_minor_has_augmented_and_dim7():
    triads = diatonic_chords("A", "harmonic minor")
    assert triads["chords"][2]["chord_type"] == "augmented"
    sevenths = diatonic_chords("A", "harmonic minor", sevenths=True)
    assert sevenths["chords"][6]["chord_type"] == "diminished 7"
    assert sevenths["chords"][0]["chord_type"] == "minor major 7"


def test_octave_aware_diatonic():
    result = diatonic_chords("C4", "major")
    five = result["chords"][4]
    assert five["notes"] == ["G4", "B4", "D5"]
    assert five["midi"] == [67, 71, 74]


def test_degrees_to_chords_formats():
    expected = ["C", "G", "Am", "F"]
    for degrees in ([1, 5, 6, 4], "1-5-6-4", "I V vi IV", ["I", "V", "vi", "IV"], "1,5,6,4"):
        result = degrees_to_chords("C", "major", degrees)
        assert result["symbols"] == expected, f"failed for {degrees!r}"
    assert degrees_to_chords("C", "major", ["ii", "V", "I"])["symbols"] == ["Dm", "G", "C"]
    # quality marks on numerals are tolerated
    assert degrees_to_chords("C", "major", ["vii°", "V7", "I"])["symbols"] == ["Bdim", "G", "C"]


def test_degrees_to_chords_minor():
    # degrees are positions, so VI and VII resolve to the scale's own chords
    result = degrees_to_chords("A", "minor", "i VI VII i")
    assert result["symbols"] == ["Am", "F", "G", "Am"]


def test_degrees_validation():
    with pytest.raises(ValueError, match="out of range"):
        degrees_to_chords("C", "major", [8])
    with pytest.raises(ValueError, match="Chromatic alterations"):
        degrees_to_chords("C", "major", ["bVII"])
    with pytest.raises(ValueError, match="Invalid scale degree"):
        degrees_to_chords("C", "major", ["XIV"])
    with pytest.raises(ValueError):
        degrees_to_chords("C", "major", [])


def test_degree_seven_as_string():
    assert degrees_to_chords("C", "major", "7")["symbols"] == ["Bdim"]


def test_pentatonic_diatonic_has_no_romans():
    result = diatonic_chords("C", "major pentatonic")
    assert len(result["chords"]) == 5
    assert all("roman" not in c for c in result["chords"])
