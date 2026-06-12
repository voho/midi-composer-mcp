import pytest

from midi_composer_mcp.scales import (
    SCALES,
    list_scales,
    match_scales,
    resolve_scale_type,
    scale_info,
)


def test_resolve_aliases():
    assert resolve_scale_type("maj").name == "major"
    assert resolve_scale_type("MAJOR").name == "major"
    assert resolve_scale_type("Ionian").name == "major"
    assert resolve_scale_type("min").name == "natural minor"
    assert resolve_scale_type("aeolian").name == "natural minor"
    assert resolve_scale_type("Melodic Minor").name == "melodic minor"
    with pytest.raises(ValueError, match="Unknown scale type"):
        resolve_scale_type("klingon")


def test_c_major():
    info = scale_info("major", "c")
    assert info["notes"] == ["C", "D", "E", "F", "G", "A", "B", "C"]
    assert info["intervals"] == [0, 2, 4, 5, 7, 9, 11]
    assert info["degrees"] == ["1", "2", "3", "4", "5", "6", "7"]
    assert "midi" not in info  # no octave on the root


def test_proper_flat_and_sharp_spelling():
    assert scale_info("major", "F")["notes"] == ["F", "G", "A", "Bb", "C", "D", "E", "F"]
    assert scale_info("minor", "D")["notes"] == ["D", "E", "F", "G", "A", "Bb", "C", "D"]
    assert scale_info("major", "F#")["notes"] == ["F#", "G#", "A#", "B", "C#", "D#", "E#", "F#"]
    assert scale_info("harmonic minor", "A")["notes"] == ["A", "B", "C", "D", "E", "F", "G#", "A"]
    assert scale_info("blues", "C")["notes"] == ["C", "Eb", "F", "Gb", "G", "Bb", "C"]


def test_octave_aware_scale():
    info = scale_info("major", "C5")
    assert info["notes"] == ["C5", "D5", "E5", "F5", "G5", "A5", "B5", "C6"]
    assert info["midi"] == [72, 74, 76, 77, 79, 81, 83, 84]
    # octave carry mid-scale
    a_min = scale_info("minor", "A3")
    assert a_min["notes"] == ["A3", "B3", "C4", "D4", "E4", "F4", "G4", "A4"]
    assert a_min["midi"][0] == 57 and a_min["midi"][-1] == 69


def test_scale_without_root_is_db_entry():
    info = scale_info("dorian")
    assert info["intervals"] == [0, 2, 3, 5, 7, 9, 10]
    assert "notes" not in info


def test_list_scales():
    listing = list_scales()
    assert listing["count"] == len(SCALES)
    names = {s["scale_type"] for s in listing["scales"]}
    assert {"major", "natural minor", "blues", "chromatic"} <= names


def test_match_contains():
    result = match_scales("c e g")
    assert result["input_notes"] == ["C", "E", "G"]
    assert result["match_count"] > 0
    top = result["matches"][0]
    # smallest scale rooted on the first input note sorts first
    assert top["root"] == "C"
    assert top["scale_type"] == "major pentatonic"
    pairs = {(m["root"], m["scale_type"]) for m in result["matches"]}
    assert ("C", "major") in pairs
    c_major = next(m for m in result["matches"] if m["scale_type"] == "major" and m["root"] == "C")
    assert c_major["match"] == "contains"
    assert c_major["added_notes"] == ["A", "B", "D", "F"]


def test_match_exact_modes():
    result = match_scales(["C", "D", "E", "F", "G", "A", "B"], exact_only=True)
    assert all(m["match"] == "exact" for m in result["matches"])
    pairs = {(m["root"], m["scale_type"]) for m in result["matches"]}
    # all seven modes of the same pitch-class set match exactly
    assert ("C", "major") in pairs
    assert ("D", "dorian") in pairs
    assert ("A", "natural minor") in pairs
    # C-rooted interpretation sorts first
    assert result["matches"][0]["root"] == "C"
    assert result["matches"][0]["scale_type"] == "major"


def test_match_respects_input_spelling():
    result = match_scales(["Bb", "D", "F"])
    roots = {m["root"] for m in result["matches"]}
    assert "Bb" in roots and "A#" not in roots


def test_match_octaves_ignored():
    with_octaves = match_scales(["C4", "E5", "G3"])
    without = match_scales(["C", "E", "G"])
    strip = lambda r: [(m["root"], m["scale_type"], m["match"]) for m in r["matches"]]
    assert strip(with_octaves) == strip(without)


def test_chromatic_excluded_from_matching():
    result = match_scales("c e g", limit=1000)
    assert all(m["scale_type"] != "chromatic" for m in result["matches"])


def test_match_limit():
    result = match_scales("c", limit=5)
    assert len(result["matches"]) == 5
    assert result["match_count"] > 5
