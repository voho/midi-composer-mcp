"""Matching ignores octaves: notes like C5 match the same as C."""

from midi_composer_mcp.chords import match_chords
from midi_composer_mcp.scales import match_scales


def _summary_scales(result):
    return [(m["root"], m["scale_type"], m["match"]) for m in result["matches"]]


def _summary_chords(result):
    return [(m["symbol"], m["match"]) for m in result["matches"]]


def test_scale_match_ignores_octaves():
    plain = match_scales(["C", "E", "G"], limit=1000)
    octaved = match_scales(["C5", "E5", "G5"], limit=1000)
    mixed = match_scales(["C4", "E5", "G3"], limit=1000)
    spread = match_scales("C2 E6 G4", limit=1000)
    assert _summary_scales(plain) == _summary_scales(octaved) == _summary_scales(mixed) == _summary_scales(spread)
    # output notes carry no octave
    assert all(not any(ch.isdigit() for ch in n)
               for m in octaved["matches"] for n in m["notes"])


def test_chord_match_ignores_octaves():
    plain = match_chords(["C", "E", "G"], limit=1000)
    octaved = match_chords(["C5", "E5", "G5"], limit=1000)
    assert _summary_chords(plain) == _summary_chords(octaved)
    assert octaved["matches"][0]["symbol"] == "C"


def test_chord_inversion_detected_with_octaves():
    # first note is the bass, octave ignored -> C/E first inversion
    result = match_chords(["E3", "G4", "C5"])
    assert result["matches"][0]["symbol"] == "C/E"
    assert result["matches"][0]["inversion"] == "first inversion"


def test_match_seventh_with_octaves():
    result = match_chords(["C4", "E4", "G4", "B5"])
    assert result["matches"][0]["symbol"] == "Cmaj7"
