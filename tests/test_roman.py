"""roman_to_chords, progression_library, the shared recognition helpers and borrowing sources.

Unit tests pin the spec examples; the sweeps below check the properties in all
keys (every heptatonic scale x 14 tonic spellings), transposition, chaining
into the voicing/rendering tools, and that bad input only ever raises ValueError.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from midi_composer_mcp import harmony
from midi_composer_mcp.chords import CHORDS, chord_notes, parse_chord_symbol
from midi_composer_mcp.diatonic import (
    PARALLEL_MODES,
    borrowed_sources,
    borrowing_sources,
    degrees_to_chords,
    diatonic_chords,
    numeral_base,
    roman_suffix,
    split_tokens,
)
from midi_composer_mcp.harmony import voice_leading
from midi_composer_mcp.masters import bach_chorale_voicing
from midi_composer_mcp.midi_io import _parse_chord_list, render_chords
from midi_composer_mcp.notes import LETTERS, parse_note
from midi_composer_mcp.roman import (
    PROGRESSION_CATEGORIES,
    PROGRESSIONS,
    applied_reading,
    figure_for_bass,
    progression_library,
    read_chord,
    roman_to_chords,
    special_reading,
)
from midi_composer_mcp.scales import SCALES, resolve_scale_type, scale_notes

TONICS = ["C", "G", "D", "A", "E", "B", "F#", "C#", "F", "Bb", "Eb", "Ab", "Db", "Gb"]
HEPTATONIC = [s.name for s in SCALES.values() if len(s.intervals) == 7]

EXAMPLE_MAJOR = "I V6 vi IV64 bVII V7/V iv N6 Ger65 Cad64 V7 I"
EXAMPLE_MINOR = "i iv V vii°7 VI It6"


def sym(token, key="C", scale="major"):
    return roman_to_chords(token, key, scale)["symbols"][0]


def chord(token, key="C", scale="major"):
    return roman_to_chords(token, key, scale)["chords"][0]


def pc(name: str) -> int:
    return parse_note(name).pitch_class


def rel_pcs(names, tonic: str) -> list[int]:
    t = pc(tonic)
    return [(pc(n) - t) % 12 for n in names]


def letter_offset(name: str, tonic: str) -> int:
    return (LETTERS.index(parse_note(name).letter) - LETTERS.index(parse_note(tonic).letter)) % 7


# --------------------------------------------------------------- spec examples

def test_example_major_line():
    r = roman_to_chords(EXAMPLE_MAJOR, "C")
    assert r["symbols"] == ["C", "G/B", "Am", "F/C", "Bb", "D7", "Fm", "Db/F",
                            ["Ab", "C", "Eb", "F#"], "C/G", "G7", "C"]
    assert r["key"] == "C major"
    assert r["bass"] == ["C", "B", "A", "C", "Bb", "D", "F", "F", "Ab", "G", "G", "C"]
    c5 = r["chords"][5]
    assert c5["token"] == "V7/V" and c5["symbol"] == "D7" and c5["kind"] == "applied"
    assert c5["applied_to"] == "G" and c5["in_key"] is False
    assert c5["non_scale_notes"] == ["F#"] and c5["fit"] == 1


def test_example_minor_line():
    r = roman_to_chords(EXAMPLE_MINOR, "A", "natural minor")
    assert r["symbols"] == ["Am", "Dm", "E", "G#dim7", "F", ["F", "A", "D#"]]


@pytest.mark.parametrize("token,expected", [
    ("V/V/V", "A"), ("viiø7/V", "F#m7b5"), ("bII7/V", "Ab7"), ("ii65", "Dm7/F"),
    ("I6/9", "C6/9"), ("iim6", "Dm6"), ("Vsus4", "Gsus4"), ("V7b5", "G7b5"), ("IΔ9", "Cmaj9"),
    ("Iadd6", "C6"), ("VI/vi", "F"), ("IV/vi", "D"), ("N", "Db/F"), ("N6", "Db/F"), ("bII", "Db"),
    ("IV7", "F7"),  # Kostka & Payne: uppercase + 7 is a dominant seventh
    ("IVΔ7", "Fmaj7"), ("V7/V", "D7"), ("vii°7/V", "F#dim7"), ("V65/vi", "E7/G#"), ("IV/IV", "Bb"),
    ("ii7/V", "Am7"), ("V7/bVII", "F7"), ("viiø", "Bm7b5"), ("IΔ", "Cmaj7"), ("V2", "G7/F"),
    ("V42", "G7/F"), ("V43", "G7/D"), ("I64", "C/G"), ("vii°6", "Bdim/D"), ("viio7", "Bdim7"),
    ("viih7", "Bm7b5"), ("IM7", "Cmaj7"), ("iΔ7", "CmMaj7"), ("bIII+", "Ebaug"), ("III+Δ7", "Emaj7#5"),
    ("V+7", "G7#5"), ("ii9", "Dm9"), ("iiadd9", "Dmadd9"), ("V9", "G9"), ("V/N", "Ab"),
    ("♭VII", "Bb"), ("#iv°7", "F#dim7"), ("II7", "D7"),
])
def test_tokens_in_c(token, expected):
    assert sym(token) == expected


def test_minor_mode_degrees_six_and_seven():
    assert sym("vi", "A", "natural minor") == "F#m"
    assert sym("VII", "A", "natural minor") == "G"
    assert sym("VII", "A", "harmonic minor") == "G"        # the subtonic, not G#
    assert sym("vii°", "A", "natural minor") == "G#dim"
    assert sym("VI", "A", "melodic minor") == "F"
    assert sym("bVII", "A", "natural minor") == "G"        # the dialect: music21 would say Gb
    assert sym("bVI", "A", "natural minor") == "F"
    assert sym("VI", "D", "dorian") == "B"                 # no Minor67 outside the three minors


def test_specials():
    assert chord("Ger65")["notes"] == ["Ab", "C", "Eb", "F#"]
    assert chord("Ger65", "A", "natural minor")["notes"] == ["F", "A", "C", "D#"]
    assert chord("Sw43")["notes"] == ["Ab", "C", "D#", "F#"]
    assert chord("Fr43")["notes"] == ["Ab", "C", "D", "F#"]
    assert chord("It6")["notes"] == ["Ab", "C", "F#"]
    for alias, canon in (("It+6", "It6"), ("It", "It6"), ("Fr+6", "Fr43"), ("Fr", "Fr43"),
                         ("Ger+6", "Ger65"), ("Ger", "Ger65"), ("N", "N6")):
        assert chord(alias)["roman"] == canon
    assert sym("Cad64", "A", "natural minor") == "Am/E"
    assert sym("Cad64") == "C/G"
    ger = chord("Ger65")
    assert ger["chord_type"] is None and ger["root"] is None and ger["kind"] == "augmented sixth"
    assert ger["bass"] == "Ab" and ger["figure"] == "65"
    assert ger["enharmonic_symbol"] == "Ab7"
    assert chord("Sw43")["enharmonic_symbol"] == "Ab7"
    assert chord("Fr43")["enharmonic_symbol"] == "D7b5/Ab"
    assert chord("It6")["enharmonic_symbol"] is None
    n6 = chord("N")
    assert n6["kind"] == "neapolitan" and n6["figure"] == "6" and n6["inversion"] == 1
    assert chord("Cad64")["kind"] == "cadential 6-4" and chord("Cad64")["in_key"] is True


def test_output_fields_and_kinds():
    iv = chord("iv")
    assert iv["kind"] == "borrowed"
    assert iv["borrowed_from"] == ["C harmonic major", "C harmonic minor", "C natural minor",
                                   "C phrygian", "C locrian"]
    assert iv["non_scale_notes"] == ["Ab"] and iv["fit"] == 1
    assert chord("V")["kind"] == "diatonic" and "borrowed_from" not in chord("V")
    assert chord("#iv°7")["kind"] == "chromatic"              # F# A C Eb: no parallel mode holds it
    assert chord("#iv°7")["fit"] == 2                         # capped at 2
    v65 = chord("V65")
    assert v65 == {"token": "V65", "roman": "V65", "symbol": "G7/B", "root": "G",
                   "chord_type": "dominant 7", "notes": ["G", "B", "D", "F"], "bass": "B",
                   "figure": "65", "inversion": 1, "kind": "diatonic", "in_key": True,
                   "non_scale_notes": [], "fit": 0}
    applied = chord("V65/vi")
    assert applied["applied_to"] == "Am" and applied["non_scale_notes"] == ["G#"]
    assert chord("V/IV")["kind"] == "applied" and chord("V/IV")["in_key"] is True
    assert chord("Ger65")["non_scale_notes"] == ["Ab", "Eb", "F#"]   # bass first


def test_normalized_romans():
    assert chord("viiø")["roman"] == "viiø7"
    assert chord("V2")["roman"] == "V42"
    assert chord("viio7")["roman"] == "vii°7"
    assert chord("viih65")["roman"] == "viiø65"
    assert chord("IM7")["roman"] == "IΔ7"
    assert chord("VII°")["roman"] == "vii°"                   # marks override case; case follows the third
    assert chord("♭VII")["roman"] == "bVII"
    assert chord("V/V/V")["roman"] == "V/V/V"
    assert chord("Vdom7")["roman"] == "V7"


def test_token_splitting_and_lists():
    assert roman_to_chords("I V | vi IV", "C")["symbols"] == ["C", "G", "Am", "F"]
    assert roman_to_chords("ii–V–I", "C")["symbols"] == ["Dm", "G", "C"]
    assert roman_to_chords("I, IV, V", "C")["symbols"] == ["C", "F", "G"]
    assert roman_to_chords(["I", " V7/V ", "V"], "C")["symbols"] == ["C", "D7", "G"]
    assert split_tokens("I V | vi", bar_lines=True) == ["I", "V", "vi"]
    assert split_tokens("I V | vi") == ["I", "V", "|", "vi"]   # degrees_to_chords keeps its old split


@pytest.mark.parametrize("token", ["V/vii°", "V96", "Vsus46", "viiø6", "IIm7", "VIII", "bbbII", "Iv",
                                   "X", "", "N6/V", "V/It6", "IΔ6", "vii°Δ7", "#bII", "ii7#9", "It7",
                                   "V/", "V7(b9)", "|"])
def test_invalid_tokens(token):
    with pytest.raises(ValueError):
        roman_to_chords(token, "C")


def test_error_messages_explain_the_rule():
    with pytest.raises(ValueError, match="cannot tonicize"):
        roman_to_chords("V/vii°", "C")
    with pytest.raises(ValueError, match="Hookpad"):
        roman_to_chords("V96", "C")
    with pytest.raises(ValueError, match="ii7"):
        roman_to_chords("IIm7", "C")
    with pytest.raises(ValueError, match="65/43/42"):
        roman_to_chords("viiø6", "C")


def test_requires_heptatonic_scale_and_valid_input():
    with pytest.raises(ValueError, match="7-note"):
        roman_to_chords("I V", "C", "major pentatonic")
    for bad in (None, 7, [], [None], {}, ["I", 5]):
        with pytest.raises(ValueError):
            roman_to_chords(bad, "C")


def test_degrees_to_chords_warns_but_keeps_positions():
    r = degrees_to_chords("C", "major", "IV iv I")
    assert r["symbols"] == ["F", "F", "C"]
    assert len(r["warnings"]) == 1 and "roman_to_chords" in r["warnings"][0]
    assert r["warnings"][0] == "iv resolved to F (the scale's own chord); for F minor use roman_to_chords"
    # the contract of tests/test_diatonic.py stays: marks are tolerated
    r = degrees_to_chords("C", "major", ["vii°", "V7", "I"])
    assert r["symbols"] == ["Bdim", "G", "C"] and len(r["warnings"]) == 1   # the ignored 7 is reported
    assert degrees_to_chords("A", "minor", "i VI VII i")["warnings"] == []
    assert degrees_to_chords("C", "major", [1, 5, "6", 4])["warnings"] == []
    assert degrees_to_chords("C", "major", "V", sevenths=True)["warnings"] == []   # 7th came from sevenths
    assert len(degrees_to_chords("C", "major", "V6 I")["warnings"]) == 1          # ignored inversion
    assert len(degrees_to_chords("A", "harmonic minor", "VII")["warnings"]) == 1  # VII reads G, got G#dim
    assert degrees_to_chords("C", "major pentatonic", "V7")["warnings"]          # non-heptatonic suffix


# -------------------------------------------------------------- shared helpers

def test_numeral_base_moved_and_reexported():
    assert harmony._numeral_base is numeral_base
    assert numeral_base(parse_note("Gb"), parse_note("C")) == ("V", "b")
    assert numeral_base(parse_note("F#"), parse_note("C")) == ("IV", "#")


def test_roman_does_not_import_harmony_at_module_level():
    src = pathlib.Path(__file__).parents[1] / "src" / "midi_composer_mcp" / "roman.py"
    tree = ast.parse(src.read_text())
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            assert node.module not in ("harmony", "server", "masters", "midi_io"), node.module
        if isinstance(node, ast.Import):
            assert all("harmony" not in a.name for a in node.names)


@pytest.mark.parametrize("symbol,figure", [
    ("C", ""), ("C/E", "6"), ("C/G", "64"), ("G7", "7"), ("G7/B", "65"), ("G7/D", "43"), ("G7/F", "42"),
    ("Bdim/D", "6"), ("Cmaj7/B", "42"), ("Bm7b5/F", "43"), ("Csus4/F", None), ("C6/E", None),
    ("C6", ""), ("C9", ""), ("C/D", None), ("G7b5/B", None),
])
def test_figure_for_bass(symbol, figure):
    r = read_chord(symbol)
    assert figure_for_bass(r["chord_type"], r["root"], r["bass"]) == figure
    assert figure_for_bass(None, r["root"], r["bass"]) is None
    assert figure_for_bass(r["chord_type"].name, r["root"], None) in ("", "7")


def test_read_chord_bass_rules():
    assert read_chord(["E3", "G3", "C4"])["bass"].name == "E"      # lowest MIDI note
    low = read_chord(["C4", "E3", "G3"])
    assert low["bass"].name == "E" and low["root"].name == "C"      # E3 is lowest though written 2nd
    assert read_chord(["E", "G", "C"])["bass"].name == "E"          # no octaves: first note
    it6 = read_chord(["Ab", "C", "F#"])
    assert it6["root"] is None and it6["chord_type"] is None and it6["written"] is True
    g7b = read_chord("G7/B")
    assert g7b["bass"].name == "B" and g7b["chord_type"].name == "dominant 7" and not g7b["written"]
    assert read_chord(g7b) is g7b
    with pytest.raises(ValueError):
        read_chord(5)


def test_applied_reading():
    assert applied_reading("A7", "C")["applied"] == "V7/ii"
    assert applied_reading("D7", "C")["applied"] == "V7/V"
    assert applied_reading("C7", "C")["applied"] == "V7/IV"
    assert applied_reading("C", "C") is None                      # in key: never 'V/IV'
    assert applied_reading("G7", "C") is None
    f = applied_reading("F#dim7", "C")
    assert f["applied"] == "vii°7/V" and f["target_symbol"] == "G" and f["type"] == "leading-tone"
    assert applied_reading("F#m7b5", "C")["applied"] == "viiø7/V"
    assert applied_reading("E", "C")["applied"] == "V/vi"
    assert applied_reading("E9", "C")["applied"] == "V9/vi"
    assert applied_reading("Bb7", "C") is None                     # Eb is not diatonic
    assert applied_reading("Bdim7", "C") is None                   # vii°7 of the tonic: mixture
    assert applied_reading("E7", "A", "natural minor") == {"applied": None,
                                                           "function_note": "harmonic-minor dominant"}
    assert applied_reading("G#dim7", "A", "natural minor")["function_note"] == "harmonic-minor dominant"
    assert applied_reading("D7", "A", "natural minor")["applied"] == "V7/bVII"
    assert applied_reading("E7", "A", "harmonic minor") is None    # in key there
    assert applied_reading("E7", "C", "major pentatonic") is None
    # the reading round-trips through roman_to_chords
    for key in TONICS:
        for c in roman_to_chords("V7/ii V/V vii°7/V viiø7/vi V7/IV V9/iii", key)["chords"]:
            got = applied_reading(c["symbol"], key)
            assert sym(got["applied"], key) == c["symbol"], (key, c["token"])


def test_special_reading():
    assert special_reading("Db/F", "C") == {"special": "Neapolitan", "roman_figured": "N6"}
    assert special_reading("Db", "C")["roman_figured"] == "bII"
    assert special_reading("Db/Ab", "C")["roman_figured"] == "bII64"
    assert special_reading("C#/E#", "C") is None                   # spelled on C: #I, not the lowered 2nd
    assert special_reading(["Ab", "C", "F#"], "C")["special"] == "It6"
    assert special_reading(["Ab", "C", "Eb", "F#"], "C")["special"] == "Ger65"
    assert special_reading(["Ab", "C", "D", "F#"], "C")["special"] == "Fr43"
    assert special_reading(["Ab", "C", "D#", "F#"], "C")["special"] == "Sw43"
    assert special_reading(["Ab3", "C4", "Eb4", "F#4"], "C")["special"] == "Ger65"
    assert special_reading(["F", "A", "C", "D#"], "A", "natural minor")["special"] == "Ger65"
    assert special_reading("Ab7", "C") == {"enharmonic_to": "Ger65"}
    assert special_reading(["Ab", "C", "Eb", "Gb"], "C") == {"enharmonic_to": "Ger65"}  # misspelled
    assert special_reading("D7b5/Ab", "C") == {"enharmonic_to": "Fr43"}
    assert special_reading(["C", "Ab", "Eb", "F#"], "C") is None    # b6 not in the bass
    assert special_reading("C/G", "C", next_chord="G7") == {
        "special": "Cad64", "roman_figured": "Cad64", "function_note": "dominant (cadential 6/4)"}
    assert special_reading("C/G", "C", next_chord="Dm") is None
    assert special_reading("C/G", "C") is None
    assert special_reading("Am/E", "A", "natural minor", "E")["special"] == "Cad64"
    assert special_reading("G7", "C") is None


def test_borrowed_sources_rule():
    assert borrowed_sources(["F", "Ab", "C"], "C") == ["C harmonic major", "C harmonic minor",
                                                        "C natural minor", "C phrygian", "C locrian"]
    distances = {s["label"]: s["distance"] for s in borrowing_sources("C", "major")}
    assert distances == {"C lydian": 1, "C mixolydian": 1, "C melodic minor": 1, "C harmonic major": 1,
                         "C dorian": 2, "C harmonic minor": 2, "C natural minor": 3, "C phrygian": 4,
                         "C locrian": 5}
    assert "C major" not in distances and len(PARALLEL_MODES) == 10
    assert borrowed_sources([0, 4, 7], "C") == ["C lydian", "C mixolydian", "C harmonic major"]
    assert borrowed_sources([0, 4, 7], "C", include_home=True, fifths_steps=1) == [
        "C major", "C lydian", "C mixolydian", "C harmonic major", "G major", "F major"]
    keys = [s["label"] for s in borrowing_sources("C", "major", modes=[], fifths_steps=2)]
    assert keys == ["G major", "F major", "D major", "Bb major"]
    keys = [s["label"] for s in borrowing_sources("A", "natural minor", modes=[], fifths_steps=1)]
    assert keys == ["E natural minor", "D natural minor"]
    assert borrowed_sources(["B", "D", "F#"], "C", modes=[], fifths_steps=1) == ["G major"]
    assert borrowed_sources(["E", "G#", "B"], "A", "aeolian") == [
        "A harmonic minor", "A melodic minor", "A harmonic major", "A major", "A lydian"]
    for bad in ({"fifths_steps": 7}, {"fifths_steps": -1}, {"modes": ["nope"]}, {"modes": 5}):
        with pytest.raises(ValueError):
            borrowing_sources("C", "major", **bad)
    with pytest.raises(ValueError):
        borrowed_sources([None], "C")


@pytest.mark.parametrize("tonic", TONICS)
def test_borrowed_sources_are_key_relative(tonic):
    base = borrowed_sources(["F", "Ab", "C"], "C")
    t = pc(tonic)
    moved = [(p + t) % 12 for p in (5, 8, 0)]
    got = borrowed_sources(moved, tonic)
    assert [g.split(" ", 1)[1] for g in got] == [b.split(" ", 1)[1] for b in base]


# ----------------------------------------------------------------- invariants

@pytest.mark.parametrize("scale", HEPTATONIC)
def test_every_diatonic_roman_parses_back(scale):
    """(1) diatonic_chords' romans round-trip to the same root and pitch classes (and the same text)."""
    for tonic in TONICS:
        for sevenths in (False, True):
            rows = diatonic_chords(tonic, scale, sevenths)["chords"]
            romans = [c["roman"] for c in rows]
            for c in rows:
                if "?" in c["roman"]:
                    continue
                r = chord(c["roman"], tonic, scale)
                assert r["root"] == c["root"], (scale, tonic, c["roman"])
                assert sorted(map(pc, r["notes"])) == sorted(map(pc, c["notes"])), (scale, tonic, c["roman"])
                assert r["roman"] == c["roman"]
                assert r["kind"] == "diatonic" and r["in_key"]
            # a numeral that means exactly what degrees_to_chords resolves draws no warning
            assert degrees_to_chords(tonic, scale, romans, sevenths)["warnings"] == []


@pytest.mark.parametrize("tonic", TONICS)
def test_examples_are_transposition_invariant(tonic):
    """(2) both example lines in every key: C's pitch classes shifted, root letters = tonic + (n-1)."""
    for line, scale, home in ((EXAMPLE_MAJOR, "major", "C"), (EXAMPLE_MINOR, "natural minor", "A")):
        ref = roman_to_chords(line, home, scale)["chords"]
        got = roman_to_chords(line, tonic, scale)["chords"]
        for a, b in zip(ref, got):
            assert rel_pcs(b["notes"], tonic) == rel_pcs(a["notes"], home), (tonic, a["token"])
            assert [letter_offset(n, tonic) for n in b["notes"]] == \
                   [letter_offset(n, home) for n in a["notes"]], (tonic, a["token"])
            assert letter_offset(b["bass"], tonic) == letter_offset(a["bass"], home)
            for key in ("kind", "figure", "inversion", "in_key", "fit", "roman"):
                assert a[key] == b[key], (tonic, a["token"], key)
            assert len(b["non_scale_notes"]) == len(a["non_scale_notes"])
    assert sym("V7/V", "Gb") == "Ab7"
    assert sym("V7/V", "F#") == "G#7"


def test_symbols_chain_into_voicing_and_rendering(tmp_path):
    """(3) symbols feed voice_leading, chords_to_midi and bach_chorale_voicing; 'G7/B' keeps B in the bass."""
    symbols = roman_to_chords(EXAMPLE_MAJOR, "C")["symbols"]
    vl = voice_leading(symbols)
    assert len(vl["voicings"]) == len(symbols)
    g7b = roman_to_chords("I V65 I", "C")["symbols"]
    assert g7b[1] == "G7/B"
    voiced = voice_leading(g7b)["voicings"][1]
    assert parse_note(min(voiced["notes"], key=lambda n: parse_note(n).midi)).letter == "B"
    rendered = render_chords(symbols, output_dir=str(tmp_path))
    assert rendered["chord_count"] == len(symbols)
    bach = bach_chorale_voicing(g7b, root="C")
    assert parse_note(bach["voices"]["bass"][1]).pitch_class == pc("B")
    minor = roman_to_chords(EXAMPLE_MINOR, "A", "natural minor")["symbols"]
    assert render_chords(minor, output_dir=str(tmp_path))["chord_count"] == len(minor)
    assert len(voice_leading(minor)["voicings"]) == len(minor)


_TOKEN_SWEEP = ("I ii iii IV V vi vii° I6 ii65 V43 V42 vii°7 viiø7 IΔ7 iΔ7 III+ bVII iv bVI bIII V7/V "
                "vii°7/V V65/vi IV/IV V/V/V bII7/V N6 N It6 Fr43 Ger65 Sw43 Cad64 I6/9 V9 Vsus4 ii9 "
                "iim6 Iadd6 IΔ9 V7b9 V7#9 #iv°7 II7 bVII7 iv7 V+7 viih43 IM65 V2 VII°").split()


@pytest.mark.parametrize("scale", ["major", "natural minor", "harmonic minor", "dorian"])
def test_normalized_roman_reads_the_same_chord(scale):
    for tonic in ("C", "F#", "Bb"):
        for c in roman_to_chords(_TOKEN_SWEEP, tonic, scale)["chords"]:
            again = chord(c["roman"], tonic, scale)
            assert again["symbol"] == c["symbol"] and again["roman"] == c["roman"], (tonic, c["token"])


# ---------------------------------------------------------- progression library

def test_every_entry_resolves_in_c_or_a():
    for name, spec in PROGRESSIONS.items():
        mode = spec[1]
        e = progression_library(name, root="C" if mode == "major" else "A")
        n = len(e["numerals"].split())
        assert len(e["chords"]) == n and len(e["bass"]) == n, name
        assert e["key"] == ("C major" if mode == "major" else "A natural minor")
        assert "warning" not in e
        assert e["category"] in PROGRESSION_CATEGORIES


def test_progression_examples():
    a = progression_library("andalusian", root="E")
    assert a["numerals"] == "i VII VI V" and a["chords"] == ["Em", "D", "C", "B"]
    assert a["bass"] == ["E", "D", "C", "B"]
    p = progression_library("prinner", root="G")
    assert p["chords"] == ["C", "G/B", "F#dim/A", "G"]
    assert p["bass"] == ["C", "B", "A", "G"] and p["melody"] == ["E", "D", "C", "B"]
    assert progression_library("fonte", root="C")["bass"] == ["C#", "D", "B", "C"]
    assert progression_library("monte", root="C")["bass"] == ["E", "F", "F#", "G"]
    assert progression_library("lament", root="A")["bass"] == ["A", "G", "F", "E"]
    assert progression_library("pop_axis", root="C")["chords"] == ["C", "G", "Am", "F"]
    assert progression_library("twelve_bar_blues", root="A")["chords"][4] == "D7"
    assert "melody" not in progression_library("fonte", root="C")


def test_progression_lookup():
    assert progression_library("Canon")["name"] == "pachelbel"
    assert progression_library("D2 (-5/+4)")["name"] == "descending_fifths"
    assert progression_library("d3 (-4/+2)")["name"] == "pachelbel"
    assert progression_library("ii-V-I")["name"] == "ii_V_I"
    assert progression_library("Pop Axis")["name"] == "pop_axis"
    assert progression_library("i-V-vi-iv")["name"] == "pop_axis"
    unresolved = progression_library("prinner")
    assert "chords" not in unresolved and unresolved["bass_degrees"] == [4, 3, 2, 1]
    listing = progression_library()
    assert listing["count"] == len(PROGRESSIONS)
    keys = [(e["category"], e["name"]) for e in listing["progressions"]]
    assert keys == sorted(keys)
    schemas = progression_library(category="schema")
    assert {e["name"] for e in schemas["progressions"]} == {
        "prinner", "meyer", "romanesca", "do_re_mi", "fenaroli", "fonte", "monte"}
    resolved = progression_library(category="jazz", root="F")
    assert all("chords" in e for e in resolved["progressions"])


def test_schema_melodies_are_chord_tones():
    """Each schema's melody note belongs to the chord under it (the table is self-consistent)."""
    for name, spec in PROGRESSIONS.items():
        if spec[7] is None:
            continue
        e = progression_library(name, root="C")
        for symbol, note in zip(e["chords"], e["melody"], strict=True):
            tones = {t.pitch_class for t in _parse_chord_list([symbol])[0]["tones"]}
            assert pc(note) in tones, (name, symbol, note)


def test_progression_mode_mismatch_warns():
    e = progression_library("andalusian", root="C", scale_type="major")
    assert "warning" in e and e["key"] == "C major"
    assert "warning" not in progression_library("andalusian", root="D", scale_type="dorian")
    assert "warning" in progression_library("pop_axis", root="A", scale_type="natural minor")


def test_progression_library_errors():
    for kwargs in ({"name": "nope"}, {"category": "metal"}, {"name": 5}, {"name": ""},
                   {"scale_type": "major"}, {"root": 5}, {"name": "prinner", "category": "jazz"},
                   {"name": "prinner", "root": "C", "scale_type": "major pentatonic"}):
        with pytest.raises(ValueError):
            progression_library(**kwargs)


def _degree_pc(degree, tonic: str, scale: str = "major") -> int:
    notes = scale_notes(resolve_scale_type(scale), parse_note(tonic))
    if isinstance(degree, str):
        shift = degree.count("#") - degree.count("b")
        return (notes[int(degree.lstrip("#b")) - 1].pitch_class + shift) % 12
    return notes[degree - 1].pitch_class


@pytest.mark.parametrize("tonic", TONICS)
def test_progressions_in_every_key(tonic):
    """Every entry resolves in every key, transposition-invariant, bass count = numeral count."""
    for name, spec in PROGRESSIONS.items():
        mode = spec[1]
        home = "C" if mode == "major" else "A"
        scale = "major" if mode == "major" else "natural minor"
        ref = progression_library(name, root=home)
        got = progression_library(name, root=tonic)
        assert len(got["bass"]) == len(got["numerals"].split())
        ref_parsed, got_parsed = _parse_chord_list(ref["chords"]), _parse_chord_list(got["chords"])
        for a, b in zip(ref_parsed, got_parsed):
            assert rel_pcs([t.name for t in b["tones"]], tonic) == rel_pcs([t.name for t in a["tones"]], home)
        assert rel_pcs(got["bass"], tonic) == rel_pcs(ref["bass"], home)
        if "bass_degrees" in got:  # the schema bass, chromatic notes included (#1, #4)
            assert [pc(b) for b in got["bass"]] == [_degree_pc(d, tonic, scale) for d in got["bass_degrees"]]
        if "melody" in got:
            assert [pc(m) for m in got["melody"]] == [_degree_pc(d, tonic, scale) for d in got["melody_degrees"]]


def test_progressions_chain_into_rendering(tmp_path):
    for name in ("royal_road", "fonte", "lament", "ragtime"):
        e = progression_library(name, root="Eb")
        assert render_chords(e["chords"], output_dir=str(tmp_path))["chord_count"] == len(e["chords"])
        assert len(voice_leading(e["chords"])["voicings"]) == len(e["chords"])
    meyer = progression_library("meyer", root="G")
    bach = bach_chorale_voicing(meyer["chords"], root="G", melody=meyer["melody"])
    assert [parse_note(n).pitch_class for n in bach["voices"]["soprano"]] == [pc(m) for m in meyer["melody"]]


# ------------------------------------------------------- only ValueErrors, never crashes

_BAD_VALUES = [None, 7, -1, 3.5, 2.0, True, "", "zz", [], [None], {}, {"a": 1}, ["C", 5], "C" * 5,
               ["C", "Db", "D"], [["C", "Db", "D"], "G"], "0 4 7", ["C", ".", "E"]]


def test_new_tools_reject_bad_input_with_value_errors():
    calls = {
        roman_to_chords: {"numerals": "I V vi IV", "root": "C", "scale_type": "major"},
        progression_library: {"name": None, "root": None, "scale_type": None, "category": None},
        degrees_to_chords: {"root": "C", "scale_type": "major", "degrees": "I iv V7", "sevenths": False},
    }
    failures = []
    for fn, good in calls.items():
        for pname in good:
            for bad in _BAD_VALUES:
                try:
                    fn(**dict(good, **{pname: bad}))
                except ValueError:
                    pass
                except Exception as e:  # noqa: BLE001 - the list is the point
                    failures.append(f"{fn.__name__}({pname}={bad!r}) raised {type(e).__name__}: {e}")
    assert not failures, "\n".join(failures)


def test_chord_types_in_the_table_are_reachable():
    """Every table chord type is reachable as numeral + roman_suffix, as analyze_progression writes it."""
    for ctype in CHORDS.values():
        lower = 3 in ctype.pitch_classes and 4 not in ctype.pitch_classes
        numeral = "ii" if lower else "II"
        token = numeral + roman_suffix(ctype, lower)
        got = chord(token)
        assert got["chord_type"] == ctype.name, token
        assert got["notes"] == [n.name for n in chord_notes(ctype, parse_note("D"))]
        assert parse_chord_symbol(got["symbol"])[1] is ctype
