"""find_cadences (analysis.py): the spec examples, the textbook rules, all-key invariants and chaining."""

from __future__ import annotations

import itertools
import random

import pytest

from midi_composer_mcp.analysis import CADENCE_TYPES, find_cadences
from midi_composer_mcp.chords import CHORDS
from midi_composer_mcp.harmony import analyze_progression
from midi_composer_mcp.masters import bach_chorale_voicing
from midi_composer_mcp.melody import notes_from_degrees
from midi_composer_mcp.notes import parse_note
from midi_composer_mcp.roman import progression_library, roman_to_chords
from midi_composer_mcp.structure import plan_sections

TONICS = ["C", "G", "D", "A", "E", "B", "F#", "C#", "F", "Bb", "Eb", "Ab", "Db", "Gb"]
MINORS = ("natural minor", "harmonic minor")
KEYS = [(t, s) for t in TONICS for s in ("major",) + MINORS]


def _only(result):
    assert len(result["cadences"]) == 1
    return result["cadences"][0]


def _labels(result):
    """Everything key-independent: the cadences without their chord spellings."""
    return {
        "phrase_ends": result["phrase_ends"],
        "summary": result["summary"],
        "cadences": [{k: v for k, v in c.items() if k != "chords"} for c in result["cadences"]],
    }


# ============================================================ the spec examples

def test_example_cadential_64_pac():
    r = find_cadences("C F C/G G7 C", "C", soprano="E5 F5 E5 D5 C5")
    c = _only(r)
    assert (c["index"], c["type"], c["subtype"]) == (4, "authentic", "perfect")
    assert c["span"] == [2, 4]
    assert c["romans"] == ["Cad64", "V7", "I"]
    assert c["soprano_degrees"] == [3, 2, 1]
    assert c["bass_degrees"] == [5, 5, 1]
    assert c["chords"] == ["C/G", "G7", "C"]
    assert c["cadential_64"] is True and c["picardy"] is False and c["soprano_checked"] is True
    assert r["key"] == "C major" and r["phrase_ends"] == [4]
    assert r["soprano"] == ["E5", "F5", "E5", "D5", "C5"]
    assert r["summary"] == {"authentic": 1, "half": 0, "plagal": 0, "deceptive": 0, "none": 0}


def test_example_phrase_length_authentic_unknown_soprano_and_deceptive():
    r = find_cadences(["C", "F", "G7", "C", "Am", "Dm", "G", "Am"], "C", phrase_length=4)
    assert r["phrase_ends"] == [3, 7]
    first, second = r["cadences"]
    assert (first["index"], first["type"], first["subtype"]) == (3, "authentic", None)
    assert first["reason"] == "give the soprano to decide PAC vs IAC"
    assert first["soprano_checked"] is False and first["soprano_degrees"] == [None, None]
    assert (second["index"], second["type"], second["subtype"]) == (7, "deceptive", None)
    assert second["romans"] == ["V", "vi"]


def test_example_phrygian_half_cadence():
    c = _only(find_cadences(["Am", "Dm/F", "E"], "A", "natural minor"))
    assert (c["index"], c["type"], c["subtype"]) == (2, "half", "phrygian")
    assert c["romans"] == ["iv6", "V"] and c["bass_degrees"] == [6, 5]
    assert c["caveats"] == []


# ================================================================== unit rules

def test_inverted_tonic_is_an_iac():
    c = _only(find_cadences("C F G7 C/E", "C"))
    assert (c["type"], c["subtype"], c["reason"]) == ("authentic", "imperfect", "inversion")
    assert c["romans"] == ["V7", "I6"]


def test_soprano_on_3_is_an_iac():
    c = _only(find_cadences("C F G C", "C", soprano="C5 C5 D5 E5"))
    assert (c["type"], c["subtype"], c["reason"]) == ("authentic", "imperfect", "soprano on ^3")
    assert c["soprano_degrees"] == [2, 3]


def test_leading_tone_chord_is_an_iac():
    c = _only(find_cadences("C Bdim/D C", "C"))
    assert (c["type"], c["subtype"], c["reason"]) == ("authentic", "imperfect", "leading-tone chord")
    # the leading-tone chord of a minor key, on the raised 7th
    c = _only(find_cadences("Am G#dim7 Am", "A", "natural minor"))
    assert (c["type"], c["subtype"], c["reason"]) == ("authentic", "imperfect", "leading-tone chord")


def test_inversion_and_soprano_reasons_are_joined():
    c = _only(find_cadences("C G/B C", "C", soprano="C5 D5 E5"))
    assert c["reason"] == "inversion; soprano on ^3"


def test_never_a_pac_without_1_in_the_soprano():
    for soprano in ("E5 D5 E5", "E5 D5 G5"):
        assert _only(find_cadences("C G C", "C", soprano=soprano))["subtype"] == "imperfect"
    assert _only(find_cadences("C G C", "C", soprano="E5 D5 C5"))["subtype"] == "perfect"
    assert _only(find_cadences("C G C", "C"))["subtype"] is None


def test_plagal_deceptive_and_none():
    assert _only(find_cadences("C F C", "C"))["type"] == "plagal"
    assert _only(find_cadences("Am Dm Am", "A", "natural minor"))["type"] == "plagal"
    assert _only(find_cadences("C F/A C", "C"))["type"] == "plagal"   # IV6 -> I
    assert _only(find_cadences("C G Am", "C"))["type"] == "deceptive"
    c = _only(find_cadences("Am E F", "A", "natural minor"))
    assert (c["type"], c["subtype"], c["reason"]) == ("deceptive", None, "V -> VI instead of i")
    c = _only(find_cadences("C Am", "C"))
    assert (c["type"], c["subtype"], c["reason"]) == ("none", None, "ends on vi")


def test_deceptive_subtypes():
    c = _only(find_cadences("C G7 Ab", "C"))
    assert (c["type"], c["subtype"]) == ("deceptive", "bVI")
    c = _only(find_cadences("C G F/A", "C"))
    assert (c["type"], c["subtype"], c["romans"]) == ("deceptive", "IV6", ["V", "IV6"])
    c = _only(find_cadences("Am E Dm/F", "A", "natural minor"))
    assert (c["type"], c["subtype"]) == ("deceptive", "IV6")
    # V -> IV in root position is a retrogression, not a deceptive cadence
    assert _only(find_cadences("C G F", "C"))["type"] == "none"


def test_picardy_third():
    c = _only(find_cadences(["E7", "A"], "A", "natural minor"))
    assert c["type"] == "authentic" and c["picardy"] is True
    assert c["romans"] == ["V7", "I"]      # not analyze_progression's context-free V/iv
    assert roman_to_chords(c["romans"], "A", "natural minor")["symbols"] == ["E7", "A"]
    assert _only(find_cadences("Am E7 Am", "A", "natural minor"))["picardy"] is False
    assert _only(find_cadences("C G C", "C"))["picardy"] is False
    c = _only(find_cadences("Am Dm A", "A", "harmonic minor"))
    assert (c["type"], c["picardy"]) == ("plagal", True)
    c = _only(find_cadences("Am E A/C#", "A", "natural minor"))
    assert c["romans"] == ["V", "I6"] and c["reason"] == "inversion"


def test_all_scan_reports_half_cadences_only_at_the_end():
    r = find_cadences("C G C F G Am", "C", phrase_ends="all")
    assert [(c["index"], c["type"]) for c in r["cadences"]] == [(2, "authentic"), (5, "deceptive")]
    assert r["phrase_ends"] == [2, 5]
    r = find_cadences("C G C F G", "C", phrase_ends="ALL")
    assert [(c["index"], c["type"]) for c in r["cadences"]] == [(2, "authentic"), (4, "half")]
    r = find_cadences("C G C F C Am", "C", phrase_ends="all")   # the last chord is reported even as 'none'
    assert [(c["index"], c["type"]) for c in r["cadences"]] == [(2, "authentic"), (4, "plagal"), (5, "none")]


def test_bach_chorale_soprano_decides_iac():
    voices = bach_chorale_voicing("C F G7 C", root="C")["voices"]
    assert voices["soprano"][-1] == "G4"
    c = _only(find_cadences(["C", "F", "G7", "C"], "C", soprano=voices["soprano"]))
    assert (c["type"], c["subtype"], c["reason"]) == ("authentic", "imperfect", "soprano on ^5")


@pytest.mark.parametrize("kwargs", [
    {"phrase_ends": [9]},
    {"phrase_ends": [-1]},
    {"phrase_length": -1},
    {"phrase_ends": [3], "phrase_length": 2},
    {"phrase_ends": "last"},
    {"phrase_ends": []},
    {"phrase_ends": [True]},
    {"phrase_ends": [1.0]},
    {"phrase_length": 5},
    {"phrase_length": 2.0},
    {"soprano": "C5 D5"},
    {"scale_type": "major pentatonic"},
])
def test_bad_phrase_and_soprano_arguments_raise(kwargs):
    with pytest.raises(ValueError):
        find_cadences(["C", "F", "G7", "C"], "C", **kwargs)


def test_non_heptatonic_key_raises():
    with pytest.raises(ValueError, match="seven-note"):
        find_cadences("C G C", "C", "blues")


def test_half_cadence_caveats():
    c = _only(find_cadences("C F G", "C"))
    assert (c["type"], c["reason"], c["caveats"]) == ("half", "ends on V", [])
    for chords in ("C F G7", "C F G/B", "C F G7/D"):
        c = _only(find_cadences(chords, "C"))
        assert c["type"] == "half"
        assert c["caveats"] == ["Caplin: a half cadence ends on a root-position dominant triad"]
    # an inverted V after iv6 is no Phrygian cadence (the bass does not fall a semitone)
    c = _only(find_cadences("Am Dm/F E/G#", "A", "harmonic minor"))
    assert (c["type"], c["subtype"]) == ("half", None)
    # a Phrygian cadence is a minor-key figure
    assert _only(find_cadences("C Dm/F E", "C"))["subtype"] is None


def test_cadential_64():
    c = _only(find_cadences("C F C/G G", "C"))
    assert (c["type"], c["cadential_64"], c["span"], c["romans"]) == ("half", True, [2, 3], ["Cad64", "V"])
    c = _only(find_cadences("C F C/G G7 Am", "C"))
    assert (c["type"], c["cadential_64"], c["span"]) == ("deceptive", True, [2, 4])
    c = _only(find_cadences("Am Dm Am/E E7 Am", "A", "harmonic minor"))
    assert (c["type"], c["cadential_64"], c["span"], c["romans"]) == \
        ("authentic", True, [2, 4], ["Cad64", "V7", "i"])
    assert _only(find_cadences("C F G7 C", "C"))["cadential_64"] is False


def test_a_cadential_64_is_no_tonic_arrival():
    r = find_cadences("C G C/G G", "C", phrase_ends=[2, 3])
    c2, c3 = r["cadences"]
    assert c2["type"] == "none" and c2["romans"] == ["V", "Cad64"] and "cadential 6/4" in c2["reason"]
    assert (c3["type"], c3["cadential_64"]) == ("half", True)
    assert find_cadences("C G C/G G", "C", phrase_ends="all")["phrase_ends"] == [3]


def test_none_reasons():
    assert _only(find_cadences("Am Em Am", "A", "natural minor"))["reason"] == \
        "v -> i: a minor v has no leading tone"
    assert "plagal cadence needs" in _only(find_cadences("C F/C C", "C"))["reason"]
    assert "plagal cadence needs" in _only(find_cadences("C F C/E", "C"))["reason"]
    assert _only(find_cadences("C Gsus4 C", "C"))["reason"] == \
        "Vsus4 -> I: the tonic is not approached from V, vii° or IV"
    assert _only(find_cadences("Am Dm Em", "A", "natural minor"))["reason"] == "ends on v"
    c = _only(find_cadences("C G", "C", phrase_ends=[0]))
    assert (c["type"], c["reason"], c["span"]) == ("none", "no chord before it", [0, 0])


def test_augmented_sixth_is_not_a_bvi():
    ger = ["Ab", "C", "Eb", "F#"]
    assert analyze_progression(["C", ger, "G"], "C")["chords"][1]["special"] == "Ger65"
    c = _only(find_cadences(["C", ger, "G"], "C"))
    assert (c["type"], c["romans"]) == ("half", ["Ger65", "V"])
    assert _only(find_cadences(["C", "G7", ger], "C"))["type"] == "none"


def test_spelling_is_read_by_letter():
    # Gb sounds like the tonic of F# major but is spelled as a second degree
    c = _only(find_cadences("C#7 Gb", "F#"))
    assert c["type"] == "none"
    assert c["caveats"] and "respell it to count as ^1" in c["caveats"][0]
    assert _only(find_cadences("C#7 F#", "F#"))["type"] == "authentic"
    # G# major after G in C is #V, not a deceptive bVI
    assert _only(find_cadences("C G G#", "C"))["type"] == "none"


def test_blues_tonic_caveat():
    c = _only(find_cadences("C7 F7 G7 C7", "C"))
    assert c["type"] == "authentic"
    assert any("blues tonic" in x for x in c["caveats"])


def test_note_arrays_give_soprano_and_bass():
    pac = [["C3", "E4", "G4", "C5"], ["G2", "D4", "G4", "B4"], ["C3", "E4", "G4", "C5"]]
    c = _only(find_cadences(pac, "C"))
    assert (c["type"], c["subtype"]) == ("authentic", "perfect")
    assert c["soprano_degrees"] == [7, 1] and c["bass_degrees"] == [5, 1]
    assert c["chords"] == [["G2", "D4", "G4", "B4"], ["C3", "E4", "G4", "C5"]]
    iac = [["C4", "E4", "G4", "C5"], ["B3", "D4", "G4", "G5"], ["C4", "E4", "G4", "E5"]]
    c = _only(find_cadences(iac, "C"))
    assert (c["subtype"], c["reason"]) == ("imperfect", "inversion; soprano on ^3")
    # without octaves the soprano stays unknown
    c = _only(find_cadences([["C", "E", "G"], ["G", "B", "D"], ["C", "E", "G"]], "C"))
    assert c["subtype"] is None and c["soprano_checked"] is False
    # an explicit soprano wins over the arrays' top notes
    c = _only(find_cadences(iac, "C", soprano="C5 D5 C5"))
    assert c["soprano_degrees"] == [2, 1]


def test_four_part_chords_with_an_omitted_fifth_are_read():
    """G B F (V7 without its fifth) and C C E C (a tonic without its fifth), as four-part writing allows."""
    arrays = [["C3", "E3", "C4", "G4"], ["F2", "F3", "C4", "A4"], ["G2", "E3", "C4", "G4"],
              ["G2", "F3", "B3", "G4"], ["C3", "C4", "E4", "C5"]]
    assert analyze_progression(arrays, "C")["chords"][3]["chord_type"] == "unknown"   # the table alone cannot
    c = _only(find_cadences(arrays, "C"))
    assert (c["type"], c["subtype"], c["romans"]) == ("authentic", "perfect", ["Cad64", "V7", "I"])
    assert c["cadential_64"] is True and c["chords"][1] == ["G2", "F3", "B3", "G4"]   # echoed as written
    c = _only(find_cadences([["C", "E", "G"], ["G", "B", "F"], ["C", "E"]], "C"))
    assert (c["type"], c["romans"]) == ("authentic", ["V7", "I"])
    # an augmented sixth is never 'completed' into another chord
    c = _only(find_cadences([["C4", "E4", "G4"], ["Ab3", "C4", "F#4"], ["G3", "B3", "D4"]], "C"))
    assert (c["type"], c["romans"]) == ("half", ["It6", "V"])


def test_phrase_ends_are_sorted_and_deduplicated():
    r = find_cadences("C G C F G C", "C", phrase_ends=[5, 2, 5])
    assert r["phrase_ends"] == [2, 5]
    assert r["summary"] == {"authentic": 2, "half": 0, "plagal": 0, "deceptive": 0, "none": 0}
    assert set(r["summary"]) == set(CADENCE_TYPES)


def test_phrase_length_ignores_a_trailing_partial_phrase():
    assert find_cadences("C F G C Am F G", "C", phrase_length=3)["phrase_ends"] == [2, 5]


def test_deterministic():
    args = ("C F C/G G7 C Am F G Am", "C")
    assert find_cadences(*args, phrase_ends="all") == find_cadences(*args, phrase_ends="all")


# ============================================================ all-key invariants

# (numerals in roman_to_chords' dialect, soprano degrees or None, kwargs), written once per mode
_MAJOR_CASES = [
    ("I IV Cad64 V7 I", [3, 4, 3, 2, 1], {}),
    ("I IV V7 I vi ii V vi", None, {"phrase_length": 4}),
    ("I IV V7 I6", None, {}),
    ("I IV V I", [1, 1, 2, 3], {}),
    ("I vii°6 I", None, {}),
    ("I V6 I", [1, 2, 3], {}),
    ("I IV I", None, {}),
    ("I V vi", None, {}),
    ("I V7 bVI", None, {}),
    ("I V IV6", None, {}),
    ("I vi", None, {}),
    ("I IV Cad64 V", None, {}),
    ("I IV V7", None, {}),
    ("I V I IV V vi", None, {"phrase_ends": "all"}),
    ("I V7 I IV I vi ii V", None, {"phrase_ends": [0, 2, 4, 7]}),
    ("I IV64 I", None, {}),
    ("ii65 V7 I", [4, 2, 1], {}),
    ("I V65/V V", None, {}),
    ("I Ger65 V", None, {}),
    ("I Vsus4 I", None, {}),
]
_MINOR_CASES = [
    ("i iv6 V", None, {}),
    ("i V VI", None, {}),
    ("i V iv6", None, {}),
    ("i V7 I", [3, 2, 1], {}),
    ("i iv I", None, {}),
    ("i iv Cad64 V7 i", [3, 4, 3, 2, 1], {}),
    ("i vii°7 i", None, {}),
    ("i v i", None, {}),
    ("i iv i", None, {}),
    ("i VII VI V", None, {}),
    ("i V i iv V VI", None, {"phrase_ends": "all"}),
    ("i iv V7 i VI iv V i", None, {"phrase_length": 4}),
    ("iiø65 V7 i", [4, 2, 1], {}),
]


def _run(numerals, tonic, scale, soprano_degrees, kwargs):
    symbols = roman_to_chords(numerals, tonic, scale)["symbols"]
    soprano = None
    if soprano_degrees is not None:
        soprano = notes_from_degrees(tonic + "5", scale, soprano_degrees)["notes"]
    return find_cadences(symbols, tonic, scale, soprano=soprano, **kwargs)


@pytest.mark.parametrize("scale", ("major",) + MINORS)
def test_labels_are_identical_in_every_key(scale):
    cases = _MAJOR_CASES if scale == "major" else _MINOR_CASES
    for numerals, sop, kwargs in cases:
        reference = _labels(_run(numerals, "C", scale, sop, kwargs))
        for tonic in TONICS:
            assert _labels(_run(numerals, tonic, scale, sop, kwargs)) == reference, (numerals, tonic, scale)


def test_major_and_minor_labels_in_c():
    """The invariant references are themselves right (spot-check a few in C and C minor)."""
    by = {n: _only(_run(n, "C", "major", s, k)) for n, s, k in _MAJOR_CASES if "phrase" not in str(k)}
    assert (by["I IV Cad64 V7 I"]["subtype"], by["I IV Cad64 V7 I"]["cadential_64"]) == ("perfect", True)
    assert by["I V7 bVI"]["subtype"] == "bVI" and by["I V IV6"]["subtype"] == "IV6"
    assert by["I V65/V V"]["type"] == "half" and by["I Ger65 V"]["type"] == "half"
    assert by["ii65 V7 I"]["subtype"] == "perfect"
    mi = {n: _only(_run(n, "C", "harmonic minor", s, k)) for n, s, k in _MINOR_CASES if "phrase" not in str(k)}
    assert mi["i iv6 V"]["subtype"] == "phrygian"
    assert mi["i V7 I"]["picardy"] is True and mi["i V7 I"]["subtype"] == "perfect"
    assert mi["i v i"]["type"] == "none" and mi["i VII VI V"]["type"] == "half"
    assert mi["iiø65 V7 i"]["subtype"] == "perfect"


@pytest.mark.parametrize("tonic,scale", KEYS)
def test_romans_round_trip_through_roman_to_chords(tonic, scale):
    cases = _MAJOR_CASES if scale == "major" else _MINOR_CASES
    for numerals, sop, kwargs in cases:
        symbols = roman_to_chords(numerals, tonic, scale)["symbols"]
        r = _run(numerals, tonic, scale, sop, kwargs)
        for c in r["cadences"]:
            back = roman_to_chords(c["romans"], tonic, scale)["symbols"]
            assert back == symbols[c["span"][0]:c["index"] + 1], (numerals, tonic, c["romans"])


@pytest.mark.parametrize("tonic,scale", KEYS)
def test_bach_chorale_voices_chain_in_every_key(tonic, scale):
    numerals = "I IV Cad64 V7 I" if scale == "major" else "i iv Cad64 V7 i"
    symbols = roman_to_chords(numerals, tonic, scale)["symbols"]
    bach = bach_chorale_voicing(symbols, root=tonic, scale_type=scale)
    by_soprano = find_cadences(symbols, tonic, scale, soprano=bach["voices"]["soprano"])
    # the same SATB chords as note arrays: soprano = highest, bass = lowest note
    arrays = [[row["soprano"], row["alto"], row["tenor"], row["bass"]] for row in bach["chords"]]
    by_arrays = find_cadences(arrays, tonic, scale)
    assert _labels(by_arrays) == _labels(by_soprano)
    c = _only(by_soprano)
    assert c["type"] == "authentic" and c["cadential_64"] is True and c["soprano_checked"] is True
    assert c["subtype"] in ("perfect", "imperfect")
    assert (c["subtype"] == "perfect") == (c["soprano_degrees"][-1] == 1)


# ================================================================== chaining

_LIBRARY_EXPECT = {
    "lament": ("half", "phrygian"),
    "andalusian": ("half", None),
    "plagal_amen": ("plagal", None),
    "double_plagal": ("plagal", None),
    "ii_V_I": ("authentic", None),
    "minor_ii_V_i": ("authentic", None),
    "prinner": ("authentic", "imperfect"),
    "royal_road": ("none", None),
    "twelve_bar_blues": ("half", None),
    "monte": ("half", None),
    "fonte": ("authentic", "imperfect"),
    "pop_axis": ("none", None),
}


def test_progression_library_chains_and_is_key_invariant():
    names = [e["name"] for e in progression_library()["progressions"]]
    for name in names:
        ref = None
        for tonic in TONICS:
            entry = progression_library(name, root=tonic)
            scale = "major" if entry["mode"] == "major" else "natural minor"
            r = find_cadences(entry["chords"], tonic, scale, soprano=entry.get("melody"))
            if ref is None:
                ref = _labels(r)
                c = _only(r)
                if name in _LIBRARY_EXPECT:
                    assert (c["type"], c["subtype"]) == _LIBRARY_EXPECT[name], name
            else:
                assert _labels(r) == ref, (name, tonic)


def test_prinner_melody_is_a_soprano():
    entry = progression_library("prinner", root="G")
    c = _only(find_cadences(entry["chords"], "G", soprano=entry["melody"]))
    assert c["reason"] == "leading-tone chord" and c["soprano_degrees"] == [4, 3]


def test_plan_sections_gives_phrase_ends():
    plan = plan_sections("verse chorus", bars=4)
    ends = [s["start_bar"] + s["bars"] - 1 for s in plan["sections"]]   # one chord per bar
    chords = ["C", "Am", "F", "G", "C", "F", "G7", "C"]
    r = find_cadences(chords, "C", phrase_ends=ends)
    assert [(c["index"], c["type"]) for c in r["cadences"]] == [(3, "half"), (7, "authentic")]
    assert find_cadences(chords, "C", phrase_length=4) == r


def test_voice_chords_arrays_with_octaves_carry_the_soprano():
    from midi_composer_mcp.voicing import voice_chords
    voiced = voice_chords(["Dm7", "G7", "Cmaj7"], style="drop2")["chords"]
    r = find_cadences(voiced, "C")
    c = _only(r)
    assert c["type"] == "authentic" and c["soprano_checked"] is True
    assert r["soprano"][-1] == max(voiced[-1], key=lambda n: parse_note(n).midi)
    # drop2 puts Cmaj7's fifth in the bass and its seventh on top: an IAC for two reasons
    assert (c["subtype"], c["reason"], c["romans"]) == ("imperfect", "inversion; soprano on ^7", ["V7", "IΔ43"])


def test_analyze_progression_output_feeds_back():
    """roman_figured from analyze_progression round-trips into find_cadences' romans."""
    chords = ["C", "A7", "Dm", "G7/B", "C"]
    romans = [c["roman_figured"] for c in analyze_progression(chords, "C")["chords"]]
    c = _only(find_cadences(chords, "C"))
    assert c["romans"] == romans[-2:]
    assert (c["type"], c["reason"]) == ("authentic", "inversion")


# ============================================================ bad-input fuzz

_BAD_VALUES = [None, 7, -1, 3.5, 2.0, True, "", "zz", [], [None], {}, {"a": 1}, ["C", 5], "C" * 5,
               ["C", "Db", "D"], [["C", "Db", "D"], "G"], "0 4 7", ["C", ".", "E"], "all", [0, 9], [-1],
               (1,), [1.0], [True], "C D E", ["C4", "D4"], [["C4", "E4"], ["G4"]], 100, "blues", "C4", "H",
               [["C4", "E4", "G4"], None], ["G7", {"symbol": "C"}]]


def test_only_value_errors_on_bad_input():
    base = {"chords": ["C", "G7", "C"], "root": "C"}
    failures = []
    for pname in ("chords", "root", "scale_type", "soprano", "phrase_ends", "phrase_length"):
        for bad in _BAD_VALUES:
            try:
                find_cadences(**dict(base, **{pname: bad}))
            except ValueError:
                pass
            except Exception as e:  # noqa: BLE001 - collect them all
                failures.append(f"{pname}={bad!r}: {type(e).__name__}: {e}")
    assert not failures, "\n".join(failures)


def test_bach_rows_are_rejected_with_a_value_error():
    rows = bach_chorale_voicing("C F G7 C", root="C")["chords"]
    with pytest.raises(ValueError):
        find_cadences(rows, "C")


def test_random_progressions_never_crash():
    """Every chord type on every root, in every key, as P and F: a well-formed report or a ValueError."""
    rng = random.Random(11)
    roots = ["C", "C#", "Db", "D", "Eb", "E", "F", "F#", "Gb", "G", "G#", "Ab", "A", "Bb", "B", "Cb", "E#"]
    symbols = [f"{r}{t.symbol}" for r in roots for t in CHORDS.values()]
    clusters = [["C4", "Db4", "D4"], ["Ab", "C", "F#"], ["Ab3", "C4", "D4", "F#4"], ["B3", "F4", "G#4", "D5"]]
    for _ in range(400):
        tonic, scale = rng.choice(KEYS + [("D", "dorian"), ("G", "mixolydian"), ("B", "locrian")])
        chords = [rng.choice(symbols + clusters) for _ in range(rng.randint(1, 5))]
        kwargs = rng.choice([{}, {"phrase_ends": "all"}, {"phrase_length": 1}])
        r = find_cadences(chords, tonic, scale, **kwargs)
        for c in r["cadences"]:
            assert c["type"] in CADENCE_TYPES
            assert c["span"][1] == c["index"] and len(c["romans"]) == len(c["chords"]) == len(c["bass_degrees"])
        assert sum(r["summary"].values()) == len(r["cadences"])
    for p, f in itertools.product(["G7", "Bdim", "F", "Fm/Ab"], ["C", "Am", "Ab", "F/A", "G"]):
        assert find_cadences([p, f], "C")["cadences"][0]["type"] in CADENCE_TYPES
