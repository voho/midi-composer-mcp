"""analyze_progression's figures / applied / special / borrowed_from fields, and next_chords.

Unit tests pin the spec examples; the sweeps check the properties in all keys
(14 tonic spellings x the relevant modes): roman_figured round-trips through
roman_to_chords, every field is transposition invariant, the pre-extension
fields are unchanged, next_chords' tokens realize their candidates, and bad
input only ever raises ValueError.
"""

from __future__ import annotations

import pytest

from midi_composer_mcp import harmony
from midi_composer_mcp.analysis import find_cadences
from midi_composer_mcp.chords import CHORDS, chord_notes
from midi_composer_mcp.diatonic import (
    _HARMONIC_FUNCTIONS,
    _ROMAN_BASE,
    borrowing_sources,
    diatonic_chords,
    roman_suffix,
)
from midi_composer_mcp.diatonic import numeral_base as _numeral_base
from midi_composer_mcp.harmony import (
    PISTON_1941,
    PISTON_TABLE,
    _root_and_quality,
    _voicing_cost,
    analyze_progression,
    next_chords,
    piston_sentence,
    voice_leading,
)
from midi_composer_mcp.masters import bach_chorale_voicing
from midi_composer_mcp.notes import LETTERS, parse_note, parse_notes, transpose
from midi_composer_mcp.roman import read_chord, roman_to_chords
from midi_composer_mcp.scales import MAJOR_DEGREES, SCALES, resolve_scale_type, scale_notes
from midi_composer_mcp.voicing import voice_chords

TONICS = ["C", "G", "D", "A", "E", "B", "F#", "C#", "F", "Bb", "Eb", "Ab", "Db", "Gb"]
MINOR_TONICS = ["A", "E", "B", "F#", "C#", "G#", "D#", "D", "G", "C", "F", "Bb", "Eb", "Ab"]


def pc(note) -> int:
    return parse_note(note).pitch_class if isinstance(note, str) else note.pitch_class


def rel_pc(note, tonic: str) -> int:
    return (pc(note) - pc(tonic)) % 12


def letter_offset(note: str, tonic: str) -> int:
    return (LETTERS.index(parse_note(note).letter) - LETTERS.index(parse_note(tonic).letter)) % 7


def chords_of(chords, root="C", scale="major"):
    return analyze_progression(chords, root, scale)["chords"]


def one(chord, root="C", scale="major"):
    return chords_of([chord], root, scale)[0]


# ------------------------------------------------------------ analyze: the spec example

def test_spec_example_line():
    res = chords_of(["C", "A7", "Dm", "G7/B", "C", "Fm", "Db/F", "C/G", "G7", "C"])
    a7, g7b, fm, db, cg = res[1], res[3], res[5], res[6], res[7]
    assert (a7["roman"], a7["applied"], a7["roman_figured"], a7["in_key"]) == ("VI7", "V7/ii", "V7/ii", False)
    assert (g7b["figure"], g7b["roman_figured"], g7b["function"]) == ("65", "V65", "dominant")
    assert fm["borrowed_from"] == ["C harmonic major", "C harmonic minor", "C natural minor", "C phrygian",
                                   "C locrian"]
    assert (db["special"], db["roman_figured"]) == ("Neapolitan", "N6")
    assert (cg["special"], cg["function"], cg["function_note"]) == ("Cad64", "tonic", "dominant (cadential 6/4)")
    assert cg["roman_figured"] == "Cad64" and cg["figure"] == "64"
    assert [c["roman_figured"] for c in res] == ["I", "V7/ii", "ii", "V65", "I", "iv", "N6", "Cad64", "V7", "I"]


def test_german_sixth_array_example():
    ger = one(["Ab", "C", "Eb", "F#"])
    assert ger["special"] == "Ger65" and ger["roman_figured"] == "Ger65"
    assert ger["non_scale_notes"] == ["Ab", "Eb", "F#"]      # the written F#, not Gb
    assert ger["chord_type"] is None and ger["figure"] == "65"
    assert "borrowed_from" not in ger and "applied" not in ger


# ------------------------------------------------------------ analyze: unit rules

def test_applied_chords():
    assert chords_of(["A7", "Dm"])[0]["applied"] == "V7/ii"
    assert one("D7")["applied"] == "V7/V"
    assert one("C7")["applied"] == "V7/IV"
    assert "applied" not in one("C")                           # a diatonic I is never 'V/IV'
    f = one("F#dim7")
    assert f["applied"] == "vii°7/V" and f["roman"] == "#iv°7" and f["roman_figured"] == "vii°7/V"
    assert one("F#m7b5")["applied"] == "viiø7/V"
    assert one("E/G#")["roman_figured"] == "V6/vi"
    assert one("E7/D")["roman_figured"] == "V42/vi"
    assert one("C#dim7/G")["roman_figured"] == "vii°43/ii"
    assert one("E9")["roman_figured"] == "V9/vi"               # an embellished chord: root-position numeral
    assert one("E9")["figure"] == ""


def test_harmonic_minor_dominant_is_not_applied():
    e7 = one("E7", "A", "natural minor")
    assert e7["function_note"] == "harmonic-minor dominant" and "applied" not in e7
    assert "borrowed_from" not in e7 and e7["roman_figured"] == "V7"
    g = one("G#dim7", "A", "natural minor")
    assert g["function_note"] == "harmonic-minor dominant" and "applied" not in g
    assert g["roman_figured"] == "vii°7"
    assert one("D7", "A", "natural minor")["applied"] == "V7/bVII"   # applied_reading's target numeral


@pytest.mark.parametrize("chord,figure,roman_figured", [
    ("C", "", "I"), ("C/E", "6", "I6"), ("C/G", "64", "I64"), ("G7", "7", "V7"), ("G7/B", "65", "V65"),
    ("G7/D", "43", "V43"), ("G7/F", "42", "V42"), ("Bm7b5/F", "43", "viiø43"), ("Cmaj7/B", "42", "IΔ42"),
    ("Bdim/D", "6", "vii°6"), ("Dm7/F", "65", "ii65"), ("G9", "", "V9"), ("Csus4", "", "Isus4"), ("Cmaj9", "", "Imaj9"),
])
def test_figures(chord, figure, roman_figured):
    r = one(chord)
    assert r["figure"] == figure and r["roman_figured"] == roman_figured
    assert "bass_degree" not in r


def test_null_figures_give_bass_degree():
    sus = one("Csus4/F")
    assert sus["figure"] is None and sus["bass_degree"] == 4 and sus["roman_figured"] == "Isus4"
    assert one("C/D")["bass_degree"] == 2
    assert one("C6/E")["figure"] is None and one("C6/E")["bass_degree"] == 3
    assert one("C/F#")["bass_degree"] == "#4"                  # a chromatic bass keeps its accidental
    assert one("C/Bb")["bass_degree"] == "b7"
    cluster = one(["C", "Db", "D"])
    assert cluster["figure"] is None and cluster["bass_degree"] == 1 and cluster["chord_type"] == "unknown"


def test_note_array_bass():
    low = one(["E3", "G3", "C4"])                              # lowest MIDI note
    assert low["figure"] == "6" and low["roman_figured"] == "I6"
    mixed = one(["C4", "E3", "G3"])                            # E3 is the lowest although written second
    assert mixed["figure"] == "6" and mixed["root"] == "C"
    assert one(["E", "G", "C"])["figure"] == "6"               # no octaves: the first note is the bass
    assert one(["G2", "B3", "D4", "F4"])["roman_figured"] == "V7"
    assert one(["B2", "G3", "D4", "F4"])["roman_figured"] == "V65"


def test_specials():
    assert one("Db/F")["roman_figured"] == "N6"
    assert one("Db")["roman_figured"] == "bII" and one("Db")["special"] == "Neapolitan"
    assert one("Db/Ab")["roman_figured"] == "bII64"
    for notes, name in ((["Ab", "C", "F#"], "It6"), (["Ab", "C", "Eb", "F#"], "Ger65"),
                        (["Ab", "C", "D", "F#"], "Fr43"), (["Ab", "C", "D#", "F#"], "Sw43"),
                        (["Ab2", "C4", "Eb4", "F#4"], "Ger65")):
        r = one(notes)
        assert r["special"] == name and r["roman_figured"] == name and r["chord_type"] is None, notes
        assert "applied" not in r                               # Fr43 is not read as V7b5/V
    ab7 = one("Ab7")
    assert ab7["roman"] == "bVI7" and ab7["enharmonic_to"] == "Ger65" and "special" not in ab7
    assert ab7["roman_figured"] == "bVI7" and ab7["chord_type"] == "dominant 7"
    assert one(["Ab", "C", "Eb", "Gb"])["enharmonic_to"] == "Ger65"   # misspelled: not a German sixth
    assert "special" not in one(["C", "Ab", "Eb", "F#"])               # b6 not in the bass
    assert one(["F", "A", "C", "D#"], "A", "natural minor")["special"] == "Ger65"


def test_cadential_six_four():
    res = chords_of(["C/G", "G7", "C"])
    assert res[0]["special"] == "Cad64" and res[0]["function"] == "tonic"
    assert res[0]["function_note"] == "dominant (cadential 6/4)" and res[0]["roman_figured"] == "Cad64"
    assert "special" not in chords_of(["C/G", "Dm"])[0]
    assert "special" not in chords_of(["C/G"])[0]
    minor = chords_of(["Am/E", "E7", "Am"], "A", "natural minor")[0]
    assert minor["roman_figured"] == "Cad64" and minor["function"] == "tonic"
    borrowed = chords_of(["Cm/G", "G7"])[0]                    # a borrowed tonic 6/4: 'Cad64' would read C/G
    assert borrowed["special"] == "Cad64" and borrowed["roman_figured"] == "i64"


def test_borrowed_from():
    assert one("Fm")["borrowed_from"] == ["C harmonic major", "C harmonic minor", "C natural minor",
                                          "C phrygian", "C locrian"]
    assert one("Bb")["borrowed_from"] == ["C mixolydian", "C dorian", "C natural minor"]
    assert "borrowed_from" not in one("C")                      # in key
    assert "borrowed_from" not in one("A7")                     # applied
    assert "borrowed_from" not in one("Db/F")                   # special
    assert one("Bm", "A", "natural minor")["borrowed_from"] == ["A dorian", "A mixolydian", "A melodic minor",
                                                                "A major"]
    assert one("D", "A", "natural minor")["applied"] == "V/bVII"      # applied, so not 'borrowed'
    assert one("F#m", "C")["borrowed_from"] == []               # outside every parallel mode


def test_borrowed_from_judges_letters_not_semitones():
    """Mixture is built from the parallel mode's spelled degrees: G# is not C minor's bVI (Ab)."""
    assert one("Ab")["borrowed_from"] == ["C harmonic minor", "C natural minor", "C phrygian", "C locrian"]
    g_sharp = chords_of(["C", "G#", "C"])[1]
    assert g_sharp["roman"] == "#V" and g_sharp["borrowed_from"] == []
    assert chords_of(["C", "F#", "C"])[1]["borrowed_from"] == []    # C locrian has Gb, not F#
    assert one("Gb")["borrowed_from"] == ["C locrian"]
    for symbol in ("G#m", "Abm", "G+"):                              # #v, bvi (needs Cb), V+ (needs D#)
        assert one(symbol)["borrowed_from"] == [], symbol
    assert "borrowed_from" not in one("E")                          # V/vi: applied, not mixture
    assert one(["Ab3", "C4", "Eb4"])["borrowed_from"] == one("Ab")["borrowed_from"]
    assert one(["G#3", "C4", "D#4"])["borrowed_from"] == []         # the same pitches, misspelled


_MIXTURE_LINE = "iv bVI bIII bVII ii° bII II v i° biii bV V+ III bvi #v #IV #V iiø7 bVII7 IV7"


@pytest.mark.parametrize("tonic", TONICS)
def test_borrowed_from_sources_spell_the_chord_in_every_key(tonic):
    spelled = {s["label"]: {n.pitch_class_name for n in scale_notes(s["scale"], s["tonic"])}
               for s in borrowing_sources(tonic, "major")}
    ref = chords_of(roman_to_chords(_MIXTURE_LINE, "C")["symbols"], "C")
    got = chords_of(roman_to_chords(_MIXTURE_LINE, tonic)["symbols"], tonic)
    for token, a, b in zip(_MIXTURE_LINE.split(), ref, got, strict=True):
        assert [s.split(" ", 1)[1] for s in b.get("borrowed_from", [])] == \
               [s.split(" ", 1)[1] for s in a.get("borrowed_from", [])], (tonic, token)
        notes = {n.pitch_class_name for n in parse_notes(roman_to_chords(token, tonic)["chords"][0]["notes"])}
        for label in b.get("borrowed_from", []):
            assert notes <= spelled[label], (tonic, token, label)


def test_minor_numerals_follow_roman_to_chords():
    res = chords_of(["Am", "C", "F", "G", "G#dim", "F#m", "Fm", "Gm", "Bdim"], "A", "natural minor")
    assert [c["roman"] for c in res] == ["i", "bIII", "bVI", "bVII", "vii°", "vi", "bvi", "bvii", "ii°"]
    assert [c["roman_figured"] for c in res] == ["i", "III", "VI", "VII", "vii°", "vi", "bvi", "bvii", "ii°"]
    for c, symbol in zip(res, ["Am", "C", "F", "G", "G#dim", "F#m", "Fm", "Gm", "Bdim"]):
        assert roman_to_chords([c["roman_figured"]], "A", "natural minor")["symbols"][0] == symbol


def test_unwritable_roots_have_no_roman_figured():
    # a root on the parallel major's degree that the key lacks has no numeral in roman_to_chords' dialect
    assert one("C#m", "A", "natural minor")["roman_figured"] is None
    assert one("Em", "C", "dorian")["roman_figured"] is None
    assert one("C#m", "A", "natural minor")["roman"] == "iii"    # the existing field is untouched


def test_clusters_have_no_roman_figured():
    # '?' is no numeral roman_to_chords reads: an unnamed cluster gets null (its roman keeps the marker)
    for notes, roman in ((["C", "Db", "D"], "I?"), (["G", "B", "Db"], "V?")):
        r = one(notes)
        assert r["roman"] == roman and r["roman_figured"] is None and r["chord_type"] == "unknown", notes
    it6 = one(["Ab", "C", "F#"])
    assert it6["special"] == "It6" and it6["roman_figured"] == "It6"   # a special still names itself
    ab7 = one(["Ab", "C", "Gb"])            # spelled as Ab7 less its fifth: a writable numeral, the It6 sound
    assert (ab7["roman_figured"], ab7["enharmonic_to"], ab7["omitted_fifth"]) == ("bVI7", "It6", "Eb")


def test_fifthless_four_part_chords_are_read():
    # bach_chorale_voicing drops a V7's fifth; analyze_progression now reads it as find_cadences does
    bach = bach_chorale_voicing("C F C/G G7 C", root="C")
    rows = [[row[v] for v in ("soprano", "alto", "tenor", "bass")] for row in bach["chords"]]
    res = chords_of(rows)
    assert [c["roman_figured"] for c in res] == ["I", "IV", "Cad64", "V7", "I"]
    assert res[2]["special"] == "Cad64" and res[3]["omitted_fifth"] == "D" and res[3]["in_key"] is True
    assert find_cadences(rows, "C")["cadences"][0]["romans"] == ["Cad64", "V7", "I"]
    shell = voice_chords(["Dm7", "G7", "Cmaj7"], "shell")["chords"]
    assert [c["roman_figured"] for c in chords_of(shell)] == ["ii7", "V7", "IΔ7"]


@pytest.mark.parametrize("tonic", TONICS)
def test_voiced_drafts_round_trip_in_every_key(tonic):
    """bach_chorale_voicing and shell voicings: no '?' numeral, and roman_figured reads the chords back."""
    for line, scale in (("I IV Cad64 V7 I", "major"), ("I vi ii65 V7 I", "major"), ("i iv V7 i", "natural minor")):
        bach = bach_chorale_voicing(roman_to_chords(line, tonic, scale)["symbols"], root=tonic, scale_type=scale)
        rows = [[row[v] for v in ("soprano", "alto", "tenor", "bass")] for row in bach["chords"]]
        figured = [c["roman_figured"] for c in chords_of(rows, tonic, scale)]
        assert all(f is not None and "?" not in f for f in figured), (tonic, line, figured)
        for row, back in zip(rows, roman_to_chords(figured, tonic, scale)["chords"]):
            assert {pc(n) for n in row} <= {pc(n) for n in back["notes"]}, (tonic, line)
            assert pc(back["bass"]) == pc(min(parse_notes(row), key=lambda n: n.midi)), (tonic, line)
    shell = voice_chords(roman_to_chords("ii7 V7 IΔ7", tonic)["symbols"], "shell")["chords"]
    assert [c["roman_figured"] for c in chords_of(shell, tonic)] == ["ii7", "V7", "IΔ7"]


def test_unwritable_slash_bass_is_kept_in_bass_degree():
    # roman_figured has no way to write a bass the figure grammar cannot (sus/add/9 inversions, pedals):
    # it is the root-position numeral, and bass_degree carries the bass to re-add in the new key
    draft = ["C", "Cadd9/E", "F", "Gsus4/C", "G9/B", "G/C", "Dm/G", "C"]
    res = chords_of(draft)
    assert [c["roman_figured"] for c in res] == ["I", "Iadd9", "IV", "Vsus4", "V9", "V", "ii", "I"]
    assert [c.get("bass_degree") for c in res] == [None, 3, None, 1, 7, 1, 5, None]
    for tonic in TONICS:                     # the documented chain: roman_figured + bass_degree in another key
        notes = scale_notes(resolve_scale_type("major"), parse_note(tonic))
        moved = roman_to_chords([c["roman_figured"] for c in res], tonic)["symbols"]
        for c, symbol, original in zip(res, moved, draft):
            if c["figure"] is None:
                rebuilt = f"{symbol}/{notes[c['bass_degree'] - 1].pitch_class_name}"
                ref = chords_of([rebuilt], tonic)[0]
                assert ref["roman"] == one(original)["roman"] and ref.get("bass_degree") == c["bass_degree"]


@pytest.mark.parametrize("scale", [s.name for s in SCALES.values() if len(s.intervals) == 7])
def test_applied_roman_figured_round_trips_in_every_mode(scale):
    """V- and vii°-family chords read as applied realize themselves again (enigmatic, hungarian major ...)."""
    symbols = [f"{letter}{acc}{q}" for letter in "CDEFGAB" for acc in ("", "#", "b")
               for q in ("", "7", "dim7", "m7b5")]
    for tonic in TONICS:
        for s, entry in zip(symbols, chords_of(symbols, tonic, scale)):
            if not entry.get("applied") or entry["roman_figured"] is None:
                continue
            back = roman_to_chords([entry["roman_figured"]], tonic, scale)["chords"][0]
            assert pc(back["root"]) == pc(entry["root"]), (scale, tonic, s, entry["roman_figured"])
            assert sorted(map(pc, back["notes"])) == sorted(n.pitch_class for n in read_chord(s)["notes"])
    assert "applied" not in one("C#", "C", "enigmatic")             # not 'V/bII' (V of Db is Ab)
    assert "applied" not in one("A#", "C", "hungarian major")       # not 'V/v' (V of G minor is D)


def test_non_heptatonic_keys_keep_figures_only():
    r = one("G7/B", "C", "major pentatonic")
    assert r["figure"] == "65" and r["roman_figured"] == "V65"
    assert "applied" not in one("D7", "C", "major pentatonic") and "special" not in one("Db/F", "C", "blues")


# ------------------------------------------------------------ analyze: invariants

_ROUND_TRIP_MAJOR = "I V6 vi ii65 V7/V V65/vi vii°7/V Cad64 V7 I N6 Ger65"
_ROUND_TRIP_MINOR = "i iv6 V7 VI It6"


@pytest.mark.parametrize("tonic", TONICS)
def test_roman_figured_round_trips_the_spec_lines(tonic):
    for line, scale in ((_ROUND_TRIP_MAJOR, "major"), (_ROUND_TRIP_MINOR, "natural minor")):
        realized = roman_to_chords(line, tonic, scale)
        got = chords_of(realized["symbols"], tonic, scale)
        assert [c["roman_figured"] for c in got] == [c["roman"] for c in realized["chords"]], (tonic, scale)


@pytest.mark.parametrize("tonic", TONICS)
def test_chromatic_numerals_keep_their_roman(tonic):
    tokens = "I bVII iv bVI vii°7 #iv°7 II7".split()
    got = chords_of(roman_to_chords(tokens, tonic)["symbols"], tonic)
    assert [c["roman"] for c in got] == tokens
    assert got[5]["roman_figured"] == "vii°7/V" and got[6]["roman_figured"] == "V7/V"
    assert [c["roman_figured"] for c in got[:5]] == tokens[:5]


_SWEEP = ("I ii iii IV V vi vii° I6 I64 ii65 V43 V42 vii°7 viiø7 viiø43 IΔ7 IΔ42 iΔ7 III+ bVII iv bVI bIII "
          "V7/V vii°7/V V65/vi viiø7/ii vii°/ii V/V V6/V V7/IV V9/iii N6 N It6 Fr43 Ger65 Sw43 Cad64 I6/9 V9 "
          "Vsus4 ii9 iim6 Iadd6 IΔ9 V7b9 V7#9 #iv°7 II7 bVII7 iv7 V+7 viih43 IM65 V2 VII° bII bII64 i iv6 VI "
          "VII v bvi bvii vi° vii°64").split()
# chords whose applied target is not a diatonic triad of that mode, or whose root that mode cannot write
_SWEEP_SKIP = {("natural minor", "V65/vi"), ("harmonic minor", "V65/vi"), ("melodic minor", "V65/vi"),
               ("dorian", "V65/vi"), ("mixolydian", "V9/iii"), ("phrygian", "V7/V"), ("phrygian", "V/V"),
               ("phrygian", "V6/V")}


@pytest.mark.parametrize("scale", ["major", "natural minor", "harmonic minor", "melodic minor", "dorian",
                                   "phrygian", "lydian", "mixolydian", "locrian", "harmonic major"])
def test_roman_figured_reads_back_the_same_chord(scale):
    """Every chord roman_to_chords builds is read back to a numeral that builds it again."""
    for tonic in ("C", "F#", "Bb", "Eb"):
        realized = roman_to_chords(_SWEEP, tonic, scale)
        for token, want, got in zip(_SWEEP, realized["chords"], chords_of(realized["symbols"], tonic, scale)):
            if (scale, token) in _SWEEP_SKIP:
                assert got["roman_figured"] is None or got.get("applied") is None, (scale, token)
                continue
            assert got["roman_figured"] is not None, (scale, tonic, token)
            back = roman_to_chords([got["roman_figured"]], tonic, scale)["chords"][0]
            assert sorted(map(pc, back["notes"])) == sorted(map(pc, want["notes"])), (scale, tonic, token)
            assert pc(back["bass"]) == pc(want["bass"]), (scale, tonic, token)


_FIELDS = ("figure", "roman_figured", "roman", "applied", "special", "function", "function_note", "in_key",
           "degree", "chord_type", "bass_degree", "enharmonic_to", "note")
_MAJOR_LINE = ("I V6 vi ii65 V7/V V65/vi vii°7/V Cad64 V7 I N6 Ger65 iv bVII bVI It6 Fr43 Sw43 Isus4 V9 viiø43 "
               "IΔ42 bII64 II7 #iv°7 V7/IV viiø7/ii")
_MINOR_LINE = "i iv6 V7 VI It6 iiø65 VII bII N6 V7/iv vii°7/v Cad64 V Ger65 III+ IV vii°7 bvi V7/VI"


@pytest.mark.parametrize("tonic_index", range(len(TONICS)))
def test_every_field_is_transposition_invariant(tonic_index):
    for line, scale, tonics in ((_MAJOR_LINE, "major", TONICS), (_MINOR_LINE, "natural minor", MINOR_TONICS)):
        home, tonic = tonics[0], tonics[tonic_index]
        ref = chords_of(roman_to_chords(line, home, scale)["symbols"], home, scale)
        got = chords_of(roman_to_chords(line, tonic, scale)["symbols"], tonic, scale)
        for token, a, b in zip(line.split(), ref, got):
            for field in _FIELDS:
                assert a.get(field) == b.get(field), (tonic, scale, token, field)
            assert rel_pc(a["root"], home) == rel_pc(b["root"], tonic)
            assert letter_offset(a["root"], home) == letter_offset(b["root"], tonic)
            assert [rel_pc(n, home) for n in a.get("non_scale_notes", [])] == \
                   [rel_pc(n, tonic) for n in b.get("non_scale_notes", [])]
            assert [s.split(" ", 1)[1] for s in a.get("borrowed_from", [])] == \
                   [s.split(" ", 1)[1] for s in b.get("borrowed_from", [])]
            assert a.keys() == b.keys()


# The analyze_progression body before the extension, kept verbatim to prove the old fields did not move.
def _old_analyze(chords, root, scale_type="major"):
    scale = resolve_scale_type(scale_type)
    tonic = parse_notes(root)[0].without_octave()
    intervals = scale.intervals
    scale_pcs = {(tonic.pitch_class + i) % 12 for i in intervals}
    out = []
    for item in chords:
        croot, ctype, bass = _root_and_quality(item)
        croot = croot.without_octave()
        rel = (croot.pitch_class - tonic.pitch_class) % 12
        if rel in intervals and len(intervals) == 7:
            idx = intervals.index(rel)
            offset = intervals[idx] - MAJOR_DEGREES[idx]
            base, accidental = _ROMAN_BASE[idx], "#" * offset if offset > 0 else "b" * -offset
        else:
            base, accidental = _numeral_base(croot, tonic)
        if ctype is None:
            tones = [n.without_octave() for n in parse_notes(item)]
            rel_pcs = {(t.pitch_class - croot.pitch_class) % 12 for t in tones}
        else:
            tones = chord_notes(ctype, croot)
            rel_pcs = ctype.pitch_classes
        minor_third = 3 in rel_pcs and 4 not in rel_pcs
        numeral = base.lower() if minor_third else base
        suffix = roman_suffix(ctype, minor_third)
        if bass is not None:
            tones = [bass.without_octave(), *tones]
        outside = []
        for t in tones:
            if t.pitch_class not in scale_pcs and t.pitch_class_name not in outside:
                outside.append(t.pitch_class_name)
        in_key = not outside
        entry = {
            "symbol": item if isinstance(item, str) else (
                f"{croot.pitch_class_name}{ctype.symbol}" if ctype else " ".join(t.name for t in tones)),
            "root": croot.pitch_class_name,
            "chord_type": ctype.name if ctype else "unknown",
            "roman": f"{accidental}{numeral}{suffix}",
            "in_key": in_key,
        }
        if rel in intervals:
            idx = intervals.index(rel)
            entry["degree"] = idx + 1
            if in_key and len(intervals) == 7:
                entry["function"] = _HARMONIC_FUNCTIONS[idx]
        if not in_key:
            entry["non_scale_notes"] = outside
            entry["note"] = "chromatic / borrowed"
        out.append(entry)
    return out


_OLD_FIELDS = ("symbol", "root", "chord_type", "roman", "in_key", "degree", "function", "non_scale_notes", "note")


@pytest.mark.parametrize("key", [("C", "major"), ("A", "natural minor"), ("Eb", "dorian"), ("F#", "harmonic minor"),
                                 ("G", "major pentatonic")])
def test_existing_fields_are_unchanged(key):
    """Symbols (all types, all roots, root position and first inversion) and note arrays (two roots)."""
    tonic, scale = key
    symbols, arrays = [], []
    for ctype in CHORDS.values():
        for root in ("C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"):
            tones = [n.name for n in chord_notes(ctype, parse_note(root))]
            symbols += [f"{root}{ctype.symbol}", f"{root}{ctype.symbol}/{tones[1]}"]
            if root in ("C", "F#"):
                arrays += [tones, tones[1:] + tones[:1]]
    for items in (symbols, arrays):
        new = chords_of(items, tonic, scale)
        for item, old, entry in zip(items, _old_analyze(items, tonic, scale), new):
            if entry.get("special") in ("It6", "Fr43", "Ger65", "Sw43"):
                assert entry["chord_type"] is None      # the one documented change: no table type
                old = dict(old, chord_type=None)
            for field in _OLD_FIELDS:
                assert entry.get(field) == old.get(field), (key, item, field)


def test_random_arrays_and_symbols_never_crash():
    import random
    rng = random.Random(7)
    names = ["C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B", "Cb",
             "E#", "Fb", "B#"]
    for _ in range(400):
        k = rng.randint(1, 5)
        notes = [rng.choice(names) + (str(rng.randint(2, 5)) if rng.random() < 0.5 else "") for _ in range(k)]
        if any(n[-1].isdigit() for n in notes) and not all(n[-1].isdigit() for n in notes):
            notes = [n.rstrip("0123456789") for n in notes]
        key = (rng.choice(TONICS), rng.choice(["major", "natural minor", "dorian", "harmonic minor"]))
        for entry in chords_of([notes, rng.choice(["G7", "C/E", "Fm", "D7/F#"])], *key):
            if entry["roman_figured"] is not None:
                assert isinstance(entry["roman_figured"], str)


# ------------------------------------------------------------ next_chords: Piston's table

def test_piston_table_is_pinned_verbatim():
    assert [piston_sentence(row) for row in PISTON_1941] == [
        "I is followed by IV or V, sometimes VI, less often II or III.",
        "II is followed by V, sometimes IV or VI, less often I or III.",
        "III is followed by VI, sometimes IV, less often I, II or V.",
        "IV is followed by V, sometimes I or II, less often III or VI.",
        "V is followed by I, sometimes IV or VI, less often II or III.",
        "VI is followed by II or V, sometimes III or IV, less often I.",
        "VII is followed by III, sometimes I.",
    ]
    assert len(PISTON_1941) == 7 and all(len(row) == 4 for row in PISTON_1941)
    assert [row[0] for row in PISTON_1941] == list(_ROMAN_BASE)
    assert PISTON_TABLE == "Piston 1941, ch. 3"


def test_c_am_example_order():
    r = next_chords(["C", "Am"], "C")
    assert r["symbols"] == ["Dm", "G", "F", "Em", "C", "A7", "D7", "B7", "C7", "Ddim", "Fm", "Eb", "Ab", "Bb"]
    assert r["tokens"] == ["ii", "V", "IV", "iii", "I", "V7/ii", "V7/V", "V7/iii", "V7/IV", "ii°", "iv", "bIII",
                           "bVI", "bVII"]
    assert [c["tier"] for c in r["candidates"]] == [1, 1, 2, 2, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6]
    assert [c["tier_name"] for c in r["candidates"]][::4] == ["usual", "less often", "applied", "unlisted"]
    assert r["candidates"][0]["rule"] == "Piston 1941: VI is usually followed by II or V"
    assert r["candidates"][2]["common_tones"] == 2
    assert r["last"] == {"symbol": "Am", "roman": "vi", "degree": 6}
    assert r["table"] == "Piston 1941, ch. 3" and r["key"] == "C major"
    dm = r["candidates"][0]
    assert dm["root_motion"] == "ascending (strong)" and dm["in_key"] is True and dm["notes"] == ["D", "F", "A"]
    assert r["candidates"][5]["roman"] == "VI7" and r["candidates"][5]["in_key"] is False
    assert r["candidates"][-1]["rule"].endswith("Piston 1941 lists no VII after VI")
    assert "Berklee" in r["candidates"][-1]["rule"] and "Aldwell & Schachter" in r["candidates"][9]["rule"]


def test_tier_zero_and_rows():
    a7 = next_chords(["A7"], "C")
    assert a7["candidates"][0]["symbol"] == "Dm" and a7["candidates"][0]["tier"] == 0
    assert a7["candidates"][0]["rule"] == "applied chord resolves to its target"
    assert a7["last"]["note"] == "row chosen by root letter" and a7["last"]["degree"] == 6
    assert sum(c["symbol"] == "Dm" for c in a7["candidates"]) == 1       # the duplicate keeps tier 0
    assert next_chords(["G"], "C")["symbols"][0] == "C"
    assert next_chords(["F#dim7"], "C")["candidates"][0]["symbol"] == "G"
    assert next_chords(["Bb"], "C")["last"]["degree"] == 7                # bVII uses row VII
    assert next_chords(["C", "Bdim"], "C")["symbols"][:2] == ["Em", "C"]   # VII: III, sometimes I


def _chord_key(symbol, tonic="C", scale="major"):
    r = read_chord(symbol)
    root = parse_note(one(symbol, tonic, scale)["root"])
    pcs = {n.pitch_class for n in r["notes"]} | {r["bass"].pitch_class}
    if "omitted_fifth" in r:
        pcs.add(r["omitted_fifth"].pitch_class)
    return root.pitch_class, frozenset(pcs)


def test_last_chord_is_never_its_own_candidate():
    # a repetition is no root progression: the applied and mixture tiers used to offer the last chord again
    for last in ("A7", "D7", "E7", "B7", "C7", "Fm", "Ab", "Bb", "Eb", "Ddim", "A7/C#"):
        r = next_chords(["C", last], "C", limit=64)
        assert _chord_key(last) not in {_chord_key(s) for s in r["symbols"]}, last
        assert next_chords(["C", last], "C", sort="movement")["candidates"][0]["movement"] > 0, last
    sev = next_chords(["C", "Fm7"], "C", sevenths=True, sort="movement")
    assert "Fm7" not in sev["symbols"] and sev["candidates"][0]["movement"] > 0
    assert next_chords(["C", "D7"], "C", sort="movement", limit=3)["symbols"] == ["F", "C7", "E7"]
    fifthless = next_chords(["C", ["D3", "F#3", "C4"]], "C", limit=64)
    assert fifthless["last"]["roman"] == "II7" and "D7" not in fifthless["symbols"]   # its fifth restored


@pytest.mark.parametrize("tonic", TONICS)
def test_no_self_candidate_in_every_key(tonic):
    lasts = roman_to_chords("V7/ii V7/V V7/vi V7/iii V7/IV iv bVI bVII bIII ii° vii°7/V", tonic)["symbols"]
    for last in lasts:
        for sevenths in (False, True):
            r = next_chords([last], tonic, sevenths=sevenths, limit=64)
            assert _chord_key(last, tonic) not in {_chord_key(s, tonic) for s in r["symbols"]}, (tonic, last)


def test_augmented_sixths_resolve_to_v_in_tier_zero():
    # tier 0 follows analyze_progression: Fr43 is a special, not an applied V7b5/V
    fr = next_chords([["Ab", "C", "D", "F#"]], "C")
    assert [(c["symbol"], c["tier"]) for c in fr["candidates"][:2]] == [("G", 0), ("C/G", 0)]
    assert fr["candidates"][0]["rule"].startswith("augmented sixth resolves to V")
    assert all(c["rule"] != "applied chord resolves to its target" for c in fr["candidates"])
    minor = next_chords([["F", "A", "B", "D#"]], "A", "natural minor")   # the major V, not Em, comes first
    assert minor["symbols"][:2] == ["E", "Am/E"] and minor["tokens"][:2] == ["V", "Cad64"]
    # the German sixth usually goes through the cadential 6/4 (parallel fifths), and Sw43 is spelled for it
    ger = next_chords(roman_to_chords("I IV Ger65", "C")["symbols"], "C")
    assert ger["tokens"][:2] == ["Cad64", "V"] and ger["candidates"][0]["tier"] == 0
    assert "I" not in ger["tokens"]                                  # Cad64 is the tonic triad's best entry
    it6 = next_chords([["Ab", "C", "F#"]], "C", sevenths=True)
    assert it6["symbols"][:2] == ["G7", "C/G"]
    assert next_chords(["Db/F"], "C")["symbols"][0] == "G"            # N6: the II row already puts V first


@pytest.mark.parametrize("tonic", TONICS)
def test_augmented_sixths_in_every_key(tonic):
    for scale, home in (("major", tonic), ("natural minor", tonic)):
        for special, first in (("It6", "V"), ("Fr43", "V"), ("Ger65", "Cad64"), ("Sw43", "Cad64")):
            notes = roman_to_chords([special], home, scale)["symbols"][0]
            r = next_chords([notes], home, scale, limit=64)
            tier0 = [c for c in r["candidates"] if c["tier"] == 0]
            assert [c["token"] for c in tier0] == ([first, "Cad64"] if first == "V" else ["Cad64", "V"]), \
                (tonic, scale, special)
            v = next(c for c in tier0 if c["token"] == "V")
            assert v["symbol"] == transpose(parse_note(home), 7, 4).name, (tonic, scale, special)
            for c, back in zip(tier0, roman_to_chords([c["token"] for c in tier0], home, scale)["chords"]):
                assert sorted(map(pc, back["notes"])) == sorted(map(pc, c["notes"])), (tonic, special)


def test_unwritable_candidates_have_a_null_token():
    r = next_chords(["C"], "C", "double harmonic")
    by_token = {c["token"]: c for c in r["candidates"]}
    assert by_token[None]["symbol"] == ["G", "B", "Db"] and "V?" not in r["tokens"]
    assert len(r["tokens"]) == len(r["symbols"])                     # kept aligned with the symbols
    assert next_chords(["C"], "C", "double harmonic", sort="movement")["candidates"]  # sorts without a token


@pytest.mark.parametrize("scale", [s.name for s in SCALES.values() if len(s.intervals) == 7])
def test_tokens_realize_their_candidates_in_every_mode(scale):
    """Every non-null token round-trips through roman_to_chords; null only for a note-array candidate."""
    for tonic in ("C", "F#", "Bb", "E"):
        for sevenths in (False, True):
            r = next_chords([tonic], tonic, scale, sevenths=sevenths, limit=64)
            for c in r["candidates"]:
                if c["token"] is None:
                    assert not isinstance(c["symbol"], str), (scale, tonic, c["symbol"])
                    continue
                back = roman_to_chords([c["token"]], tonic, scale)["chords"][0]
                assert sorted(map(pc, back["notes"])) == sorted(map(pc, c["notes"])), (scale, tonic, c["token"])


def test_movement_sort():
    diatonic = next_chords(["C"], "C", sort="movement", include_chromatic=False)["candidates"]
    assert [(c["symbol"], c["movement"]) for c in diatonic[:4]] == [("Em", 1), ("Am", 2), ("F", 3), ("G", 3)]
    full = next_chords(["C"], "C", sort="movement")["candidates"]
    assert [(c["symbol"], c["movement"], c["tier"]) for c in full[:7]] == [
        ("Em", 1, 3), ("Am", 2, 2), ("C7", 2, 4), ("Ab", 2, 5), ("Fm", 2, 5), ("F", 3, 1), ("G", 3, 1)]
    for c in full:  # voice_leading's own search: the cost it minimized for the appended chord
        v = voice_leading(["C", c["symbol"]])["voicings"]
        assert c["movement"] == _voicing_cost(v[0]["midi"], v[1]["midi"])


def test_natural_minor_adds_the_raised_dominant_and_no_mixture():
    r = next_chords(["Am"], "A", "natural minor")
    by_symbol = {c["symbol"]: c for c in r["candidates"]}
    assert by_symbol["E"]["token"] == "V" and by_symbol["Em"]["token"] == "v"
    assert by_symbol["E"]["tier"] == by_symbol["Em"]["tier"] == 1
    assert by_symbol["E"]["rule"].endswith("raised leading tone (minor-mode dominant)")
    assert not [c for c in r["candidates"] if c["tier"] in (5, 6)]
    sev = next_chords(["Am"], "A", "natural minor", sevenths=True)["symbols"]
    assert "E7" in sev and "Em7" in sev
    # Piston lists VII after no degree, so neither VII nor the raised vii° is ever a diatonic candidate
    for last in ("Am", "Bdim", "C", "Dm", "Em", "E", "F", "G"):
        for c in next_chords([last], "A", "natural minor")["candidates"]:
            assert not (c["tier"] in (1, 2, 3) and read_chord(c["symbol"])["root"].letter == "G"), last


def test_sevenths_mixture_includes_the_diminished_seventh():
    r = next_chords(["C"], "C", sevenths=True)
    by_token = {c["token"]: c for c in r["candidates"]}
    assert by_token["vii°7"]["symbol"] == "Bdim7" and by_token["vii°7"]["notes"] == ["B", "D", "F", "Ab"]
    assert by_token["vii°7"]["tier"] == 6            # mixture, but Piston lists no VII after I: 'unlisted'
    assert by_token["iiø7"]["tier"] == by_token["iv7"]["tier"] == 5 and by_token["bVII7"]["tier"] == 6
    assert [c["symbol"] for c in r["candidates"] if c["tier"] == 1] == ["Fmaj7", "G7"]


def test_mixture_is_for_major_keys_only():
    for scale in ("natural minor", "harmonic minor", "dorian", "mixolydian", "lydian"):
        assert not [c for c in next_chords(["C"], "C", scale)["candidates"] if c["tier"] in (5, 6)], scale
    assert not [c for c in next_chords(["C"], "C", include_chromatic=False)["candidates"] if c["tier"] > 3]


@pytest.mark.parametrize("kwargs", [
    {"sort": "popular"}, {"sort": None}, {"scale_type": "major pentatonic"}, {"scale_type": "blues"},
    {"limit": 0}, {"limit": 65}, {"limit": True}, {"limit": 2.0}, {"octave": 10}, {"sevenths": "yes"},
    {"include_chromatic": 1}, {"chords": []}, {"chords": ""}, {"chords": [{"root": 1}]}, {"root": "H"},
])
def test_next_chords_rejects_bad_input(kwargs):
    args = dict({"chords": ["C"], "root": "C"}, **kwargs)
    with pytest.raises(ValueError):
        next_chords(**args)


def test_limit_and_determinism():
    for n in (1, 3, 64):
        r = next_chords(["C", "F", "G7"], "C", limit=n)
        assert len(r["candidates"]) == min(n, len(next_chords(["C", "F", "G7"], "C", limit=64)["candidates"]))
        assert len(r["symbols"]) == len(r["tokens"]) == len(r["candidates"])
    assert next_chords("C Am F", "C", sort="movement") == next_chords(["C", "Am", "F"], "C", sort="movement")


def test_odd_last_chords_still_rank():
    for last in (["Ab", "C", "Eb", "F#"], ["C", "Db", "D"], ["E3", "G3", "C4"], "Csus4/F", "C#m"):
        r = next_chords(["C", last], "C")
        assert r["candidates"] and r["last"]["degree"] in range(1, 8)
    assert next_chords([["Ab", "C", "Eb", "F#"]], "C")["last"]["degree"] == 6
    assert next_chords(["Db", "C"], "C#")["last"]["degree"] == 7   # C in C# major is the leading-tone chord


# ------------------------------------------------------------ next_chords: invariants

_NEXT_KEYS = ["major", "natural minor", "harmonic minor", "dorian"]


def _next_sweep(tonic, scale):
    lasts = [[c["symbol"]] for c in diatonic_chords(tonic, scale)["chords"] if c["degree"] in (1, 2, 5, 6)]
    lasts.append(roman_to_chords(["I" if scale == "major" else "i", "V7/V"], tonic, scale)["symbols"])
    return [next_chords(last, tonic, scale, sevenths=(i % 2 == 1), limit=64) for i, last in enumerate(lasts)]


@pytest.mark.parametrize("scale", _NEXT_KEYS)
def test_next_chords_is_transposition_invariant(scale):
    home_tonic = "C"
    ref = _next_sweep(home_tonic, scale)
    for tonic in TONICS[1:]:
        got = _next_sweep(tonic, scale)
        for a, b in zip(ref, got):
            assert a["tokens"] == b["tokens"], (tonic, scale)
            for ca, cb in zip(a["candidates"], b["candidates"]):
                for field in ("tier", "tier_name", "rule", "common_tones", "root_motion", "in_key", "roman"):
                    assert ca[field] == cb[field], (tonic, scale, ca["token"], field)
                assert [rel_pc(n, home_tonic) for n in ca["notes"]] == [rel_pc(n, tonic) for n in cb["notes"]]
                assert [letter_offset(n, home_tonic) for n in ca["notes"]] == \
                       [letter_offset(n, tonic) for n in cb["notes"]], (tonic, scale, ca["token"])
            assert a["last"]["degree"] == b["last"]["degree"] and a["last"]["roman"] == b["last"]["roman"]


@pytest.mark.parametrize("scale", _NEXT_KEYS)
def test_tokens_realize_their_candidates(scale):
    for tonic in ("C", "F#", "Bb"):
        for r in _next_sweep(tonic, scale):
            realized = roman_to_chords(r["tokens"], tonic, scale)["chords"]
            for c, back in zip(r["candidates"], realized):
                assert sorted(map(pc, back["notes"])) == sorted(map(pc, c["notes"])), (tonic, c["token"])
                assert pc(back["root"]) == pc(read_chord(c["symbol"])["root"])
                info = one(c["symbol"], tonic, scale)
                if c["tier"] != 4 or not info["in_key"]:   # V7/III in minor is VII7, a diatonic chord
                    assert info["roman_figured"] == c["token"], (tonic, scale, c["token"])


def test_rule_order_within_tiers():
    for scale in _NEXT_KEYS:
        for r in _next_sweep("Eb", scale):
            cands = r["candidates"]
            tiers = [c["tier"] for c in cands]
            assert tiers == sorted(tiers)
            keys = [(c["symbol"] if isinstance(c["symbol"], str) else tuple(c["symbol"])) for c in cands]
            assert len(set(keys)) == len(keys)                     # no chord twice
            for t in (1, 2, 3, 6):
                ct = [c["common_tones"] for c in cands if c["tier"] == t]
                assert ct == sorted(ct, reverse=True)


# ------------------------------------------------------------ only ValueErrors, never crashes

_BAD_VALUES = [None, 7, -1, 3.5, 2.0, True, "", "zz", [], [None], {}, {"a": 1}, ["C", 5], "C" * 5,
               ["C", "Db", "D"], [["C", "Db", "D"], "G"], "0 4 7", ["C", ".", "E"], [[]], [["C", "E"]],
               [{"root": "C", "chord_type": None, "bass": None, "notes": []}], "H", "C major", 64, 0]


def test_new_code_rejects_bad_input_with_value_errors():
    calls = {
        analyze_progression: {"chords": ["C", "G"], "root": "C", "scale_type": "major"},
        next_chords: {"chords": ["C", "G"], "root": "C", "scale_type": "major", "sevenths": False,
                      "include_chromatic": True, "sort": "rule", "octave": 4, "limit": 16},
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


def test_masters_is_imported_lazily():
    import ast
    import pathlib
    tree = ast.parse(pathlib.Path(harmony.__file__).read_text())
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not any(getattr(n, "module", "") == "masters" for n in top)
