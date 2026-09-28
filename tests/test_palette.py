"""chord_palette: in-key chords plus borrowed (modal-interchange / neighbour-key) chords.

Unit tests pin the spec examples; the sweeps check the properties in all keys
(every scale x 14 tonic spellings), transposition, the round trip of every token
through roman_to_chords, chaining into the rendering/voicing tools, and that bad
input only ever raises ValueError.
"""

from __future__ import annotations

import pytest

from midi_composer_mcp.chords import CHORDS
from midi_composer_mcp.diatonic import (
    PARALLEL_MODES,
    _chord_family,
    borrowed_sources,
    borrowing_sources,
    chord_palette,
    diatonic_chords,
)
from midi_composer_mcp.harmony import voice_leading
from midi_composer_mcp.midi_io import _parse_chord_list, render_chords
from midi_composer_mcp.notes import LETTER_PCS, LETTERS, parse_note
from midi_composer_mcp.roman import roman_to_chords
from midi_composer_mcp.scales import SCALES, resolve_scale_type, scale_notes

TONICS = ["C", "G", "D", "A", "E", "B", "F#", "C#", "F", "Bb", "Eb", "Ab", "Db", "Gb"]
HEPTATONIC = [s.name for s in SCALES.values() if len(s.intervals) == 7]

ENTRY_FIELDS = ["symbol", "root", "notes", "chord_type", "size", "degree", "family", "core", "roman",
                "degree_function", "in_key", "source", "distance", "non_home_notes", "same_notes_as",
                "sources"]


def pc(name: str) -> int:
    return parse_note(name).pitch_class


def pcs(names) -> frozenset[int]:
    return frozenset(pc(n) for n in names)


def rel(names, tonic: str) -> list[int]:
    return [(pc(n) - pc(tonic)) % 12 for n in names]


def by_symbol(palette: dict) -> dict:
    return {c["symbol"]: c for c in palette["chords"] if isinstance(c["symbol"], str)}


def scale_pcs(tonic: str, scale: str) -> frozenset[int]:
    return frozenset((pc(tonic) + i) % 12 for i in resolve_scale_type(scale).intervals)


def misspelled_tones(c: dict) -> list[str]:
    """Tones of a named entry that are not on their chord degree's letter above the root.

    A tone may leave its letter only when that spelling would need more than a double accidental.
    """
    if c["chord_type"] == "unknown":
        return []
    root = parse_note(c["root"])
    wrong = []
    for (label, semis), name in zip(CHORDS[c["chord_type"]].degrees, c["notes"], strict=True):
        letter = LETTERS[(LETTERS.index(root.letter) + int(label.lstrip("#b")) - 1) % 7]
        note = parse_note(name)
        needed = (root.pitch_class + semis - LETTER_PCS[letter] + 6) % 12 - 6
        if note.letter != letter and abs(needed) <= 2:
            wrong.append(name)
    return wrong


# --------------------------------------------------------------- spec examples

def test_default_palette_is_the_diatonic_triads():
    p = chord_palette("C")
    assert p["symbols"] == ["C", "Dm", "Em", "F", "G", "Am", "Bdim"]
    assert p["tokens"] == ["I", "ii", "iii", "IV", "V", "vi", "vii°"]
    assert p["key"] == "C major" and p["count"] == 7 and p["borrowed_count"] == 0
    c = p["chords"][0]
    assert list(c) == ENTRY_FIELDS
    assert c == {"symbol": "C", "root": "C", "notes": ["C", "E", "G"], "chord_type": "major", "size": 3,
                 "degree": 1, "family": "major", "core": True, "roman": "I", "degree_function": "tonic",
                 "in_key": True, "source": "C major", "distance": 0, "non_home_notes": [],
                 "same_notes_as": [], "sources": ["C major"]}
    assert [c["degree_function"] for c in p["chords"]] == [
        "tonic", "subdominant", "tonic", "subdominant", "dominant", "tonic", "dominant"]
    assert [c["family"] for c in p["chords"]] == ["major", "minor", "minor", "major", "major", "minor", "other"]


def test_extended_palette_counts_and_order():
    full = chord_palette("C", extended=True)
    assert full["count"] == 42 and full["borrowed_count"] == 0
    three = chord_palette("C", extended=True, max_notes=3)
    assert three["count"] == 17
    spec_set = "C Csus2 Csus4 Dm Dsus2 Dsus4 Em Esus4 F Fsus2 G Gsus2 Gsus4 Am Asus2 Asus4 Bdim".split()
    assert set(three["symbols"]) == set(spec_set)
    # sorted by (size, degree, harmonize_melody's type preference): sus4 ranks before sus2
    assert three["symbols"] == ["C", "Csus4", "Csus2", "Dm", "Dsus4", "Dsus2", "Em", "Esus4", "F", "Fsus2",
                                "G", "Gsus4", "Gsus2", "Am", "Asus4", "Asus2", "Bdim"]
    assert "Fsus4" not in full["symbols"] and "Esus2" not in full["symbols"]
    sizes = [c["size"] for c in full["chords"]]
    assert sizes == sorted(sizes) and sizes.count(3) == 17 and sizes.count(4) == 25
    assert full["symbols"][17:21] == ["Cmaj7", "C6", "Cadd9", "Cadd4"]


BORROWED_FROM_C = [  # (symbol, roman, source)
    ("D", "II", "C lydian"), ("F#dim", "#iv°", "C lydian"), ("Bm", "vii", "C lydian"),
    ("Edim", "iii°", "C mixolydian"), ("Gm", "v", "C mixolydian"), ("Bb", "bVII", "C mixolydian"),
    ("Cm", "i", "C melodic minor"), ("Ebaug", "bIII+", "C melodic minor"), ("Adim", "vi°", "C melodic minor"),
    ("Ddim", "ii°", "C harmonic major"), ("Fm", "iv", "C harmonic major"), ("Abaug", "bVI+", "C harmonic major"),
    ("Eb", "bIII", "C dorian"),
    ("Ab", "bVI", "C harmonic minor"),
    ("Db", "bII", "C phrygian"), ("Gdim", "v°", "C phrygian"), ("Bbm", "bvii", "C phrygian"),
    ("Cdim", "i°", "C locrian"), ("Ebm", "biii", "C locrian"), ("Gb", "bV", "C locrian"),
]


def test_borrowed_chords_from_c_major():
    p = chord_palette("C", borrow=True)
    assert p["symbols"][:7] == ["C", "Dm", "Em", "F", "G", "Am", "Bdim"]
    assert p["count"] == 27 and p["borrowed_count"] == 20
    got = [(c["symbol"], c["roman"], c["source"]) for c in p["chords"][7:]]
    assert got == BORROWED_FROM_C
    assert p["tokens"][7:] == [r for _, r, _ in BORROWED_FROM_C]
    assert all(not c["in_key"] and c["core"] for c in p["chords"][7:])
    fm = by_symbol(p)["Fm"]
    assert fm["sources"] == ["C harmonic major", "C harmonic minor", "C natural minor", "C phrygian",
                             "C locrian"]
    assert fm["non_home_notes"] == ["Ab"] and fm["degree"] == 4 and fm["distance"] == 1
    assert by_symbol(p)["Cm"]["sources"] == ["C melodic minor", "C dorian", "C harmonic minor",
                                             "C natural minor", "C phrygian"]
    assert by_symbol(p)["C"]["sources"][0] == "C major"   # the home is a considered source


def test_source_distances_from_major():
    """distance = the source's pitch classes outside the home scale (the spec's table)."""
    p = chord_palette("C", borrow=True)
    dist = {c["source"]: c["distance"] for c in p["chords"]}
    assert dist == {"C major": 0, "C lydian": 1, "C mixolydian": 1, "C melodic minor": 1,
                    "C harmonic major": 1, "C dorian": 2, "C harmonic minor": 2, "C phrygian": 4,
                    "C locrian": 5}
    # natural minor (distance 3) adds nothing new: all its chords came earlier
    assert "C natural minor" not in dist
    assert all("C natural minor" in by_symbol(p)[s]["sources"] for s in ("Cm", "Ddim", "Eb", "Fm", "Gm", "Ab", "Bb"))


def test_fifths_steps_borrows_from_neighbour_keys():
    p = chord_palette("C", fifths_steps=1)
    got = [(c["symbol"], c["roman"], c["source"], c["distance"], c["degree"]) for c in p["chords"][7:]]
    assert got == [("Bm", "vii", "G major", 1, 3), ("D", "II", "G major", 1, 5),
                   ("F#dim", "#iv°", "G major", 1, 7), ("Gm", "v", "F major", 1, 2),
                   ("Bb", "bVII", "F major", 1, 4), ("Edim", "iii°", "F major", 1, 7)]
    assert by_symbol(p)["D"]["degree_function"] == "dominant"   # D is V of G major
    assert by_symbol(p)["G"]["sources"] == ["C major", "G major"]
    three = chord_palette("C", fifths_steps=3)
    assert [c["source"] for c in three["chords"][7:]][::3][:6] == [
        "G major", "F major", "D major", "Bb major", "A major", "Eb major"]


def test_fifths_neighbours_keep_the_home_spelling():
    """A neighbour key is spelled on the home tonic's letters (Cb major's is Fb major, not E major), so its
    chords read the numerals they have in C; only the source label is the practical enharmonic key."""
    def tail(p, start=7):
        return [(c["symbol"], c["roman"], c["source"]) for c in p["chords"][start:]]

    assert tail(chord_palette("Cb", fifths_steps=1)) == [
        ("Bbm", "vii", "Gb major"), ("Db", "II", "Gb major"), ("Fdim", "#iv°", "Gb major"),
        ("Gbm", "v", "E major"), ("Bbb", "bVII", "E major"), ("Ebdim", "iii°", "E major")]   # Fb major
    assert tail(chord_palette("C#", fifths_steps=1)) == [
        ("B#m", "vii", "Ab major"), ("D#", "II", "Ab major"), ("F##dim", "#iv°", "Ab major"),  # G# major
        ("G#m", "v", "F# major"), ("B", "bVII", "F# major"), ("E#dim", "iii°", "F# major")]
    # A minor's neighbours give F#dim Bm D (E minor) and Edim Gm Bb (D minor): v°, bvii, bII
    assert tail(chord_palette("Ab", "natural minor", fifths_steps=1), 10) == [
        ("Ebdim", "v°", "C# natural minor"), ("Gbm", "bvii", "C# natural minor"), ("Bbb", "bII", "C# natural minor")]
    gb = chord_palette("Gb", fifths_steps=2)["chords"][16:]   # from Fb major: C's Cm Eb Adim from Bb major
    assert [(c["symbol"], c["roman"], c["source"]) for c in gb] == [
        ("Gbm", "i", "E major"), ("Bbb", "bIII", "E major"), ("Ebdim", "vi°", "E major")]


def test_six_fifths_name_the_tritone_key_once():
    """+6 and -6 fifths reach one scale (F# = Gb major from C): it is one source, under the +6 spelling."""
    p = by_symbol(chord_palette("C", fifths_steps=6))
    assert p["E#dim"]["sources"] == ["F# major"]
    assert p["B"]["sources"] == ["E major", "B major", "F# major"]
    assert not any("Gb major" in c["sources"] or c["source"] == "Gb major" for c in p.values())
    # F#'s +6 key is B# major (label 'C major'), spelled as C's F# major is: the same tokens
    assert chord_palette("F#", fifths_steps=6)["tokens"] == chord_palette("C", fifths_steps=6)["tokens"]
    assert by_symbol(chord_palette("F#", fifths_steps=6))["A##dim"]["source"] == "C major"


def test_extended_palette_keeps_only_chords_the_scale_spells():
    """Letters + semitones: Ab B Eb is no Abm (Ab-B is an augmented 2nd) — a 7-note scale's chords are the
    ones its own notes spell; the harmonic-minor triads are i ii° III+ iv V VI vii° (Kostka & Payne)."""
    three = chord_palette("C", "harmonic minor", extended=True, max_notes=3)
    assert three["symbols"] == ["Cm", "Csus4", "Csus2", "Ddim", "Ebaug", "Fm", "Fsus2", "G", "Gsus4", "Ab", "Bdim"]
    four = by_symbol(chord_palette("C", "harmonic minor", extended=True))
    assert not {"Abm", "Fdim", "Abdim", "Gaug", "Baug", "Ddim7"} & set(four)
    assert four["Bdim7"]["notes"] == ["B", "D", "F", "Ab"] and four["Ebaug"]["roman"] == "bIII+"
    hmaj = by_symbol(chord_palette("C", "harmonic major", extended=True))
    assert not {"E", "Caug", "Eaug"} & set(hmaj) and hmaj["Abaug"]["notes"] == ["Ab", "C", "E"]
    borrowed = by_symbol(chord_palette("C", borrow=True, extended=True, max_notes=3))
    assert "E" not in borrowed and borrowed["Abaug"]["source"] == "C harmonic major"
    # other scale sizes have no letter per degree: pitch classes decide, the chord spells itself
    wt = chord_palette("C", "whole tone")
    assert wt["symbols"] == ["Caug", "Daug", "Eaug", "F#aug", "G#aug", "Bbaug"]
    assert [c["notes"] for c in wt["chords"]][1:3] == [["D", "F#", "A#"], ["E", "G#", "B#"]]
    chromatic = by_symbol(chord_palette("C", "chromatic", extended=True, max_notes=3))
    assert chromatic["E"]["notes"] == ["E", "G#", "B"] and len(chromatic) == 72   # 12 roots x 6 triad types


def test_extended_palette_respects_max_notes_and_lists_only_table_chords():
    """extended=true is every chord-table type with 3..max_notes notes: no unnamed stacked sets, whatever
    `sevenths` (which only moves the core flag)."""
    p = chord_palette("C", "hungarian major", extended=True, sevenths=True, max_notes=3)
    assert max(c["size"] for c in p["chords"]) == 3
    assert all(c["chord_type"] != "unknown" for c in chord_palette("C", "double harmonic", extended=True)["chords"])
    assert chord_palette("C", "hirajoshi", extended=True, sevenths=True, max_notes=3)["count"] == \
        chord_palette("C", "hirajoshi", extended=True, max_notes=3)["count"]
    for scale in SCALES:
        for max_notes in range(3, 7):
            counts = set()
            for sevenths in (False, True):
                p = chord_palette("C", scale, extended=True, sevenths=sevenths, max_notes=max_notes)
                assert all(3 <= c["size"] <= max_notes for c in p["chords"]), (scale, sevenths, max_notes)
                assert all(c["chord_type"] != "unknown" for c in p["chords"]), (scale, sevenths, max_notes)
                counts.add(tuple(p["symbols"]))
            assert len(counts) == 1, (scale, max_notes)


def test_harmonic_minor_extended_has_the_leading_tone_chords():
    p = by_symbol(chord_palette("A", "harmonic minor", extended=True))
    assert p["G#dim7"]["roman"] == "vii°7" and p["G#dim7"]["notes"] == ["G#", "B", "D", "F"]
    assert p["Caug"]["roman"] == "bIII+" and p["Caug"]["core"]
    assert p["Caug"]["notes"] == ["C", "E", "G#"]


def test_pentatonic_home_has_no_romans():
    p = chord_palette("C", "major pentatonic")
    # diatonic_chords stacks C E A (= Am/C) etc.; the palette lists each in root position, on its root
    assert p["symbols"] == ["C", "Csus2", "Dsus2", "Gsus2", "Am"]
    assert p["tokens"] == []
    assert all("roman" not in c and "degree_function" not in c for c in p["chords"])
    assert by_symbol(p)["Am"]["degree"] == 5 and by_symbol(p)["Am"]["notes"] == ["A", "C", "E"]
    ext = chord_palette("C", "major pentatonic", extended=True)
    assert all(pcs(c["notes"]) <= scale_pcs("C", "major pentatonic") for c in ext["chords"])
    assert {"C6", "Am7", "Cadd9"} <= set(ext["symbols"])


def test_same_notes_as_links_identical_pitch_sets():
    p = by_symbol(chord_palette("C", extended=True))
    assert p["Am7"]["same_notes_as"] == ["C6"] and p["C6"]["same_notes_as"] == []
    assert p["Gsus4"]["same_notes_as"] == ["Csus2"] and p["Csus2"]["same_notes_as"] == []
    assert p["Dm7"]["same_notes_as"] == [] and p["F6"]["same_notes_as"] == ["Dm7"]   # earlier symbols only
    assert p["Fadd9"]["same_notes_as"] == []


def test_a_minor_home_borrows_from_dorian_and_harmonic_minor():
    p = by_symbol(chord_palette("A", "natural minor", borrow=True))
    want = {"E": ("V", "A harmonic minor"), "Bm": ("ii", "A dorian"), "D": ("IV", "A dorian"),
            "F#dim": ("vi°", "A dorian")}
    for symbol, (roman, source) in want.items():
        assert (p[symbol]["roman"], p[symbol]["source"]) == (roman, source)
        assert "roman_note" not in p[symbol]
    assert p["F#dim"]["distance"] == 1 and p["E"]["non_home_notes"] == ["G#"]


def test_sevenths_borrow():
    p = by_symbol(chord_palette("C", sevenths=True, borrow=True))
    assert p["Fm7"]["roman"] == "iv7"
    assert p["Bbmaj7"]["roman"] == "bVIIΔ7" and p["Bbmaj7"]["source"] == "C mixolydian"
    assert p["Cmaj7"]["in_key"] and p["G7"]["roman"] == "V7"


def test_source_modes_and_their_precedence():
    only = chord_palette("C", source_modes="dorian")
    assert [(c["symbol"], c["roman"]) for c in only["chords"][7:]] == [
        ("Cm", "i"), ("Eb", "bIII"), ("Gm", "v"), ("Adim", "vi°"), ("Bb", "bVII")]
    assert all(c["distance"] == 2 for c in only["chords"][7:])
    assert chord_palette("C", source_modes=["dorian"], borrow=True) == only   # given modes win
    assert by_symbol(only)["Gm"]["sources"] == ["C dorian"]                    # only considered sources
    assert chord_palette("C", source_modes=[])["borrowed_count"] == 0
    assert chord_palette("C", source_modes=["major"])["borrowed_count"] == 0  # the home mode adds nothing
    lyd_dom = chord_palette("C", source_modes="lydian dominant")   # any 7-note mode may be a source
    assert [(c["symbol"], c["roman"]) for c in lyd_dom["chords"][7:]] == [
        ("D", "II"), ("Edim", "iii°"), ("F#dim", "#iv°"), ("Gm", "v"), ("Bbaug", "bVII+")]


def test_core_flag_follows_sevenths():
    tri = by_symbol(chord_palette("C", extended=True))
    assert tri["Dm"]["core"] and not tri["Dm7"]["core"] and not tri["Dsus4"]["core"]
    sev = by_symbol(chord_palette("C", extended=True, sevenths=True))
    assert sev["Dm7"]["core"] and not sev["Dm"]["core"] and sev["G7"]["core"]
    assert chord_palette("C", sevenths=True)["symbols"] == ["Cmaj7", "Dm7", "Em7", "Fmaj7", "G7", "Am7", "Bm7b5"]


def test_dyads_and_max_notes():
    assert "C5" not in chord_palette("C", extended=True)["symbols"]
    dy = by_symbol(chord_palette("C", extended=True, include_dyads=True))
    assert dy["C5"]["size"] == 2 and dy["C5"]["roman"] == "I5" and dy["C5"]["family"] == "other"
    assert list(dy)[:7] == ["C5", "D5", "E5", "F5", "G5", "A5", "C"]   # size first
    six = chord_palette("C", extended=True, max_notes=6)
    assert {"G13", "Dm11", "Cmaj9", "G9"} <= set(six["symbols"])
    assert max(c["size"] for c in six["chords"]) == 6


def test_limit_truncates_but_counts_the_total():
    p = chord_palette("C", borrow=True, limit=9)
    assert len(p["chords"]) == len(p["symbols"]) == len(p["tokens"]) == 9
    assert p["count"] == 27 and p["borrowed_count"] == 20
    assert p["symbols"] == chord_palette("C", borrow=True)["symbols"][:9]


def test_family_reads_letters_not_semitones():
    c = parse_note("C")
    assert _chord_family(CHORDS["dominant 7 sharp 9"], c, []) == "major"   # the #9 is a ninth
    assert _chord_family(CHORDS["minor 6"], c, []) == "minor"
    assert _chord_family(CHORDS["minor 11"], c, []) == "minor"
    assert _chord_family(CHORDS["dominant 7 sharp 11"], c, []) == "major"  # #11 is no fifth
    for name in ("diminished", "augmented", "half-diminished", "suspended 4", "power chord",
                 "dominant 7 flat 5", "augmented 7", "diminished 7"):
        assert _chord_family(CHORDS[name], c, []) == "other", name
    g = parse_note("G")
    assert _chord_family(None, g, [g, parse_note("B"), parse_note("Db")]) == "other"   # diminished 5th
    assert _chord_family(None, g, [g, parse_note("B"), parse_note("D")]) == "major"


def test_unnamed_stacked_chords_stay_note_lists():
    p = chord_palette("C", "double harmonic")
    odd = [c for c in p["chords"] if not isinstance(c["symbol"], str)]
    assert [(c["symbol"], c["roman"], c["degree"]) for c in odd] == [
        (["G", "B", "Db"], "V?", 5), (["B", "Db", "F"], "VII?", 7)]
    assert all(c["chord_type"] == "unknown" and c["family"] == "other" for c in odd)
    # diatonic_chords names hungarian major's (C D# E F# G A Bb) G Bb D# as D#/G and Bb D# F# as
    # D#m/Bb by pitch class, but by letters they are no such chords (D# major needs F## and A#):
    # they stay the stacked note lists, on their own degree
    hm = chord_palette("C", "hungarian major")
    assert hm["symbols"] == ["C", "D#dim", "Edim", "F#dim", ["G", "Bb", "D#"], "Am", ["Bb", "D#", "F#"]]
    assert hm["tokens"] == ["I", "#ii°", "iii°", "#iv°", "v?", "vi", "bVII?"]
    g = hm["chords"][4]
    assert g["chord_type"] == "unknown" and g["degree"] == 5 and g["root"] == "G" and g["core"]
    assert g["family"] == "other"   # G-Bb-D#: a minor third and an augmented fifth
    # the inversions whose scale spelling does spell the chord keep their name, in root position
    assert chord_palette("C", "major pentatonic")["symbols"] == ["C", "Csus2", "Dsus2", "Gsus2", "Am"]


def test_non_home_notes_are_key_spelled():
    p = by_symbol(chord_palette("C", borrow=True, fifths_steps=1))
    assert p["Bb"]["non_home_notes"] == ["Bb"]
    assert p["F#dim"]["non_home_notes"] == ["F#"] and p["F#dim"]["source"] == "C lydian"
    assert p["Ebm"]["non_home_notes"] == ["Eb", "Gb", "Bb"]
    flats = by_symbol(chord_palette("Db", borrow=True))
    assert flats["Cbm"]["notes"] == ["Cb", "Ebb", "Gb"] and flats["Cbm"]["roman"] == "bvii"
    assert flats["Ebb"]["notes"] == ["Ebb", "Gb", "Bbb"] and flats["Ebb"]["roman"] == "bII"


def test_roman_note_marks_the_dialect_gap_of_non_major_homes():
    """A bare numeral names the home's own degree: Em borrowed into C minor ('iii') reads back as Ebm."""
    p = by_symbol(chord_palette("C", "natural minor", borrow=True))
    em = p["Em"]
    assert em["roman"] == "iii" and "Ebm" in em["roman_note"] and "C major" in em["roman_note"]
    assert roman_to_chords("iii", "C", "natural minor")["symbols"] == ["Ebm"]
    assert roman_to_chords("iii", "C", "major")["symbols"] == ["Em"]
    assert "roman_note" not in p["Ab"] and "roman_note" not in p["D"]
    # in minor, ^6/^7 without an accidental follow the chord's third (music21 Minor67Default)
    mm = by_symbol(chord_palette("A", "melodic minor", extended=True, borrow=True))
    assert mm["F#sus4"]["roman"] == "VIsus4" and "Fsus4" in mm["F#sus4"]["roman_note"]
    assert "roman_note" not in mm["F#m7b5"] and "roman_note" not in mm["G#dim"]
    assert "G#aug" not in mm   # G# C E is Caug (bIII+) spelled wrong: a G# augmented triad needs B#
    am = by_symbol(chord_palette("A", "natural minor", extended=True, borrow=True))
    assert am["F#sus4"]["roman"] == "VIsus4" and "Fsus4" in am["F#sus4"]["roman_note"]
    assert am["F#sus4"]["source"] == "A mixolydian" and "roman_note" not in am["F#m"]
    assert not any("roman_note" in c for c in chord_palette("A", "major", extended=True, borrow=True,
                                                             fifths_steps=2)["chords"])


def test_chord_palette_errors():
    for kwargs in ({"max_notes": 1}, {"max_notes": 7}, {"max_notes": 2},
                   {"scale_type": "major pentatonic", "borrow": True},
                   {"scale_type": "blues", "fifths_steps": 1},
                   {"scale_type": "whole tone", "source_modes": "dorian"},
                   {"source_modes": "nope"}, {"source_modes": "major pentatonic"},
                   {"source_modes": ["dorian", "blues"]}, {"fifths_steps": 7}, {"fifths_steps": -1},
                   {"limit": -1}, {"root": "C E"}, {"root": 5}, {"borrow": "yes"}):
        with pytest.raises(ValueError):
            chord_palette(**{"root": "C", **kwargs})


# ------------------------------------------------------------------ sweeps

@pytest.mark.parametrize("scale", list(SCALES))
def test_in_key_entries_fit_and_counts_match_in_every_key(scale):
    counts = set()
    for tonic in TONICS:
        home = scale_pcs(tonic, scale)
        for extended in (False, True):
            p = chord_palette(tonic, scale, extended=extended)
            assert p["borrowed_count"] == 0 and all(c["in_key"] for c in p["chords"])
            for c in p["chords"]:
                assert pcs(c["notes"]) <= home, (tonic, c["symbol"])
                assert c["notes"][0] == c["root"] and 1 <= c["degree"] <= len(resolve_scale_type(scale).intervals)
                # the notes spell the symbol: letters + semitones (Ab B Eb would be no Abm)
                assert not misspelled_tones(c), (tonic, extended, c["symbol"], c["notes"])
            counts.add((extended, p["count"]))
    assert len(counts) == 2, counts   # one count per mode of the universe, whatever the tonic


@pytest.mark.parametrize("scale", ["major", "natural minor", "dorian", "harmonic minor", "lydian dominant"])
def test_no_borrowed_entry_is_a_subset_of_the_home_scale(scale):
    configs = [{"borrow": True}, {"borrow": True, "sevenths": True}, {"fifths_steps": 3}]
    for tonic in TONICS:
        home = scale_pcs(tonic, scale)
        heavy = [{"extended": True, "borrow": True, "fifths_steps": 1}] if tonic in ("C", "F#", "Db") else []
        for kwargs in configs + heavy:
            p = chord_palette(tonic, scale, **kwargs)
            considered = borrowing_sources(tonic, scale, list(PARALLEL_MODES) if kwargs.get("borrow") else [],
                                           kwargs.get("fifths_steps", 0), include_home=True)
            spelled = {s["label"]: {n.pitch_class_name for n in scale_notes(s["scale"], s["tonic"])}
                       for s in considered}
            for c in p["chords"]:
                assert (pcs(c["notes"]) <= home) == c["in_key"], (tonic, kwargs, c["symbol"])
                assert (c["distance"] == 0) == c["in_key"]
                assert (c["non_home_notes"] == []) == c["in_key"]
                assert not misspelled_tones(c), (tonic, kwargs, c["symbol"], c["notes"])
                # sources judge spelling: each one's own notes spell the chord, so the nearest is its source
                assert c["sources"][0] == c["source"], (tonic, kwargs, c["symbol"], c["sources"])
                assert all(set(c["notes"]) <= spelled[label] for label in c["sources"]), (tonic, c["symbol"])
                assert (c["sources"][0] == p["key"]) == c["in_key"]
                if tonic == "C":  # the one shared rule: palette sources are borrowed_sources' labels
                    assert c["sources"] == borrowed_sources(
                        c["notes"], tonic, scale, include_home=True, fifths_steps=kwargs.get("fifths_steps", 0),
                        modes=list(PARALLEL_MODES) if kwargs.get("borrow") else [])
            keys = [(pc(c["root"]), c["chord_type"]) for c in p["chords"]]
            assert len(keys) == len(set(keys))                        # no repeats
            borrowed = [c for c in p["chords"] if not c["in_key"]]
            assert [c["distance"] for c in borrowed] == sorted(c["distance"] for c in borrowed)


def _letter_steps(note: str, root: str) -> int:
    return (LETTERS.index(parse_note(note).letter) - LETTERS.index(parse_note(root).letter)) % 7


def _assert_transposed(ref: dict, got: dict, tonic: str) -> None:
    """`got` (in `tonic`) is `ref` (in C) moved by one spelled interval: tokens, notes, letters, sources."""
    assert got["tokens"] == ref["tokens"], tonic
    assert [c["distance"] for c in got["chords"]] == [c["distance"] for c in ref["chords"]]
    assert [c["degree"] for c in got["chords"]] == [c["degree"] for c in ref["chords"]]
    assert [c["family"] for c in got["chords"]] == [c["family"] for c in ref["chords"]]
    for a, b in zip(ref["chords"], got["chords"], strict=True):
        assert rel(b["notes"], tonic) == rel(a["notes"], "C"), (tonic, b["symbol"])
        assert b["chord_type"] == a["chord_type"]
        assert [s.split(" ", 1)[1] for s in b["sources"]] == [s.split(" ", 1)[1] for s in a["sources"]]
        assert [rel([s.split(" ", 1)[0]], tonic) for s in b["sources"]] == \
               [rel([s.split(" ", 1)[0]], "C") for s in a["sources"]], (tonic, b["symbol"])
        assert ("roman_note" in a) == ("roman_note" in b)
        # spelled: the root on the same letter above the tonic, every chord tone on its own letter above the root
        assert _letter_steps(b["root"], tonic) == _letter_steps(a["root"], "C"), (tonic, b["symbol"])
        assert [_letter_steps(n, b["root"]) for n in b["notes"]] == [_letter_steps(n, a["root"]) for n in a["notes"]]


@pytest.mark.parametrize("scale", ["major", "natural minor"])
@pytest.mark.parametrize("sevenths", [False, True])
def test_borrowing_is_transposition_invariant(scale, sevenths):
    ref = chord_palette("C", scale, borrow=True, sevenths=sevenths)
    for tonic in TONICS:
        _assert_transposed(ref, chord_palette(tonic, scale, borrow=True, sevenths=sevenths), tonic)


@pytest.mark.parametrize("scale, tonics", [("major", TONICS + ["Cb"]),
                                           ("natural minor", TONICS + ["G#", "D#", "A#"]),
                                           ("harmonic minor", TONICS + ["G#"])])
@pytest.mark.parametrize("steps", range(1, 7))
def test_fifths_borrowing_is_transposition_invariant(scale, tonics, steps):
    """Neighbour keys keep the home's letters (Cb major's Fb major, C# major's G# major), so every key reads
    the numerals C does, up to six steps round the circle; the tritone key is one source everywhere."""
    configs = [{"fifths_steps": steps}] + ([{"fifths_steps": steps, "borrow": True, "sevenths": True}]
                                           if steps in (2, 6) else [])
    for kwargs in configs:
        ref = chord_palette("C", scale, **kwargs)
        for tonic in tonics:
            _assert_transposed(ref, chord_palette(tonic, scale, **kwargs), tonic)


def _assert_round_trip(p: dict, tonic: str, scale: str) -> None:
    """Every numeral reads back in the parallel major, and in the home key unless it carries a roman_note."""
    check = [c for c in p["chords"] if "?" not in c["roman"]]
    for c in p["chords"]:
        if "?" in c["roman"]:
            assert c["chord_type"] == "unknown"
            with pytest.raises(ValueError):
                roman_to_chords(c["roman"], tonic, scale)
    if not check:
        return
    tokens = [c["roman"] for c in check]
    home = roman_to_chords(tokens, tonic, scale)["chords"]
    major = roman_to_chords(tokens, tonic, "major")["chords"]
    for c, h, m in zip(check, home, major, strict=True):
        assert pcs(m["notes"]) == pcs(c["notes"]), (tonic, scale, c["symbol"], c["roman"], m["symbol"])
        assert (pcs(h["notes"]) == pcs(c["notes"])) == ("roman_note" not in c), (tonic, scale, c["roman"])
        if scale == "major":
            assert "roman_note" not in c


@pytest.mark.parametrize("scale", HEPTATONIC)
def test_every_token_round_trips_through_roman_to_chords(scale):
    for tonic in TONICS:
        _assert_round_trip(chord_palette(tonic, scale, borrow=True), tonic, scale)
    for tonic in ("C", "F#", "Gb", "Eb"):
        _assert_round_trip(chord_palette(tonic, scale, extended=True), tonic, scale)
        _assert_round_trip(chord_palette(tonic, scale, borrow=True, sevenths=True, fifths_steps=2), tonic, scale)
    _assert_round_trip(chord_palette("Db", scale, extended=True, max_notes=6, include_dyads=True, borrow=True),
                       "Db", scale)


def test_in_key_romans_agree_with_diatonic_chords():
    for scale in HEPTATONIC:
        for tonic in ("C", "Eb", "F#"):
            p = chord_palette(tonic, scale)
            stacked = diatonic_chords(tonic, scale)["chords"]
            for c in p["chords"]:
                same = [s for s in stacked if s["symbol"] == c["symbol"]]
                if same:  # a root-position stacked chord keeps diatonic_chords' own numeral
                    assert c["roman"] == same[0]["roman"]
                    assert c["degree_function"] == same[0]["harmonic_function"]


def test_symbols_chain_into_rendering_and_voicing(tmp_path):
    palettes = [chord_palette("C", borrow=True, fifths_steps=2, extended=True),
                chord_palette("C", "double harmonic"), chord_palette("Eb", "major pentatonic", extended=True),
                chord_palette("F#", "harmonic minor", sevenths=True, borrow=True),
                chord_palette("Gb", "natural minor", borrow=True, extended=True, include_dyads=True)]
    for p in palettes:
        symbols = p["symbols"]
        parsed = _parse_chord_list(symbols)
        for entry, c in zip(parsed, p["chords"], strict=True):
            assert {t.pitch_class for t in entry["tones"]} == pcs(c["notes"]), c["symbol"]
        assert render_chords(symbols, output_dir=str(tmp_path))["chord_count"] == len(symbols)
    small = chord_palette("C", borrow=True)["symbols"]
    assert len(voice_leading(small)["voicings"]) == len(small)


def test_palette_is_deterministic():
    kwargs = {"extended": True, "borrow": True, "fifths_steps": 2, "sevenths": True}
    assert chord_palette("Bb", "dorian", **kwargs) == chord_palette("Bb", "dorian", **kwargs)


# ------------------------------------------------------- only ValueErrors, never crashes

_BAD_VALUES = [None, 7, -1, 3.5, 2.0, True, "", "zz", [], [None], {}, {"a": 1}, ["C", 5], "C" * 5,
               ["C", "Db", "D"], [["C", "Db", "D"], "G"], "0 4 7", ["C", ".", "E"]]


def test_chord_palette_rejects_bad_input_with_value_errors():
    good = {"root": "C", "scale_type": "major", "extended": False, "sevenths": False, "max_notes": 4,
            "include_dyads": False, "borrow": True, "source_modes": None, "fifths_steps": 1, "limit": 0}
    failures = []
    for pname in good:
        for bad in _BAD_VALUES:
            try:
                chord_palette(**dict(good, **{pname: bad}))
            except ValueError:
                pass
            except Exception as e:  # noqa: BLE001 - the list is the point
                failures.append(f"chord_palette({pname}={bad!r}) raised {type(e).__name__}: {e}")
    assert not failures, "\n".join(failures)


def test_odd_tonic_spellings_never_crash():
    """Double sharps/flats and theoretical keys (B## major, Fbb minor) still give a palette."""
    tonics = [letter + acc for letter in LETTERS for acc in ("", "#", "b", "##", "bb")]
    cases = [("major", {"borrow": True, "sevenths": True, "fifths_steps": 6}),
             ("lydian augmented", {"extended": True, "borrow": True}),
             ("altered", {"extended": True}), ("blues", {"extended": True})]
    failures = []
    for tonic in tonics:
        for scale, kwargs in cases:
            try:
                p = chord_palette(tonic, scale, **kwargs)
            except Exception as e:  # noqa: BLE001 - the list is the point
                failures.append(f"{tonic} {scale} {kwargs}: {type(e).__name__}: {e}")
                continue
            if not all(pcs(c["notes"]) <= scale_pcs(tonic, scale) for c in p["chords"] if c["in_key"]):
                failures.append(f"{tonic} {scale} {kwargs}: an in-key chord leaves the scale")
    assert not failures, "\n".join(failures[:20])
