"""voice_chords (voicing.py): drop-2/3/2&4, open, shell and Levine rootless voicings.

Unit tests pin the spec examples (Levine's rootless ii-V-I, the drop voicings of
Cmaj7); the sweeps check the properties over all 14 tonic spellings and every
chord type, that style 'close' and the replicated greedy search agree with
voice_leading, chaining into the renderers and check_voice_leading, and that bad
input only ever raises ValueError.
"""

from __future__ import annotations

import pytest

from midi_composer_mcp.analysis import check_voice_leading
from midi_composer_mcp.chords import CHORDS, chord_notes, parse_chord_symbol
from midi_composer_mcp.diatonic import degrees_to_chords, diatonic_chords
from midi_composer_mcp.harmony import _voicing_cost, voice_leading
from midi_composer_mcp.midi_io import _parse_chord_list, render_arrangement, render_chords
from midi_composer_mcp.notes import LETTERS, parse_note, transpose
from midi_composer_mcp.voicing import STYLES, _greedy, _read, voice_chords

TONICS = ["C", "G", "D", "A", "E", "B", "F#", "C#", "F", "Bb", "Eb", "Ab", "Db", "Gb"]


def _notes(chords, style="drop2", **kw):
    return voice_chords(chords, style, **kw)["chords"]


def _gaps(midis):
    return tuple(b - a for a, b in zip(midis, midis[1:]))


# ------------------------------------------------------------ spec examples

def test_levine_rootless_ii_v_i():
    r = voice_chords(["Dm7", "G7", "Cmaj7"], "rootless")
    assert r["style"] == "rootless"
    assert r["chords"] == [["F3", "A3", "C4", "E4"], ["F3", "A3", "B3", "E4"], ["E3", "G3", "B3", "D4"]]
    assert [v["variant"] for v in r["voicings"]] == ["A", "B", "A"]
    assert [v["degrees"] for v in r["voicings"]] == [["b3", "5", "b7", "9"], ["b7", "9", "3", "13"],
                                                     ["3", "5", "7", "9"]]
    # the costs the greedy chain compared (spec: B 1 vs A 23, then A 5 vs B 19)
    dm7 = r["voicings"][0]["midi"]
    g7a, g7b = (voice_chords(["G7"], s)["voicings"][0]["midi"] for s in ("rootless_a", "rootless_b"))
    assert (_voicing_cost(dm7, g7b), _voicing_cost(dm7, g7a)) == (1, 23)
    ca, cb = (voice_chords(["Cmaj7"], s)["voicings"][0]["midi"] for s in ("rootless_a", "rootless_b"))
    assert (_voicing_cost(g7b, ca), _voicing_cost(g7b, cb)) == (5, 19)
    assert r["total_movement"] == 1 + 5


def test_drop_voicings_of_cmaj7():
    assert _notes(["Cmaj7"], "drop2") == [["G3", "C4", "E4", "B4"]]
    assert _notes(["Cmaj7"], "drop3") == [["E3", "C4", "G4", "B4"]]
    assert _notes(["Cmaj7"], "drop24") == [["C3", "G3", "E4", "B4"]]
    assert voice_chords(["Cmaj7"], "drop2")["voicings"][0]["degrees"] == ["5", "1", "3", "7"]


def test_close_is_voice_leading():
    corpus = [["C", "G", "Am", "F"], ["Dm7", "G7", "Cmaj7"], ["C", "C/E", "F/A", "G/B", "C"],
              ["Cmaj9", "Fmaj7#11", "Bb13", "Ebmaj7", "Abm6", "Db7#9"], "C Am F G",
              [["C", "E", "G"], ["E", "G", "C"], "G7"], [["C", "Db", "D"], "G"], ["F#m7b5", "B7b9", "Em"],
              ["Cdim7", "C#dim7", "Dm7", "G7sus4", "C5", "Csus2"]]
    for chords in corpus:
        for octave in (2, 4, 6):
            assert _notes(chords, "close", octave=octave) == voice_leading(chords, octave=octave)["chords"]


def test_replicated_search_matches_voice_leading():
    """The greedy search used for top_notes and every other style IS voice_leading's search."""
    corpus = [["C", "G", "Am", "F", "C/E", "Dm7", "G7", "C"], ["Cmaj9", "A7b9", "Dm11", "G13", "Cmaj7#11"],
              ["Bb", "Eb/G", "Ab", "Gb", "Cb", "E"], [["C", "E", "G", "C"], ["A", "C", "E"], "F6"]]
    for chords in corpus:
        items = chords.split() if isinstance(chords, str) else chords
        read = [_read(e, i) for e, i in zip(_parse_chord_list(chords), items)]
        for octave in (3, 4, 5):
            ours = [v["notes"] for v in _greedy(read, "close", octave, True, None)]
            assert ours == voice_leading(chords, octave=octave)["chords"]


def test_rootless_rejects_triads_sus_and_dim7():
    for chord in ("C", "Cm", "Csus4", "C7sus4", "Cdim7", "Cadd9", "C5"):
        with pytest.raises(ValueError):
            voice_chords([chord], "rootless")


def test_top_note_with_octave():
    r = voice_chords(["Cmaj7"], "drop2", top_notes=["E5"])
    assert r["voicings"][0]["midi"][-1] == parse_note("E5").midi
    assert r["chords"] == [["C4", "G4", "B4", "E5"]]


def test_top_note_outside_the_chord_raises():
    with pytest.raises(ValueError, match="F#.*C"):
        voice_chords(["C"], "drop2", top_notes="F#")
    with pytest.raises(ValueError, match="C7"):          # a tone, but the shell's bass: never on top
        voice_chords(["C7"], "shell", top_notes=["C"])
    with pytest.raises(ValueError, match="G"):           # the 5th is omitted from a drop-2 13th chord
        voice_chords(["C13"], "drop2", top_notes=["G"])
    with pytest.raises(ValueError, match="one note per chord"):
        voice_chords(["C", "F"], "drop2", top_notes=["E5"])


def test_slash_bass_stays_lowest():
    v = voice_chords(["C/E"], "drop2")["voicings"][0]
    assert v["notes"][0] == "E3" and v["midi"][0] == min(v["midi"])
    assert v["notes"] == ["E3", "G3", "C4", "E4", "C5"]
    for style in STYLES:
        chords = ["Cmaj7/E", "Dm7/A", "G7/B", "Cmaj7/D"]
        for v, bass in zip(voice_chords(chords, style)["voicings"], ["E", "A", "B", "D"]):
            assert parse_note(v["notes"][0]).pitch_class == parse_note(bass).pitch_class, (style, v)
            assert v["midi"][0] < min(v["midi"][1:]), (style, v)


def test_thirteenth_chord_omits_fifth_then_root():
    v = voice_chords(["C13"], "drop2")["voicings"][0]
    assert sorted(parse_note(n).name[:-1] for n in v["notes"]) == ["A", "Bb", "D", "E"]
    assert sorted(v["degrees"]) == ["13", "3", "9", "b7"]
    assert voice_chords(["Cmaj9"], "drop2")["voicings"][0]["degrees"].count("1") == 1   # only the 5th goes
    assert "5" not in voice_chords(["Cmaj9"], "drop2")["voicings"][0]["degrees"]


def test_unknown_style_raises():
    for bad in ("drop5", "", "quartal", None, 3):
        with pytest.raises(ValueError):
            voice_chords(["C"], bad)


# ------------------------------------------------------------ the other styles

def test_style_names_are_forgiving():
    assert voice_chords(["C"], "Drop 2")["style"] == "drop2"
    assert voice_chords(["C"], "drop-2&4")["style"] == "drop24"
    assert voice_chords(["Cmaj7"], "rootless A")["style"] == "rootless_a"
    assert voice_chords(["Cmaj7"], "ROOTLESS_B")["style"] == "rootless_b"


def test_open_voicing():
    assert _notes(["Cmaj7"], "open") == [["C3", "G3", "E4", "B4"]]
    assert _notes(["Cmaj9"], "open") == [["C3", "G3", "E4", "B4", "D5"]]
    assert _notes(["C"], "open") == [["C3", "G3", "E4"]]
    assert _notes(["C5"], "open") == [["C3", "G3"]]                     # no 3rd: bass and 5th
    assert _notes(["C/G"], "open") == [["G3", "C4", "E4"]]               # a 5th in the bass is not doubled
    assert _notes(["C/D"], "open") == [["D3", "G3", "C4", "E4"]]         # a foreign bass: root and 3rd above


def test_shell_voicing():
    a, b = (voice_chords(["Cmaj7"], "shell", connect=False)["voicings"][0],
            voice_chords(["Cmaj7"], "shell", top_notes="E")["voicings"][0])
    assert (a["notes"], a["variant"]) == (["C3", "E3", "B3"], "A")
    assert (b["notes"], b["variant"]) == (["C3", "B3", "E4"], "B")
    assert _notes(["C6"], "shell", connect=False) == [["C3", "E3", "A3"]]   # the 6th replaces the 7th
    assert _notes(["C"], "shell", connect=False) == [["C3", "E3", "G3"]]    # a triad uses its 5th
    assert _notes(["C7sus4"], "shell", connect=False) == [["C3", "F3", "Bb3"]]   # the sus 4th stands in
    assert _notes(["Cdim7"], "shell", connect=False) == [["C3", "Eb3", "Bbb3"]]
    with pytest.raises(ValueError):
        voice_chords(["C5"], "shell")
    # shells keep the bass in one octave (octave-1 by letter) and choose A or B by movement
    r = voice_chords(["Dm7", "G7", "Cmaj7"], "shell")
    assert r["chords"] == [["D3", "F3", "C4"], ["G3", "B3", "F4"], ["C3", "B3", "E4"]]


def test_levine_forms_per_quality():
    def a(sym):
        return voice_chords([sym], "rootless_a")["voicings"][0]

    def b(sym):
        return voice_chords([sym], "rootless_b")["voicings"][0]

    assert a("C6")["degrees"] == ["3", "5", "6", "9"] and b("C6")["degrees"] == ["6", "9", "3", "5"]
    assert a("Cm6")["notes"] == ["Eb3", "G3", "A3", "D4"]
    assert a("CmMaj7")["degrees"] == ["b3", "5", "7", "9"]
    assert a("G7b9")["notes"] == ["B3", "E4", "F4", "Ab4"]            # the chord's own b9
    assert a("G7#9")["notes"] == ["B3", "E4", "F4", "A#4"]
    assert a("G7#5")["notes"] == ["B3", "D#4", "F4", "A4"]            # an altered 5th replaces the 13
    assert a("C7b5")["degrees"] == ["3", "b5", "b7", "9"]
    assert a("Bm7b5")["notes"] == ["D3", "F3", "A3", "B3"]            # b3 b5 b7 1: the root stays
    assert b("Bm7b5")["notes"] == ["A3", "B3", "D4", "F4"]
    assert a("C13")["notes"] == ["E3", "A3", "Bb3", "D4"]
    assert a("Ebmaj7")["notes"] == ["G3", "Bb3", "D4", "F4"]
    assert a("Db7")["notes"] == ["F3", "Bb3", "Cb4", "Eb4"]           # spelled by letters: b7 of Db is Cb
    # the bottom note is the lowest of its class at or above D(octave-1)
    assert a("Bbmaj7")["notes"][0] == "D3" and a("Amaj7")["notes"][0] == "C#4"
    assert voice_chords(["Dm7"], "rootless_a", octave=3)["chords"][0] == ["F2", "A2", "C3", "E3"]


def test_connect_false_takes_rotation_zero_and_variant_a():
    for style in ("drop2", "drop3", "drop24", "open", "close"):
        r = voice_chords(["C", "Am7", "F", "G7"], style, connect=False)
        assert [v["variant"] for v in r["voicings"]] == ["rotation 0"] * 4
    r = voice_chords(["Dm7", "G7", "Cmaj7"], "rootless", connect=False)
    assert [v["variant"] for v in r["voicings"]] == ["A", "A", "A"]
    assert _notes(["C", "Am7"], "close", connect=False) == [["C4", "E4", "G4"], ["A4", "C5", "E5", "G5"]]


def test_top_notes_melody_on_top():
    melody = ["E5", "F5", "D5", "C5"]
    for style in ("drop2", "drop3", "drop24", "close"):
        r = voice_chords(["Cmaj7", "Dm7", "G7", "C6"], style, top_notes=melody)
        assert [v["notes"][-1] for v in r["voicings"]] == melody, style
    r = voice_chords(["Cmaj7", "Dm7", "G7"], "open", top_notes=melody[:2] + ["B4"])
    assert r["chords"] == [["C4", "G4", "B4", "E5"], ["D4", "A4", "C5", "F5"], ["G3", "D4", "F4", "B4"]]
    with pytest.raises(ValueError, match="G7"):     # an open G7 has its 5th just above the bass
        voice_chords(["G7"], "open", top_notes=["D5"])
    r = voice_chords(["Dm7", "G7", "Cmaj7"], "rootless", top_notes="E A D")   # pitch classes only
    assert [parse_note(v["notes"][-1]).name[0] for v in r["voicings"]] == ["E", "A", "D"]
    r = voice_chords(["Dm7", "G7", "Cmaj7"], "rootless", top_notes=["E5", "A4", "G4"])   # exact: shifts -1..+1
    assert r["chords"] == [["F4", "A4", "C5", "E5"], ["B3", "E4", "F4", "A4"], ["B3", "D4", "E4", "G4"]]
    assert [v["variant"] for v in r["voicings"]] == ["A", "A", "B"]
    r = voice_chords(["Cmaj7"], "shell", top_notes=["B4"])
    assert r["chords"] == [["C4", "E4", "B4"]]


def test_note_arrays_are_read_as_their_chord():
    assert _notes([["D", "F", "A", "C"]], "drop2") == _notes(["Dm7"], "drop2")
    assert _notes([["C4", "E4", "G4", "Bb4"]], "rootless_a") == _notes(["C7"], "rootless_a")
    v = voice_chords([["A#", "C##", "E#", "G#"]], "drop2")["voicings"][0]   # the caller's spelling is kept
    assert v["symbol"] == "A#7" and any(n.startswith("C##") for n in v["notes"])
    with pytest.raises(ValueError, match="close"):
        voice_chords([["C", "Db", "D"]], "drop2")                    # a cluster: only 'close'
    assert voice_chords([["C", "Db", "D"]], "close")["voicings"][0]["degrees"] is None


def test_drop_styles_need_three_tones():
    with pytest.raises(ValueError):
        voice_chords(["C5"], "drop2")
    assert _notes(["C"], "drop2") == [["G3", "C4", "E4", "C5"]]      # R-3-5-R with the 5th dropped
    assert voice_chords(["C"], "drop2", top_notes="E")["voicings"][0]["degrees"] == ["1", "3", "5", "3"]


def test_string_and_list_input_agree():
    assert voice_chords("Dm7 G7 Cmaj7", "rootless") == voice_chords(["Dm7", "G7", "Cmaj7"], "rootless")
    assert voice_chords("C4maj7, G7", "drop2")["voicings"][0]["symbol"] == "Cmaj7"


@pytest.mark.parametrize("octave", [0, 1, 8, 9])
def test_register_extremes_stay_in_midi_range(octave):
    for style in STYLES:
        try:
            r = voice_chords(["Cmaj7", "A7", "Dm9", "G13", "C6"], style, octave=octave)
        except ValueError:
            continue
        for v in r["voicings"]:
            assert all(0 <= m <= 127 for m in v["midi"])


# ------------------------------------------------------------ invariants

@pytest.mark.parametrize("tonic", TONICS)
def test_fixed_rootless_forms_keep_their_shape_in_every_key(tonic):
    syms = degrees_to_chords(tonic, "major", "ii V I", sevenths=True)["symbols"]
    for style, home in (("rootless_a", [(4, 3, 4), (5, 1, 4), (3, 4, 3)]),
                        ("rootless_b", [(4, 1, 4), (4, 2, 5), (3, 2, 3)])):
        r = voice_chords(syms, style)
        assert [_gaps(v["midi"]) for v in r["voicings"]] == home, (tonic, style)
        for v in r["voicings"]:
            assert 50 <= v["midi"][0] < 62                             # bottom in D3..C#4


@pytest.mark.parametrize("tonic", TONICS)
def test_auto_rootless_picks_a_or_b_in_every_key(tonic):
    """Levine's register floor makes the A/B choice key-dependent (in A the I chord takes form B),
    but every voicing is exactly the chord's A or B form and the choice follows the cost rule."""
    syms = degrees_to_chords(tonic, "major", "ii V I", sevenths=True)["symbols"]
    r = voice_chords(syms, "rootless")
    prev = None
    for sym, v in zip(syms, r["voicings"]):
        a = voice_chords([sym], "rootless_a")["voicings"][0]["midi"]
        b = voice_chords([sym], "rootless_b")["voicings"][0]["midi"]
        assert v["midi"] == (a if v["variant"] == "A" else b)
        if prev is None:
            assert v["variant"] == "A"
        else:
            ca, cb = _voicing_cost(prev, a), _voicing_cost(prev, b)
            assert v["variant"] == ("A" if (ca, -len(set(prev) & set(a))) <= (cb, -len(set(prev) & set(b)))
                                    else "B")
        prev = v["midi"]


@pytest.mark.parametrize("scale", ["major", "harmonic minor", "melodic minor"])
@pytest.mark.parametrize("style", ["drop2", "drop3", "drop24", "open"])
def test_drop_and_open_over_diatonic_sevenths_keep_their_shape_in_every_key(scale, style):
    """The whole diatonic seventh-chord cycle (and back to I) voices to the same shapes in all keys.

    Not asserted for every scale or for 'close': an exact tie (equal cost and common tones) falls
    to voice_leading's register-anchored tie-break (no octave shift, then lower rotation), which
    can land differently by key — voice_leading itself voices the C and G major cycles differently.
    """
    shapes = set()
    for tonic in TONICS:
        syms = [c["symbol"] for c in diatonic_chords(tonic, scale, sevenths=True)["chords"]]
        r = voice_chords(syms + syms[:1], style)
        shapes.add(tuple(_gaps(v["midi"]) for v in r["voicings"]))
    assert len(shapes) == 1, (scale, style)


@pytest.mark.parametrize("prog", ["ii V I", "I vi ii V", "I IV vii iii vi ii V I"])
@pytest.mark.parametrize("style", ["drop2", "drop3", "drop24", "open", "close", "rootless_a", "rootless_b"])
def test_major_key_progressions_keep_their_shape_in_every_key(prog, style):
    shapes = set()
    for tonic in TONICS:
        syms = degrees_to_chords(tonic, "major", prog, sevenths=True)["symbols"]
        shapes.add(tuple(_gaps(v["midi"]) for v in voice_chords(syms, style)["voicings"]))
    assert len(shapes) == 1, (prog, style)


@pytest.mark.parametrize("style", STYLES)
def test_total_movement_is_the_sum_of_voice_leading_costs(style):
    for tonic in TONICS:
        syms = degrees_to_chords(tonic, "major", "I vi ii V I", sevenths=True)["symbols"]
        r = voice_chords(syms, style)
        costs = [_voicing_cost(a["midi"], b["midi"]) for a, b in zip(r["voicings"], r["voicings"][1:])]
        assert r["total_movement"] == sum(costs)
        assert r["chords"] == [v["notes"] for v in r["voicings"]]
        assert r["render_hint"]["tracks"][0]["chords"] == r["chords"]


@pytest.mark.parametrize("style", STYLES)
def test_every_chord_type_on_every_root_voices_or_raises_cleanly(style):
    """Every table chord in every key is voiced from its own spelled tones (or refused with a ValueError)."""
    for chord in CHORDS.values():
        for tonic in TONICS:
            symbol = f"{tonic}{chord.symbol}"
            try:
                v = voice_chords([symbol], style)["voicings"][0]
            except ValueError:
                assert style.startswith(("rootless", "drop", "shell")), symbol
                continue
            root, ctype, _ = parse_chord_symbol(symbol)
            names = {n.name for n in chord_notes(ctype, root)}
            names |= {transpose(root, 14, 8).name, transpose(root, 21, 12).name}   # Levine's added 9 / 13
            assert {parse_note(n).pitch_class_name for n in v["notes"]} <= names, (style, symbol, v)
            assert v["midi"] == sorted(v["midi"]) and len(set(v["midi"])) == len(v["midi"]), (style, symbol)
            if style.startswith("drop"):
                assert len(v["notes"]) == 4 and v["midi"][-1] - v["midi"][0] <= 24
            if style.startswith("rootless"):
                assert len(v["notes"]) == 4 and root.pitch_class not in {
                    parse_note(n).pitch_class for n in v["notes"]} or ctype.name == "half-diminished"
            if style == "shell":
                assert len(v["notes"]) == 3


def test_levine_intervals_are_spelled_by_letter_in_every_key():
    for tonic in TONICS:
        v = voice_chords([f"{tonic}13"], "rootless_a")["voicings"][0]
        third, thirteenth, seventh, ninth = (parse_note(n) for n in v["notes"])
        root = parse_note(tonic)
        steps = [(LETTERS.index(n.letter) - LETTERS.index(root.letter)) % 7 for n in (third, thirteenth, seventh, ninth)]
        assert steps == [2, 5, 6, 1], tonic


# ------------------------------------------------------------ chaining

def test_voicings_chain_into_the_renderers(tmp_path):
    for style in STYLES:
        r = voice_chords(["Dm7", "G7", "Cmaj7", "A7b9"], style)
        out = render_chords(r["chords"], output_dir=str(tmp_path))
        assert out["chord_count"] == 4
        song = render_arrangement(r["render_hint"]["tracks"], output_dir=str(tmp_path))
        assert song["total_beats"] == 16


def test_voicings_chain_into_check_voice_leading():
    for style in ("rootless", "drop2", "drop3", "drop24", "close"):
        r = voice_chords(["Dm7", "G7", "Cmaj7", "Fmaj7"], style)
        report = check_voice_leading(voicings=r["chords"], root="C")
        assert "violations" in report


def test_symbols_from_roman_to_chords_chain():
    from midi_composer_mcp.roman import roman_to_chords
    syms = roman_to_chords("ii7 V7 Imaj7 vi7", "Bb")["symbols"]
    assert len(voice_chords(syms, "rootless")["chords"]) == 4


# ------------------------------------------------------------ bad input

_BAD = [None, 7, -1, 3.5, 2.0, True, "", "zz", [], [None], {}, {"a": 1}, ["C", 5], "C" * 5,
        ["C", "Db", "D"], [["C", "Db", "D"], "G"], "0 4 7", ["C", ".", "E"], "C Db D", ["E9"], "H"]


def test_bad_input_only_raises_value_errors():
    good = {"chords": ["C", "G"], "style": "drop2", "octave": 4, "top_notes": None, "connect": True}
    failures = []
    for param in good:
        for bad in _BAD:
            for style in STYLES:
                args = dict(good, style=style, **{param: bad}) if param != "style" else dict(good, style=bad)
                try:
                    voice_chords(**args)
                except ValueError:
                    pass
                except Exception as e:  # noqa: BLE001 - the list of offenders is the point
                    failures.append(f"voice_chords({param}={bad!r}, style={args['style']!r}): {type(e).__name__}: {e}")
    assert not failures, "\n".join(failures[:20])
