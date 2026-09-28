"""Historical melody rules, harmony rules named after their authors, and modern techniques."""

import pytest

from midi_composer_mcp.chant import cantus_firmus, check_melody, church_mode, guido_vowel_melody, solmization
from midi_composer_mcp.counterpoint import species_counterpoint
from midi_composer_mcp.masters import (
    bach_chorale_voicing, bartok_axis, coltrane_changes, neo_riemannian, rameau_fundamental_bass,
    schoenberg_progressions,
)
from midi_composer_mcp.modern import additive_process, phase_shift, pitch_class_set, twelve_tone_matrix
from midi_composer_mcp.notes import parse_note
from midi_composer_mcp.scales import scale_info

# ------------------------------------------------------------ church modes


@pytest.mark.parametrize("mode,final,tenor,low", [
    (1, "D4", "A4", "D4"), (2, "D4", "F4", "A3"), (3, "E4", "C5", "E4"), (4, "E4", "A4", "B3"),
    (5, "F4", "C5", "F4"), (6, "F4", "A4", "C4"), (7, "G4", "D5", "G4"), (8, "G4", "C5", "D4"),
])
def test_church_modes_finals_tenors_ambitus(mode, final, tenor, low):
    m = church_mode(mode)
    assert (m["final"], m["tenor"], m["ambitus"][0]) == (final, tenor, low)
    assert len(m["notes"]) == 8                      # an octave of white keys
    assert m["type"] == ("authentic" if mode % 2 else "plagal")


def test_church_mode_names_and_numerals():
    assert church_mode("Hypomixolydian")["mode"] == 8
    assert church_mode("VI")["name"] == "Hypolydian"
    with pytest.raises(ValueError):
        church_mode(9)


# ------------------------------------------------------------ solmization

def test_solmization_mutates_on_a_pivot():
    r = solmization("C D E F G A B C5")
    assert r["syllables"] == ["ut", "re", "mi", "fa", "sol", "re", "mi", "fa"]
    assert r["mutations"][0]["note"] == "A"


def test_solmization_mutates_per_re_up_per_la_down():
    """'Per re quidem sursum mutatur, per la deorsum' (Grove, 'Solmisation')."""
    cases = {
        "C D E F G A Bb C5": "ut re mi fa re mi fa sol",
        "D5 C5 B4 A4 G4 F4 E4 D4": "sol fa mi la sol fa mi re",
        "F E D C B3 A3 G3": "fa mi re fa mi re ut",   # E (la) would follow the semitone F-E: keep it fa-mi
        "G A B C5 D5 E5 F5": "ut re mi fa re mi fa",
        "B3 C4 D4 E4 F4 G4 A4 Bb4": "mi fa re mi fa re mi fa",   # through naturale, never a direct mutation
    }
    for line, sung in cases.items():
        r = solmization(line)
        assert " ".join(r["syllables"]) == sung, (line, r["syllables"])
        assert not any("note_detail" in m for m in r["mutations"])


def test_mi_fa_is_always_the_semitone():
    for line in ("G A B C5 D5 E5 F5 G5", "F G A Bb C5 D5", "D E F G A Bb A G F E D", "Bb4 A4 G4 F4 E4 D4 C4 B3",
                 "D E F# G", "G A B C5 D5 E5 F#5 G5", "C C# D", "G Ab G", "A B C# D E"):
        r = solmization(line)
        notes = [parse_note(n if n[-1].isdigit() else n + "4") for n in r["notes"]]
        for (a, b), (sa, sb) in zip(zip(notes, notes[1:]), zip(r["syllables"], r["syllables"][1:])):
            if abs(b.midi - a.midi) == 1 and a.letter != b.letter:  # a diatonic semitone (C-C# is chromatic)
                assert {sa, sb} == {"mi", "fa"}, (line, sa, sb)


def test_solmization_ficta():
    r = solmization("D E F# G")
    assert r["syllables"] == ["ut", "re", "mi", "fa"] and r["hexachords"][2] == "ficta on D"
    assert solmization("G Ab G")["syllables"] == ["mi", "fa", "mi"]


# ------------------------------------------------------------ Guido's vowels

def test_guido_vowels_follow_the_table_and_close_on_the_final():
    r = guido_vowel_melody("Ut queant laxis resonare fibris", "Dorian")
    assert r["notes"][-1] == "D4"
    # Guido's table from gamma, in the modes' octave: the protus final D carries u, E a, F e, G i, A o
    assert r["vowel_pitches"] == {"a": ["E4", "C5"], "e": ["F4", "D5"], "i": ["G4"], "o": ["A4"], "u": ["D4", "B4"]}
    for v, note in zip(r["vowels"], r["notes"]):
        assert note in r["vowel_pitches"][v]
    wide = guido_vowel_melody("Ut queant laxis", "Dorian", rows=2)["vowel_pitches"]
    assert set(wide["i"]) == {"G4", "D4", "B4"}  # the second row (from B) adds candidates


@pytest.mark.parametrize("text,vowels", [
    ("Ut queant laxis", "ueaai"), ("famuli tuorum", "auiuou"), ("Deus", "eu"), ("filii", "iii"),
    ("Sancte Iohannes", "aeoae"), ("Linguam refrenans", "iaeea"), ("Kyrie eleison", "iieeeio"), ("gaudium", "aiu"),
])
def test_latin_syllables(text, vowels):
    from midi_composer_mcp.chant import _text_vowels
    assert "".join(_text_vowels(text)) == vowels


# ------------------------------------------------------------ cantus firmus

def test_textbook_cantus_passes():
    fux = ["D4", "F4", "E4", "D4", "G4", "F4", "A4", "G4", "F4", "E4", "D4"]
    r = check_melody(fux, "D", "dorian")
    assert r["valid"], r["violations"]


@pytest.mark.parametrize("bad,rule", [
    (["C4", "D4", "E4", "F4", "B4", "C5", "B4", "A4", "D4", "C4"], "interval"),     # F-B tritone leap
    (["C4", "D4", "D4", "E4", "F4", "E4", "D4", "C4"], "interval"),                 # repeated note
    (["C4", "A4", "B4", "G4", "F4", "E4", "D4", "C4"], "leap_recovery"),            # leap up not recovered down
    (["C4", "E4", "G4", "F4", "G4", "E4", "D4", "C4"], "climax"),                  # climax twice
    (["C4", "D4", "E4", "F4", "E4", "G4", "E4", "C4"], "final_by_step"),
])
def test_rule_breaks_are_reported(bad, rule):
    r = check_melody(bad, "C")
    assert not r["valid"]
    assert rule in {v["rule"] for v in r["violations"]}, r["violations"]


@pytest.mark.parametrize("key,scale", [("D", "dorian"), ("C", "major"), ("A", "natural minor"), ("E", "phrygian"),
                                       ("G", "mixolydian"), ("F", "lydian")])
@pytest.mark.parametrize("variant", [0, 1, 2])
def test_generated_cantus_obeys_every_rule(key, scale, variant):
    r = cantus_firmus(key, scale, 11, variant)
    assert r["check"]["valid"] and not r["check"]["warnings"]
    assert len(r["notes"]) == 11
    # and it works as input to the counterpoint engine
    cp = species_counterpoint(r["notes"], key, scale, species=1)
    assert cp["counterpoint"]


def test_cantus_variants_differ_and_repeat():
    a, b = cantus_firmus("C", "major", 10, 0)["notes"], cantus_firmus("C", "major", 10, 1)["notes"]
    assert a != b and a == cantus_firmus("C", "major", 10, 0)["notes"]
    for key, scale in (("C", "major"), ("D", "dorian")):
        assert len({tuple(cantus_firmus(key, scale, 10, v)["notes"]) for v in range(12)}) == 12


def test_long_cantus_is_fast_and_found():
    import time
    t = time.time()
    for v in range(6):
        for scale in ("major", "lydian", "harmonic minor"):
            assert cantus_firmus("C", scale, 16, v)["check"]["valid"]
    assert time.time() - t < 10


def test_modal_cadence_may_raise_the_seventh():
    ok = check_melody("A3 C4 B3 D4 C4 E4 D4 C4 B3 G#3 A3", "A", "natural minor")
    assert ok["valid"], ok["violations"]
    early = check_melody("A3 G#3 A3 C4 B3 D4 C4 B3 A3", "A", "natural minor")
    assert "diatonic" in {v["rule"] for v in early["violations"]}
    sub = check_melody("A3 C4 B3 D4 C4 E4 D4 C4 B3 G3 A3", "A", "natural minor")
    assert "subtonium" in {w["rule"] for w in sub["warnings"]}
    r = cantus_firmus("A", "natural minor", 10, 0)
    assert r["notes"][-2] in ("G#4", "G#3", "B4", "B3")


def test_same_direction_leaps_are_spelled_and_tritones_measured_between_turns():
    aug5 = check_melody("A3 B3 C4 E4 G#4 A4 F4 E4 D4 C4 B3 A3", "A", "harmonic minor")
    assert "leaps_same_direction" in {v["rule"] for v in aug5["violations"]}
    tri = check_melody("F4 G4 B4 A4 G4 F4 E4 F4", "F", "lydian")
    assert "outlined_tritone" in {w["rule"] for w in tri["warnings"]}


def test_fux_own_cantus_passes_when_not_strict():
    fux_f = "F4 G4 A4 F4 D4 E4 F4 C5 A4 F4 G4 F4"
    assert not check_melody(fux_f, "F", "lydian")["valid"]
    assert check_melody(fux_f, "F", "lydian", strict=False)["valid"]


# ------------------------------------------------------------ Rameau / Schoenberg

def test_rameau_cadences_and_fundamental_bass():
    r = rameau_fundamental_bass(["C", "F/A", "G7", "C", "G", "Am"], "C")
    assert r["fundamental_bass"] == ["C", "F", "G", "C", "G", "A"]  # roots, not the slash bass
    cad = [m.get("cadence", "") for m in r["motions"]]
    assert cad[2].startswith("cadence parfaite") and cad[4].startswith("cadence rompue")
    assert cad[0].startswith("cadence irrégulière") is False and cad[3].startswith("cadence irrégulière")


def test_schoenberg_classes():
    r = schoenberg_progressions(["C", "F", "Dm", "G", "C", "D", "G", "B"])
    classes = [p["class"] for p in r["progressions"]]
    assert classes == ["ascending (strong)", "ascending (strong)", "ascending (strong)", "ascending (strong)",
                       "superstrong", "ascending (strong)", "descending"]


# ------------------------------------------------------------ Bach chorale

def _midis(names):
    return [parse_note(n).midi for n in names]


def test_chorale_voicing_obeys_the_rules():
    r = bach_chorale_voicing(["C", "Am", "F", "G7", "C", "Dm", "G", "C"], "C")
    assert r["rule_breaks"] == []
    rows = [[parse_note(c[v]).midi for v in ("soprano", "alto", "tenor", "bass")] for c in r["chords"]]
    for s, a, t, b in rows:
        assert 60 <= s <= 79 and 55 <= a <= 74 and 48 <= t <= 67 and 40 <= b <= 60
        assert s > a > t > b and s - a <= 12 and a - t <= 12 and t - b <= 19
    for prev, cur in zip(rows, rows[1:]):
        for i in range(4):
            for j in range(i + 1, 4):
                pi, ci = (prev[i] - prev[j]) % 12, (cur[i] - cur[j]) % 12
                moving = prev[i] != cur[i] and (cur[i] - prev[i]) * (cur[j] - prev[j]) > 0
                assert not (ci in (0, 7) and pi == ci and moving), (prev, cur)
    # the G7's seventh (F) resolves down by step into the C chord
    g7, c = r["chords"][3], r["chords"][4]
    f_voice = next(v for v in ("soprano", "alto", "tenor", "bass") if g7[v].startswith("F"))
    assert parse_note(g7[f_voice]).midi - parse_note(c[f_voice]).midi in (1, 2)


def test_chorale_bass_follows_slash_chords():
    r = bach_chorale_voicing(["C", "G/B", "Am", "C/G", "F", "G", "C"], "C")
    assert [b.rstrip("0123456789") for b in r["voices"]["bass"]] == ["C", "B", "A", "G", "F", "G", "C"]


# ------------------------------------------------------------ neo-Riemannian, Bartók, Coltrane

def test_plr():
    assert neo_riemannian("C", "P")["symbols"] == ["C", "Cm"]
    assert neo_riemannian("C", "R")["symbols"] == ["C", "Am"]
    assert neo_riemannian("C", "L")["symbols"] == ["C", "Em"]
    assert neo_riemannian("C", "LRLR")["symbols"] == ["C", "Em", "G", "Bm", "D"]
    for op in "PLR":  # each operation is an involution
        assert neo_riemannian("Eb", op + op)["symbols"][-1] == "Eb"


def test_bartok_axes_cover_all_twelve_keys():
    r = bartok_axis("C")
    keys = r["tonic_axis"]["keys"] + r["subdominant_axis"]["keys"] + r["dominant_axis"]["keys"]
    assert len({parse_note(k).pitch_class for k in keys}) == 12
    assert r["tonic_axis"]["counterpole"] == "F#"


def test_coltrane_changes():
    assert coltrane_changes("C")["coltrane"] == ["Dm7", "Eb7", "Abmaj7", "B7", "Emaj7", "G7", "Cmaj7"]


# ------------------------------------------------------------ modern

def test_twelve_tone_matrix_forms():
    m = twelve_tone_matrix("E F G C# F# D# D B C A Bb G#")
    assert m["forms"]["P4"][:3] == ["E", "F", "G"]
    assert m["forms"]["I4"][:3] == ["E", "D#", "C#"]
    assert m["forms"]["R4"] == m["forms"]["P4"][::-1]
    assert len(m["forms"]) == 48
    for r in m["matrix_numbers"]:
        assert sorted(r) == list(range(12))


@pytest.mark.parametrize("notes,prime,vector", [
    ("C E G", [0, 3, 7], [0, 0, 1, 1, 1, 0]),
    ("A C E", [0, 3, 7], [0, 0, 1, 1, 1, 0]),
    ("C D E F# G# A#", [0, 2, 4, 6, 8, 10], [0, 6, 0, 6, 0, 3]),
    ("B D F Ab", [0, 3, 6, 9], [0, 0, 4, 0, 0, 2]),
])
def test_pitch_class_sets(notes, prime, vector):
    r = pitch_class_set(notes)
    assert r["prime_form"] == prime and r["interval_vector"] == vector


def test_messiaen_modes():
    assert scale_info("messiaen mode 3", "C")["intervals"] == [0, 2, 3, 4, 6, 7, 8, 10, 11]
    assert scale_info("messiaen mode 1", "C")["scale_type"] == "whole tone"
    assert pitch_class_set(scale_info("messiaen mode 3", "C")["notes"])["transpositional_symmetry"] == 3


def test_minimalist_processes():
    assert additive_process("C D E")["notes"] == ["C4", "D4", "C4", "D4", "E4"]
    ph = phase_shift("C D E F", repeats_per_stage=1)
    assert ph["voice_2"][4:8] == ["D4", "E4", "F4", "C4"] and ph["voice_1"][-4:] == ph["voice_2"][-4:]


def test_phasing_keeps_both_voices_on_the_same_pitches():
    """Every repeat and both voices must be the same concrete pitches (no octave drift)."""
    pattern = "E F# B C# D F# E C# B F# D C#"
    ph = phase_shift(pattern, repeats_per_stage=2)
    pitches = {parse_note(n).midi for n in ph["pattern"]}
    assert {parse_note(n).midi for n in ph["voice_1"] + ph["voice_2"]} == pitches
    fig = additive_process("C E G B", mode="both")
    for st in fig["stages"]:
        assert fig["notes"][st["start"]:st["start"] + st["length"]] == fig["figure"][:st["length"]]

def test_clapping_music_renders(tmp_path):
    from midi_composer_mcp.midi_io import render_arrangement
    ph = phase_shift("C C C . C C . C . C C .", repeats_per_stage=1)
    out = render_arrangement(ph["render_hint"]["tracks"], output_dir=str(tmp_path))
    assert [t["event_count"] for t in out["tracks"]] == [len(ph["voice_1"])] * 2
    assert out["total_beats"] == 13 * 12 * 0.25


def test_clapping_music_rotates_rests_with_the_pattern():
    ph = phase_shift("C C C . C C . C . C C .", repeats_per_stage=1)
    assert ph["rhythm_1"][:12] == "ooo.oo.o.oo." and ph["rhythm_2"][12:24] == "oo.oo.o.oo.o"
    assert ph["rhythm_1"][-12:] == ph["rhythm_2"][-12:] and ph["stages"] == 13
    assert len(ph["voice_1"]) == ph["rhythm_1"].count("o")


def test_numbers_as_strings():
    assert pitch_class_set("0 4 7")["prime_form"] == [0, 3, 7]
    assert pitch_class_set(["0", "3", "7"])["prime_form"] == [0, 3, 7]
    assert twelve_tone_matrix("0 1 2 3 4 5 6 7 8 9 10 11")["row_numbers"] == list(range(12))


def test_bare_mode_numbers_are_not_messiaen_aliases():
    from midi_composer_mcp.scales import resolve_scale_type
    with pytest.raises(ValueError):
        resolve_scale_type("mode 3")  # would clash with Gregorian mode III
    assert resolve_scale_type("messiaen 3").name == "messiaen mode 3"


def test_chorale_keeps_the_melody_in_the_soprano():
    tune = ["E4", "D4", "C4", "D4", "E4", "E4", "E4"]
    chords = ["C", "G", "Am", "G", "C", "Em", "C"]
    r = bach_chorale_voicing(chords, "C", melody=tune)
    assert [parse_note(n).pitch_class for n in r["voices"]["soprano"]] == [parse_note(n).pitch_class for n in tune]
    with pytest.raises(ValueError):
        bach_chorale_voicing(["C", "F"], "C", melody=["E4", "G4"])  # G is not in F major


# ------------------------------------------------------------ historical-accuracy regressions

def test_d_rooted_sevenths_keep_their_seventh():
    """C is pitch class 0: a D7's seventh must still be found (and resolved), and D9 must be voiceable."""
    r = bach_chorale_voicing(["G", "C", "D7", "G"], "G")
    d7, g = r["chords"][2], r["chords"][3]
    for voice in ("soprano", "alto", "tenor", "bass"):
        if parse_note(d7[voice]).pitch_class == 0:
            assert parse_note(d7[voice]).midi - parse_note(g[voice]).midi in (1, 2)
    assert r["rule_breaks"] == []
    bach_chorale_voicing(["C", "D9", "G"], "G")


def test_root_motion_is_read_from_the_sounding_interval():
    classes = {(p["from"], p["to"]): (p["root_motion"], p["class"])
               for p in schoenberg_progressions(["Abmaj7", "B7", "E", "Ab", "C", "G#"])["progressions"]}
    assert classes[("Abmaj7", "B7")] == ("third up", "descending")
    assert classes[("E", "Ab")] == ("third up", "descending")
    assert classes[("C", "G#")] == ("third down", "ascending (strong)")


def test_contrary_fifths_and_octaves_are_faults():
    from midi_composer_mcp.masters import _pair_parallels
    assert any("contrary" in f for f in _pair_parallels((71, 69, 63, 59), (72, 67, 64, 48)))
    r = bach_chorale_voicing(["C", "G", "Am"], "C")
    assert r["rule_breaks"] == []


def test_forced_doubling_is_reported():
    r = bach_chorale_voicing(["C", "G/B", "C"], "C", melody=["C", "B", "C"])
    assert {"chord": 2, "rule": "doubled leading tone"} in r["rule_breaks"]


def test_note_arrays_in_inversion_find_their_root():
    r = bach_chorale_voicing([["B", "D", "F", "G"], "C"], "C")
    first = r["chords"][0]
    assert parse_note(first["bass"]).pitch_class == 11
    assert parse_note(first["soprano"]).pitch_class == 5 and parse_note(r["chords"][1]["soprano"]).pitch_class == 4


def test_clusters_raise_clear_errors_or_pass():
    rameau_fundamental_bass([["C", "D", "E"], "G", "C"], "C")
    with pytest.raises(ValueError):
        neo_riemannian(["C", "Db", "D"], "P")


def test_rameau_cadences_need_the_right_motion_and_a_seven_note_key():
    assert "cadence" not in rameau_fundamental_bass(["A7", "C"], "C", "major pentatonic")["motions"][0]


def test_cycles_close_without_double_accidentals():
    assert coltrane_changes("Db")["coltrane"] == ["Ebm7", "E7", "Amaj7", "C7", "Fmaj7", "Ab7", "Dbmaj7"]
    assert neo_riemannian("C", "PLPLPL")["symbols"][-1] == "C"
    assert neo_riemannian("C", "PRPRPRPR")["symbols"][-1] == "C"
    for key in ("C#", "F#", "Gb", "Cb"):
        ax = bartok_axis(key)
        for name in ("tonic_axis", "subdominant_axis", "dominant_axis"):
            assert all("##" not in k and "bb" not in k for k in ax[name]["keys"])


def test_phase_shift_list_items_and_size_caps():
    assert phase_shift(["C E", "G", "B"], repeats_per_stage=1)["pattern"] == phase_shift("C E G B")["pattern"]
    assert phase_shift(["C E", ".", "G"])["pattern"] == ["C4", "E4", ".", "G4"]
    assert phase_shift(["C", "", "E"])["pattern"] == ["C4", "E4"]
    with pytest.raises(ValueError):
        phase_shift(" ".join(["C"] * 400), repeats_per_stage=16)
    with pytest.raises(ValueError):
        additive_process(" ".join(["C", "D"] * 500), mode="both", repeats=16)


def test_pitch_class_numbers_must_be_0_to_11():
    for row in ("1 2 3 4 5 6 7 8 9 10 11 12", list(range(-12, 0)), "0 1 2 3 4 5 6 7 8 9 10 23"):
        with pytest.raises(ValueError):
            twelve_tone_matrix(row)
    with pytest.raises(ValueError):
        pitch_class_set("99")


def test_chorale_keeps_altered_fifths_and_slash_sevenths():
    first = bach_chorale_voicing("C7b5 F")["chords"][0]
    assert "Gb" in {parse_note(first[v]).name.rstrip("0123456789") for v in ("soprano", "alto", "tenor", "bass")}
    dim7 = bach_chorale_voicing(["C", "Bdim7", "G"], root="C")["chords"][1]
    assert {parse_note(dim7[v]).pitch_class for v in ("soprano", "alto", "tenor", "bass")} == {11, 2, 5, 8}
    assert bach_chorale_voicing("Am/G D", root="G")["rule_breaks"] == \
        bach_chorale_voicing("Am7/G D", root="G")["rule_breaks"]


def test_dim7_array_is_read_by_its_spelling():
    r = bach_chorale_voicing([["D", "F", "Ab", "B"], "C"], root="C")
    assert r["rule_breaks"] == [] and parse_note(r["chords"][0]["bass"]).pitch_class == 2


def test_masters_edge_cases():
    assert "cadence" in rameau_fundamental_bass("G7#9 C", root="C")["motions"][0]
    assert bach_chorale_voicing(["Am", "E7", "Am"])["key"].startswith("A minor")
    with pytest.raises(ValueError, match="more than four voices"):
        bach_chorale_voicing("C13 G7 C")
    with pytest.raises(ValueError):
        neo_riemannian("C", ",")
    assert coltrane_changes("Fb")["original"] == ["F#m7", "B7", "Emaj7"]


def test_pivot_never_breaks_a_semitone():
    for line, sung in (("F4 E4 B3", "fa mi mi"), ("B4 C5 F5", "mi fa fa"), ("E4 F4 Bb4", "mi fa fa")):
        assert " ".join(solmization(line)["syllables"]) == sung


def test_cantus_needs_a_seven_note_scale_and_fails_fast():
    import time
    for scale in ("minor pentatonic", "blues", "hungarian major"):
        t = time.time()
        with pytest.raises(ValueError):
            cantus_firmus("C", scale, 16, 0)
        assert time.time() - t < 2


def test_ficta_hexachords_keep_every_semitone_and_the_bookkeeping():
    import random
    cases = {"B3 C4 D4 Eb4": "mi fa mi fa", "E4 F4 G4 Ab4": "mi fa mi fa",
             "C4 D4 Eb4 F4 G4 Ab4 Bb4 C5": "re mi fa re mi fa sol la", "E4 F#4 G4 A4 Bb4": "re mi fa mi fa",
             "C#4 D#4 E#4 F#4": "ut re mi fa"}
    for line, sung in cases.items():
        assert " ".join(solmization(line)["syllables"]) == sung, line
    rng = random.Random(3)
    pool = ["C", "D", "E", "F", "F#", "G", "A", "Bb", "B", "Eb", "C#", "Ab"]
    for _ in range(400):
        line = [rng.choice(pool) + str(rng.choice((3, 4))) for _ in range(rng.randint(2, 9))]
        r = solmization(line)
        notes = [parse_note(n) for n in r["notes"]]
        semis = [False] + [abs(b.midi - a.midi) == 1 and a.letter != b.letter for a, b in zip(notes, notes[1:])]
        for i in range(1, len(notes)):  # two semitones in a row cannot both be mi-fa; every other one is
            if semis[i] and not semis[i - 1] and not (i + 1 < len(notes) and semis[i + 1]):
                assert {r["syllables"][i - 1], r["syllables"][i]} == {"mi", "fa"}, (line, r["syllables"])
        for m in r["mutations"]:  # every reported mutation matches the returned hexachords
            assert r["hexachords"][m["index"]] != r["hexachords"][m["index"] - 1]
            assert m["to"].endswith(f"({r['hexachords'][m['index']]})") or m["to"] == r["hexachords"][m["index"]]
        changes = sum(1 for a, b in zip(r["hexachords"], r["hexachords"][1:]) if a != b)
        assert changes == len(r["mutations"])


def test_latin_ligatures_diaeresis_and_rows_type():
    from midi_composer_mcp.chant import _text_vowels
    assert "".join(_text_vowels("sæcula sæculórum")) == "auaauou"
    assert "".join(_text_vowels("Israël")) == "iae"
    with pytest.raises(ValueError):
        guido_vowel_melody("Ut", 1, 1.0)
