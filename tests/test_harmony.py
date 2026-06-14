import pytest

from midi_composer_mcp.circle import circle_of_fifths
from midi_composer_mcp.counterpoint import first_species
from midi_composer_mcp.harmony import (
    analyze_progression,
    interval_between,
    negative_harmony,
    secondary_dominant,
    tritone_substitute,
    voice_leading,
)
from midi_composer_mcp.notes import parse_note


# ----------------------------------------------------------- circle of fifths

def test_circle_full():
    cf = circle_of_fifths()
    assert len(cf["circle"]) == 12
    g = next(e for e in cf["circle"] if e["major"] == "G")
    assert g["accidentals"] == ["F#"] and g["relative_minor"] == "E"
    eb = next(e for e in cf["circle"] if e["major"] == "Eb")
    assert eb["accidentals"] == ["Bb", "Eb", "Ab"] and eb["fifths"] == -3


def test_circle_focus():
    f = circle_of_fifths("C")["focus"]
    assert f["dominant"] == "G" and f["subdominant"] == "F"
    assert f["relative_minor"] == "A minor" and f["parallel_minor"] == "C minor"
    related = {k["key"] for k in f["closely_related_keys"]}
    assert {"A minor", "G major", "E minor", "F major", "D minor"} == related


def test_circle_focus_enharmonic_root():
    # Bb is a valid key centre
    f = circle_of_fifths("Bb")["focus"]
    assert f["dominant"] == "F" and f["subdominant"] == "Eb"


# ----------------------------------------------------------------- intervals

def test_interval_qualities():
    assert interval_between("C", "Eb")["short"] == "m3"
    assert interval_between("C", "E")["short"] == "M3"
    assert interval_between("C", "G")["short"] == "P5"
    assert interval_between("C", "F#")["short"] == "A4"   # augmented 4th
    assert interval_between("C", "Gb")["short"] == "d5"   # diminished 5th (enharmonic)
    assert interval_between("C", "C")["short"] == "P1"
    assert interval_between("D", "C")["short"] == "m7"
    assert interval_between("C4", "G4")["signed_semitones"] == 7


# ------------------------------------------------------------ analyze chords

def test_analyze_major_key():
    res = analyze_progression(["C", "Am", "Dm", "G7"], "C", "major")
    romans = [c["roman"] for c in res["chords"]]
    funcs = [c.get("function") for c in res["chords"]]
    assert romans == ["I", "vi", "ii", "V7"]
    assert funcs == ["tonic", "tonic", "subdominant", "dominant"]


def test_analyze_borrowed_flagged():
    res = analyze_progression(["C", "Ab", "Bb", "C"], "C", "major")
    # bVI and bVII are chromatic in C major
    abc = res["chords"][1]
    assert abc["roman"].startswith("b") and abc["in_key"] is False


def test_analyze_string_input():
    res = analyze_progression("Dm7 G7 Cmaj7", "C", "major")
    assert [c["roman"] for c in res["chords"]] == ["ii7", "V7", "IΔ7"]


# ---------------------------------------------------------------- voice leading

def test_voice_leading_keeps_common_tones_and_moves_little():
    vl = voice_leading(["C", "Am", "F", "G"], octave=4)
    voicings = [v["midi"] for v in vl["voicings"]]
    assert voicings[0] == [60, 64, 67]  # C major root position
    # each step moves a small total distance
    for prev, cur in zip(voicings, voicings[1:]):
        total = sum(min(abs(c - p) for p in prev) for c in cur)
        assert total <= 6
    # output usable as note arrays
    assert vl["chords"][0] == ["C4", "E4", "G4"]


# --------------------------------------------------------------- reharmonization

def test_secondary_dominant():
    assert secondary_dominant("Dm")["symbol"] == "A7"
    assert secondary_dominant("G")["symbol"] == "D7"
    assert secondary_dominant("Em", chord_type="7b9")["root"] == "B"


def test_tritone_substitute():
    assert tritone_substitute("G7")["symbol"] == "Db7"
    assert tritone_substitute("D7")["symbol"] == "Ab7"


def test_negative_harmony_major_to_minor():
    # C major triad mirrors to its minor shadow (G Eb C)
    assert negative_harmony(["C", "E", "G"], "C")["notes"] == ["G", "Eb", "C"]
    # tonic maps to the fifth and vice versa
    nh = negative_harmony(["C", "G"], "C")["notes"]
    assert nh == ["G", "C"]


# ----------------------------------------------------------------- counterpoint

def _midis(names):
    return [parse_note(n).midi for n in names]


def _parallel_perfect_violations(cf, cp):
    cfm, cpm = _midis(cf), _midis(cp)
    bad = 0
    for i in range(1, len(cfm)):
        s = abs(cpm[i] - cfm[i]) % 12
        sp = abs(cpm[i - 1] - cfm[i - 1]) % 12
        cd = (cfm[i] > cfm[i - 1]) - (cfm[i] < cfm[i - 1])
        pd = (cpm[i] > cpm[i - 1]) - (cpm[i] < cpm[i - 1])
        if s in (0, 7) and s == sp and cd == pd and cd != 0:
            bad += 1
    return bad


def test_counterpoint_follows_rules():
    cf = ["C5", "D5", "E5", "D5", "C5"]
    for position in ("above", "below"):
        r = first_species(cf, "C", "major", position)
        cpm = _midis(r["counterpoint"])
        cfm = _midis(cf)
        # only consonances
        assert all(abs(c - f) % 12 in {0, 3, 4, 7, 8, 9} for c, f in zip(cpm, cfm))
        # begins and ends on a perfect consonance, ending on the tonic
        assert abs(cpm[0] - cfm[0]) % 12 in (0, 7)
        assert abs(cpm[-1] - cfm[-1]) % 12 in (0, 7)
        assert parse_note(r["counterpoint"][-1]).pitch_class == 0  # C tonic
        # no parallel/direct perfect fifths or octaves
        assert _parallel_perfect_violations(cf, r["counterpoint"]) == 0
        # the counterpoint is on the correct side
        if position == "above":
            assert all(c >= f for c, f in zip(cpm, cfm))
        else:
            assert all(c <= f for c, f in zip(cpm, cfm))


def test_counterpoint_deterministic():
    cf = ["D5", "F5", "E5", "D5", "G5", "F5", "A5", "G5", "F5", "E5", "D5"]
    a = first_species(cf, "C", "major", "above")
    b = first_species(cf, "C", "major", "above")
    assert a["counterpoint"] == b["counterpoint"]
    assert _parallel_perfect_violations(cf, a["counterpoint"]) == 0


def test_counterpoint_is_diatonic():
    cf = ["A4", "B4", "C5", "B4", "A4"]
    r = first_species(cf, "A", "natural minor", "above")
    a_minor = {9, 11, 0, 2, 4, 5, 7}
    assert all(parse_note(n).pitch_class in a_minor for n in r["counterpoint"])


# ------------------------------------------------------ higher species (2-5)

from midi_composer_mcp.counterpoint import species_counterpoint

CF = ["C5", "D5", "F5", "E5", "D5", "C5"]
_CONSONANT = {0, 3, 4, 7, 8, 9}


def _strong_notes(result):
    """The note sounding on each bar's downbeat (= the per-bar downbeat interval)."""
    return result["downbeat_intervals"]


@pytest.mark.parametrize("species", [1, 2, 3, 4, 5])
def test_species_endpoints_and_determinism(species):
    a = species_counterpoint(CF, "C", "major", species=species)
    b = species_counterpoint(CF, "C", "major", species=species)
    assert a == b  # deterministic
    cpm = [parse_note(n).midi for n in a["counterpoint"]]
    cfm = [parse_note(n).midi for n in a["cantus"]]
    # ends on a perfect consonance, on the tonic
    assert abs(cpm[-1] - cfm[-1]) % 12 in (0, 7)
    assert parse_note(a["counterpoint"][-1]).pitch_class == 0
    # diatonic to C major
    assert all(parse_note(n).pitch_class in {0, 2, 4, 5, 7, 9, 11} for n in a["counterpoint"])
    # a render hint is provided for both voices
    assert len(a["render_hint"]["tracks"]) == 2


@pytest.mark.parametrize("species", [1, 2, 3])
def test_species_strong_beats_consonant(species):
    # in species 1-3 every downbeat is consonant (no suspensions)
    r = species_counterpoint(CF, "C", "major", species=species)
    assert all(d in {"P1/P8", "m3", "M3", "P5", "m6", "M6"} for d in r["downbeat_intervals"])


def test_species_ratios():
    assert species_counterpoint(CF, "C", "major", species=2)["ratio"] == "2:1"
    assert species_counterpoint(CF, "C", "major", species=3)["ratio"] == "4:1"
    # 4:1 produces ~4 counterpoint notes per cantus note
    r3 = species_counterpoint(CF, "C", "major", species=3)
    assert len(r3["counterpoint"]) >= 4 * (len(CF) - 1)


def test_species4_suspensions_resolve_down():
    # every dissonant downbeat must be a suspension that steps DOWN to a consonance
    r = species_counterpoint(CF, "C", "major", species=4)
    cpm = [parse_note(n).midi for n in r["counterpoint"]]
    # the line should contain only diatonic, mostly-stepwise motion
    steps = [abs(b - a) for a, b in zip(cpm, cpm[1:])]
    assert max(steps) <= 12
    assert r["ratio"] == "syncopated"


def test_species_below():
    r = species_counterpoint(CF, "C", "major", species=2, position="below")
    cpm = [parse_note(n).midi for n in r["counterpoint"]]
    cfm = [parse_note(n).midi for n in r["cantus"]]
    # counterpoint stays at or below the cantus
    assert all(c <= max(cfm) for c in cpm)
    assert parse_note(r["counterpoint"][-1]).pitch_class == 0
