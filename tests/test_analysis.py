"""detect_key and check_voice_leading (analysis.py), and the masters refactor they share."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import itertools
import json
import os
import random
import re

import pytest

from midi_composer_mcp.analysis import PROFILES, _slice_chord, check_voice_leading, detect_key
from midi_composer_mcp.chant import cantus_firmus
from midi_composer_mcp.counterpoint import species_counterpoint
from midi_composer_mcp.diatonic import diatonic_chords
from midi_composer_mcp.harmony import voice_leading
from midi_composer_mcp.masters import (
    _VOICES, _augmented_sixth, _consecutive_perfect, _direct_perfect, _melodic_augmented, _overlap, _pair_parallels,
    _transition_faults, bach_chorale_voicing,
)
from midi_composer_mcp.melody import melodic_walk, notes_from_degrees
from midi_composer_mcp.notes import LETTERS, parse_note, transpose
from midi_composer_mcp.roman import progression_library, roman_to_chords
from midi_composer_mcp.voicing import voice_chords

TONICS = ["C", "G", "D", "A", "E", "B", "F#", "C#", "F", "Bb", "Eb", "Ab", "Db", "Gb"]
GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "golden")


def _shift(note: str, tonic: str, home: str = "C", near: bool = False) -> str:
    """Transpose a note name from the key of `home` up (or, with `near`, the nearer way) to the
    key of `tonic`, spelled by letters."""
    h, t = parse_note(home), parse_note(tonic)
    semis = (t.pitch_class - h.pitch_class) % 12
    steps = (LETTERS.index(t.letter) - LETTERS.index(h.letter)) % 7
    if near and semis > 6:
        semis, steps = semis - 12, steps - 7
    return transpose(parse_note(note), semis, steps).name


def _rows(r):
    return [(e["index"], e["voices"], e["rule"]) for e in r]


# ======================================================== bach golden regression

def _golden_module():
    spec = importlib.util.spec_from_file_location("make_bach_golden", os.path.join(GOLDEN_DIR, "make_bach_golden.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_bach_chorale_voicing_matches_the_golden_corpus_byte_for_byte():
    """Pinned on main before the masters refactor (regenerated once, on purpose, for the
    melodic-augmented-interval rule): every case's full JSON output must be unchanged."""
    gold = _golden_module()
    with open(gold.GOLDEN, encoding="utf-8") as fh:
        text = fh.read()
    stored = json.loads(text)
    assert len(stored) > 2000
    fresh = gold.build()
    mismatches = [(s["args"], s.get("error"), f.get("error")) for s, f in zip(stored, fresh) if s != f]
    assert not mismatches, mismatches[:3]
    assert gold.dump(fresh) == text


def test_golden_digest_is_the_full_output():
    gold = _golden_module()
    case = next(c for c in json.load(open(gold.GOLDEN, encoding="utf-8")) if "sha256" in c)
    out = bach_chorale_voicing(**case["args"])
    assert hashlib.sha256(gold.canonical(out).encode("utf-8")).hexdigest() == case["sha256"]


# frozen copies of the pre-refactor masters code (commit 6e86e27), the oracle for the predicates

def _old_pair_parallels(prev, cur):
    bad = []
    for i in range(4):
        for j in range(i + 1, 4):
            p_int = (prev[i] - prev[j]) % 12
            c_int = (cur[i] - cur[j]) % 12
            if c_int not in (0, 7) or p_int != c_int:
                continue
            what = "octaves" if c_int == 0 else "fifths"
            motion = (cur[i] - prev[i]) * (cur[j] - prev[j])
            if motion > 0:
                bad.append(f"parallel {what} {_VOICES[i]}/{_VOICES[j]}")
            elif motion < 0:
                bad.append(f"consecutive {what} by contrary motion {_VOICES[i]}/{_VOICES[j]}")
    return bad


def _old_direct_and_overlap(prev, cur):
    faults = []
    outer = (cur[0] - cur[3]) % 12
    if (outer in (0, 7) and (prev[0] - prev[3]) % 12 != outer
            and (cur[0] - prev[0]) * (cur[3] - prev[3]) > 0 and abs(cur[0] - prev[0]) > 2):
        faults.append("direct fifth/octave in the outer voices with a soprano leap")
    for i in range(3):
        if cur[i] < prev[i + 1] or cur[i + 1] > prev[i]:
            faults.append(f"overlap {_VOICES[i]}/{_VOICES[i + 1]}")
    return faults


def _satb(rng):
    b = rng.randint(40, 60)
    t = rng.randint(b + 1, b + 19)
    a = rng.randint(t + 1, t + 12)
    return (rng.randint(a + 1, a + 12), a, t, b)


def test_refactored_predicates_agree_with_the_old_inline_rules():
    rng = random.Random(7)
    for _ in range(20000):
        prev, cur = _satb(rng), _satb(rng)
        if rng.random() < 0.3:  # many near-identical moves, where the rules bite
            cur = tuple(p + rng.choice((-2, -1, 0, 0, 1, 2, 7, -7, 12)) for p in prev)
            if not cur[0] > cur[1] > cur[2] > cur[3]:
                continue
        assert _pair_parallels(prev, cur) == _old_pair_parallels(prev, cur)
        faults, _cost = _transition_faults(prev, cur, -1, -1, None)   # (no key: the tonal rules stay out)
        assert faults == _old_pair_parallels(prev, cur) + _old_direct_and_overlap(prev, cur)


def test_masters_predicates():
    assert _consecutive_perfect(72, 65, 74, 67) == ("parallel", 7)
    assert _consecutive_perfect(67, 60, 74, 55) == ("contrary", 7)
    assert _consecutive_perfect(84, 65, 86, 67) == ("parallel", 7)       # compound fifths count
    assert _consecutive_perfect(72, 65, 72, 65) is None                  # static
    assert _consecutive_perfect(72, 60, 74, 62) == ("parallel", 0)
    assert _direct_perfect(64, 48, 72, 53)                                # E4/C3 -> C5/F3, soprano leaps
    assert not _direct_perfect(71, 48, 72, 53)                            # B4/C3 -> C5/F3, soprano steps
    assert not _direct_perfect(72, 53, 74, 55)                            # a true parallel is not "direct"
    assert _overlap(64, 60, 59, 57) and _overlap(64, 60, 67, 65) and not _overlap(64, 60, 65, 62)


# ================================================================== detect_key

def _top(r, n=4):
    return [(x["root"], x["scale_type"], x["correlation"]) for x in r["ranking"][:n]]


def test_c_major_scale_matches_music21():
    r = detect_key(notes="C4 D4 E4 F4 G4 A4 B4 C5")
    assert (r["root"], r["scale_type"], r["mode"]) == ("C", "major", "major")
    assert r["correlation"] == 0.9014 and r["certainty"] == 1.1914
    assert _top(r) == [("C", "major", 0.9014), ("A", "natural minor", 0.7563), ("F", "major", 0.6125),
                       ("G", "major", 0.5998)]
    assert len(r["ranking"]) == 24 and r["profile"] == "krumhansl"
    assert r["histogram"] == {"C": 2.0, "D": 1.0, "E": 1.0, "F": 1.0, "G": 1.0, "A": 1.0, "B": 1.0}
    assert "regions" not in r
    assert detect_key(notes="C4 D4 E4 F4 G4 A4 B4 C5", profile="temperley")["correlation"] == 0.9163


def test_harmonic_minor_line_and_profiles():
    for profile, r_expected in (("krumhansl", 0.8466), ("temperley", 0.8979)):
        r = detect_key(notes="A B C D E F G# A", profile=profile)
        assert (r["root"], r["scale_type"], r["correlation"]) == ("A", "natural minor", r_expected)


def test_chords_are_read_as_their_tones():
    r = detect_key(chords="C Am F G")
    assert (r["root"], r["scale_type"], r["correlation"]) == ("C", "major", 0.9348)
    r = detect_key(chords="Am Dm E7 Am")
    assert (r["root"], r["scale_type"], round(r["correlation"], 3)) == ("A", "natural minor", 0.874)
    # a slash bass sounds as an extra note, as it renders
    assert detect_key(chords=["C/E"])["histogram"] == {"C": 4.0, "E": 8.0, "G": 4.0}


def test_aarden_reads_the_c_major_scale_as_a_minor():
    r = detect_key(notes="C4 D4 E4 F4 G4 A4 B4 C5", profile="aarden")
    assert _top(r, 2) == [("A", "natural minor", 0.8244), ("C", "major", 0.8181)]


def test_the_tonic_keeps_its_written_spelling():
    r = detect_key(notes="Gb Ab Bb Cb Db Eb F Gb")
    assert (r["root"], r["scale_type"]) == ("Gb", "major")
    assert r["ranking"][1]["root"] == "Eb" and "Cb" in r["histogram"]
    r = detect_key(notes="F# G# A# B C# D# E# F#")
    assert r["root"] == "F#"


def test_unwritten_tonics_take_the_smaller_signature():
    from midi_composer_mcp.analysis import _spell_tonic
    assert _spell_tonic(6, "major", {}).name == "F#" and _spell_tonic(3, "minor", {}).name == "Eb"
    assert _spell_tonic(1, "major", {}).name == "Db" and _spell_tonic(8, "minor", {}).name == "G#"
    assert _spell_tonic(10, "minor", {}).name == "Bb" and _spell_tonic(11, "major", {}).name == "B"
    # a written but theoretical key is made practical (A# major -> Bb major)
    assert _spell_tonic(10, "major", {10: parse_note("A#")}).name == "Bb"


def test_rhythm_durations_equal_explicit_durations():
    by_rhythm = detect_key(notes="C D E G", rhythm="O.o..o.O", step_beats=0.5)
    explicit = detect_key(notes="C D E G", durations=[1.0, 1.5, 1.0, 0.5])
    assert by_rhythm == explicit
    # onsets take the notes cyclically, as a notes track does
    assert detect_key(notes="C E", rhythm="oooo")["histogram"] == {"C": 2.0, "E": 2.0}
    # simultaneous notes share one slot
    assert detect_key(notes=[["C4", "E4", "G4"], "D4"], durations=[3, 1])["histogram"] == \
        {"C": 3.0, "D": 1.0, "E": 3.0, "G": 3.0}


def test_windows_find_the_modulation():
    c_part = "C4 D4 E4 F4 G4 E4 D4 C4"          # 8 beats in C major
    g_part = "G4 A4 B4 D5 F#4 A4 G4 G4"          # 8 beats in G major, with F#
    for hop in (0.0, 8.0):
        r = detect_key(notes=f"{c_part} {g_part}", window_beats=8, hop_beats=hop)
        assert [(x["root"], x["scale_type"]) for x in r["regions"]] == [("C", "major"), ("G", "major")]
        assert r["regions"][0]["start_beat"] == 0.0 and r["regions"][-1]["end_beat"] == 16.0
    exact = detect_key(notes=f"{c_part} {g_part}", window_beats=8, hop_beats=8)["regions"]
    assert (exact[0]["end_beat"], exact[1]["start_beat"]) == (8.0, 8.0)
    # overlapping windows: the change sits midway through the overlap of the two windows that disagree
    halves = detect_key(notes=f"{c_part} {g_part}", window_beats=8)["regions"]
    assert (halves[0]["end_beat"], halves[1]["start_beat"]) == (6.0, 6.0)
    # whole windows only, so the tail is not read from a fragment (the README example)
    song = detect_key(chords="C Am F G C Am F G G Em C D G Em C D", window_beats=16, hop_beats=16)["regions"]
    assert [(x["start_beat"], x["end_beat"], x["root"], x["scale_type"], x["mean_correlation"]) for x in song] == \
        [(0.0, 32.0, "C", "major", 0.9348), (32.0, 64.0, "G", "major", 0.9348)]
    assert len(detect_key(chords="C Am F G C Am F G G Em C D G Em C D", window_beats=16)["regions"]) == 2
    # a window longer than the music is the whole piece
    whole = detect_key(chords="C G", window_beats=100)
    assert whole["regions"] == [{"start_beat": 0.0, "end_beat": 8.0, "root": whole["root"],
                                 "scale_type": whole["scale_type"], "mean_correlation": whole["correlation"]}]


def test_silent_windows_copy_their_neighbours():
    track = {"type": "notes", "notes": ["C4", "E4", "G4", "C5"], "rhythm": "....oooo", "step_beats": 1}
    r = detect_key(tracks=[track], window_beats=2, hop_beats=2)
    assert [x["root"] for x in r["regions"]] == [r["root"]] and r["regions"][0]["start_beat"] == 0.0


def test_tracks_are_timed_as_they_render_and_drums_ignored():
    tracks = [{"type": "chords", "chords": ["Am", "Dm", "E7", "Am"]},
              {"type": "notes", "notes": ["A2", "D3", "E3", "A2"], "step_beats": 4},
              {"type": "drums", "lanes": {"kick": "O...O..."}}]
    r = detect_key(tracks=tracks)
    assert (r["root"], r["scale_type"]) == ("A", "natural minor")
    hint = bach_chorale_voicing("C F G7 C", root="C")["render_hint"]["tracks"]
    assert detect_key(tracks=hint)["root"] == "C"


def test_every_profile_is_music21s():
    expected = {
        "krumhansl": ([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
                      [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]),
        "temperley": ([0.748, 0.060, 0.488, 0.082, 0.670, 0.460, 0.096, 0.715, 0.104, 0.366, 0.057, 0.400],
                      [0.712, 0.084, 0.474, 0.618, 0.049, 0.460, 0.105, 0.747, 0.404, 0.067, 0.133, 0.330]),
        "bellman": ([16.80, 0.86, 12.95, 1.41, 13.49, 11.93, 1.25, 20.28, 1.80, 8.04, 0.62, 10.57],
                    [18.16, 0.69, 12.99, 13.34, 1.07, 11.15, 1.38, 21.07, 7.49, 1.53, 0.92, 10.21]),
        "aarden": ([17.7661, 0.145624, 14.9265, 0.160186, 19.8049, 11.3587, 0.291248, 22.062, 0.145624,
                    8.15494, 0.232998, 4.95122],
                   [18.2648, 0.737619, 14.0499, 16.8599, 0.702494, 14.4362, 0.702494, 18.6161, 4.56621,
                    1.93186, 7.37619, 1.75623]),
        "simple": ([2, 0, 1, 0, 1, 1, 0, 2, 0, 1, 0, 1], [2, 0, 1, 1, 0, 1, 0, 2, 1, 0, 0.5, 0.5]),
    }
    assert set(PROFILES) == set(expected)
    for name, (major, minor) in expected.items():
        assert list(PROFILES[name]["major"]) == major and list(PROFILES[name]["minor"]) == minor
        assert len(major) == len(minor) == 12


@pytest.mark.parametrize("kwargs", [
    {},
    {"tracks": [{"type": "notes", "notes": ["C4"]}], "notes": "C4"},
    {"tracks": [{"type": "notes", "notes": ["C4"]}], "chords": "C"},
    {"notes": "C C# D D# E F F# G G# A A# B"},
    {"notes": "C4 E4", "profile": "zz"},
    {"notes": "C4 E4 G4", "durations": [1, 2]},
    {"notes": "C4 E4 G4", "durations": [1, 2, 0]},
    {"notes": "C4 E4 G4", "durations": [1, 1, 1], "rhythm": "ooo"},
    {"chords": "C", "durations": [1]},
    {"notes": "C4", "rhythm": "...."},
    {"notes": "C4", "hop_beats": 2},
    {"notes": "C4 D4", "durations": [5000, 5000], "window_beats": 10, "hop_beats": 0.5},
    {"tracks": [{"type": "drums", "lanes": {"kick": "O..."}}]},
])
def test_detect_key_rejects_bad_input(kwargs):
    with pytest.raises(ValueError):
        detect_key(**kwargs)


_KEY_INPUT = ["C4", "E4", "G4", "F4", "D4", "B3", "C4", "A4", "G4", "F#4", "G4", "E4", "C4"]
_KEY_DURATIONS = [2, 1, 1, 1, 0.5, 0.5, 2, 1, 1, 0.5, 1.5, 1, 3]


@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_detect_key_is_transposition_equivariant(profile):
    ref = detect_key(notes=_KEY_INPUT, durations=_KEY_DURATIONS, profile=profile)
    ref_r = [x["correlation"] for x in ref["ranking"]]
    # exact ties (the 'simple' profile has some) are ordered by pitch class, which does not rotate
    ref_rel = sorted((x["correlation"], parse_note(x["root"]).pitch_class, x["scale_type"]) for x in ref["ranking"])
    for tonic in TONICS:
        k = parse_note(tonic).pitch_class
        moved = [_shift(n, tonic) for n in _KEY_INPUT]
        r = detect_key(notes=moved, durations=_KEY_DURATIONS, profile=profile)
        assert [x["correlation"] for x in r["ranking"]] == ref_r
        assert sorted((x["correlation"], (parse_note(x["root"]).pitch_class - k) % 12, x["scale_type"])
                      for x in r["ranking"]) == ref_rel
        assert r["certainty"] == ref["certainty"]
        written = {parse_note(n).pitch_class: parse_note(n).name.rstrip("0123456789") for n in moved}
        assert r["root"] == written.get(parse_note(r["root"]).pitch_class, r["root"])
        assert r["root"] == _shift(ref["root"] + "4", tonic)[:-1]


@pytest.mark.parametrize("tonic", TONICS)
def test_detect_key_chains_into_diatonic_chords(tonic):
    for scale, degrees in (("major", [1, 2, 3, 4, 5, 6, 7, 8, 5, 1]), ("harmonic minor", [1, 3, 5, 4, 7, 8, 5, 1])):
        line = notes_from_degrees(tonic + "4", scale, degrees)["notes"]
        r = detect_key(notes=line)
        assert parse_note(r["root"]).pitch_class == parse_note(tonic).pitch_class
        assert r["scale_type"] == ("major" if scale == "major" else "natural minor")
        chords = diatonic_chords(r["root"], r["scale_type"])["chords"]
        assert chords[0]["root"] == r["root"]


def test_detect_key_reads_random_generator_output():
    from midi_composer_mcp.scales import scale_info
    walk = melodic_walk(scale_info("major", "D4")["notes"], length=24, seed=3)["notes"]
    assert detect_key(notes=walk)["ranking"][0]["correlation"] > 0


# ======================================================== check_voice_leading

def test_parallel_fifths_example():
    r = check_voice_leading(voices=[["C5", "D5"], ["F4", "G4"]])
    assert r["valid"] is False
    assert r["violations"] == [{"index": 1, "voices": [0, 1], "rule": "parallel_fifths",
                                "message": "C5/F4 -> D5/G4: parallel perfect fifths"}]
    assert r["motion"] == {"0-1": {"parallel": 1, "similar": 0, "contrary": 0, "oblique": 0, "static": 0}}
    assert r["dissonances"] == [] and r["voice_order"] == ["voice 0", "voice 1"]


def test_bach_chorale_output_is_clean():
    tracks = bach_chorale_voicing("C F G7 C", root="C")["render_hint"]["tracks"]
    r = check_voice_leading(voices=tracks, root="C")
    assert r["valid"] is True and r["voice_order"] == ["soprano", "alto", "tenor", "bass"]
    assert r["key"] == "C major" and [s["beat"] for s in r["slices"]] == [0.0, 2.0, 4.0, 6.0]


def test_contrary_and_compound_fifths():
    assert _rows(check_voice_leading(voices=[["G4", "D5"], ["C4", "G3"]])["violations"]) == [(1, [0, 1], "contrary_fifths")]
    assert _rows(check_voice_leading(voices=[["C6", "D6"], ["F4", "G4"]])["violations"]) == [(1, [0, 1], "parallel_fifths")]
    octaves = check_voice_leading(voices=[["C5", "D5"], ["C4", "D4"]])
    assert _rows(octaves["violations"]) == [(1, [0, 1], "parallel_octaves")]
    assert "perfect octaves" in octaves["violations"][0]["message"]


def test_unequal_fifths():
    clean = check_voice_leading(voices=[["G4", "F4"], ["C4", "B3"]])    # P5 -> d5
    assert clean["valid"] and clean["warnings"] == []
    r = check_voice_leading(voices=[["F4", "G4"], ["B3", "C4"]])        # d5 -> P5 against the bass
    assert r["valid"] and _rows(r["warnings"]) == [(1, [0, 1], "unequal_fifths")]


def test_crossing_and_overlap():
    r = check_voice_leading(voices=[["E4", "C4"], ["C4", "F4"]])
    assert sorted(_rows(r["violations"])) == [(1, [0, 1], "voice_crossing"), (1, [0, 1], "voice_overlap")]
    r = check_voice_leading(voices=[["E4", "B3"], ["C4", "A3"]])        # the upper voice dips below C4
    assert _rows(r["violations"]) == [(1, [0, 1], "voice_overlap")]


def test_melodic_augmented_intervals():
    assert check_voice_leading(voices=[["C5", "C#5"], ["A3", "A3"]])["valid"]           # chromatic A1
    r = check_voice_leading(voices=[["F4", "G#4"], ["D3", "E3"]])
    assert _rows(r["violations"]) == [(1, [0], "melodic_augmented")]
    r = check_voice_leading(voices=[["F4", "B4"], ["D3", "D3"]])
    assert _rows(r["violations"]) == [(1, [0], "melodic_augmented")]                   # the tritone A4
    r = check_voice_leading(voices=[["C4", "E5"], ["C3", "C3"]])
    assert _rows(r["violations"]) == [(1, [0], "melodic_leap_beyond_octave")]


def test_seventh_resolution():
    rise = check_voice_leading(voices=[["D5", "E5"], ["F4", "G4"], ["B3", "C4"], ["G2", "C3"]], root="C")
    assert _rows(rise["violations"]) == [(1, [1], "seventh_resolution")]
    held = check_voice_leading(voices=[["D5", "D5"], ["F4", "F4"], ["B3", "A3"], ["G2", "D3"]], root="C")  # G7 -> Dm
    assert held["valid"] and (1, [1], "seventh_resolution") in _rows(held["warnings"])
    good = check_voice_leading(voices=[["D5", "E5"], ["F4", "E4"], ["B3", "C4"], ["G2", "C3"]], root="C")
    assert good["valid"]
    # a G7 without its fifth is still read as G7
    no_fifth = check_voice_leading(voices=[["G4", "G4"], ["F4", "G4"], ["B3", "C4"], ["G2", "C3"]], root="C")
    assert (1, [1], "seventh_resolution") in _rows(no_fifth["violations"])


def test_leading_tone():
    sop = check_voice_leading(voices=[["B4", "G4"], ["G4", "E4"], ["D4", "C4"], ["G2", "C3"]], root="C")
    assert _rows(sop["violations"]) == [(1, [0], "leading_tone")]
    alto = check_voice_leading(voices=[["D5", "C5"], ["B4", "G4"], ["G4", "E4"], ["G3", "C3"]], root="C")
    assert alto["valid"] and _rows(alto["warnings"]) == [(1, [1], "leading_tone")]
    assert "frustrated leading tone" in alto["warnings"][0]["message"]
    minor = check_voice_leading(voices=[["G#4", "E4"], ["E4", "C4"], ["B3", "A3"], ["E3", "A3"]],
                                root="A", scale_type="harmonic minor")
    assert (1, [0], "leading_tone") in _rows(minor["violations"])


def test_doubled_leading_tone():
    r = check_voice_leading(voices=[["B4"], ["G4"], ["B3"], ["G2"]], root="C")
    assert _rows(r["violations"]) == [(0, [0, 2], "doubled_leading_tone")]


def test_warnings_spacing_range_leaps_and_enharmonic_fifths():
    r = check_voice_leading(voices=[["C6", "B5", "C6"], ["E4", "D4", "E4"], ["G3", "G3", "G3"], ["C3", "G2", "C3"]])
    rules = {w["rule"] for w in r["warnings"]}
    assert {"spacing", "range"} <= rules
    r = check_voice_leading(voices=[["E4", "C5", "D5"], ["C4", "A3", "G3"]])
    assert _rows(r["warnings"]) == [(1, [0], "leap_not_recovered")]
    assert check_voice_leading(voices=[["E4", "C5", "B4"], ["C4", "A3", "G3"]])["warnings"] == []   # stepped back
    r = check_voice_leading(voices=[["D4", "C5", "B4"], ["Bb3", "A3", "G3"]])
    assert _rows(r["warnings"]) == [(1, [0], "melodic_seventh")] and r["valid"]
    r = check_voice_leading(voices=[["C5"], ["E#4"]])
    assert _rows(r["warnings"]) == [(0, [0, 1], "enharmonic_fifth")]


def test_dissonances_are_spelled_and_include_the_fourth():
    r = check_voice_leading(voices=[["F4", "E4", "C5"], ["C4", "C4", "E#4"]])
    assert [(d["index"], d["interval"]) for d in r["dissonances"]] == [(0, "P4"), (2, "d6")]


def test_tracks_align_on_every_attack_with_held_and_silent_voices():
    melody = {"type": "notes", "name": "melody", "notes": ["E5", "F5", "G5", "E5"], "step_beats": 1}
    bass = {"type": "notes", "name": "bass", "notes": ["C3", "G2"], "rhythm": "O.O.", "step_beats": 1, "sustain": True}
    gap = {"type": "notes", "name": "inner", "notes": ["G4"], "rhythm": "o...", "step_beats": 1}
    r = check_voice_leading(voices=[bass, gap, melody])
    assert r["voice_order"] == ["melody", "inner", "bass"] and r["track_order"] == [2, 1, 0]
    assert [s["notes"] for s in r["slices"]] == [["E5", "G4", "C3"], ["F5", None, "C3"],
                                                ["G5", None, "G2"], ["E5", None, "G2"]]
    assert r["motion"]["0-2"] == {"parallel": 0, "similar": 0, "contrary": 1, "oblique": 2, "static": 0}
    assert r["motion"]["0-1"] == {"parallel": 0, "similar": 0, "contrary": 0, "oblique": 0, "static": 0}


def test_voicings_input():
    vl = voice_leading(["C", "G", "Am", "F"])["chords"]
    r = check_voice_leading(voicings=vl)
    assert len(r["slices"]) == 4 and r["voice_order"] == ["voice 0", "voice 1", "voice 2"]
    assert r["slices"][0]["notes"] == sorted(vl[0], key=lambda n: -parse_note(n).midi)
    assert check_voice_leading(voicings=[["C4", "E4", "G4"], ["D4", "F4", "A4"]])["violations"]  # parallel triads


# ------------------------------------------------ regression tests (review findings)

def test_augmented_sixths_are_not_dominant_sevenths():
    """Finding 11/41: the #4 of It6/Ger65 (F# over Ab) rises; it is no seventh of Ab7 that must fall."""
    ger = check_voice_leading(voicings=[["C5", "F#4", "Eb4", "Ab3"], ["C5", "G4", "E4", "G3"]], root="C")
    it6 = check_voice_leading(voicings=[["C5", "F#4", "C4", "Ab2"], ["B4", "G4", "D4", "G2"]], root="C")
    fr = check_voice_leading(voicings=[["F#4", "D4", "C4", "Ab2"], ["G4", "E4", "C4", "G2"]], root="C")
    two = check_voice_leading(voices=[["F#4", "G4"], ["C4", "B3"], ["Ab2", "G2"]], root="C")
    for r in (ger, it6, fr, two):
        assert r["valid"] and r["warnings"] == [], r
    # the wrong way round: the #4 falls (bach_chorale_voicing used to write F#4 -> E4)
    wrong = check_voice_leading(voices=[["G4", "A4", "F#4", "E4"], ["C4", "C4", "C4", "C4"],
                                        ["E3", "F3", "Eb3", "G3"], ["C3", "F2", "Ab2", "G2"]], root="C")
    assert _rows(wrong["violations"]) == [(3, [0], "augmented_sixth")]
    assert "does not rise a semitone" in wrong["violations"][0]["message"]
    falls = check_voice_leading(voicings=[["C5", "F#4", "Eb4", "Ab3"], ["C5", "G4", "E4", "Bb3"]], root="C")
    assert (1, [3], "augmented_sixth") in _rows(falls["violations"])       # b6 must fall to ^5
    # a genuinely spelled Ab7 still resolves its seventh (Gb) down
    ab7 = check_voice_leading(voicings=[["Gb4", "Eb4", "C4", "Ab2"], ["G4", "F4", "Db4", "Db3"]], root="Db")
    assert _rows(ab7["violations"]) == [(1, [0], "seventh_resolution")]
    # C7b5 with C in the bass is a V7b5 of F, not a French sixth of Bb
    v7b5 = check_voice_leading(voicings=[["Bb4", "Gb4", "E4", "C3"], ["A4", "F4", "C4", "F3"]], root="F")
    assert v7b5["valid"] and _rows(v7b5["warnings"]) == [(1, [2], "leading_tone")]


def test_readme_augmented_sixth_chain_is_clean():
    sy = roman_to_chords("i iv N6 Ger65 Cad64 V7 i", "D", "natural minor")["symbols"]
    b = bach_chorale_voicing(sy, root="D", scale_type="harmonic minor")
    assert b["rule_breaks"] == []
    assert b["voices"]["alto"][3:5] == ["G#4", "A4"] and b["voices"]["bass"][3:5] == ["Bb3", "A3"]
    r = check_voice_leading(voices=b["render_hint"]["tracks"], root="D", scale_type="harmonic minor")
    assert r["valid"], r["violations"]
    major = bach_chorale_voicing(roman_to_chords("I IV Ger65 Cad64 V7 I", "C")["symbols"], root="C")
    assert major["voices"]["soprano"][2:4] == ["F#4", "G4"] and major["rule_breaks"] == []


_AUG6_MAJOR = ["I IV It6 V I", "I IV Ger65 Cad64 V7 I", "I ii6 Fr43 V I", "I IV Sw43 Cad64 V I", "I vi It6 V7 I"]
_AUG6_MINOR = ["i iv It6 V i", "i iv Ger65 Cad64 V7 i", "i ii°6 Fr43 V i", "i iv N6 Ger65 Cad64 V7 i",
               "i VI Ger65 V i"]


@pytest.mark.parametrize("tonic", TONICS)
def test_augmented_sixths_resolve_outward_in_every_key(tonic):
    for scale, progressions in (("major", _AUG6_MAJOR), ("harmonic minor", _AUG6_MINOR)):
        for numerals in progressions:
            sy = roman_to_chords(numerals, tonic, scale)["symbols"]
            b = bach_chorale_voicing(sy, root=tonic, scale_type=scale)
            assert b["rule_breaks"] == [], (tonic, numerals, b["rule_breaks"])
            r = check_voice_leading(voices=b["render_hint"]["tracks"], root=tonic, scale_type=scale)
            assert r["valid"], (tonic, numerals, r["violations"])
            k = next(i for i, s in enumerate(sy) if isinstance(s, list))
            here = [parse_note(b["chords"][k][v]) for v in _VOICES]
            there = [parse_note(b["chords"][k + 1][v]) for v in _VOICES]
            low, high = _augmented_sixth(here, here[3])
            for a, c in zip(here, there):          # #4 up a semitone, b6 down a semitone
                if a.without_octave() == high:
                    assert c.midi - a.midi == 1, (tonic, numerals, a.name, c.name)
                if a.without_octave() == low:
                    assert c.midi - a.midi == -1, (tonic, numerals, a.name, c.name)


def test_two_voice_dyads_are_not_power_or_sus_chords():
    """Finding 12: a bare octave, a second or an open fifth is no chord for the key rules."""
    assert _slice_chord([parse_note("B3"), parse_note("B4")]) is None
    assert _slice_chord([parse_note("B3"), parse_note("A4")]) is None
    assert _slice_chord([parse_note("G3"), parse_note("F4")]) is None
    root, ctype = _slice_chord([parse_note("B3"), parse_note("D4")], {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7,
                                                                       "A": 9, "B": 11})
    assert (root.name, ctype.name) == ("B", "diminished")                 # the key's fifth: vii°
    root, ctype = _slice_chord([parse_note("G2"), parse_note("B3"), parse_note("F4")])
    assert (root.name, ctype.name) == ("G", "dominant 7")                 # an omitted fifth still reads
    r = check_voice_leading(voices=[["G4", "A4", "G4"], ["B3", "B3", "C4"]], root="C")
    assert r["valid"]                                                     # a neighbour over a held B
    cf = cantus_firmus("C", "major", 10, 2)["notes"]
    two = species_counterpoint(cf, "C", "major", 2, "above")
    rules = {v["rule"] for v in check_voice_leading(voices=two["render_hint"]["tracks"], root="C")["violations"]}
    assert "doubled_leading_tone" not in rules and "leading_tone" not in rules
    held = check_voice_leading(voices=[["B4", "B4"], ["G4", "G4"], ["D4", "E4"], ["G2", "C3"]], root="C")
    assert held["valid"]                                                  # V -> Imaj7 keeps B
    # a bare octave or open fifth on the TONIC still takes the leading tone (Fux's cadence)
    assert check_voice_leading(voices=[["B4", "C5"], ["D4", "C4"]], root="C")["valid"]
    octave = check_voice_leading(voices=[["D5", "C5"], ["B3", "C3"]], root="C")
    assert _rows(octave["violations"]) == [(1, [1], "leading_tone")]
    assert "tonic of C (octave)" in octave["violations"][0]["message"]
    assert check_voice_leading(voices=[["B4", "A4"], ["D4", "A3"]], root="C")["valid"]   # an A octave is no vi


@pytest.mark.parametrize("tonic", TONICS)
def test_species_counterpoint_key_rules_read_real_chords(tonic):
    for variant, species, position in itertools.product(range(2), range(1, 6), ("above", "below")):
        cf = cantus_firmus(tonic, "major", 10, variant)["notes"]
        r = species_counterpoint(cf, tonic, "major", species, position)
        if r.get("rules_broken") or "warning" in r:
            continue
        for v in check_voice_leading(voices=r["render_hint"]["tracks"], root=tonic)["violations"]:
            assert v["rule"] != "doubled_leading_tone", (tonic, variant, species, position, v)
            if v["rule"] == "leading_tone":   # V/vii° read as real triads; a bare arrival only on ^1
                before, after = re.search(r"leading tone of (\S+) does not rise to the tonic of (.+)$",
                                          v["message"]).groups()
                assert not re.fullmatch(r"[A-G][#b]*(5|sus\d)", before), v
                if re.fullmatch(r"[A-G][#b]*(5|sus\d| \(octave\))", after):
                    assert after.startswith(tonic) and after[len(tonic):len(tonic) + 1] in ("5", "s", " "), v


def test_passing_leading_tone_in_a_descending_bass():
    """Finding 13/56: I V6 vi (bass 1-7-6, the Romanesca) is no leading-tone fault."""
    r = check_voice_leading(voices=[["G4", "G4", "A4"], ["E4", "D4", "C4"], ["G3", "G3", "E3"], ["C3", "B2", "A2"]],
                            root="C")
    assert r["valid"] and r["warnings"] == []
    for chords in ("C G/B F/A G", "C G/B Am G"):                          # the same bass under IV6 and vi
        b = bach_chorale_voicing(chords, root="C")
        assert b["voices"]["bass"][:3] == ["C3", "B2", "A2"] and b["rule_breaks"] == []
        assert check_voice_leading(voices=b["render_hint"]["tracks"], root="C")["valid"], chords
    leap = check_voice_leading(voices=[["G4", "G4", "A4"], ["E4", "D4", "C4"], ["B3", "G3", "E3"],
                                       ["E3", "B2", "A2"]], root="C")
    assert (2, [3], "leading_tone") in _rows(leap["violations"])          # ^7 leapt into: no passing tone
    sop = check_voice_leading(voices=[["C5", "B4", "A4"], ["G4", "G4", "E4"], ["E4", "D4", "C4"], ["C3", "G2", "A2"]],
                              root="C")
    assert (2, [0], "leading_tone") in _rows(sop["violations"])           # the soprano rule stays


@pytest.mark.parametrize("tonic", TONICS)
def test_library_schemas_voiced_by_bach_pass_the_checker(tonic):
    """Findings 13/56 and 59: romanesca, lament and andalusian come back clean in every key."""
    for name in ("romanesca", "lament", "andalusian"):
        pl = progression_library(name, root=tonic)
        scale = pl["key"].split(" ", 1)[1]
        b = bach_chorale_voicing(pl["chords"], root=tonic, scale_type=scale)
        assert b["rule_breaks"] == [], (name, tonic, b["rule_breaks"])
        r = check_voice_leading(voices=b["render_hint"]["tracks"], root=tonic, scale_type=scale)
        assert r["valid"], (name, tonic, r["violations"])


def test_direct_fifths_from_an_enharmonic_interval():
    """Finding 14: a d6 (seven semitones) into a spelled P5 with a leap is a direct fifth (music21)."""
    r = check_voice_leading(voices=[["Ab4", "C5"], ["C#4", "F4"]])
    assert _rows(r["violations"]) == [(1, [0, 1], "direct_fifths")]
    r = check_voice_leading(voices=[["B#4", "E5"], ["C4", "E4"]])
    assert _rows(r["violations"]) == [(1, [0, 1], "direct_octaves")]
    stepped = check_voice_leading(voices=[["Ab4", "Bb4"], ["C#4", "Eb4"]])
    assert stepped["valid"]                                               # the soprano steps: allowed
    true_parallel = check_voice_leading(voices=[["C5", "E5"], ["F4", "A4"]])
    assert _rows(true_parallel["violations"]) == [(1, [0, 1], "parallel_fifths")]


@pytest.mark.parametrize("tonic", TONICS)
def test_direct_fifths_from_an_enharmonic_interval_in_every_key(tonic):
    for voices, rule in (([["Ab4", "C5"], ["C#4", "F4"]], "direct_fifths"),
                         ([["B#4", "E5"], ["C4", "E4"]], "direct_octaves")):
        moved = [[_shift(n, tonic, near=True) for n in v] for v in voices]
        assert _rows(check_voice_leading(voices=moved)["violations"]) == [(1, [0, 1], rule)], (tonic, moved)


def test_secondary_dominant_on_the_seventh_degree_may_double_its_root():
    """Finding 16: B D# F# (V/iii in C) doubles its root B; only V and vii° hold a leading tone."""
    r = check_voice_leading(voicings=[["B4", "F#4", "D#4", "B2"], ["B4", "G4", "E4", "E3"]], root="C")
    assert r["valid"]
    assert check_voice_leading(voicings=[["B4", "F#4", "D#4", "B2"], ["B4", "G4", "E4", "E3"]], root="E")["valid"]
    dim = check_voice_leading(voices=[["B4"], ["F4"], ["D4"], ["B2"]], root="C")
    assert _rows(dim["violations"]) == [(0, [0, 3], "doubled_leading_tone")]  # vii° still counts


@pytest.mark.parametrize("tonic", TONICS)
def test_secondary_dominant_on_the_seventh_degree_in_every_key(tonic):
    moved = [[_shift(n, tonic, near=True) for n in ch] for ch in (["B4", "F#4", "D#4", "B2"], ["B4", "G4", "E4", "E3"])]
    assert check_voice_leading(voicings=moved, root=tonic)["valid"], (tonic, moved)
    dim = [[_shift(n, tonic, near=True)] for n in ("B4", "F4", "D4", "B2")]
    assert _rows(check_voice_leading(voices=dim, root=tonic)["violations"]) == [(0, [0, 3], "doubled_leading_tone")]


def test_voicings_of_different_sizes_are_aligned():
    """Finding 51: voice_leading / voice_chords output mixing triads and sevenths goes in as is."""
    vl = voice_leading(["C", "F", "G7", "C"])["chords"]
    r = check_voice_leading(voicings=vl, root="C")
    assert len(r["voice_order"]) == 4 and len(r["slices"]) == 4
    assert all(sum(n is not None for n in s["notes"]) == len(c) for s, c in zip(r["slices"], vl))
    g7_to_c = check_voice_leading(voicings=[["G4", "F4", "D4", "B3"], ["G4", "E4", "C4"]], root="C")
    assert g7_to_c["slices"][1]["notes"] == ["G4", "E4", None, "C4"]     # F falls to E, B rises to C
    assert g7_to_c["valid"]
    bad = check_voice_leading(voicings=[["G4", "F4", "D4", "B3"], ["A4", "E4", "C4"]], root="C")
    assert bad["valid"]                                                   # G -> A: no seventh or LT fault
    drop2 = check_voice_leading(voicings=voice_chords(["C", "G/B", "C"], "drop2")["chords"])
    assert len(drop2["voice_order"]) == 5


@pytest.mark.parametrize("tonic", TONICS)
def test_voice_leading_chains_into_voicings_in_every_key(tonic):
    for numerals in ("I V7 I", "I IV V7 I", "ii7 V7 I", "I V65 I", "I vi ii V7 I"):
        symbols = roman_to_chords(numerals, tonic)["symbols"]
        for voiced in (voice_leading(symbols)["chords"], voice_chords(symbols, "drop2")["chords"],
                       voice_chords(symbols, "close")["chords"]):
            r = check_voice_leading(voicings=voiced, root=tonic)
            assert [sum(n is not None for n in s["notes"]) for s in r["slices"]] == [len(c) for c in voiced]


def test_bach_avoids_the_melodic_augmented_second():
    """Finding 59: iv6 -> V and VI -> V in minor no longer write b6 -> #7 as an A2."""
    lament = bach_chorale_voicing(["Cm", "Gm/Bb", "Fm/Ab", "G"], root="C", scale_type="natural minor")
    assert lament["voices"]["alto"] == ["C4", "D4", "C4", "B3"] and lament["rule_breaks"] == []
    for chords, tonic in ((["Am", "F", "E", "Am"], "A"), (["Cm", "Ab", "G", "Cm"], "C")):
        b = bach_chorale_voicing(chords, root=tonic, scale_type="harmonic minor")
        assert b["rule_breaks"] == []
        r = check_voice_leading(voices=b["render_hint"]["tracks"], root=tonic, scale_type="harmonic minor")
        assert "melodic_augmented" not in {v["rule"] for v in r["violations"]}
    # when the melody forces it, the A2 is reported, never silent
    forced = bach_chorale_voicing(["Am", "Dm", "E", "Am"], root="A", scale_type="harmonic minor",
                                  melody=["C5", "F4", "G#4", "A4"])
    assert {"chord": 3, "rule": "melodic augmented second in the soprano"} in forced["rule_breaks"]
    assert _melodic_augmented("Ab3", "B3") == "augmented second" and _melodic_augmented("C4", "C#4") is None


@pytest.mark.parametrize("kwargs", [
    {},
    {"voices": [["C5"], ["C4"]], "voicings": [["C4", "E4"]]},
    {"voices": [["C5", "D"], ["C4", "D4"]]},
    {"voices": [["C5", "D5"], ["C4"]]},
    {"voices": [["C5"]]},
    {"voices": [["C5"], {"type": "notes", "notes": ["C4"]}]},
    {"voices": [{"type": "chords", "chords": ["C"]}, {"type": "notes", "notes": ["C4"]}]},
    {"voicings": [["C4"], ["C4", "E4", "G4"]]},
    {"voicings": [["C", "E"]]},
    {"voices": [["C5"], ["C4"]], "root": "C", "scale_type": "major pentatonic"},
])
def test_check_voice_leading_rejects_bad_input(kwargs):
    with pytest.raises(ValueError):
        check_voice_leading(**kwargs)


# ---------------------------------------------------------------- properties

_HARD = {"parallel_fifths", "parallel_octaves", "contrary_fifths", "contrary_octaves", "direct_fifths",
         "direct_octaves", "voice_overlap", "voice_crossing", "melodic_augmented", "augmented_sixth"}
_PROGRESSIONS = ["C Am F G7 C", "C F G7 C", "C Am Dm G7 C", "C F Bdim Em Am Dm G C", "C G Am Em F C F G",
                 "C Dm7 G7 C", "C G/B Am C/G F G C", "C E7 Am D7 G7 C", "Cmaj7 Am7 Dm7 G7 C",
                 "C G/B Am C/E", "C G/B Am Em/G F C/E Dm G C", "C F Fm/Ab G C", "C Am F G B7 Em"]


@pytest.mark.parametrize("tonic", TONICS)
def test_clean_chorales_pass_the_checker(tonic):
    """Whenever bach_chorale_voicing reports no rule break, the checker agrees on the shared rules
    (the soprano leading tone, sevenths, augmented sixths and melodic augmented intervals too)."""
    for prog in _PROGRESSIONS:
        chords = [_shift_symbol(c, tonic) for c in prog.split()]
        b = bach_chorale_voicing(chords, root=tonic)
        if b["rule_breaks"]:
            continue
        r = check_voice_leading(voices=b["render_hint"]["tracks"], root=tonic)
        for v in r["violations"]:
            assert v["rule"] not in _HARD, (prog, tonic, v)
            assert v["rule"] != "seventh_resolution", (prog, tonic, v)
            assert not (v["rule"] == "leading_tone" and v["voices"] == [0]), (prog, tonic, v)


def _shift_symbol(symbol: str, tonic: str) -> str:
    import re
    root, quality, bass = re.match(r"^([A-G][#b]*)(.*?)(?:/([A-G][#b]*))?$", symbol).groups()
    return _shift(root, tonic) + quality + ("/" + _shift(bass, tonic) if bass else "")


_CANTUS_SHAPES = [[1, 2, 3, 2, 4, 3, 2, 1], [1, 3, 2, 4, 3, 5, 4, 2, 1], [1, 5, 4, 3, 2, 3, 2, 1], [1, 6, 5, 4, 3, 2, 1]]


@pytest.mark.parametrize("species", [1, 2, 3, 4, 5])
def test_counterpoint_has_no_parallels_on_adjacent_attacks(species):
    for (key, scale), shape, position in itertools.product(
            [("C", "major"), ("Bb", "major"), ("A", "natural minor"), ("D", "dorian")], _CANTUS_SHAPES, ("above", "below")):
        cf = notes_from_degrees(key + "4", scale, shape)["notes"]
        r = species_counterpoint(cf, key, scale, species=species, position=position)
        if "warning" in r:
            continue
        rules = {v["rule"] for v in check_voice_leading(voices=r["render_hint"]["tracks"])["violations"]}
        assert not {x for x in rules if x.startswith(("parallel_", "contrary_"))}, (species, key, shape, position)


_LINT_ME = [["E5", "D5", "C5", "B4", "C5", "F5", "E5"], ["G4", "G4", "F4", "G#4", "A4", "B4", "G4"],
            ["C4", "B3", "A3", "F4", "E4", "D4", "C4"], ["C3", "G2", "A2", "E3", "A2", "G2", "C3"]]


@pytest.mark.parametrize("tonic", TONICS)
def test_check_voice_leading_is_transposition_invariant(tonic):
    ref = check_voice_leading(voices=_LINT_ME, root="C")
    assert ref["violations"] and ref["warnings"] and ref["dissonances"]
    moved = [[_shift(n, tonic, near=True) for n in voice] for voice in _LINT_ME]
    r = check_voice_leading(voices=moved, root=tonic)
    assert _rows(r["violations"]) == _rows(ref["violations"])
    # SATB ranges are absolute pitches, the one thing a transposition may change
    assert [w for w in _rows(r["warnings"]) if w[2] != "range"] == _rows(ref["warnings"])
    assert [(d["index"], d["voices"], d["interval"]) for d in r["dissonances"]] == \
        [(d["index"], d["voices"], d["interval"]) for d in ref["dissonances"]]
    assert r["motion"] == ref["motion"]


# ------------------------------------------------------------------- fuzzing

_BAD_VALUES = [None, 7, -1, 3.5, 2.0, True, "", "zz", [], [None], {}, {"a": 1}, ["C", 5], "C" * 5,
               ["C", "Db", "D"], [["C", "Db", "D"], "G"], "0 4 7", ["C", ".", "E"]]


@pytest.mark.parametrize("fn,base", [(detect_key, {"notes": ["C4", "E4", "G4"]}),
                                     (detect_key, {"chords": ["C", "G"]}),
                                     (detect_key, {}),
                                     (check_voice_leading, {"voices": [["C5", "D5"], ["E4", "G4"]]}),
                                     (check_voice_leading, {"voicings": [["C4", "E4"], ["D4", "F4"]]}),
                                     (check_voice_leading, {})])
def test_analysis_tools_only_raise_value_errors(fn, base):
    failures = []
    for pname in inspect.signature(fn).parameters:
        for bad in _BAD_VALUES:
            try:
                fn(**dict(base, **{pname: bad}))
            except ValueError:
                pass
            except Exception as e:  # noqa: BLE001 — collect them all
                failures.append(f"{fn.__name__}({pname}={bad!r}) raised {type(e).__name__}: {e}")
    assert not failures, "\n".join(failures)

