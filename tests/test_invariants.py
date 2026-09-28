"""Invariant sweeps that guard whole classes of bugs, not single examples.

Each test states a property that must hold for *every* input in a family
(all 12 keys, every chord type, every species/position...). The bugs these
were written against slipped through because the original tests only used
C major or one hand-picked input:

- key-relative tools that were only right in C (negative harmony);
- chord inputs accepted by one tool but not another ("D" as a triad);
- key membership judged by the root alone (E7 "in" A minor);
- counterpoint rule violations that only show up over many cantus lines;
- tools that are documented to chain but disagree on conventions
  (negative scale degrees between motif_grammar and notes_from_degrees).
"""

from __future__ import annotations

import itertools

import pytest

from midi_composer_mcp.chords import CHORDS, chord_notes, parse_chord_symbol
from midi_composer_mcp.counterpoint import species_counterpoint
from midi_composer_mcp.diatonic import degrees_to_chords, diatonic_chords
from midi_composer_mcp.forms import resolve_form
from midi_composer_mcp.harmony import (
    analyze_progression,
    harmonize_melody,
    interval_between,
    negative_harmony,
    voice_leading,
)
from midi_composer_mcp.melody import (
    melodic_sequence,
    motif_grammar,
    notes_from_degrees,
    snap_to_scale,
    tintinnabuli_voice,
)
from midi_composer_mcp.midi_io import render_arrangement
from midi_composer_mcp.notes import parse_note, parse_notes
from midi_composer_mcp.scales import SCALES, resolve_scale_type
from midi_composer_mcp.structure import plan_sections, render_song_structure

# Every tonic spelling a caller is likely to use, naturals, sharps and flats.
TONICS = ["C", "G", "D", "A", "E", "B", "F#", "C#", "F", "Bb", "Eb", "Ab", "Db", "Gb"]
HEPTATONIC = [s.name for s in SCALES.values() if len(s.intervals) == 7]


def pcs(notes) -> list[int]:
    return [parse_note(n).pitch_class if isinstance(n, str) else n.pitch_class for n in notes]


def rel(notes, tonic: str) -> list[int]:
    t = parse_note(tonic).pitch_class
    return [(p - t) % 12 for p in pcs(notes)]


# ------------------------------------------------------ transposition equivariance

@pytest.mark.parametrize("tonic", TONICS)
def test_negative_harmony_is_key_relative(tonic):
    """Mirroring in any key equals mirroring in C, transposed to that key."""
    c_rel = [0, 4, 7, 2, 5, 9, 11, 1, 6]   # a spread of scale and chromatic notes
    t = parse_note(tonic).pitch_class
    notes = [spell for spell in (["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"][(t + r) % 12]
                                 for r in c_rel)]
    got = rel(negative_harmony(notes, tonic)["notes"], tonic)
    expected = [(7 - r) % 12 for r in c_rel]
    assert got == expected


@pytest.mark.parametrize("tonic", TONICS)
def test_negative_harmony_flips_tonic_major_to_tonic_minor(tonic):
    major = chord_notes(CHORDS["major"], parse_note(tonic))
    minor = chord_notes(CHORDS["minor"], parse_note(tonic))
    mirrored = negative_harmony([n.name for n in major], tonic)["notes"]
    assert sorted(pcs(mirrored)) == sorted(pcs(minor))
    # proper spelling: the mirror of a spelled chord uses the minor chord's letters
    assert sorted(mirrored) == sorted(n.name for n in minor)


@pytest.mark.parametrize("tonic", TONICS)
def test_negative_harmony_is_an_involution(tonic):
    notes = ["C4", "E4", "G4", "Bb4", "D5", "F#3", "A2", "Eb5"]
    once = negative_harmony(notes, tonic)["notes"]
    twice = negative_harmony(once, tonic)["notes"]
    assert pcs(twice) == pcs(notes)
    # octaves stay near the original and inside MIDI
    for a, b in zip(notes, once):
        assert abs(parse_note(a).midi - parse_note(b).midi) <= 6


def test_negative_harmony_never_leaves_midi_range():
    for tonic in TONICS:
        for note in ["G9", "F#9", "C-1", "C#-1"]:
            out = negative_harmony([note], tonic)["notes"][0]
            assert 0 <= parse_note(out).midi <= 127


@pytest.mark.parametrize("scale", ["major", "natural minor", "dorian", "harmonic minor", "major pentatonic"])
def test_key_relative_melody_tools_are_transposition_invariant(scale):
    base = None
    for tonic in TONICS:
        degs = notes_from_degrees(tonic + "4", scale, [1, 3, -1, 5, 8, -3])["notes"]
        snapped = snap_to_scale(["C4", "C#4", "D#4", "F#4", "A#4", "B4"], tonic, scale)["notes"]
        seq = melodic_sequence(degs[:3], tonic, scale, step=-1, count=3)["notes"]
        t = parse_note(tonic).pitch_class
        # express each result relative to the tonic (degrees/sequence) or as-is (snap input is fixed)
        shape = (
            [parse_note(n).midi - parse_note(tonic + "4").midi for n in degs],
            [parse_note(n).midi - parse_note(degs[0]).midi for n in seq],
        )
        if base is None:
            base = shape
        assert shape == base, tonic
        # snapping lands in the scale and moves each note at most a whole step (a tritone for gappy scales)
        scale_pcs = {(t + i) % 12 for i in resolve_scale_type(scale).intervals}
        for src, dst in zip(["C4", "C#4", "D#4", "F#4", "A#4", "B4"], snapped):
            assert parse_note(dst).pitch_class in scale_pcs
            assert abs(parse_note(dst).midi - parse_note(src).midi) <= 3


@pytest.mark.parametrize("species", [1, 2, 3, 4, 5])
def test_counterpoint_is_transposition_invariant(species):
    cf_c = ["C4", "D4", "F4", "E4", "G4", "F4", "E4", "D4", "C4"]
    ref = species_counterpoint(cf_c, "C", "major", species=species)
    ref_shape = [parse_note(n).midi - 60 for n in ref["counterpoint"]]
    for shift, tonic in [(2, "D"), (5, "F"), (-2, "Bb"), (4, "E"), (-4, "Ab")]:
        cf = [parse_note(n).midi + shift for n in cf_c]
        from midi_composer_mcp.notes import note_from_midi
        names = [note_from_midi(m, prefer_flats="b" in tonic or tonic == "F").name for m in cf]
        r = species_counterpoint(names, tonic, "major", species=species)
        assert [parse_note(n).midi - 60 - shift for n in r["counterpoint"]] == ref_shape, tonic
        assert r["counterpoint_rhythm"] == ref["counterpoint_rhythm"]


# ---------------------------------------------------------- analysis round trips

@pytest.mark.parametrize("scale", HEPTATONIC)
@pytest.mark.parametrize("sevenths", [False, True])
def test_analyze_progression_round_trips_diatonic_chords(scale, sevenths):
    for tonic in ["C", "F#", "Bb", "E", "Db"]:
        dia = diatonic_chords(tonic, scale, sevenths)["chords"]
        res = analyze_progression([c["symbol"] for c in dia], tonic, scale)["chords"]
        for d, a in zip(dia, res):
            assert a["in_key"] is True, (tonic, scale, d["symbol"])
            if not isinstance(d["symbol"], str) or "/" in d["symbol"]:
                continue  # not a stacked-thirds chord: named from its real root (or as notes)
            assert a["degree"] == d["degree"]
            assert a["function"] == d["harmonic_function"]
            # same numeral letters and accidental as diatonic_chords
            strip = lambda r: r.rstrip("°ø+Δ7")  # noqa: E731
            assert strip(a["roman"]).upper() == strip(d["roman"]).upper(), (tonic, scale, a, d)


@pytest.mark.parametrize("tonic", ["C", "A", "Eb", "F#"])
@pytest.mark.parametrize("scale", ["major", "natural minor", "harmonic minor", "dorian"])
def test_in_key_means_every_chord_tone_is_in_the_scale(tonic, scale):
    scale_pcs = {(parse_note(tonic).pitch_class + i) % 12 for i in resolve_scale_type(scale).intervals}
    symbols = []
    for ctype in CHORDS.values():
        for root in ["C", "D", "E", "F", "G", "A", "B", "Bb", "F#", "Eb", "G#"]:
            symbols.append((f"{root}{ctype.symbol}", ctype, root))
    res = analyze_progression([s for s, _, _ in symbols], tonic, scale)["chords"]
    for (sym, ctype, root), entry in zip(symbols, res):
        tones = {n.pitch_class for n in chord_notes(ctype, parse_note(root))}
        assert entry["in_key"] == tones.issubset(scale_pcs), (sym, tonic, scale)
        if not entry["in_key"]:
            assert entry["non_scale_notes"]


def test_dominant_of_minor_key_is_chromatic_in_natural_minor():
    e7 = analyze_progression(["Am", "E7", "Am"], "A", "natural minor")["chords"][1]
    assert e7["in_key"] is False and e7["non_scale_notes"] == ["G#"] and e7["degree"] == 5
    e7h = analyze_progression(["Am", "E7", "Am"], "A", "harmonic minor")["chords"][1]
    assert e7h["in_key"] is True and e7h["function"] == "dominant"


@pytest.mark.parametrize("tonic", TONICS)
def test_analyze_numerals_follow_spelling(tonic):
    t = parse_note(tonic)
    # the chord a diminished fifth above (b5) and an augmented fourth above (#4)
    from midi_composer_mcp.notes import transpose
    b5 = transpose(t, 6, 4)
    s4 = transpose(t, 6, 3)
    res = analyze_progression([b5.name, s4.name], tonic, "major")["chords"]
    assert res[0]["roman"] == "bV" and res[1]["roman"] == "#IV"


@pytest.mark.parametrize("scale", sorted(SCALES))
def test_diatonic_symbols_always_chain(scale, tmp_path):
    """Every symbol diatonic_chords/degrees_to_chords returns is valid renderer input."""
    from midi_composer_mcp.midi_io import render_chords
    for sevenths in (False, True):
        chords = diatonic_chords("Eb", scale, sevenths)["chords"]
        symbols = [c["symbol"] for c in chords]
        render_chords(symbols, output_dir=str(tmp_path), file_name="x.mid")
        romans = [c["roman"] for c in chords if "roman" in c]
        if romans and "?" not in "".join(romans):  # the printed numerals parse back
            back = degrees_to_chords("Eb", scale, romans, sevenths)["symbols"]
            assert back == symbols


# ---------------------------------------------------------- chord-input vocabulary

def test_every_chord_symbol_parses_on_every_root():
    roots = ["C", "D", "E", "F", "G", "A", "B", "C#", "Db", "F#", "Gb", "Bb", "Eb", "Ab", "G#"]
    for ctype, root in itertools.product(CHORDS.values(), roots):
        r, parsed, bass = parse_chord_symbol(f"{root}{ctype.symbol}")
        assert parsed is ctype and r.name == root and bass is None


@pytest.mark.parametrize("triad", ["D", "Dm", "F#", "Bb", "Ebm", "D F# A", "D,F#,A", ["D", "F#", "A"], ["Dm"], "D/F#"])
def test_tintinnabuli_accepts_every_chord_spelling(triad):
    out = tintinnabuli_voice(["D5", "E5", "F#5", "G5"], triad)
    root = parse_note(out["triad"][0]).pitch_class
    assert root == parse_note("D").pitch_class or triad in ("F#", "Bb", "Ebm")
    for n in out["t_voice"]:
        assert parse_note(n).pitch_class in pcs(out["triad"])


@pytest.mark.parametrize("chord", ["D", "Dm", "D7", "D/F#", ["D", "F#", "A"], "D F# A"])
def test_chord_consuming_tools_agree_on_chord_inputs(chord):
    """A chord accepted by one tool is accepted by the others that take chords."""
    as_list = [chord] if not isinstance(chord, str) or " " not in chord else [chord.split()]
    voice_leading(as_list + ["G"])
    analyze_progression(as_list + ["G"], "D", "major")
    tintinnabuli_voice(["D5", "E5"], chord)


def test_voice_leading_keeps_slash_basses_at_the_bottom():
    vl = voice_leading(["C", "C/E", "F/A", "G/B", "C"])["voicings"]
    for v, bass in zip(vl[1:4], ["E", "A", "B"]):
        assert parse_note(v["notes"][0]).pitch_class == parse_note(bass).pitch_class
        assert v["midi"][0] == min(v["midi"])


@pytest.mark.parametrize("octave", [0, 1, 7, 8, 9])
def test_voice_leading_survives_register_extremes(octave):
    try:
        out = voice_leading(["C", "A", "F", "G", "B", "E"], octave=octave)
    except ValueError:
        assert octave == 9  # only a first chord that cannot fit at all may be rejected
        return
    for v in out["voicings"]:
        assert all(0 <= m <= 127 for m in v["midi"])


# ---------------------------------------------------------- intervals with octaves

def test_interval_names_agree_with_semitones_and_letters():
    notes = [f"{l}{a}{o}" for l in "CDEFGAB" for a in ("", "#", "b") for o in (3, 4, 5)]
    for x, y in itertools.product(notes[::2], notes[1::3]):
        r = interval_between(x, y)
        a, b = parse_note(x), parse_note(y)
        assert r["signed_semitones"] == b.midi - a.midi
        assert abs(r["semitones"]) == abs(b.midi - a.midi)  # B#3 -> Cb4 is a (negative) dd2
        lo, hi = sorted((a, b), key=lambda n: ("CDEFGAB".index(n.letter) + 7 * n.octave, n.midi))
        assert r["number"] == ("CDEFGAB".index(hi.letter) + 7 * hi.octave) - ("CDEFGAB".index(lo.letter) + 7 * lo.octave) + 1


def test_compound_intervals():
    assert interval_between("C4", "C5")["short"] == "P8"
    assert interval_between("C4", "E5")["name"] == "major tenth"
    assert interval_between("G4", "C4")["direction"] == "descending"
    assert interval_between("G4", "C4")["short"] == "P5"
    assert interval_between("G4", "D4")["short"] == "P4"
    assert interval_between("C5", "B3")["name"] == "minor ninth"
    assert interval_between("B#3", "C4")["short"] == "d2"
    assert interval_between("C", "E")["short"] == "M3"      # pitch classes: simple, upward
    assert "direction" not in interval_between("C", "E")


# ----------------------------------------------------------- degree conventions

@pytest.mark.parametrize("scale", ["major", "dorian", "minor pentatonic", "blues", "diminished whole-half"])
def test_degree_number_line_has_no_gap(scale):
    """Consecutive degrees ... -2, -1, 1, 2 ... are consecutive scale steps."""
    line = [-9, -8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    midis = [parse_note(n).midi for n in notes_from_degrees("C4", scale, line)["notes"]]
    assert midis == sorted(midis) and len(set(midis)) == len(midis)
    n = len(resolve_scale_type(scale).intervals)
    assert midis[line.index(-n)] == 48 and midis[line.index(1)] == 60 and midis[line.index(n + 1)] == 72


@pytest.mark.parametrize("tonic", ["C4", "A3", "F#4", "Eb4"])
def test_motif_grammar_degrees_chain_into_notes_from_degrees(tonic):
    """Transposing/inverting in degree space equals doing it on the resolved notes."""
    motif = [1, 2, 3, 5]
    g = motif_grammar("ABC", {"A": motif, "B": {"vary": "A", "transpose": -2},
                              "C": {"vary": "A", "invert": True}}, kind="degrees")
    notes = notes_from_degrees(tonic, "major", g["degrees"])["notes"]
    a, b, c = notes[:4], notes[4:8], notes[8:]
    down2 = melodic_sequence(a, tonic[:-1], "major", step=-2, count=2)["notes"][4:]
    assert b == down2
    # the inversion mirrors the scale-step contour around the first note
    ladder = notes_from_degrees(tonic, "major", list(range(-14, 0)) + list(range(1, 16)))["notes"]
    idx = [ladder.index(n) for n in a]
    inv = [ladder.index(n) for n in c]
    assert [i - idx[0] for i in idx] == [-(j - inv[0]) for j in inv]


# ------------------------------------------------------------------- forms

@pytest.mark.parametrize("word", ["verse", "chorus", "intro", "bridge", "hook", "outro", "breakdown"])
def test_single_word_form_is_one_section(word):
    assert resolve_form(word) == [word]
    assert plan_sections(word, bars=4)["form"] == [word]


def test_letter_forms_still_split():
    assert resolve_form("AABA") == ["A", "A", "B", "A"]
    assert resolve_form("abab", known={"a": 1, "b": 2}) == ["a", "b", "a", "b"]
    assert motif_grammar("abab", {"a": "C4", "b": "D4"})["notes"] == ["C4", "D4", "C4", "D4"]


# ---------------------------------------------------------------- rendering

def test_many_tracks_fail_cleanly_and_channels_do_not_collide(tmp_path):
    tracks = [{"type": "notes", "notes": ["C4"], "name": f"t{i}"} for i in range(16)]
    with pytest.raises(ValueError, match="15 melodic channels"):
        render_arrangement(tracks, output_dir=str(tmp_path))
    mixed = [{"type": "notes", "notes": ["C4"], "channel": 0, "program": 40},
             {"type": "notes", "notes": ["E4"], "program": 0},
             {"type": "drums", "lanes": {"kick": "O"}, "channel": 3}]
    out = render_arrangement(mixed, output_dir=str(tmp_path))
    chans = [t["channel"] for t in out["tracks"]]
    assert chans[0] == 0 and chans[1] != 0 and chans[2] == 9


def test_song_time_signature_follows_beats_per_bar(tmp_path):
    import mido
    out = render_song_structure({"verse": {"bars": 2, "tracks": [{"type": "chords", "chords": ["C", "G"]}]}},
                                form="verse", beats_per_bar=3, output_dir=str(tmp_path))
    ts = [m for m in mido.MidiFile(out["file"]).tracks[0] if m.type == "time_signature"][0]
    assert (ts.numerator, ts.denominator) == (3, 4)


@pytest.mark.parametrize("bad_tempo", [0, -5, 1000])
def test_structure_tools_validate_tempo(bad_tempo, tmp_path):
    with pytest.raises(ValueError):
        plan_sections("verse chorus", tempo=bad_tempo)
    with pytest.raises(ValueError):
        render_song_structure({"v": {"tracks": [{"type": "notes", "notes": ["C4"]}]}},
                              tempo=bad_tempo, output_dir=str(tmp_path))


def test_render_hints_chain_into_arrange(tmp_path):
    cp = species_counterpoint(["C4", "D4", "E4", "D4", "C4"], "C", "major", species=4)
    out = render_arrangement(cp["render_hint"]["tracks"], output_dir=str(tmp_path))
    ends = {t["end_beat"] for t in out["tracks"]}
    assert ends == {5 * 4.0}  # cantus and counterpoint end together
    hm = harmonize_melody(["E4", "D4", "C4", "G4"], "C", "major", in_scale=True)
    render_arrangement(hm["render_hint"]["tracks"], output_dir=str(tmp_path))


# ------------------------------------------------------------- counterpoint rules

_CONS = {0, 3, 4, 7, 8, 9}
_PERF = {0, 7}


def _cp_violations(r, species, position) -> list[str]:
    """Independent Fux checker over the rendered rhythm (not the solver's own bookkeeping)."""
    cf = [parse_note(n).midi for n in r["cantus"]]
    bar = r["cantus_step_beats"]
    step = r["counterpoint_step_beats"]
    notes = [parse_note(n).midi for n in r["counterpoint"]]
    onsets = [i for i, ch in enumerate(r["counterpoint_rhythm"]) if ch == "O"]
    assert len(onsets) == len(notes)
    ev = []
    for k, i in enumerate(onsets):
        end = onsets[k + 1] if k + 1 < len(onsets) else len(r["counterpoint_rhythm"])
        ev.append((notes[k], i * step, (end - i) * step))
    errs = []
    if abs(sum(d for *_, d in ev) - len(cf) * bar) > 1e-9:
        errs.append("length mismatch")

    def cf_at(t):
        return cf[min(int(t // bar), len(cf) - 1)]

    for k, (m, s, d) in enumerate(ev):
        c = cf_at(s)
        iv = abs(m - c) % 12
        strong = abs(s % bar) < 1e-9
        tied_over = int((s + d - 1e-9) // bar) > int(s // bar)
        if (strong or tied_over) and iv not in _CONS:
            errs.append(f"struck dissonance at {s}")
        if not strong and not tied_over and iv not in _CONS:
            prev = ev[k - 1][0] if k else None
            nxt = ev[k + 1][0] if k + 1 < len(ev) else None
            if prev is None or nxt is None or abs(m - prev) > 2 or abs(nxt - m) > 2:
                errs.append(f"unprepared passing dissonance at {s}")
        if tied_over:
            c2 = cf_at((int(s // bar) + 1) * bar)
            if abs(m - c2) % 12 not in _CONS:
                nxt = ev[k + 1][0] if k + 1 < len(ev) else None
                if nxt is None or not 1 <= m - nxt <= 2 or abs(nxt - c2) % 12 not in _CONS:
                    errs.append(f"unresolved suspension at {s}")
                elif position == "below" and abs(nxt - c2) % 12 == 0:
                    errs.append(f"7-8 suspension below at {s}")
        if (position == "above" and m <= c) or (position == "below" and m >= c):
            errs.append(f"voice crossing at {s}")
        if k and (abs(m - ev[k - 1][0]) > 12 or abs(m - ev[k - 1][0]) in (6, 10, 11)):
            errs.append(f"leap over an octave, tritone or seventh at {s}")
    # no dissonance next to another dissonance, and none left or approached by a repeat
    for k in range(1, len(ev)):
        (m0, s0, _), (m1, s1, _) = ev[k - 1], ev[k]
        d0, d1 = abs(m0 - cf_at(s0)) % 12 not in _CONS, abs(m1 - cf_at(s1)) % 12 not in _CONS
        if d0 and d1:
            errs.append(f"dissonance next to a dissonance at {s1}")
        if (d0 or d1) and m0 == m1:
            errs.append(f"dissonance restruck at {s1}")
    if position == "below" and abs(ev[0][0] - cf[0]) % 12 != 0:
        errs.append("lower voice does not start on the unison/octave")
    if len(ev) > 1 and abs(ev[-1][0] - ev[-2][0]) > 2:
        errs.append("final not reached by step")

    def cp_at(t):
        return next(mm for mm, s, d in ev if s <= t + 1e-9 < s + d)

    times = sorted({s for _, s, _ in ev} | {b * bar for b in range(len(cf))})
    for t0, t1 in zip(times, times[1:]):
        m0, m1, c0, c1 = cp_at(t0), cp_at(t1), cf_at(t0), cf_at(t1)
        if abs(m1 - c1) % 12 in _PERF and m0 != m1 and c0 != c1 and (m1 - m0) * (c1 - c0) > 0:
            errs.append(f"parallel/direct perfect into {t1}")
    if species == 4:  # parallels between the struck (weak-beat) notes
        struck = [(m, cf_at(s)) for m, s, d in ev if abs(s % bar) > 1e-9]
        pairs = zip(struck, struck[1:])
    else:  # parallels between successive struck downbeats (a tied one is a syncopation)
        struck_at = {s for _, s, _ in ev}
        dbs = [(cp_at(b * bar), cf[b]) if b * bar in struck_at else None for b in range(len(cf))]
        pairs = [(x, y) for x, y in zip(dbs, dbs[1:]) if x and y]
    for (m0, c0), (m1, c1) in pairs:
        i0, i1 = abs(m0 - c0) % 12, abs(m1 - c1) % 12
        if i0 == i1 and i1 in _PERF and (m1 - m0) * (c1 - c0) > 0:
            errs.append("parallel perfects on successive structural beats")
    if abs(ev[0][0] - cf[0]) % 12 not in _PERF or abs(ev[-1][0] - cf[-1]) % 12 not in _PERF:
        errs.append("endpoints not perfect")
    return errs


_CANTUS_SHAPES = [
    [1, 2, 3, 2, 4, 3, 2, 1], [1, 3, 2, 4, 3, 5, 4, 2, 1], [1, -1, 2, 3, 5, 4, 3, 2, 1],
    [1, 4, 3, 2, 5, 4, 2, 3, 2, 1], [1, 5, 4, 3, 2, 3, 2, 1], [1, 3, 4, 2, 1], [1, 2, 1],
    [1, 6, 5, 4, 3, 2, 1],
]
_CP_KEYS = [("C", "major"), ("D", "major"), ("Bb", "major"), ("A", "natural minor"),
            ("E", "natural minor"), ("D", "dorian"), ("G", "mixolydian"), ("A", "harmonic minor")]


@pytest.mark.parametrize("species", [1, 2, 3, 4, 5])
def test_species_counterpoint_obeys_the_rules_it_promises(species):
    runs = warned = 0
    for (key, scale), shape, position in itertools.product(_CP_KEYS, _CANTUS_SHAPES, ("above", "below")):
        cf = notes_from_degrees(key + "4", scale, shape)["notes"]
        r = species_counterpoint(cf, key, scale, species=species, position=position)
        errs = _cp_violations(r, species, position)
        runs += 1
        if "warning" in r:
            warned += 1  # the solver admits no clean line exists; it must say so
            assert r["rules_broken"] >= 1
        else:
            assert not errs, (species, key, scale, position, cf, r["counterpoint"], errs)
    assert warned <= runs // 20  # a clean line is the norm


# ------------------------------------------------------- chord-symbol misreads

@pytest.mark.parametrize("symbol,expected", [
    ("C7sus", "dominant 7 sus 4"), ("C7M", "major 7"), ("C7+", "augmented 7"), ("C7-5", "dominant 7 flat 5"),
    ("C7-9", "dominant 7 flat 9"), ("C-9", "minor 9"), ("C-6", "minor 6"), ("C-maj7", "minor major 7"),
    ("C-7b5", "half-diminished"), ("C-69", "minor six nine"), ("CM7", "major 7"), ("CM9", "major 9"),
    ("CM6/9", "six nine"), ("CMadd9", "add 9"), ("CMIN7", "minor 7"), ("C47", "dominant 7"), ("C4m7", "minor 7"),
])
def test_jazz_symbol_variants(symbol, expected):
    assert parse_chord_symbol(symbol)[1].name == expected


def test_major_and_minor_markers_are_never_crossed():
    """'M…' never parses as a minor chord, '-…' never as a major one; unknowns are rejected."""
    for ctype in CHORDS.values():
        minorless = ctype.symbol[1:] if ctype.symbol.startswith("m") and not ctype.symbol.startswith("maj") else ctype.symbol
        for text in ("M" + ctype.symbol, "-" + minorless, "-" + ctype.symbol):
            try:
                _root, parsed, _bass = parse_chord_symbol("C" + text)
            except ValueError:
                continue
            is_minor = parsed.symbol.startswith("m") and not parsed.symbol.startswith("maj")
            assert is_minor == text.startswith("-"), (text, parsed.name)
    for bad in ("CM11", "CM7b5", "C9sus4", "C7alt"):
        with pytest.raises(ValueError):
            parse_chord_symbol(bad)


# ------------------------------------------------------------ spelling agreement

@pytest.mark.parametrize("tonic", ["F", "Bb", "Eb", "Ab", "Db", "Gb", "G", "D", "A", "E", "B"])
def test_matches_keep_the_callers_spelling(tonic):
    scale = [n.name for n in __import__("midi_composer_mcp.scales", fromlist=["x"]).scale_notes(
        resolve_scale_type("major"), parse_note(tonic))][:-1]
    # the exact major-scale match is named after the tonic as spelled
    names = [m["name"] for m in __import__("midi_composer_mcp.scales", fromlist=["x"]).match_scales(
        scale, exact_only=True, limit=50)["matches"]]
    assert f"{tonic} major" in names
    # a chord root the caller did not write is spelled to keep as many of their notes
    # as any spelling of that root could (D F -> Bb D F, never A# C## E#)
    from midi_composer_mcp.chords import match_chords, resolve_chord_type
    from midi_composer_mcp.notes import Note
    dyad = [scale[1], scale[3]]
    for m in match_chords(dyad, limit=60)["matches"]:
        if m["root"] in dyad:
            continue
        ctype = resolve_chord_type(m["chord_type"])
        agree = lambda notes: sum(n in notes for n in dyad)  # noqa: E731
        pc = parse_note(m["root"]).pitch_class
        options = [Note(l, a) for l in "CDEFGAB" for a in (-1, 0, 1)
                   if (parse_note(l).pitch_class + a) % 12 == pc]
        best = max(agree([t.name for t in chord_notes(ctype, o)]) for o in options)
        assert agree(m["notes"]) == best, m
    # harmonize_melody's chords contain the melody note exactly as written
    for entry in harmonize_melody(scale[:4])["chords"]:
        assert entry["melody_note"] in entry["notes"]


@pytest.mark.parametrize("semitones", range(-6, 7))
def test_transpose_notes_moves_every_note_by_the_same_letters(semitones):
    from midi_composer_mcp.melody import transpose_notes
    for tonic in ["C", "F", "Bb", "D", "E"]:
        scale = notes_from_degrees(tonic + "4", "major", [1, 2, 3, 4, 5, 6, 7])["notes"]
        moved = [parse_note(n) for n in transpose_notes(scale, semitones)["notes"]]
        shifts = {("CDEFGAB".index(m.letter) - "CDEFGAB".index(parse_note(o).letter)) % 7
                  for m, o in zip(moved, scale)}
        assert len(shifts) == 1, (tonic, semitones, moved)
        assert [m.midi - parse_note(o).midi for m, o in zip(moved, scale)] == [semitones] * 7


@pytest.mark.parametrize("scale", HEPTATONIC)
def test_respelling_never_moves_a_pitch(scale):
    """snap_to_scale respells in-scale notes (B -> Cb) but must keep the exact pitch."""
    for tonic in ["C#", "Gb", "Eb", "Ab", "B", "F"]:
        probe = ["B3", "C4", "B#3", "Cb4", "E4", "Fb4", "E#4", "F4"]
        out = snap_to_scale(probe, tonic, scale)["notes"]
        t = parse_note(tonic).pitch_class
        scale_pcs = {(t + i) % 12 for i in resolve_scale_type(scale).intervals}
        for src, dst in zip(probe, out):
            if parse_note(src).pitch_class in scale_pcs:
                assert parse_note(dst).midi == parse_note(src).midi, (tonic, scale, src, dst)


# --------------------------------------------------------------- MIDI range

@pytest.mark.parametrize("call", [
    lambda: notes_from_degrees("G9", "major", [1, 2, 3]),
    lambda: notes_from_degrees("C0", "major", [-15]),
    lambda: __import__("midi_composer_mcp.chords", fromlist=["x"]).chord_info("13", "G9"),
    lambda: diatonic_chords("C8", "major", True),
    lambda: __import__("midi_composer_mcp.melody", fromlist=["x"]).arpeggiate_notes(["C9", "E9"], octaves=2),
    lambda: __import__("midi_composer_mcp.scales", fromlist=["x"]).scale_info("major", "Cb-1"),
    lambda: melodic_sequence(["C4", "E4"], "C", "major", step=-7, count=64),
])
def test_out_of_range_pitches_raise(call):
    with pytest.raises(ValueError):
        call()


# ------------------------------------------------------- rendering contracts

def test_rendered_files_last_as_long_as_reported(tmp_path):
    import mido
    from midi_composer_mcp.generate import GROOVES
    from midi_composer_mcp.midi_io import render_drums, render_notes
    for name, (pattern, step, _desc) in GROOVES.items():
        r = render_drums({"clave": pattern}, step_beats=step, output_dir=str(tmp_path), file_name="g.mid")
        assert abs(mido.MidiFile(r["file"]).length - r["duration_seconds"]) < 1e-6, name
    r = render_notes(["C4"], rhythm="O.......", output_dir=str(tmp_path), file_name="n.mid")
    assert abs(mido.MidiFile(r["file"]).length - r["duration_seconds"]) < 1e-6


def test_song_parts_keep_their_instruments(tmp_path):
    import mido
    song = render_song_structure({
        "verse": {"bars": 1, "tracks": [
            {"type": "notes", "notes": ["C4"], "step_beats": 4, "program": 48},
            {"type": "notes", "notes": ["C3"], "step_beats": 4, "program": 33},
            {"type": "chords", "name": "keys", "chords": ["C"], "program": 0}]},
        "bridge": {"bars": 1, "tracks": [{"type": "chords", "name": "keys", "chords": ["F"], "program": 48}]},
    }, form="verse bridge", output_dir=str(tmp_path))
    by_name = {t["name"]: t for t in song["tracks"]}
    assert by_name["notes_1"]["program"] == 48 and by_name["notes_2"]["program"] == 33
    assert by_name["notes_1"]["channel"] != by_name["notes_2"]["channel"]
    assert by_name["keys"]["program_changes"] == [48]
    changes = [(m.program, m.time) for t in mido.MidiFile(song["file"]).tracks for m in t
               if m.type == "program_change" and m.channel == by_name["keys"]["channel"]]
    assert [p for p, _ in changes] == [0, 48]
    with pytest.raises(ValueError, match="drums"):
        render_song_structure({"a": {"tracks": [{"type": "drums", "name": "x", "lanes": {"kick": "O"}}]},
                               "b": {"tracks": [{"type": "notes", "name": "x", "notes": ["C4"]}]}},
                              output_dir=str(tmp_path))


def test_nearest_octaves_do_not_drift():
    from midi_composer_mcp.midi_io import assign_octaves
    for octave in (2, 4):
        placed = assign_octaves(parse_notes(["C", "G", "A", "F"] * 16), octave, "nearest")
        midis = [n.midi for n in placed]
        assert midis[-4:] == midis[-8:-4]            # settles into a repeating register
        assert max(midis) - min(midis) <= 48 and min(midis) >= 0


@pytest.mark.parametrize("line,expected", [
    ("G A B C D E F# G", ["G4", "A4", "B4", "C5", "D5", "E5", "F#5", "G5"]),
    ("C B A G F E D C", ["C4", "B3", "A3", "G3", "F3", "E3", "D3", "C3"]),
    ("C E G C E G C", ["C4", "E4", "G4", "C5", "E5", "G5", "C6"]),
    ("C6 B A G", ["C6", "B5", "A5", "G5"]),
])
def test_nearest_octaves_leave_ordinary_lines_alone(line, expected):
    from midi_composer_mcp.midi_io import assign_octaves
    assert [n.name for n in assign_octaves(parse_notes(line), 4, "nearest")] == expected


def test_first_octave_follows_the_letter():
    from midi_composer_mcp.midi_io import assign_octaves, voice_chord
    assert assign_octaves(parse_notes(["Cb"]), 4, "nearest")[0].name == "Cb4"
    assert assign_octaves(parse_notes(["B#"]), 4, "nearest")[0].name == "B#4"
    v = voice_chord(parse_notes(["C", "E", "G"]), 4, parse_note("E3"))
    assert v[0].name == "E3" and all(n.midi > v[0].midi for n in v[1:])


def test_voice_leading_minimizes_total_movement():
    import itertools as it
    from midi_composer_mcp.midi_io import voice_chord
    symbols = ["C", "Am7", "Fmaj7", "G7", "Dm7", "E7", "Bbmaj7", "Ebm", "F#m7b5", "Ab"]
    for a, b in it.permutations(symbols, 2):
        prev, cur = voice_leading([a, b])["voicings"]
        tones = [t.without_octave() for t in parse_chord_symbol(b)[1:2][0] and
                 __import__("midi_composer_mcp.chords", fromlist=["x"]).chord_notes(
                     parse_chord_symbol(b)[1], parse_chord_symbol(b)[0])]
        cost = lambda m: sum(abs(p - c) for p, c in zip(sorted(prev["midi"]), sorted(m)))  # noqa: E731
        best = min(cost([v.midi for v in voice_chord(tones[r:] + tones[:r], o)])
                   for o in (3, 4, 5) for r in range(len(tones)))
        if len(prev["midi"]) == len(cur["midi"]):
            assert cost(cur["midi"]) == best, (a, b)


@pytest.mark.parametrize("tonic", TONICS + ["Cb", "C#"])
def test_circle_neighbours_are_spelled_fifths(tonic):
    from midi_composer_mcp.circle import circle_of_fifths
    f = circle_of_fifths(tonic)["focus"]
    key = f["key"].split()[0]
    dom, sub = interval_between(key, f["dominant"]), interval_between(key, f["subdominant"])
    assert dom["semitones"] % 12 == 7 and sub["semitones"] % 12 == 5
    if f["fifths"] < 7:   # otherwise the dominant has 8 sharps and is given enharmonically
        assert dom["short"] == "P5"
    if f["fifths"] > -7:
        assert sub["short"] == "P4"
    assert len(f["accidentals"]) == abs(f["fifths"]) <= 7


def test_random_notes_pool_counts_each_note_once():
    from midi_composer_mcp.generate import random_notes
    from midi_composer_mcp.scales import scale_info
    pool = scale_info("minor pentatonic", "A")["notes"]  # A C D E G A
    for seed in range(50):
        picks = random_notes(pool, count=5, allow_repeats=False, seed=seed)["notes"]
        assert len(set(picks)) == 5
    with pytest.raises(ValueError):
        random_notes(pool, count=6, allow_repeats=False, seed=1)


# ------------------------------------------- only ValueErrors, never crashes

_BAD_VALUES = [None, 7, -1, 3.5, 2.0, True, "", "zz", [], [None], {}, {"a": 1}, ["C", 5], "C" * 5,
               ["C", "Db", "D"], [["C", "Db", "D"], "G"], "0 4 7", ["C", ".", "E"]]  # clusters, numbers, rests


def test_tools_reject_bad_input_with_value_errors(tmp_path):
    """Every MCP tool meets malformed input with a clear ValueError, never TypeError/KeyError/StopIteration."""
    import inspect
    from midi_composer_mcp import server
    good = {"notes": ["C4", "E4", "G4"], "chords": ["C", "G"], "root": "C", "scale_type": "major",
            "degrees": [1, 3, 5], "triad": "C", "melody": ["C5", "D5"], "cantus": ["C4", "D4", "C4"],
            "form": "verse", "motifs": {"A": "C4 D4"}, "lanes": {"kick": "O..."}, "tracks": [{"type": "notes", "notes": ["C4"]}],
            "sections": {"v": {"tracks": [{"type": "notes", "notes": ["C4"]}]}}, "chord_type": "m7",
            "note_a": "C", "note_b": "E", "target": "Dm", "symbol": "G7", "tonic": "C", "name": "tresillo",
            "pulses": 3, "midi_file": None, "melody_notes": ["C5"], "semitones": 2, "mode": 1,
            "text": "Ut queant laxis", "operations": "PLR", "key": "C", "chord": "C", "row": "C C# D D# E F F# G G# A A# B"}
    skip = {"midi_to_audio", "list_scales", "list_chords", "list_grooves"}
    failures = []
    for name, fn in inspect.getmembers(server, inspect.isfunction):
        if fn.__module__ != server.__name__ or name in skip or name == "main":
            continue
        params = inspect.signature(fn).parameters
        base = {p: good[p] for p in params if p in good and params[p].default is inspect.Parameter.empty}
        if "output_dir" in params:
            base["output_dir"] = str(tmp_path)
        for pname in params:
            if pname in ("output_dir", "file_name"):
                continue
            for bad in _BAD_VALUES:
                try:
                    fn(**dict(base, **{pname: bad}))
                except ValueError:
                    pass
                except Exception as e:  # collect them all: the list is the point
                    failures.append(f"{name}({pname}={bad!r}) raised {type(e).__name__}: {e}")
    assert not failures, "\n".join(failures)


# ------------------------------------------- regressions found by the review pass

def test_transposition_spelling_edge_cases():
    from midi_composer_mcp.melody import transpose_notes
    assert transpose_notes("C#4 E#4 G#4", 12)["notes"] == ["C#5", "E#5", "G#5"]   # octave: never respelled
    assert transpose_notes("C# E# G#", 0)["notes"] == ["C#", "E#", "G#"]
    assert transpose_notes(["E4", "G#4", "B4", "D5"], -1)["notes"] == ["Eb4", "G4", "Bb4", "Db5"]
    assert transpose_notes("B3 D#4 F#4", -1)["notes"] == ["Bb3", "D4", "F4"]


def test_enharmonic_cantus_is_read_in_the_key():
    r = species_counterpoint(["Eb4", "D#4", "F4", "Eb4"], "Eb", "major", species=1)  # D# written for Eb
    assert "warning" not in r and r["cantus"][1] == "Eb4"


def test_motif_inversion_mirrors_spelling():
    inv = motif_grammar("AB", {"A": "F4 G4 A4 Bb4", "B": {"vary": "A", "invert": True}})["motifs"]["B"]
    assert inv == ["F4", "Eb4", "Db4", "C4"]


@pytest.mark.parametrize("tonic,chords,expected", [
    ("C#", ["Db", "Gb", "Ab"], ["I", "IV", "V"]),
    ("Gb", ["F#", "B", "C#"], ["I", "IV", "V"]),
])
def test_enharmonic_scale_degrees_keep_their_numerals(tonic, chords, expected):
    res = analyze_progression(chords, tonic, "major")["chords"]
    assert [c["roman"] for c in res] == expected
    assert [c["degree"] for c in res] == [1, 4, 5]


def test_voice_leading_ties_keep_common_tones():
    vl = voice_leading(["Cmaj7", "Dm7"])["voicings"]
    assert "C4" in vl[1]["notes"]


def test_diatonic_slash_entries_are_self_consistent():
    c = diatonic_chords("C", "major pentatonic")["chords"][0]
    assert c["symbol"] == "Am/C" and c["root"] == "A" and c["bass"] == "C" and c["chord_type"] == "minor"


def test_degree_strings_with_spaced_dashes():
    assert degrees_to_chords("C", "major", "I - V - vi - IV")["symbols"] == ["C", "G", "Am", "F"]
    assert degrees_to_chords("C", "major", "ii–V–I")["symbols"] == ["Dm", "G", "C"]


def test_inversion_names_only_for_real_inversions():
    from midi_composer_mcp.chords import match_chords
    def inv(notes, symbol):
        return {m["symbol"]: m.get("inversion") for m in match_chords(notes, include_partial=False)["matches"]}[symbol]
    assert inv("F C G", "Csus4/F") == "first inversion"      # the sus 4th stands in for the 3rd
    assert inv("G C E", "C/G") == "second inversion"
    assert inv("A C E G", "C6/A") == "third inversion"       # never more inversions than notes - 1
    assert inv("G C", "C5/G") == "first inversion"
    assert inv("D C E G", "Cadd9/D") == "third inversion"


@pytest.mark.parametrize("symbol,expected", [
    ("G5sus4", "suspended 4"), ("C5Maj7", "major 7"), ("CmAdd9", "minor add 9"), ("CMadd9", "add 9"), ("C-Δ7", "minor major 7"),
    ("Cdominant-7", "dominant 7"), ("CM6/9", "six nine"),
])
def test_symbol_parsing_after_review(symbol, expected):
    assert parse_chord_symbol(symbol)[1].name == expected
