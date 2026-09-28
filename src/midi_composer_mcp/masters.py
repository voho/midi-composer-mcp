"""Harmony rules named after the composers and theorists who codified them.

- Rameau (Traité de l'harmonie, 1722): the fundamental bass — the chain of chord
  roots under a progression — and his cadences.
- Schoenberg (Structural Functions of Harmony, 1954): ascending (strong),
  descending and superstrong root progressions.
- Bach: four-part chorale writing (the voice-leading rules taught from his 371
  chorales) as a voicing engine and a checker.
- Riemann / neo-Riemannian theory (Cohn, Lewin): the P, L and R transformations
  between major and minor triads.
- Bartók (as analysed by Ernő Lendvai): the axis system of tonic, subdominant and
  dominant axes.
- Coltrane: the "Giant Steps" substitution — tonal centres a major third apart.

Every function is a deterministic application of the named rule set.
"""

from __future__ import annotations

from functools import lru_cache

from .chords import CHORDS, chord_notes, parse_chord_symbol
from .harmony import _root_and_quality, interval_between
from .midi_io import _parse_chord_list, _with_octave
from .notes import Note, parse_note, parse_notes, spell_pitch_class, transpose
from .roman import _is_above, _steps
from .scales import resolve_scale_type, scale_notes

# ------------------------------------------------------------ helpers

_MOTION = {0: "unison", 1: "second up", 2: "second up", 3: "third up", 4: "third up", 5: "fourth up",
           6: "tritone", 7: "fifth up", 8: "third down", 9: "third down", 10: "second down", 11: "second down"}


def _chords(chords) -> list[tuple[Note, object, Note | None, str]]:
    items = chords
    if isinstance(items, str):
        items = [t for t in items.replace(",", " ").split() if t]
    if not isinstance(items, (list, tuple)) or not items:
        raise ValueError("chords must be a non-empty list of chord symbols or note arrays")
    out = []
    for item in items:
        root, ctype, bass = _root_and_quality(item)
        label = item if isinstance(item, str) else " ".join(parse_note(n).name for n in item)
        out.append((root.without_octave(), ctype, bass, label))
    return out


def _root_motion(a: Note, b: Note) -> tuple[int, str]:
    """Semitones and a name for the root motion a -> b, by the sounding interval (not the spelling)."""
    semis = (b.pitch_class - a.pitch_class) % 12
    return semis, _MOTION[semis]


def _plain(n: Note) -> Note:
    """Respell a double accidental, or E#/B#/Fb/Cb, as the simpler enharmonic (Fb7 -> E7)."""
    if abs(n.accidental) >= 2 or (n.accidental and n.letter + ("#" if n.accidental > 0 else "b") in ("E#", "B#", "Fb", "Cb")):
        return spell_pitch_class(n.pitch_class, prefer_flats=n.accidental < 0)
    return n


# ------------------------------------------------------------ Rameau

def rameau_fundamental_bass(chords, root: str | None = None, scale_type: str = "major") -> dict:
    """Rameau's fundamental bass: the chord roots beneath a progression, and his cadences.

    Rameau (1722) heard every chord as built on a root, whatever its inversion,
    and judged progressions by how that 'fundamental bass' moves: best by fifths
    (falling a fifth is the 'cadence parfaite' motion), then by thirds, while
    steps are a licence. Motions are named by the sounding interval. With a
    seven-note key (`root`/`scale_type`) the classic cadences are named: V(7)→I
    cadence parfaite, a fundamental bass rising a fifth (IV→I or I→V) cadence
    irrégulière, V→vi (a step up) cadence rompue. Returns the
    fundamental bass (a notes list you can render as a bass track), each root
    motion and any cadences. Deterministic.
    """
    items = _chords(chords)
    tonic = parse_notes(root)[0].without_octave() if root is not None else None
    scale = resolve_scale_type(scale_type) if root is not None else None
    degrees = {}
    if tonic is not None and len(scale.intervals) == 7:  # degree numbers only mean V, IV, vi in a 7-note key
        degrees = {n.pitch_class: i + 1 for i, n in enumerate(scale_notes(scale, tonic)[:-1])}
    motions = []
    for (a, at, _ab, al), (b, bt, _bb, bl) in zip(items, items[1:]):
        semis, name = _root_motion(a, b)
        kind = ("fifth" if name in ("fourth up", "fifth up") else "third" if "third" in name
                else "step" if "second" in name else name)
        entry = {"from": al, "to": bl, "roots": [a.name, b.name], "motion": name, "kind": kind}
        if name == "fourth up":
            entry["motion"] = "fifth down (fourth up)"
        if degrees:
            da, db = degrees.get(a.pitch_class), degrees.get(b.pitch_class)
            dominant_quality = at is not None and 4 in at.intervals and 3 not in at.intervals  # (#9 = 15, not 3)
            if da == 5 and db == 1 and name == "fourth up" and dominant_quality:
                entry["cadence"] = "cadence parfaite (perfect cadence: the fundamental bass falls a fifth to the tonic)"
            elif name == "fifth up" and ((da, db) in ((4, 1), (1, 5))):
                entry["cadence"] = "cadence irrégulière (irregular cadence: the fundamental bass rises a fifth)"
            elif da == 5 and db == 6 and name == "second up":
                entry["cadence"] = "cadence rompue (broken/deceptive cadence: V to vi)"
        motions.append(entry)
    counts = {k: sum(1 for m in motions if m["kind"] == k) for k in ("fifth", "third", "step")}
    return {
        "fundamental_bass": [a.name for a, *_ in items],
        "motions": motions,
        "summary": counts,
        "note": "Rameau preferred root motion by fifths, then thirds; stepwise root motion he explained as an "
                "elided fifth.",
    }


# ------------------------------------------------------------ Schoenberg

def schoenberg_progressions(chords) -> dict:
    """Classify each root progression as Schoenberg did (Structural Functions of Harmony, 1954).

    Ascending (strong) progressions: the root rises a fourth or falls a third —
    the new chord contains the old root as a lesser member, so harmony moves
    forward. Descending progressions: the root rises a fifth or rises a third
    (later writers call them 'weak'; Schoenberg avoided the word).
    Superstrong ('skipping') progressions: the root moves a second up or down,
    sharing no tones. Root motion is read by its sounding interval, so the
    spelling does not matter (Ab→B is a third up). Returns each step's class
    and a summary; Schoenberg advised building mostly on ascending
    progressions. Deterministic.
    """
    items = _chords(chords)
    steps = []
    for (a, _at, _ab, al), (b, _bt, _bb, bl) in zip(items, items[1:]):
        _semis, name = _root_motion(a, b)
        cls = {"fourth up": "ascending (strong)", "third down": "ascending (strong)",
               "fifth up": "descending", "third up": "descending",
               "second up": "superstrong", "second down": "superstrong"}.get(name, "none (same root)" if name == "unison" else "tritone")
        steps.append({"from": al, "to": bl, "root_motion": name, "class": cls})
    summary = {}
    for s in steps:
        summary[s["class"]] = summary.get(s["class"], 0) + 1
    return {"progressions": steps, "summary": summary}


# ------------------------------------------------------------ Bach chorale (SATB)

RANGES = {"soprano": (60, 79), "alto": (55, 74), "tenor": (48, 67), "bass": (40, 60)}  # C4-G5, G3-D5, C3-G4, E2-C4
_VOICES = ("soprano", "alto", "tenor", "bass")


def _pcs_of(tones):
    return [t.pitch_class for t in tones]


def _roles(tones: list[Note]) -> tuple[int | None, int | None]:
    """(fifth pc, seventh pc) of a chord, read from intervals above the root, not list positions.

    A minor or major seventh counts only when it is spelled as a seventh (on the 7th letter above
    the root: the Gb of Ab7), so the F# of an augmented sixth (Ab C Eb F#, an A6 above Ab) is no
    chord seventh — letters and semitones, not semitones alone."""
    root = tones[0]
    rel = {(t.pitch_class - root.pitch_class) % 12: t.pitch_class for t in tones}
    fifth = next((rel[i] for i in (7, 6, 8) if i in rel), None)
    seventh = next((rel[i] for i in (10, 11) if any(
        (t.pitch_class - root.pitch_class) % 12 == i and _steps(t, root) == 6 for t in tones)), None)
    if seventh is None and 9 in rel and 3 in rel and 6 in rel:  # the diminished seventh (bb7)
        seventh = rel[9]
    return fifth, seventh


def _augmented_sixth(tones: list[Note], bass: Note) -> tuple[Note, Note] | None:
    """(b6, #4) of an augmented-sixth chord, or None: read by spelling, in any key.

    It6 (Ab C F# in C), Fr43 (+ D), Ger65 (+ Eb) and Sw43 (+ D#) put b6 in the bass (Ab), with the
    major third above it (C) and the note an augmented sixth above it (the 6th letter, 10
    semitones: F#, not the Gb seventh of Ab7) — roman.special_reading's reading, key-free, so an
    applied augmented sixth counts too. Kostka & Payne, Aldwell & Schachter: the augmented sixth
    expands to the octave on ^5 — the upper note (#4) rises a semitone, the bass (b6) falls a
    semitone. (C E Gb Bb with C in the bass is a V7b5, not a French sixth.)
    """
    lo = bass.without_octave()
    if not any(_is_above(t, lo, 4, 2) for t in tones):
        return None
    hi = next((t for t in tones if _is_above(t, lo, 10, 5)), None)
    return (lo, hi.without_octave()) if hi is not None else None


@lru_cache(maxsize=None)
def _melodic_augmented(a: str, b: str) -> str | None:
    """The name of the melodic interval a -> b (concrete pitches such as 'Ab3', 'B3') when it is
    augmented — 'augmented second' (Ab–B), 'augmented fourth' (F–B) … — else None. The chromatic
    A1 (C–C#) is fine (Kostka & Payne). Shared by bach_chorale_voicing and check_voice_leading."""
    iv = interval_between(a, b)
    if iv["quality"] in ("augmented", "doubly augmented") and iv["number"] != 1:
        return iv["name"]
    return None


def _voicings(tones: list[Note], bass_pc: int, root_pc: int, avoid_double: set[int],
              soprano: int | None = None) -> list[tuple[int, int, int, int]]:
    """All SATB spacings of a chord: every chord tone present (the 5th may drop from a 7th chord),
    bass on `bass_pc`, ranges and spacing respected, no crossing, none of `avoid_double` doubled."""
    pcs = _pcs_of(tones)
    need = set(pcs)
    fifth, seventh = _roles(tones)
    perfect = fifth is not None and (fifth - root_pc) % 12 == 7  # an altered fifth (b5, #5) defines the chord
    optional = {fifth} if (seventh is not None and perfect and len(need) >= 4) else set()
    lo_b, hi_b = RANGES["bass"]
    out = []
    for b in range(lo_b, hi_b + 1):
        if b % 12 != bass_pc:
            continue
        for t in range(max(b + 1, RANGES["tenor"][0]), min(b + 19, RANGES["tenor"][1]) + 1):
            if t % 12 not in need:
                continue
            for a in range(max(t + 1, RANGES["alto"][0]), min(t + 12, RANGES["alto"][1]) + 1):
                if a % 12 not in need:
                    continue
                for s in range(max(a + 1, RANGES["soprano"][0]), min(a + 12, RANGES["soprano"][1]) + 1):
                    if s % 12 not in need or (soprano is not None and s != soprano):
                        continue
                    present = {b % 12, t % 12, a % 12, s % 12}
                    if not (need - optional) <= present:
                        continue
                    voices = [s % 12, a % 12, t % 12, b % 12]
                    doubled = {pc for pc in voices if voices.count(pc) > 1}
                    if doubled & avoid_double:
                        continue
                    out.append((s, a, t, b))
    return out


# MIDI-level voice-leading facts for two voices moving from one chord to the next (p_* before,
# c_* after; *_hi is the upper voice, *_lo the lower, as MIDI numbers). bach_chorale_voicing and
# analysis.check_voice_leading share them, so the two tools cannot disagree about these facts;
# the checker adds the spelled-interval test (a real P5, not a d6) on top.

def _consecutive_perfect(p_hi: int, p_lo: int, c_hi: int, c_lo: int) -> tuple[str, int] | None:
    """Two voices a perfect fifth or octave apart (7 or 0 semitones mod 12, compound or not) in
    both chords, both moving: ('parallel', 7|0) in similar motion, ('contrary', 7|0) in contrary
    motion; None otherwise (a held voice is oblique motion, not a fault)."""
    p_int = abs(p_hi - p_lo) % 12
    c_int = abs(c_hi - c_lo) % 12
    if c_int not in (0, 7) or p_int != c_int:
        return None
    motion = (c_hi - p_hi) * (c_lo - p_lo)
    if motion > 0:
        return "parallel", c_int
    if motion < 0:
        return "contrary", c_int
    return None


def _similar_leap(p_hi: int, p_lo: int, c_hi: int, c_lo: int) -> bool:
    """Similar motion (both voices moving the same way) with the upper voice leaping (more than 2
    semitones) — the motion of a direct (hidden) fifth or octave."""
    return (c_hi - p_hi) * (c_lo - p_lo) > 0 and abs(c_hi - p_hi) > 2


def _direct_perfect(p_hi: int, p_lo: int, c_hi: int, c_lo: int) -> bool:
    """A direct (hidden) fifth/octave: similar motion into a perfect fifth or octave (mod 12) from a
    different interval, with the upper voice leaping (more than 2 semitones). A true parallel is
    _consecutive_perfect's case, not this one. (check_voice_leading judges 'a different interval'
    by spelling instead: a d6 moving to a P5 is a direct fifth.)"""
    outer = abs(c_hi - c_lo) % 12
    return outer in (0, 7) and abs(p_hi - p_lo) % 12 != outer and _similar_leap(p_hi, p_lo, c_hi, c_lo)


def _overlap(p_hi: int, p_lo: int, c_hi: int, c_lo: int) -> bool:
    """Voice overlap (music21): the upper voice moves below where the lower just was, or the lower
    voice moves above where the upper just was."""
    return c_hi < p_lo or c_lo > p_hi


def _pair_parallels(prev, cur) -> list[str]:
    """Consecutive perfect fifths/octaves between two voices, in similar or in contrary motion."""
    bad = []
    for i in range(4):
        for j in range(i + 1, 4):
            hit = _consecutive_perfect(prev[i], prev[j], cur[i], cur[j])
            if hit is None:
                continue
            kind, size = hit
            what = "octaves" if size == 0 else "fifths"
            if kind == "parallel":
                bad.append(f"parallel {what} {_VOICES[i]}/{_VOICES[j]}")
            else:
                bad.append(f"consecutive {what} by contrary motion {_VOICES[i]}/{_VOICES[j]}")
    return bad


def _transition_faults(prev, cur, leading_pc, tonic_pc, prev_seventh_pc, prev_aug6=None,
                       names=None) -> tuple[list[str], int]:
    """Hard faults (rule breaks) and a style cost for moving from one SATB chord to the next.

    `prev_aug6`: the (b6, #4) pitch classes when the first chord is an augmented sixth. `names`: a
    pair of {midi: spelled name} maps for the two chords, for the melodic-augmented-interval rule."""
    faults = _pair_parallels(prev, cur)
    # direct (hidden) fifths/octaves in the outer voices with a leap in the soprano
    if _direct_perfect(prev[0], prev[3], cur[0], cur[3]):
        faults.append("direct fifth/octave in the outer voices with a soprano leap")
    # voice overlap: a voice moves past where its neighbour just was
    for i in range(3):
        if _overlap(prev[i], prev[i + 1], cur[i], cur[i + 1]):
            faults.append(f"overlap {_VOICES[i]}/{_VOICES[i + 1]}")
    # the leading tone in an outer voice rises to the tonic when the next chord holds the tonic
    for i in (0,):
        if prev[i] % 12 == leading_pc and tonic_pc in {x % 12 for x in cur} and cur[i] != prev[i] + 1:
            faults.append("leading tone in the soprano does not rise to the tonic")
    # a chord seventh resolves down by step
    if prev_seventh_pc is not None:
        for i in range(4):
            if prev[i] % 12 == prev_seventh_pc and not (1 <= prev[i] - cur[i] <= 2) and cur[i] != prev[i]:
                faults.append(f"the seventh in the {_VOICES[i]} does not resolve down by step")
    # an augmented sixth expands to the octave on ^5: #4 rises a semitone, b6 falls a semitone
    if prev_aug6 is not None:
        low_pc, high_pc = prev_aug6
        for i in range(4):
            if cur[i] == prev[i]:
                continue   # held, like a held seventh
            if prev[i] % 12 == high_pc and cur[i] != prev[i] + 1:
                faults.append(f"the augmented sixth's upper note (#4) in the {_VOICES[i]} does not rise a semitone")
            elif prev[i] % 12 == low_pc and cur[i] != prev[i] - 1:
                faults.append(f"the augmented sixth's lower note (b6) in the {_VOICES[i]} does not fall a semitone")
    # no melodic augmented interval (A2, A4 …) in any voice, judged by the spelled notes
    if names is not None:
        before, after = names
        for i in range(4):
            if cur[i] != prev[i]:
                aug = _melodic_augmented(before[prev[i]], after[cur[i]])
                if aug is not None:
                    faults.append(f"melodic {aug} in the {_VOICES[i]}")
    cost = abs(cur[1] - prev[1]) * 2 + abs(cur[2] - prev[2]) * 2 + abs(cur[0] - prev[0]) + abs(cur[3] - prev[3]) // 2
    cost += sum(3 for i in range(3) if abs(cur[i] - prev[i]) > 4)  # inner-voice leaps
    return faults, cost


def bach_chorale_voicing(chords, root: str | None = None, scale_type: str = "major", melody=None) -> dict:
    """Voice a progression in four parts (SATB) by the rules of Bach-chorale writing.

    Rules applied: soprano C4–G5, alto G3–D5, tenor C3–G4, bass E2–C4; no voice
    crossing; soprano–alto and alto–tenor within an octave, tenor–bass within a
    twelfth; every chord tone present (a seventh chord may drop its fifth); the
    bass takes the chord root, or the slash bass ('C/E'; a note array's first
    note when it is not the root); never double the leading tone or a chord
    seventh (spelled as a seventh: the F# of Ab C Eb F# is no seventh) or an
    augmented-sixth tone; no parallel fifths or octaves between any two voices, nor
    consecutive ones by contrary motion (the strict textbook rule; Bach himself
    occasionally allows contrary octaves at a cadence); no direct
    fifths/octaves in the outer voices with a soprano
    leap; no voice overlap; no melodic augmented interval (A2, A4 — the
    b6→#7 step in minor; Kostka & Payne), judged by the spelled notes; a
    soprano leading tone rises to the tonic; a chord seventh resolves down by
    step; an augmented sixth (It6, Fr43, Ger65, Sw43: b6 in the bass with the
    note an augmented sixth above it, Ab–F# in C, read by spelling) expands
    to the octave on ^5, #4 rising and b6 falling a semitone (Kostka & Payne;
    Aldwell & Schachter). Among the voicings that obey all rules the one with
    the smoothest inner voices wins (an exact search). Pass the key
    (`root`) for the leading-tone rule. Pass `melody` (one note per chord) to
    harmonize a chorale tune: the soprano then sings exactly that melody, and
    each chord must contain its melody note. Returns the four voices, per-chord SATB
    notes, any rule the search could not avoid, and a four-track `render_hint`.
    """
    parsed = _parse_chord_list(chords)
    for ch in parsed:  # a note array: find its root (it may be written in inversion)
        if ch["symbol"] is None:
            ch["label"] = " ".join(t.name for t in ch["tones"])
            root_note, ctype, _b = _root_and_quality([t.name for t in ch["tones"]])
            k = next((i for i, t in enumerate(ch["tones"]) if t.pitch_class == root_note.pitch_class), 0)
            if ctype is not None and k:
                ch["bass"] = ch["tones"][0].without_octave()
                ch["tones"] = ch["tones"][k:] + ch["tones"][:k]
    sop = None
    if melody is not None:
        from .midi_io import assign_octaves  # local import avoids a cycle
        sop = assign_octaves(parse_notes(melody), 4, "nearest")
        if len(sop) != len(parsed):
            raise ValueError(f"melody has {len(sop)} notes but there are {len(parsed)} chords (one note per chord)")
        for n, ch in zip(sop, parsed):
            if n.pitch_class not in {t.pitch_class for t in ch["tones"]}:
                raise ValueError(f"melody note {n.name} is not a tone of {ch['symbol'] or 'the chord'}")
        while any(n.midi < RANGES["soprano"][0] for n in sop):   # lift the tune into the soprano range
            sop = [Note(n.letter, n.accidental, n.octave + 1) for n in sop]
        while any(n.midi > RANGES["soprano"][1] for n in sop):
            sop = [Note(n.letter, n.accidental, n.octave - 1) for n in sop]
    tonic = parse_notes(root)[0].without_octave() if root is not None else None
    scale = resolve_scale_type(scale_type)
    key_label = f"{tonic.name} {scale.name}" if tonic is not None else None
    if tonic is None:  # read the key from the last chord's root (and its third)
        last = parsed[-1]["tones"]
        tonic = last[0].without_octave()
        minor = any((t.pitch_class - tonic.pitch_class) % 12 == 3 for t in last) and \
            not any((t.pitch_class - tonic.pitch_class) % 12 == 4 for t in last)
        key_label = f"{tonic.name} {'minor' if minor else 'major'} (from the final chord)"
    leading_pc = (tonic.pitch_class - 1) % 12
    cands, sevenths, aug6s, spelled = [], [], [], []
    big = 1000
    for k, ch in enumerate(parsed):
        tones = [t.without_octave() for t in ch["tones"]]
        bass = ch["bass"] or tones[0]
        fifth_pc, seventh_pc = _roles(tones)
        if (seventh_pc is None and bass.pitch_class not in {t.pitch_class for t in tones}
                and (bass.pitch_class - tones[0].pitch_class) % 12 in (10, 11)
                and _steps(bass.without_octave(), tones[0]) == 6):
            seventh_pc = bass.pitch_class  # a slash bass on the seventh (Am/G) is a seventh too
        aug6 = _augmented_sixth(tones, bass)
        aug6_pcs = (aug6[0].pitch_class, aug6[1].pitch_class) if aug6 is not None else None
        avoid = {leading_pc} | ({seventh_pc} if seventh_pc is not None else set()) | set(aug6_pcs or ())
        # every MIDI note a voice may sing in this chord, spelled as the chord writes it
        tone_names = tones + ([ch["bass"].without_octave()] if ch["bass"] else [])
        by_pc = {}
        for t in tone_names:
            by_pc.setdefault(t.pitch_class, t)
        spelled.append({m: _with_octave(by_pc[m % 12], m).name
                        for m in range(RANGES["bass"][0], RANGES["soprano"][1] + 1) if m % 12 in by_pc})
        pin = sop[k].midi if sop is not None else None
        options = _voicings(tones, bass.pitch_class, tones[0].pitch_class, set(), pin)
        if not options:
            label = ch["symbol"] or ch["label"]
            required = {t.pitch_class for t in tones} | {bass.pitch_class}
            if seventh_pc is not None and fifth_pc is not None and (fifth_pc - tones[0].pitch_class) % 12 == 7 \
                    and fifth_pc != bass.pitch_class:
                required.discard(fifth_pc)
            if len(required) > 4:
                raise ValueError(f"{label} needs {len(required)} different tones, more than four voices can sing "
                                 "(voice a simpler chord, e.g. a 7th instead of a 9th or 13th)")
            raise ValueError(f"no four-part voicing of {label} fits the SATB ranges"
                             + (" with that melody note" if pin is not None else ""))

        def doubling(v, tones=tones, fifth_pc=fifth_pc, avoid=avoid, seventh_pc=seventh_pc):
            """(style cost, faults): prefer doubling the root, then the fifth; a doubled leading
            tone, seventh or augmented-sixth tone is a rule break, reported if the search cannot
            avoid it."""
            voices = [x % 12 for x in v]
            doubled = [pc for pc in set(voices) if voices.count(pc) > 1]
            role = {pc: "augmented-sixth tone" for pc in avoid} | {seventh_pc: "seventh", leading_pc: "leading tone"}
            faults = [f"doubled {role[pc]}" for pc in doubled if pc in avoid]
            cost = 0 if doubled == [tones[0].pitch_class] else 2 if doubled == [fifth_pc] else 4 if doubled else 1
            return cost + big * len(faults), faults
        cands.append([(v, *doubling(v)) for v in options])
        sevenths.append(seventh_pc)
        aug6s.append(aug6_pcs)
    # Viterbi over voicings
    layer = {v: (dc + abs(v[0] - 72) // 3, [v], [(0, x) for x in df]) for v, dc, df in cands[0]}
    for k in range(1, len(cands)):
        nxt = {}
        same = {m % 12 for m in spelled[k - 1]} == {m % 12 for m in spelled[k]}   # the harmony goes on
        for v, dc, df in cands[k]:
            best = None
            for pv, (cost, path, faults) in layer.items():
                f, c = _transition_faults(pv, v, leading_pc, tonic.pitch_class, sevenths[k - 1],
                                          None if same else aug6s[k - 1], (spelled[k - 1], spelled[k]))
                total = cost + c + dc + big * len(f)
                if best is None or total < best[0]:
                    best = (total, path + [v], faults + [(k, x) for x in f] + [(k, x) for x in df])
            nxt[v] = best
        layer = nxt
    total, path, faults = min(layer.values(), key=lambda t: (t[0], t[1]))

    rows = []
    for ch, names, v in zip(parsed, spelled, path):   # each note spelled as its chord writes it
        rows.append({"symbol": ch["symbol"] or ch["label"], **{voice: names[m] for voice, m in zip(_VOICES, v)}})
    voices = {voice: [r[voice] for r in rows] for voice in _VOICES}
    return {
        "key": key_label,
        "voices": voices,
        "chords": rows,
        "rule_breaks": [{"chord": k + 1, "rule": f} for k, f in faults],
        "render_hint": {"tracks": [
            {"type": "notes", "name": voice, "notes": voices[voice], "step_beats": 2.0, "sustain": True,
             "program": 52 if voice in ("soprano", "alto") else 48 if voice == "tenor" else 42}
            for voice in _VOICES]},
    }


# ------------------------------------------------------------ neo-Riemannian

def _triad(symbol_or_notes):
    root, ctype, _bass = _root_and_quality(symbol_or_notes)
    if ctype is None or ctype.name not in ("major", "minor"):
        raise ValueError("neo-Riemannian transformations act on major and minor triads, "
                         f"not {ctype.name if ctype else symbol_or_notes!r}")
    return root.without_octave(), ctype.name == "major"


def _triad_symbol(root: Note, major: bool) -> str:
    return root.name + ("" if major else "m")


def neo_riemannian(chord, operations: str) -> dict:
    """Apply the neo-Riemannian P, L and R transformations to a major or minor triad.

    After Hugo Riemann, as formalised by Lewin and Cohn: each keeps two common
    tones and moves the third voice by step. P (parallel) swaps major and minor
    on the same root (C ↔ Cm); R (relative) moves to the relative (C ↔ Am);
    L (Leittonwechsel, 'leading-tone exchange') moves the root down a semitone
    in major (C ↔ Em). `operations` is a string such as 'PLR' or 'L R L R',
    applied left to right; each step reports the chord and which note moved.
    Chains of these (e.g. alternating L and R) trace paths on the Tonnetz
    used in film scores; roots that would need a double accidental (or E#, B#,
    Fb, Cb) are respelled, so cycles visibly close. Deterministic.
    """
    if not isinstance(operations, str) or not operations.replace(" ", "").replace(",", ""):
        raise ValueError("operations must be a string of P, L and R, e.g. 'PLR'")
    root, major = _triad(chord)
    path = [{"chord": _triad_symbol(root, major), "notes": [n.name for n in chord_notes(CHORDS["major" if major else "minor"], root)]}]
    for op in operations.upper().replace(" ", "").replace(",", ""):
        before = chord_notes(CHORDS["major" if major else "minor"], root)
        if op == "P":
            major = not major
        elif op == "R":
            root = transpose(root, 9, 5) if major else transpose(root, 3, 2)
            major = not major
        elif op == "L":
            root = transpose(root, 4, 2) if major else transpose(root, 8, 5)
            major = not major
        else:
            raise ValueError(f"unknown operation {op!r}: use P, L or R")
        root = _plain(root)  # keep the chain readable: Fb -> E, so PLPLPL closes on C
        after = chord_notes(CHORDS["major" if major else "minor"], root)
        moved = [(a.name, b.name) for a in before for b in after
                 if a.pitch_class not in {x.pitch_class for x in after} and b.pitch_class not in {x.pitch_class for x in before}]
        path.append({"operation": op, "chord": _triad_symbol(root, major), "notes": [n.name for n in after],
                     "moved": f"{moved[0][0]} → {moved[0][1]}" if moved else "none"})
    return {"start": path[0]["chord"], "operations": operations, "path": path,
            "symbols": [p["chord"] for p in path]}


# ------------------------------------------------------------ Bartók

def bartok_axis(key: str) -> dict:
    """Bartók's axis system (Ernő Lendvai): the tonic, subdominant and dominant axes of a key.

    Lendvai showed that Bartók treats keys a minor third apart as functional
    substitutes: each function is an axis of four keys forming a diminished-seventh
    cycle, with a principal pole and its counterpole a tritone away. For C:
    tonic axis C–F♯ (A–E♭), subdominant axis F–B (D–A♭), dominant axis G–C♯
    (E–B♭) — together all twelve keys. Keys that would need a double
    accidental (or E#, B#, Fb, Cb) are given their simpler name. Deterministic.
    """
    tonic = _plain(parse_notes(key)[0].without_octave())

    def axis(offset):
        pole = _plain(transpose(tonic, offset, {0: 0, 5: 3, 7: 4}[offset])) if offset else tonic
        cycle = [pole] + [_plain(transpose(pole, semis, steps)) for semis, steps in ((3, 2), (6, 3), (9, 5))]
        return {"pole": pole.name, "counterpole": cycle[2].name,
                "secondary_poles": [cycle[3].name, cycle[1].name], "keys": [n.name for n in cycle]}

    return {"key": tonic.name, "tonic_axis": axis(0), "subdominant_axis": axis(5), "dominant_axis": axis(7)}


# ------------------------------------------------------------ Coltrane

def coltrane_changes(key: str) -> dict:
    """Coltrane changes ('Giant Steps' substitution) for a ii–V–I in `key`.

    John Coltrane replaced the plain ii–V–I with a cycle through three tonal
    centres a major third apart, each reached by its own dominant:
    ii7 – V7/♭VI – ♭VImaj7 – V7/III – IIImaj7 – V7 – Imaj7. In C:
    Dm7 E♭7 A♭maj7 B7 Emaj7 G7 Cmaj7. Returns the original and the
    substituted progression (chord symbols ready for voice_leading /
    chords_to_midi) and the three tonal centres; roots that would need a double
    accidental are respelled (in D♭: E♭m7 E7 Amaj7 ...). Deterministic.
    """
    tonic = _plain(parse_notes(key)[0].without_octave())
    ii = _plain(transpose(tonic, 2, 1))
    v = _plain(transpose(tonic, 7, 4))
    flat6 = _plain(transpose(tonic, 8, 5))      # a major third below the tonic
    third = _plain(transpose(tonic, 4, 2))      # a major third above
    v_of_flat6 = _plain(transpose(flat6, 7, 4))
    v_of_third = _plain(transpose(third, 7, 4))
    subst = [f"{ii.name}m7", f"{v_of_flat6.name}7", f"{flat6.name}maj7", f"{v_of_third.name}7",
             f"{third.name}maj7", f"{v.name}7", f"{tonic.name}maj7"]
    return {
        "key": tonic.name,
        "original": [f"{ii.name}m7", f"{v.name}7", f"{tonic.name}maj7"],
        "coltrane": subst,
        "tonal_centres": [flat6.name, third.name, tonic.name],
        "note": "Tonal centres divide the octave into three major thirds, as in 'Giant Steps' and 'Countdown'.",
    }
