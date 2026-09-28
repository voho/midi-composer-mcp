"""Reading a draft back: key finding, a voice-leading lint and cadence labels.

The generators write music; these tools read it back so the LLM can check that
a draft does what it intended:

- detect_key: the Krumhansl–Schmuckler key-finding algorithm (Pearson
  correlation of a duration-weighted pitch-class histogram with a key profile),
  with the five profiles and the tonal-certainty measure exactly as music21
  implements them (analysis/discrete.py and key.py, BSD-3), and windowed key
  regions after music21's WindowedAnalysis.
- check_voice_leading: lints the caller's own parts (note lists, voicings or
  notes tracks) for parallel/contrary/direct fifths and octaves, crossing,
  overlap, melodic and key-dependent rules, with the motion types of music21's
  voiceLeading.VoiceLeadingQuartet. Its MIDI-level facts come from the same
  predicates bach_chorale_voicing uses, so the two tools cannot disagree.
- find_cadences: labels the cadence at each phrase end (authentic perfect /
  imperfect, half, Phrygian half, plagal, deceptive) by the textbook rules of
  Kostka, Payne & Almén, Caplin and Aldwell & Schachter, reading every chord
  through analyze_progression so the two tools agree.

Everything here is deterministic.
"""

from __future__ import annotations

import math

from .chords import chord_notes
from .circle import _fifths, _practical
from .diatonic import roman_suffix
from .generate import RHYTHM_REST, parse_rhythm
from .harmony import (
    _FIGURE_MARKS, _bass_degree, _chord_items, _root_and_quality, analyze_progression, interval_between,
)
from .masters import (
    RANGES, _VOICES, _augmented_sixth, _consecutive_perfect, _melodic_augmented, _overlap, _roles, _similar_leap,
)
from .midi_io import TICKS_PER_BEAT, _check_range, _parse_chord_list, assign_octaves, build_track_events
from .notes import (
    LETTER_PCS, LETTERS, Note, note_from_midi, parse_note, parse_notes, spelling_for_pcs, transpose,
)
from .roman import _is_above, _steps, read_chord
from .scales import resolve_scale_type, scale_notes

# ================================================================ detect_key

# Key profiles, index 0 = the tonic, then chromatically upwards. Copied verbatim from
# music21 analysis/discrete.py (KrumhanslKessler, TemperleyKostkaPayne, BellmanBudge,
# AardenEssen, SimpleWeights), (c) Michael Scott Asato Cuthbert and the music21 project, BSD-3.
PROFILES: dict[str, dict] = {
    "krumhansl": {
        "name": "Krumhansl–Kessler (1982)",
        "major": (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88),
        "minor": (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17),
    },
    "temperley": {
        "name": "Temperley–Kostka–Payne (Temperley 2007)",
        "major": (0.748, 0.060, 0.488, 0.082, 0.670, 0.460, 0.096, 0.715, 0.104, 0.366, 0.057, 0.400),
        "minor": (0.712, 0.084, 0.474, 0.618, 0.049, 0.460, 0.105, 0.747, 0.404, 0.067, 0.133, 0.330),
    },
    "bellman": {
        "name": "Bellman–Budge (Bellman 2005)",
        "major": (16.80, 0.86, 12.95, 1.41, 13.49, 11.93, 1.25, 20.28, 1.80, 8.04, 0.62, 10.57),
        "minor": (18.16, 0.69, 12.99, 13.34, 1.07, 11.15, 1.38, 21.07, 7.49, 1.53, 0.92, 10.21),
    },
    "aarden": {
        "name": "Aarden–Essen (Aarden 2003)",
        "major": (17.7661, 0.145624, 14.9265, 0.160186, 19.8049, 11.3587, 0.291248, 22.062, 0.145624,
                  8.15494, 0.232998, 4.95122),
        "minor": (18.2648, 0.737619, 14.0499, 16.8599, 0.702494, 14.4362, 0.702494, 18.6161, 4.56621,
                  1.93186, 7.37619, 1.75623),
    },
    "simple": {
        "name": "Simple weights (Sapp 2011)",
        "major": (2, 0, 1, 0, 1, 1, 0, 2, 0, 1, 0, 1),
        "minor": (2, 0, 1, 1, 0, 1, 0, 2, 1, 0, 0.5, 0.5),
    },
}
_MODES = ("major", "minor")
_SCALE_OF_MODE = {"major": "major", "minor": "natural minor"}
_MAX_WINDOWS = 2000
_TRACK_STEP_BEATS = 0.5      # arrange_to_midi's defaults, so tracks are timed as they render
_TRACK_BEATS_PER_CHORD = 4.0


def _centred(values) -> tuple[list[float], float]:
    mean = sum(values) / 12
    centred = [v - mean for v in values]
    return centred, sum(c * c for c in centred)


def _correlations(hist: list[float], profile: dict) -> list[tuple[float, int, str]] | None:
    """Pearson r of the histogram with the profile rotated to every tonic, for both modes:
    [(r, tonic pc, mode)], or None when the histogram has no variance (no tonal centre)."""
    d, d_ss = _centred(hist)
    if d_ss <= 1e-12 * max(1.0, max(abs(h) for h in hist)) ** 2:
        return None
    out = []
    for mode in _MODES:
        w, w_ss = _centred(profile[mode])
        den = math.sqrt(w_ss * d_ss)
        for t in range(12):
            # the same summation order for every tonic, so a transposed input gives bit-identical r
            num = sum(w[i] * d[(i + t) % 12] for i in range(12))
            out.append((num / den, t, mode))
    return out


def _ranked(corrs):
    # ties (to 12 places) fall back to the tonic's pitch class, then major before minor
    return sorted(corrs, key=lambda c: (-round(c[0], 12), c[1], _MODES.index(c[2])))


def _certainty(ranking) -> float:
    """music21 key._tonalCertaintyCorrelationCoefficient: r1 + 2·(r1 − r2), r2 the best positive runner-up."""
    r1 = ranking[0][0]
    positive = [c[0] for c in ranking[1:] if c[0] > 0]
    if not positive:
        return r1 if r1 > 0 else 0.0
    return r1 + 2 * (r1 - positive[0])


def _spell_tonic(pc: int, mode: str, written: dict[int, Note]) -> Note:
    """The caller's own spelling of that pitch class if they wrote it (made practical: at most 7
    accidentals), else the spelling with the smaller key signature (F# major and Eb minor on a tie)."""
    offset = -3 if mode == "minor" else 0
    if pc in written:
        return _practical(written[pc], offset)
    cands = [Note(letter, acc) for letter in LETTERS for acc in (0, -1, 1)
             if (LETTER_PCS[letter] + acc) % 12 == pc]

    def key(n):
        f = _fifths(n) + offset
        return abs(f), (-f if mode == "major" else f)

    return min(cands, key=key)


def _note_slots(notes) -> list[list[Note]]:
    """Notes input -> time slots; a list item that is itself a list is one slot of simultaneous notes."""
    if isinstance(notes, str):
        return [[n] for n in parse_notes(notes)]
    if not isinstance(notes, (list, tuple)) or not notes:
        raise ValueError("notes must be a non-empty list of note names (an inner list = simultaneous notes) "
                         "or a string like 'C4 E4 G4'")
    slots = []
    for item in notes:
        if isinstance(item, (list, tuple)):
            if not item:
                raise ValueError("an inner list of simultaneous notes must not be empty")
            slots.append(parse_notes(list(item)))
        elif isinstance(item, str):
            slots.extend([n] for n in parse_notes([item]))
        else:
            raise ValueError(f"Not a note name: {item!r} (expected a string like 'C4' or a list of them)")
    return slots


def _key_events(notes, chords, tracks, durations, rhythm, step_beats, beats_per_chord):
    """All pitched input as [(start beat, duration beats, Note)] on one timeline from beat 0."""
    events: list[tuple[float, float, Note]] = []
    if notes is not None:
        slots = _note_slots(notes)
        if rhythm is not None:
            pattern = parse_rhythm(rhythm)
            onsets = [i for i, s in enumerate(pattern) if s != RHYTHM_REST]
            if not onsets:
                raise ValueError("the rhythm has no onsets ('O' or 'o')")
            for n_on, step in enumerate(onsets):
                nxt = onsets[n_on + 1] if n_on + 1 < len(onsets) else len(pattern)
                dur = step_beats * (nxt - step)  # the note sustains through the following rests
                events.extend((step * step_beats, dur, n) for n in slots[n_on % len(slots)])
        else:
            if durations is not None:
                if not isinstance(durations, (list, tuple)):
                    raise ValueError("durations must be a list of positive numbers, one per note (or chord of notes)")
                if len(durations) != len(slots):
                    raise ValueError(f"durations has {len(durations)} values but there are {len(slots)} notes")
                lengths = [_check_range("each duration", d, 1e-6, 10000) for d in durations]
            else:
                lengths = [step_beats] * len(slots)
            start = 0.0
            for slot, dur in zip(slots, lengths):
                events.extend((start, dur, n) for n in slot)
                start += dur
    elif durations is not None or rhythm is not None:
        raise ValueError("durations and rhythm time the `notes`; give notes too")
    if chords is not None:
        for i, ch in enumerate(_parse_chord_list(chords)):
            seen = set()
            for t in ch["tones"]:
                if t.pitch_class not in seen:  # each distinct chord tone once
                    seen.add(t.pitch_class)
                    events.append((i * beats_per_chord, beats_per_chord, t))
            if ch["bass"] is not None:  # a slash bass sounds as its own note below the chord
                events.append((i * beats_per_chord, beats_per_chord, ch["bass"]))
    if tracks is not None:
        if not isinstance(tracks, (list, tuple)) or not tracks:
            raise ValueError("tracks must be a non-empty list of track objects (render_hint or arrange tracks)")
        for i, track in enumerate(tracks):
            built = build_track_events(track, i, _TRACK_STEP_BEATS, _TRACK_BEATS_PER_CHORD)
            if built["is_drums"]:
                continue
            for e in built["events"]:
                events.append((e["start"], e["duration"], parse_note(e["label"])))
    return events


def _histogram(events, lo: float | None = None, hi: float | None = None) -> list[float]:
    hist = [0.0] * 12
    for start, dur, note in events:
        if lo is None:
            weight = dur
        else:
            weight = min(start + dur, hi) - max(start, lo)
            if weight <= 0:
                continue
        hist[note.pitch_class] += weight
    return hist


def detect_key(notes=None, chords=None, tracks=None, durations=None, rhythm: str | None = None,
               step_beats: float = 1.0, beats_per_chord: float = 4.0, profile: str = "krumhansl",
               window_beats: float = 0.0, hop_beats: float = 0.0) -> dict:
    """Estimate the major or minor key of notes, chords or tracks (Krumhansl–Schmuckler key finding).

    The algorithm of Krumhansl & Kessler (1982) and Krumhansl (1990), as music21
    implements it: a pitch-class histogram, each note weighted by its duration,
    is correlated (Pearson r) with a key profile rotated to all 24 major and
    minor keys; the best r wins. Profile vectors are music21's
    (analysis/discrete.py, BSD-3): 'krumhansl' (Krumhansl–Kessler, the
    default), 'temperley' (Temperley–Kostka–Payne), 'bellman' (Bellman–Budge),
    'aarden' (Aarden–Essen — music21 warns that its minor weights are of
    uncertain origin and recommends them for major only; it reads a plain
    C-major scale as A minor) and 'simple' (Sapp).

    Input: `notes` and/or `chords` (summed on one timeline from beat 0), OR
    `tracks` alone. `notes`: note names (an inner list = simultaneous notes);
    each lasts its entry in `durations`, or follows `rhythm` exactly as a notes
    track does (onsets take the notes cyclically; a note sustains through the
    rests after it, each step `step_beats` long), or lasts `step_beats`.
    `chords`: symbols or note arrays; each distinct chord tone, and a slash
    bass, sounds `beats_per_chord`. `tracks`: render_hint / arrange tracks,
    timed exactly as arrange_to_midi renders them (defaults step 0.5, 4 beats
    per chord); drum tracks are ignored.

    Returns root and scale_type ('major' / 'natural minor', ready for
    diatonic_chords, analyze_progression, snap_to_scale, check_voice_leading),
    mode, correlation, certainty (music21's tonalCertainty: r1 + 2·(r1 − r2),
    r2 the best positive runner-up), the full 24-key ranking, and the
    histogram. The tonic keeps your spelling when you wrote that pitch class
    (Gb stays Gb), else the key with the smaller signature. With
    `window_beats` > 0 it also returns `regions`: whole windows [s, s+window)
    every `hop_beats` (default window/2; the last aligned to the end, as in
    music21's WindowedAnalysis), each note weighted by its overlap; a silent
    window takes its neighbour's key; runs of equal keys merge into regions
    {start_beat, end_beat, root, scale_type, mean_correlation}, a change placed
    midway through the overlap of the two windows that disagree; no smoothing
    across windows. A histogram with no variance (e.g. all twelve notes
    equally) has no tonal centre and raises.
    Example: detect_key(notes='C4 D4 E4 F4 G4 A4 B4 C5') -> C major, r 0.9014,
    certainty 1.1914, runner-up A natural minor 0.7563. Deterministic.
    """
    if not isinstance(profile, str) or profile.strip().lower() not in PROFILES:
        raise ValueError(f"profile must be one of {', '.join(PROFILES)}, got {profile!r}")
    prof_name = profile.strip().lower()
    prof = PROFILES[prof_name]
    _check_range("step_beats", step_beats, 0.0625, 16)
    _check_range("beats_per_chord", beats_per_chord, 0.25, 64)
    _check_range("window_beats", window_beats, 0, 100000)
    _check_range("hop_beats", hop_beats, 0, 100000)
    if hop_beats and not window_beats:
        raise ValueError("hop_beats needs window_beats > 0")
    if tracks is not None and (notes is not None or chords is not None):
        raise ValueError("give tracks alone, or notes and/or chords — not both")
    if notes is None and chords is None and tracks is None:
        raise ValueError("give notes, chords or tracks to analyse")
    if durations is not None and rhythm is not None:
        raise ValueError("give durations or rhythm, not both")

    events = _key_events(notes, chords, tracks, durations, rhythm, step_beats, beats_per_chord)
    written = spelling_for_pcs([n for _s, _d, n in events])
    hist = _histogram(events)
    corrs = _correlations(hist, prof)
    if corrs is None:
        raise ValueError("no tonal centre: the pitch-class histogram is flat (every pitch class equally weighted, "
                         "or no pitched notes)")
    ranking = _ranked(corrs)
    best_r, best_pc, best_mode = ranking[0]
    tonic = _spell_tonic(best_pc, best_mode, written)
    result = {
        "root": tonic.name,
        "scale_type": _SCALE_OF_MODE[best_mode],
        "mode": best_mode,
        "correlation": round(best_r, 4),
        "certainty": round(_certainty(ranking), 4),
        "profile": prof_name,
        "ranking": [{"root": _spell_tonic(pc, mode, written).name, "scale_type": _SCALE_OF_MODE[mode],
                     "correlation": round(r, 4)} for r, pc, mode in ranking],
        "histogram": {written[pc].name: round(w, 6) for pc, w in enumerate(hist) if w > 0},
    }
    if window_beats:
        result["regions"] = _regions(events, prof, written, window_beats, hop_beats or window_beats / 2,
                                     (best_pc, best_mode, best_r))
    return result


def _regions(events, prof, written, window: float, hop: float, overall: tuple) -> list[dict]:
    end = max(s + d for s, d, _n in events)
    # whole windows only, as in music21's WindowedAnalysis (a window never hangs past the end, so
    # the last one is not read from a fragment); the final window is aligned to the end
    count = max(1, math.floor((end - window) / hop + 1e-9) + 1) if end > window else 1
    if count > _MAX_WINDOWS:
        raise ValueError(f"{count} windows is too many (at most {_MAX_WINDOWS}): use a larger hop_beats")
    starts = [i * hop for i in range(count)]
    if end > window and starts[-1] + window < end - 1e-9:
        starts.append(end - window)
    keys: list[tuple[int, str] | None] = []
    rs: list[float | None] = []
    for s in starts:
        corrs = _correlations(_histogram(events, s, s + window), prof)
        if corrs is None:  # silent (or tonally flat): decided by its neighbours below
            keys.append(None)
            rs.append(None)
            continue
        r, pc, mode = _ranked(corrs)[0]
        keys.append((pc, mode))
        rs.append(r)
    for i in range(1, len(keys)):   # a silent window copies the previous key ...
        if keys[i] is None:
            keys[i] = keys[i - 1]
    for i in range(len(keys) - 2, -1, -1):   # ... and leading silent windows copy the next
        if keys[i] is None:
            keys[i] = keys[i + 1]
    if all(k is None for k in keys):  # no window has a tonal centre of its own: the whole piece's key
        keys, rs = [overall[:2]] * len(keys), [overall[2]] + [None] * (len(keys) - 1)
    regions = []
    for i, k in enumerate(keys):
        if regions and regions[-1]["key"] == k:
            regions[-1]["rs"].append(rs[i])
            continue
        regions.append({"key": k, "first": i, "rs": [rs[i]]})
    # a key change between window a and the next window b is placed midway through their overlap
    # (or the gap between them): with hop = window, exactly at the window boundary
    bounds = [0.0]
    for reg in regions[1:]:
        b = reg["first"]
        bounds.append(min(end, max(0.0, (starts[b - 1] + window + starts[b]) / 2)))
    bounds.append(end)
    out = []
    for n, reg in enumerate(regions):
        pc, mode = reg["key"]
        vals = [r for r in reg["rs"] if r is not None]
        out.append({
            "start_beat": round(float(bounds[n]), 6),
            "end_beat": round(float(bounds[n + 1]), 6),
            "root": _spell_tonic(pc, mode, written).name,
            "scale_type": _SCALE_OF_MODE[mode],
            "mean_correlation": round(sum(vals) / len(vals), 4),
        })
    return out


# ======================================================= check_voice_leading

_CONSONANT = {("perfect", 1), ("minor", 3), ("major", 3), ("perfect", 5), ("minor", 6), ("major", 6)}


def _simple_number(number: int) -> int:
    return (number - 1) % 7 + 1


def _vertical(upper: Note, lower: Note) -> dict:
    """The spelled interval between two sounding notes (interval_between from the lower-listed voice)."""
    return interval_between(lower.name, upper.name)


def _perfect(iv: dict) -> str | None:
    """'fifth' / 'octave' when a spelled interval is a perfect fifth / unison-octave (compound or not)."""
    if iv["quality"] != "perfect":
        return None
    simple = _simple_number(iv["number"])
    return "fifth" if simple == 5 else "octave" if simple == 1 else None


def _directed_generic(iv: dict) -> int:
    return -iv["number"] if iv.get("direction") == "descending" else iv["number"]


def _parts_from_lists(voices) -> list[list[Note]]:
    parts = []
    for k, v in enumerate(voices):
        notes = parse_notes(v)
        for n in notes:
            if n.octave is None:
                raise ValueError(f"voice {k}: {n.name} has no octave — write concrete pitches such as 'C4'")
        parts.append(notes)
    if len({len(p) for p in parts}) != 1:
        raise ValueError("note-list voices must all have the same number of notes (one per slot); "
                         f"got lengths {[len(p) for p in parts]}")
    return parts


def _slices_from_tracks(voices):
    """Notes tracks -> (names, original indices, slices, attacks, beats), aligned on every attack."""
    timed = []
    for i, track in enumerate(voices):
        ttype = track.get("type", "notes")
        if ttype != "notes":
            raise ValueError(f"voice {i} is a {ttype!r} track: check_voice_leading reads notes tracks, one part each "
                             "(pass a chord progression's voicings through `voicings`)")
        built = build_track_events(dict(track, type="notes"), i, _TRACK_STEP_BEATS, _TRACK_BEATS_PER_CHORD)
        evs = []
        for e in built["events"]:
            on = round(e["start"] * TICKS_PER_BEAT)
            off = max(on + 1, round((e["start"] + e["duration"]) * TICKS_PER_BEAT))
            evs.append((on, off, parse_note(e["label"])))
        if not evs:
            raise ValueError(f"voice {i} has no notes")
        timed.append((built["name"], i, evs))
    # voices ordered by mean MIDI pitch, highest first (ties keep the given order)
    timed.sort(key=lambda t: (-sum(n.midi for _a, _b, n in t[2]) / len(t[2]), t[1]))
    ticks = sorted({on for _name, _i, evs in timed for on, _off, _n in evs})
    slices, attacks = [], []
    cursor = [0] * len(timed)   # a notes track's events are in time order and never overlap
    for tick in ticks:
        row, hit = [], []
        for v, (_name, _i, evs) in enumerate(timed):
            while cursor[v] < len(evs) and evs[cursor[v]][1] <= tick:
                cursor[v] += 1
            ev = evs[cursor[v]] if cursor[v] < len(evs) and evs[cursor[v]][0] <= tick else None
            row.append(ev[2] if ev else None)
            hit.append(ev is not None and ev[0] == tick)
        slices.append(row)
        attacks.append(hit)
    beats = [round(t / TICKS_PER_BEAT, 6) for t in ticks]
    return [t[0] for t in timed], [t[1] for t in timed], slices, attacks, beats


def _align_voicings(chords: list[list[Note]]) -> list[list[Note | None]]:
    """Voiced chords (each highest note first) -> slices of one voice per slot.

    The parts are as many as the largest chord has notes. A chord of that size fills them in
    order (part k = its k-th highest note). A smaller chord (a triad among sevenths: I V7 I) keeps
    its notes in order and takes the parts nearest to where each voice last sounded (the smallest
    total distance, ties to the upper parts), leaving the others silent (None) — a voice that
    drops out or enters, as a silent track voice does. Before any voice has sounded, the first
    full-size chord is the reference. Deterministic.
    """
    n = max(len(c) for c in chords)
    last = list(next(c for c in chords if len(c) == n))   # where each part last sounded
    out = []
    for chord in chords:
        m = len(chord)
        # best[i][j]: the least total distance placing notes i.. into parts j.., order kept
        best = [[math.inf] * (n + 1) for _ in range(m)] + [[0] * (n + 1)]
        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                best[i][j] = min(best[i][j + 1], abs(chord[i].midi - last[j].midi) + best[i + 1][j + 1])
        slots, j = [], 0
        for i in range(m):   # the upper-most parts among the optimal assignments
            while abs(chord[i].midi - last[j].midi) + best[i + 1][j + 1] != best[i][j]:
                j += 1
            slots.append(j)
            j += 1
        row: list[Note | None] = [None] * n
        for note, p in zip(chord, slots):
            row[p] = note
            last[p] = note
        out.append(row)
    return out


def _tertian(ctype) -> bool:
    """A triad or a seventh-family chord: it has a third (a power chord or a sus chord has none)."""
    degrees = dict(ctype.degrees)
    return "3" in degrees or "b3" in degrees


def _slice_chord(notes: list[Note], key: dict[str, int] | None = None):
    """(root, ChordType) of a sonority read lowest note first, or None if it is no chord.

    Read by harmony._root_and_quality; failing that, as a triad or seventh chord with its fifth
    omitted (G B F = G7, C E = C), the omission four-part writing allows (Aldwell & Schachter,
    Kostka & Payne). Only a completion with a third counts: a bare unison or octave (B B), a
    second (B A) or a seventh (G F) is no chord, not a power or sus chord. The omitted fifth is the
    root's perfect fifth or, with `key` ({letter: pitch class} of a seven-note scale), the key's
    note on the fifth letter when that is a perfect or diminished fifth (B D in C = B D F, vii°).
    None when nothing reads.
    """
    if len(notes) < 2:
        return None
    asc = sorted(notes, key=lambda n: n.midi)
    names = [n.without_octave().name for n in asc]
    root, ctype, _bass = _root_and_quality(names)
    if ctype is not None:
        return root.without_octave(), ctype
    for cand in dict.fromkeys(n.without_octave() for n in asc):
        fifth = transpose(cand, 7, 4)
        key_fifth = key.get(LETTERS[(LETTERS.index(cand.letter) + 4) % 7]) if key is not None else None
        if key_fifth is not None and (key_fifth - cand.pitch_class) % 12 == 6:
            fifth = transpose(cand, 6, 4)
        root, ctype, _bass = _root_and_quality(names + [fifth.name])
        if ctype is not None and root.pitch_class == cand.pitch_class and _tertian(ctype):
            return root.without_octave(), ctype
    return None


def check_voice_leading(voices=None, voicings=None, root: str | None = None, scale_type: str = "major") -> dict:
    """Lint your own parts for voice-leading faults (the report shape of check_melody).

    Give exactly one of:
    - `voices`: at least 2 parts, either note lists with octaves (one note per
      slot, equal length, listed highest first) or notes-track objects
      ({notes, rhythm, step_beats, sustain}: render_hint tracks of
      bach_chorale_voicing, counterpoint, tintinnabuli_voice, a bass line…).
      Tracks are timed as arrange_to_midi renders them and aligned on every
      attack (as music21 does): each voice contributes the note sounding then,
      a held note is oblique motion, a silent voice drops out of that slice.
      Track voices are ordered by mean pitch, highest first (`voice_order`).
    - `voicings`: voiced note arrays (voice_leading's or voice_chords'
      `chords`, bach_chorale_voicing's rows); sizes may differ (I V7 I). There
      are as many parts as the largest chord has notes: part k is a full-size
      chord's k-th highest note, and a smaller chord keeps its notes in order
      on the parts nearest where each voice last sounded, the others silent
      for that slice (a voice drops out or enters, as in a track).

    Motion per voice pair (music21 VoiceLeadingQuartet): static, oblique,
    parallel (same direction, same generic interval), similar, contrary.
    Violations (index = the arriving slice, voices 0 = highest):
    parallel_fifths/_octaves (a spelled P5 or P1/P8, compound too, in both
    slices, both voices moving the same way); contrary_fifths/_octaves (the
    same by contrary motion — the strict textbook rule bach_chorale_voicing
    enforces); direct_fifths/_octaves (outer voices only: similar motion into a
    spelled P5/P8 from a differently spelled interval — a d6 counts, as in
    music21's hiddenFifth — with the upper voice leaping more than 2 semitones;
    Aldwell & Schachter allow it when the soprano steps); voice_crossing;
    voice_overlap (adjacent voices, music21); melodic_augmented (A2, A4… — the
    chromatic A1 is fine; Kostka & Payne; bach_chorale_voicing's own test);
    melodic_leap_beyond_octave. With `root` (a seven-note key), reading each
    slice as a tertian chord (triad or seventh; a missing fifth is restored as
    the key's fifth: B D in C = vii°; a power chord, sus chord, bare octave or
    dyad second is no harmony for these rules, except that one on the tonic is
    still an arrival on ^1, as at Fux's cadence): leading_tone (in the highest or
    lowest voice the leading tone — tonic minus a semitone, the raised 7th in
    minor — of a chord on ^5 or a diminished chord on ^7 rises to the tonic when
    the next chord's root is ^1 or ^6 and that chord lacks ^7 (Imaj7 may keep
    it); ^1-^7-^6 stepping down in the lowest voice is a passing ^7, not a
    fault (the Romanesca's 1-7-6, Gjerdingen; I V6 vi); in an inner voice a
    'frustrated leading tone' is only a warning; bach_chorale_voicing's own
    soprano rule is stricter), seventh_resolution (a chord seventh — spelled as
    a seventh above the root — falls 1–2 semitones when the harmony changes;
    held is a warning), augmented_sixth (It6, Fr43, Ger65, Sw43 — b6 in the
    lowest voice with the note an augmented sixth above it, Ab–F# in C, read
    by spelling: #4 rises a semitone and b6 falls a semitone to ^5 when the
    harmony changes, Kostka & Payne, Aldwell & Schachter; held is a warning;
    its #4 is no chord seventh), doubled_leading_tone (in a chord on ^5 or a
    diminished chord on ^7; the root of V/iii, B D# F# in C, may double).
    Warnings: unequal_fifths (d5 -> P5 in similar
    motion with the lowest voice; P5 -> d5 is fine), spacing (adjacent upper
    voices > 12 semitones, the lowest pair > 19), range (four voices: SATB
    ranges), melodic_seventh, leap_not_recovered (a leap of 8+ semitones not
    followed by a 1–2 semitone step back), enharmonic_fifth. Info:
    `dissonances` — every interval above the lowest voice that is not P1, m3,
    M3, P5, m6, M6, P8 or a compound of these (music21 isConsonant; a P4 above
    the bass is dissonant, and spelling matters). A held crossing, spacing or
    dissonance is reported once, where it is struck. Only adjacent attacks are
    compared: Fux's species 2–5 rule against fifths or octaves on successive
    downbeats is NOT checked. Note lists and voicings need octaves (tracks are
    placed as they render). Returns {valid, voice_order, slices, violations,
    warnings, motion, dissonances} (+ key, and track_order for tracks).
    Example: check_voice_leading(voices=[['C5','D5'],['F4','G4']]) -> a
    parallel_fifths violation at index 1 between voices [0, 1]. Deterministic.
    """
    if (voices is None) == (voicings is None):
        raise ValueError("give exactly one of voices (parts) or voicings (voiced chords)")
    scale = resolve_scale_type(scale_type)
    beats = track_order = None
    if voices is not None:
        if not isinstance(voices, (list, tuple)) or len(voices) < 2:
            raise ValueError("voices must be a list of at least 2 parts (note lists with octaves, or notes tracks)")
        if all(isinstance(v, dict) for v in voices):
            names, track_order, slices, attacks, beats = _slices_from_tracks(voices)
        elif all(isinstance(v, (str, list, tuple)) for v in voices):
            parts = _parts_from_lists(voices)
            names = [f"voice {k}" for k in range(len(parts))]
            slices = [list(col) for col in zip(*parts)]
            attacks = [[True] * len(parts) for _ in slices]
        else:
            raise ValueError("voices must be all note lists (with octaves) or all notes-track objects")
    else:
        if not isinstance(voicings, (list, tuple)) or not voicings:
            raise ValueError("voicings must be a non-empty list of voiced chords (note arrays with octaves)")
        slices = []
        for k, chord in enumerate(voicings):
            notes = parse_notes(chord)
            for n in notes:
                if n.octave is None:
                    raise ValueError(f"voicing {k}: {n.name} has no octave — voicings are concrete pitches such as 'C4'")
            slices.append(sorted(notes, key=lambda n: -n.midi))
        if min(len(s) for s in slices) < 2:
            raise ValueError("voicings need at least 2 notes each")
        slices = _align_voicings(slices)
        names = [f"voice {k}" for k in range(len(slices[0]))]
        attacks = [[n is not None for n in row] for row in slices]

    tonic = None
    if root is not None:
        tonic = parse_notes(root)[0].without_octave()
        if len(scale.intervals) != 7:
            raise ValueError(f"the key rules need a seven-note scale; {scale.name} has {len(scale.intervals)} notes")
    n_voices = len(names)
    violations, warnings, dissonances = [], [], []
    motion = {f"{i}-{j}": {"parallel": 0, "similar": 0, "contrary": 0, "oblique": 0, "static": 0}
              for i in range(n_voices) for j in range(i + 1, n_voices)}

    def add(target, k, vs, rule, message):
        target.append({"index": k, "voices": vs, "rule": rule, "message": message})

    def sounding(k):
        return [v for v in range(n_voices) if slices[k][v] is not None]

    # ---------------------------------------------------------- per slice
    for k, row in enumerate(slices):
        live = sounding(k)   # (a held crossing, spacing or dissonance is reported once, when struck)
        for a, i in enumerate(live):
            for j in live[a + 1:]:
                if (attacks[k][i] or attacks[k][j]) and row[i].midi < row[j].midi:
                    add(violations, k, [i, j], "voice_crossing",
                        f"{row[i].name} (voice {i}) sounds below {row[j].name} (voice {j}), the voice listed under it")
        for a, i in enumerate(live):
            for j in live[a + 1:]:
                if not (attacks[k][i] or attacks[k][j]):
                    continue
                iv = _vertical(row[i], row[j])
                if abs(row[i].midi - row[j].midi) % 12 == 7 and _simple_number(iv["number"]) != 5:
                    add(warnings, k, [i, j], "enharmonic_fifth",
                        f"{row[i].name}/{row[j].name}: seven semitones spelled as a {iv['name']}, not a fifth")
        for a, i in enumerate(live[:-1]):
            j = live[a + 1]
            limit = 19 if j == live[-1] else 12
            if (attacks[k][i] or attacks[k][j]) and row[i].midi - row[j].midi > limit:
                add(warnings, k, [i, j], "spacing",
                    f"{row[i].name}/{row[j].name}: {row[i].midi - row[j].midi} semitones apart "
                    f"(more than {'a twelfth' if limit == 19 else 'an octave'})")
        if n_voices == 4:
            for v in live:
                lo, hi = RANGES[_VOICES[v]]
                if attacks[k][v] and not lo <= row[v].midi <= hi:
                    add(warnings, k, [v], "range", f"{row[v].name} is outside the {_VOICES[v]} range "
                        f"({note_from_midi(lo).name}–{note_from_midi(hi).name})")
        if len(live) >= 2:
            low = live[-1]
            for v in live[:-1]:
                if not (attacks[k][v] or attacks[k][low]):
                    continue
                iv = _vertical(row[v], row[low])
                if (iv["quality"], _simple_number(iv["number"])) not in _CONSONANT:
                    dissonances.append({"index": k, "voices": [v, low], "interval": iv["short"],
                                        "notes": f"{row[v].name}/{row[low].name}"})

    # ---------------------------------------------------------- between slices
    for k in range(1, len(slices)):
        prev, cur = slices[k - 1], slices[k]
        both = [v for v in range(n_voices) if prev[v] is not None and cur[v] is not None]
        for a, i in enumerate(both):
            for j in both[a + 1:]:
                p_i, p_j, c_i, c_j = prev[i], prev[j], cur[i], cur[j]
                d_i, d_j = c_i.midi - p_i.midi, c_j.midi - p_j.midi
                p_iv, c_iv = _vertical(p_i, p_j), _vertical(c_i, c_j)
                if d_i == 0 and d_j == 0:
                    kind = "static"
                elif d_i == 0 or d_j == 0:
                    kind = "oblique"
                elif (d_i > 0) == (d_j > 0):
                    kind = "parallel" if _directed_generic(p_iv) == _directed_generic(c_iv) else "similar"
                else:
                    kind = "contrary"
                motion[f"{i}-{j}"][kind] += 1
                hit = _consecutive_perfect(p_i.midi, p_j.midi, c_i.midi, c_j.midi)
                if hit is not None:
                    how, size = hit
                    what = "fifth" if size == 7 else "octave"
                    if _perfect(p_iv) == what and _perfect(c_iv) == what:
                        label = ("perfect fifths" if what == "fifth" else "unisons"
                                 if p_iv["number"] == c_iv["number"] == 1 else "perfect octaves")
                        move = f"{p_i.name}/{p_j.name} -> {c_i.name}/{c_j.name}"
                        if how == "parallel":
                            add(violations, k, [i, j], f"parallel_{what}s", f"{move}: parallel {label}")
                        else:
                            add(violations, k, [i, j], f"contrary_{what}s", f"{move}: consecutive {label} by contrary motion")
        if len(both) >= 2:
            hi, lo = both[0], both[-1]
            p_h, p_l, c_h, c_l = prev[hi], prev[lo], cur[hi], cur[lo]
            move = f"{p_h.name}/{p_l.name} -> {c_h.name}/{c_l.name}"
            # into a spelled P5/P8 from a differently spelled interval (a d6 is not a P5: music21's
            # hiddenFifth), in similar motion with the upper voice leaping; a spelled P5 -> P5 is
            # the parallel rule's case
            what = _perfect(_vertical(c_h, c_l))
            if (what is not None and _perfect(_vertical(p_h, p_l)) != what
                    and _similar_leap(p_h.midi, p_l.midi, c_h.midi, c_l.midi)):
                add(violations, k, [hi, lo], f"direct_{what}s",
                    f"{move}: direct (hidden) {what} in the outer voices, the upper voice leaping")
            for v in both[:-1]:   # unequal fifths against the lowest voice
                p_iv, c_iv = _vertical(prev[v], p_l), _vertical(cur[v], c_l)
                d_v, d_l = cur[v].midi - prev[v].midi, c_l.midi - p_l.midi
                if (p_iv["quality"] == "diminished" and _simple_number(p_iv["number"]) == 5
                        and _perfect(c_iv) == "fifth" and d_v * d_l > 0):
                    add(warnings, k, [v, lo], "unequal_fifths",
                        f"{prev[v].name}/{p_l.name} -> {cur[v].name}/{c_l.name}: a diminished fifth moving to a "
                        "perfect fifth against the bass")
        for a, i in enumerate(both[:-1]):
            j = both[a + 1]
            if _overlap(prev[i].midi, prev[j].midi, cur[i].midi, cur[j].midi):
                add(violations, k, [i, j], "voice_overlap",
                    f"{prev[i].name}/{prev[j].name} -> {cur[i].name}/{cur[j].name}: "
                    "a voice moves past the note its neighbour just held")
        for v in both:   # melodic intervals, between a voice's successive notes
            a_n, b_n = prev[v], cur[v]
            if not attacks[k][v] or a_n.midi == b_n.midi:
                continue
            iv = interval_between(a_n.name, b_n.name)
            augmented = _melodic_augmented(a_n.name, b_n.name)   # bach_chorale_voicing's own test
            if augmented is not None:
                add(violations, k, [v], "melodic_augmented", f"{a_n.name} -> {b_n.name}: a melodic {augmented}")
            if abs(b_n.midi - a_n.midi) > 12:
                add(violations, k, [v], "melodic_leap_beyond_octave",
                    f"{a_n.name} -> {b_n.name}: a leap of {abs(b_n.midi - a_n.midi)} semitones, beyond an octave")
            if _simple_number(iv["number"]) == 7:
                add(warnings, k, [v], "melodic_seventh", f"{a_n.name} -> {b_n.name}: a melodic {iv['name']}")

    # leaps of a minor sixth or more, recovered by a step back (a rest starts a new line)
    for v in range(n_voices):
        line: list[tuple[int, int]] = []   # (slice index, midi) of each new pitch in the current phrase
        for k in range(len(slices)):
            note = slices[k][v]
            if note is None:
                _leap_warnings(line, v, warnings, slices)
                line = []
            elif not line or note.midi != line[-1][1]:
                line.append((k, note.midi))
        _leap_warnings(line, v, warnings, slices)

    # ---------------------------------------------------------- key rules
    if tonic is not None:
        t_pc = tonic.pitch_class
        leading = (t_pc - 1) % 12
        deg5, deg6 = (t_pc + scale.intervals[4]) % 12, (t_pc + scale.intervals[5]) % 12
        key = {n.letter: n.pitch_class for n in scale_notes(scale, tonic)[:-1]}
        chords, arrivals = [], []
        for row in slices:   # tertian harmonies only: a power or sus chord is no harmony here
            notes = [n for n in row if n is not None]
            ch = _slice_chord(notes, key)
            chords.append(ch if ch is not None and _tertian(ch[1]) else None)
            # ...but a bare unison/octave, open fifth or sus chord ON THE TONIC is still an arrival on
            # ^1 for the leading tone (Fux's cadence ends ^7-^1 on the final's octave or unison)
            if chords[-1] is not None:
                arrivals.append(chords[-1])
            elif ch is not None and ch[0].pitch_class == t_pc:
                arrivals.append(ch)
            elif notes and {n.pitch_class for n in notes} == {t_pc}:
                arrivals.append((tonic, None))
            else:
                arrivals.append(None)

        def label(ch):
            return f"{ch[0].name}{ch[1].symbol}" if ch[1] is not None else f"{ch[0].name} (octave)"

        def leading_tone_chord(ch):
            """^7 is a leading tone in a chord on ^5 (its third) or a diminished chord on ^7 (vii°,
            vii°7, viiø7) — not in a major or minor chord on ^7, whose root it is (V/iii = B D# F#
            in C; Kostka & Payne, secondary functions)."""
            return ch[0].pitch_class == deg5 or (ch[0].pitch_class == leading and ch[1].intervals[:3] == (0, 3, 6))

        def passing_bass(k, v):
            """^1-^7-^6 stepping down in the lowest voice (the Romanesca's 1-7-6, Gjerdingen; the
            descending 5-6 sequence): ^7 passes, it is no leading tone (Aldwell & Schachter)."""
            p, c = slices[k - 1][v], slices[k][v]
            if c.pitch_class != deg6 or not 1 <= p.midi - c.midi <= 2:
                return False
            for j in range(k - 2, -1, -1):   # the voice's pitch before this ^7
                e = slices[j][v]
                if e is None:
                    return False
                if e.midi != p.midi:
                    return e.pitch_class == t_pc and 1 <= e.midi - p.midi <= 2
            return False

        for k, ch in enumerate(chords):
            if ch is None or not leading_tone_chord(ch):
                continue
            holders = [v for v in sounding(k) if slices[k][v].pitch_class == leading]
            if len(holders) > 1:
                add(violations, k, holders, "doubled_leading_tone",
                    f"the leading tone {slices[k][holders[0]].without_octave().name} is doubled in {label(ch)}")
        for k in range(1, len(slices)):
            prev, cur = slices[k - 1], slices[k]
            before, after = chords[k - 1], arrivals[k]
            live = sounding(k - 1)
            if (before is not None and after is not None and leading_tone_chord(before)
                    and after[0].pitch_class in (t_pc, deg6)
                    and (after[1] is None or leading not in {t.pitch_class for t in chord_notes(after[1], after[0])})):
                for v in live:
                    if prev[v].pitch_class != leading or cur[v] is None or cur[v].midi - prev[v].midi == 1:
                        continue
                    if v == live[-1] and passing_bass(k, v):
                        continue
                    move = f"{prev[v].name} -> {cur[v].name}"
                    if v in (live[0], live[-1]):
                        add(violations, k, [v], "leading_tone",
                            f"{move}: the leading tone of {label(before)} does not rise to the tonic of {label(after)}")
                    else:
                        add(warnings, k, [v], "leading_tone",
                            f"{move}: frustrated leading tone (an inner voice of {label(before)} -> {label(after)})")
            if {n.pitch_class for n in prev if n is not None} == {n.pitch_class for n in cur if n is not None}:
                continue   # the same harmony goes on: a seventh or augmented sixth may resolve later
            sounding_prev = [n for n in prev if n is not None]
            aug6 = _augmented_sixth(sounding_prev, min(sounding_prev, key=lambda n: n.midi)) if sounding_prev else None
            if aug6 is not None:   # It6/Fr43/Ger65/Sw43: its #4 is no chord seventh
                low, high = aug6
                for v in live:
                    if cur[v] is None or prev[v].without_octave() not in aug6:
                        continue
                    upper = prev[v].without_octave() == high
                    step = cur[v].midi - prev[v].midi
                    move = f"{prev[v].name} -> {cur[v].name}"
                    what = (f"the augmented sixth's upper note ({high.name}, #4)" if upper
                            else f"the augmented sixth's lower note ({low.name}, b6)")
                    should = "rise a semitone" if upper else "fall a semitone"
                    if step == 0:
                        add(warnings, k, [v], "augmented_sixth",
                            f"{move}: {what} is held into the next chord (it should {should})")
                    elif step != (1 if upper else -1):
                        add(violations, k, [v], "augmented_sixth", f"{move}: {what} does not {should}")
                continue
            if before is None:
                continue
            _fifth_pc, seventh_pc = _roles(chord_notes(before[1], before[0]))
            if seventh_pc is None:
                continue
            dim7 = (seventh_pc - before[0].pitch_class) % 12 == 9   # a bb7 may be respelled (Cb dim7: Ab)
            for v in live:
                if prev[v].pitch_class != seventh_pc or cur[v] is None:
                    continue
                if not dim7 and _steps(prev[v], before[0]) != 6:
                    continue   # the same pitch spelled as another interval is no seventh
                fall = prev[v].midi - cur[v].midi
                move = f"{prev[v].name} -> {cur[v].name}"
                if fall == 0:
                    add(warnings, k, [v], "seventh_resolution",
                        f"{move}: the seventh of {label(before)} is held into the next chord (it should fall by step)")
                elif not 1 <= fall <= 2:
                    add(violations, k, [v], "seventh_resolution",
                        f"{move}: the seventh of {label(before)} does not fall by step")

    order = lambda e: (e["index"], e["voices"])  # noqa: E731
    violations.sort(key=order)
    warnings.sort(key=order)
    result = {
        "valid": not violations,
        "voice_order": names,
        "slices": [({"beat": beats[k]} if beats is not None else {})
                   | {"notes": [n.name if n is not None else None for n in row]} for k, row in enumerate(slices)],
        "violations": violations,
        "warnings": warnings,
        "motion": motion,
        "dissonances": dissonances,
    }
    if track_order is not None:
        result["track_order"] = track_order
    if tonic is not None:
        result["key"] = f"{tonic.name} {scale.name}"
    return result


def _leap_warnings(line, v, warnings, slices):
    for (k1, m1), (k2, m2), (_k3, m3) in zip(line, line[1:], line[2:]):
        leap, back = m2 - m1, m3 - m2
        if abs(leap) >= 8 and not (1 <= abs(back) <= 2 and (back > 0) != (leap > 0)):
            warnings.append({"index": k2, "voices": [v], "rule": "leap_not_recovered",
                             "message": f"{slices[k1][v].name} -> {slices[k2][v].name}: a leap of {abs(leap)} "
                                        "semitones not followed by a step back"})


# ============================================================= find_cadences

CADENCE_TYPES = ("authentic", "half", "plagal", "deceptive", "none")
_CAPLIN_HC = "Caplin: a half cadence ends on a root-position dominant triad"
_ASK_SOPRANO = "give the soprano to decide PAC vs IAC"
_LEADING_TONE_TYPES = ("diminished", "diminished 7", "half-diminished")
_FIRST_INVERSION = ("6", "65")      # analyze_progression's figure when the bass is the chord's third
_PLAGAL_FIGURES = ("", "7", "6", "65")
_AUGMENTED_SIXTHS = ("It6", "Fr43", "Ger65", "Sw43")
# The roots the cadence rules look for, as a spelled (semitones, letter steps) interval above the tonic.
_ON_1, _ON_4, _ON_5, _ON_b6, _ON_6, _ON_7 = (0, 0), (5, 3), (7, 4), (8, 5), (9, 5), (11, 6)


def _cadence_reading(reading: dict, tonic: Note, minor: bool) -> dict:
    """A read_chord reading -> its third, root position and cadence family (letters + semitones).

    Families: 'dominant' (root on ^5, major 3rd), 'leading-tone' (dim/dim7/m7b5 a spelled m2 below
    the tonic), 'tonic' (root on ^1, major or minor 3rd and perfect 5th: the triad or its 6th/7th
    chords), 'iv' (the same on ^4), 'submediant' (minor on ^6 in a major key, major on b6 in a minor
    key) and 'bVI' (major on b6 in a major key); None otherwise.
    """
    root, ctype, bass = reading["root"], reading["chord_type"], reading["bass"]
    degrees = dict(ctype.degrees) if ctype is not None else {}
    third = "major" if degrees.get("3") == 4 else "minor" if degrees.get("b3") == 3 else None
    triadic = third is not None and degrees.get("5") == 7

    def on(at):
        return root is not None and _is_above(root, tonic, *at)

    family = None
    if on(_ON_5) and third == "major":
        family = "dominant"
    elif on(_ON_7) and ctype.name in _LEADING_TONE_TYPES:
        family = "leading-tone"
    elif triadic and on(_ON_1):
        family = "tonic"
    elif triadic and on(_ON_4):
        family = "iv"
    elif triadic and ((minor and on(_ON_b6) and third == "major") or (not minor and on(_ON_6) and third == "minor")):
        family = "submediant"
    elif triadic and not minor and on(_ON_b6) and third == "major":
        family = "bVI"
    return {"root": root, "ctype": ctype, "bass": bass, "third": third, "family": family,
            "root_position": root is not None and bass.pitch_class == root.pitch_class,
            "minor_v": on(_ON_5) and third == "minor", "minor_seventh": degrees.get("b7") == 10}


def _spelling_caveat(root: Note | None, tonic: Note, intervals) -> str | None:
    """A caveat when a root sounds like a cadence degree but is spelled as another (Gb in F#)."""
    if root is None:
        return None
    rel = (root.pitch_class - tonic.pitch_class) % 12
    for semis, steps in (_ON_1, _ON_4, _ON_5, _ON_b6, _ON_6, _ON_7):
        if rel == semis and not _is_above(root, tonic, semis, steps):
            expected = transpose(tonic, semis, steps)
            label = _bass_degree(expected, tonic, intervals)
            return (f"{root.name} is spelled as another degree than {expected.name} (^{label}): cadence roles "
                    f"are read by letter, so respell it to count as ^{label}")
    return None


def _completed_array(item) -> list[str] | None:
    """A note array no table chord matches, with its omitted fifth restored — the omission
    four-part writing allows, read by check_voice_leading's rule (_slice_chord): G2 B3 F4 G4 is G7,
    C3 C4 E4 C5 is C. Returns the octave-less notes, bass first, or None when that reads no chord.
    """
    notes = list(dict.fromkeys(parse_notes(list(item))))
    if len(notes) > 6:
        return None
    if any(x.octave is None for x in notes):   # no octaves: the first note is the bass (read_chord)
        notes = assign_octaves(list(dict.fromkeys(x.without_octave() for x in notes)), 2, "ascending")
    found = _slice_chord(notes)
    if found is None:
        return None
    names = list(dict.fromkeys(x.without_octave().name for x in sorted(notes, key=lambda x: x.midi)))
    fifth = transpose(found[0], 7, 4)
    if all(parse_note(x).pitch_class != fifth.pitch_class for x in names):
        names.append(fifth.name)
    return names


def _tonic_roman(ctype, figure: str | None) -> str:
    """A major-third tonic written as I in roman_to_chords' dialect ('I', 'I6', 'I7', 'IΔ7')."""
    if figure is not None and ctype.name in _FIGURE_MARKS:
        return f"I{_FIGURE_MARKS[ctype.name]}{figure}"
    return f"I{roman_suffix(ctype, False)}"


def _soprano_line(soprano, items: list) -> list[Note | None]:
    """The soprano note of every chord: the `soprano` argument, else a note array's highest pitch."""
    n = len(items)
    if soprano is not None:
        line = assign_octaves(parse_notes(soprano), 4, "nearest")
        if len(line) != n:
            raise ValueError(f"soprano has {len(line)} notes but there are {n} chords (one soprano note per chord)")
        return line
    out: list[Note | None] = []
    for item in items:
        if isinstance(item, (list, tuple)):
            notes = parse_notes(list(item))
            if all(x.octave is not None for x in notes):
                out.append(max(notes, key=lambda x: x.midi))
                continue
        out.append(None)
    return out


def _phrase_end_indices(phrase_ends, phrase_length, n: int) -> tuple[list[int], bool]:
    """(sorted phrase-end indices, whether this is an 'all' scan)."""
    if phrase_length is None:
        phrase_length = 0
    if not isinstance(phrase_length, int) or isinstance(phrase_length, bool):
        raise ValueError(f"phrase_length must be a whole number of chords (0 = not used), got {phrase_length!r}")
    if phrase_length < 0:
        raise ValueError(f"phrase_length must be 0 or more, got {phrase_length}")
    if phrase_ends is not None and phrase_length:
        raise ValueError("give phrase_ends or phrase_length, not both")
    if phrase_length:
        if phrase_length > n:
            raise ValueError(f"phrase_length {phrase_length} is longer than the progression ({n} chords)")
        return list(range(phrase_length - 1, n, phrase_length)), False
    if phrase_ends is None:
        return [n - 1], False
    if isinstance(phrase_ends, str):
        if phrase_ends.strip().lower() == "all":
            return list(range(n)), True
        raise ValueError(f"phrase_ends must be a list of 0-based chord indices or 'all', got {phrase_ends!r}")
    if not isinstance(phrase_ends, (list, tuple)) or not phrase_ends:
        raise ValueError("phrase_ends must be a non-empty list of 0-based chord indices, or 'all'")
    for e in phrase_ends:
        if not isinstance(e, int) or isinstance(e, bool):
            raise ValueError(f"phrase_ends holds 0-based chord indices (integers), got {e!r}")
        if not 0 <= e < n:
            raise ValueError(f"phrase end {e} is out of range: there are {n} chords (indices 0-{n - 1})")
    return sorted(set(phrase_ends)), False


def find_cadences(chords, root: str, scale_type: str = "major", soprano=None, phrase_ends=None,
                  phrase_length: int = 0) -> dict:
    """Label the cadence at each phrase end: authentic (perfect/imperfect), half, plagal, deceptive or none.

    A textbook codification — Kostka, Payne & Almén, Tonal Harmony, ch. 10
    "Cadences, Phrases, and Periods" (PAC/IAC/HC/plagal/deceptive, Phrygian HC);
    William E. Caplin, Classical Form (1998) (a half cadence ends on a
    root-position dominant triad; the cadential 6/4); Aldwell & Schachter,
    Harmony and Voice Leading (V–IV6 deceptive motion, the Picardy third). It is
    not music21, which has cadence primitives but no classifier.

    `chords`: symbols or note arrays (a string splits on spaces/commas), read by
    analyze_progression (degree, figure, specials such as Cad64); a note array
    no table chord matches is read with its omitted fifth restored, as
    check_voice_leading reads four-part writing (G2 F3 B3 G4 = V7, C3 C4 E4 C5
    = I), so bach_chorale_voicing's SATB rows read as written. `root` /
    `scale_type`: a seven-note key; its mode is major when the scale's 3rd is 4
    semitones, else minor. Chord roots are judged by letter AND semitones from
    the tonic (Ab in C is b6, G# is not; a misspelled root gets a caveat).

    Families: DOMINANT = root on ^5 with a major 3rd (V, V7, V9 …);
    LEADING-TONE = a dim / dim7 / m7b5 chord a semitone below the tonic (the
    raised 7th in minor); TONIC = root on ^1 with a major or minor 3rd and a
    perfect 5th (its 6th/7th chords too); IV = the same on ^4; SUBMEDIANT =
    minor on ^6 in a major key, major on b6 in a minor key.

    Phrase ends (0-based chord indices): `phrase_ends` as a list; or 'all'
    (scan every chord, reporting authentic, plagal and deceptive cadences
    anywhere, a half cadence or 'none' only at the last chord); or
    `phrase_length` L > 0 (indices L−1, 2L−1, …; e.g. from plan_sections'
    section lengths); default the last chord. Not both.

    Soprano per chord: `soprano` (one note per chord — pass
    bach_chorale_voicing(...)['voices']['soprano'] or a melody), else a note
    array's highest pitch when all its notes carry octaves (its lowest is the
    bass), else unknown. `soprano` in the result shows the reading.

    Rules at end e (P = chord e−1, F = chord e; first match wins):
    1 authentic — P dominant or leading-tone, F tonic. 'perfect' only when P is
      a root-position dominant, F is in root position AND F's soprano is ^1;
      else 'imperfect' with reason 'leading-tone chord', or 'inversion' and/or
      'soprano on ^3' ('; '-joined). Soprano unknown and both in root position:
      subtype null, reason 'give the soprano to decide PAC vs IAC' — never a
      PAC without ^1.
    2 plagal — P on ^4 (IV/iv in root position or 6), F tonic in root position.
    3 deceptive — P dominant, F submediant (subtype null), a major-key bVI
      ('bVI') or IV/iv in first inversion ('IV6', Aldwell & Schachter).
    4 half — F dominant; 'phrygian' in a minor key when P is iv6 (b6 in the
      bass falling a semitone to ^5). An inverted V or a V7 is still a half
      cadence, with the caveat 'Caplin: a half cadence ends on a root-position
      dominant triad'.
    5 none — with a reason ('ends on vi', 'a minor v has no leading tone' …).
    cadential_64: a Cad64 before the dominant (authentic and deceptive: chord
    e−2, and the span starts there; half: chord e−1). picardy: a major-3rd tonic
    ending in a minor key (its roman is written I, not analyze_progression's
    context-free V/iv). A tonic with a minor 7th gets a blues-tonic caveat.

    Returns {key, phrase_ends, soprano, cadences: [{index, type, subtype,
    reason, span [start, end], chords, romans (roman_figured, which
    roman_to_chords reads back), soprano_degrees, bass_degrees (scale degrees,
    '#7' off the scale, null unknown), soprano_checked, cadential_64, picardy,
    caveats}], summary {type: count}} — the report shape of check_melody.
    e.g. find_cadences('C F C/G G7 C', 'C', soprano='E5 F5 E5 D5 C5') -> index
    4 authentic perfect, span [2, 4], romans Cad64 V7 I, soprano_degrees
    [3, 2, 1], cadential_64 true. Deterministic.
    """
    if not isinstance(root, str):
        raise ValueError(f"root must be a note name like 'C' or 'F#', got {root!r}")
    scale = resolve_scale_type(scale_type)
    intervals = scale.intervals
    if len(intervals) != 7:
        raise ValueError(f"find_cadences needs a seven-note key (major, minor or a mode); "
                         f"{scale.name} has {len(intervals)} notes")
    items = _chord_items(chords)
    analysis = analyze_progression(items, root, scale.name)
    heard = list(items)
    for i, (item, entry) in enumerate(zip(items, analysis["chords"])):
        if entry["chord_type"] == "unknown" and entry.get("special") is None:
            heard[i] = _completed_array(item) or item
    if heard != items:   # read again, so a Cad64 before a completed V7 is found too
        analysis = analyze_progression(heard, root, scale.name)
    tonic = parse_notes(root)[0].without_octave()
    minor = intervals[2] != 4
    n = len(items)
    info = [_cadence_reading(read_chord(item), tonic, minor) for item in heard]
    for chord, entry in zip(info, analysis["chords"]):
        if entry.get("special") == "Cad64":
            # a cadential 6/4 embellishes the dominant after it (Aldwell & Schachter): no tonic arrival
            chord["family"] = "cadential 6/4"
        elif entry.get("special") in _AUGMENTED_SIXTHS:
            chord["family"] = None   # an augmented sixth (Ab C Eb F#) is no bVI, whatever its table reading
    sop = _soprano_line(soprano, items)
    ends, scan_all = _phrase_end_indices(phrase_ends, phrase_length, n)
    written = [item if isinstance(item, str) else [x.name for x in parse_notes(list(item))] for item in items]

    cadences = []
    for e in ends:
        cad = _cadence_at(e, info, analysis["chords"], written, sop, tonic, intervals, minor)
        if scan_all and e != n - 1 and cad["type"] in ("half", "none"):
            continue   # an 'all' scan reports half cadences (and 'none') only at the last chord
        cadences.append(cad)
    summary = {t: 0 for t in CADENCE_TYPES}
    for cad in cadences:
        summary[cad["type"]] += 1
    return {
        "key": analysis["key"],
        "phrase_ends": [cad["index"] for cad in cadences],
        "soprano": [x.name if x is not None else None for x in sop],
        "cadences": cadences,
        "summary": summary,
    }


def _cadence_at(e: int, info: list[dict], analysed: list[dict], written: list, sop: list, tonic: Note,
                intervals, minor: bool) -> dict:
    fin = info[e]
    picardy = minor and fin["family"] == "tonic" and fin["third"] == "major"

    def roman(i):
        if i == e and picardy:
            return _tonic_roman(fin["ctype"], analysed[i]["figure"])
        a = analysed[i]
        return a["roman_figured"] if a["roman_figured"] is not None else a["roman"]

    def degree(note):
        return _bass_degree(note, tonic, intervals)

    kind, subtype, cad64, caveats = "none", None, False, []
    start = max(e - 1, 0)
    if e == 0:
        reason = "no chord before it"
    else:
        pre = info[e - 1]
        pr, fr = roman(e - 1), roman(e)
        dominant_before = e >= 2 and pre["family"] == "dominant" and analysed[e - 2].get("special") == "Cad64"
        if pre["family"] in ("dominant", "leading-tone") and fin["family"] == "tonic":
            kind = "authentic"
            if pre["family"] == "leading-tone":
                subtype, reason = "imperfect", "leading-tone chord"
            else:
                why = []
                if not (pre["root_position"] and fin["root_position"]):
                    why.append("inversion")
                if sop[e] is not None and sop[e].pitch_class != tonic.pitch_class:
                    why.append(f"soprano on ^{degree(sop[e])}")
                if why:
                    subtype, reason = "imperfect", "; ".join(why)
                elif sop[e] is None:
                    reason = _ASK_SOPRANO
                else:
                    subtype, reason = "perfect", f"root-position {pr} -> {fr} with ^1 in the soprano"
            cad64 = dominant_before
        elif (pre["family"] == "iv" and analysed[e - 1]["figure"] in _PLAGAL_FIGURES
              and fin["family"] == "tonic" and fin["root_position"]):
            kind, reason = "plagal", f"{pr} -> {fr}"
        elif pre["family"] == "dominant" and (fin["family"] in ("submediant", "bVI") or (
                fin["family"] == "iv" and analysed[e]["figure"] in _FIRST_INVERSION)):
            kind = "deceptive"
            subtype = {"bVI": "bVI", "iv": "IV6"}.get(fin["family"])
            reason = f"{pr} -> {fr} instead of {'i' if minor else 'I'}"
            cad64 = dominant_before
        elif fin["family"] == "dominant":
            kind = "half"
            if (minor and pre["family"] == "iv" and pre["third"] == "minor"
                    and analysed[e - 1]["figure"] in _FIRST_INVERSION and fin["root_position"]):
                subtype = "phrygian"
                reason = (f"{pr} -> {fr}: the bass falls a semitone, ^{degree(pre['bass'])} -> "
                          f"^{degree(fin['bass'])}")
            else:
                reason = f"ends on {fr}"
            if not fin["root_position"] or fin["ctype"].name != "major":
                caveats.append(_CAPLIN_HC)
            cad64 = analysed[e - 1].get("special") == "Cad64"
        elif fin["family"] == "tonic":
            if pre["minor_v"]:
                reason = f"{pr} -> {fr}: a minor v has no leading tone"
            elif pre["family"] == "iv":
                reason = f"{pr} -> {fr}: a plagal cadence needs IV/iv in root position or 6 and a root-position tonic"
            else:
                reason = f"{pr} -> {fr}: the tonic is not approached from V, vii° or IV"
        elif fin["family"] == "cadential 6/4":
            reason = f"ends on {fr}: a cadential 6/4 embellishes the dominant that follows; end the phrase on that V"
        else:
            reason = f"ends on {fr}"
        if cad64 and kind != "half":
            start = e - 2
    span = range(start, e + 1)
    for i in (e - 1, e) if e else (e,):
        note = _spelling_caveat(info[i]["root"], tonic, intervals)
        if note is not None and note not in caveats:
            caveats.append(note)
    if fin["family"] == "tonic" and fin["third"] == "major" and fin["minor_seventh"]:
        caveats.append("a dominant seventh on the tonic: a blues tonic; common-practice harmony hears it as an "
                       "applied V7 of the subdominant")
    return {
        "index": e,
        "type": kind,
        "subtype": subtype,
        "reason": reason,
        "span": [start, e],
        "chords": [written[i] for i in span],
        "romans": [roman(i) for i in span],
        "soprano_degrees": [degree(sop[i]) if sop[i] is not None else None for i in span],
        "bass_degrees": [degree(info[i]["bass"]) for i in span],
        "soprano_checked": sop[e] is not None,
        "cadential_64": cad64,
        "picardy": picardy,
        "caveats": caveats,
    }
