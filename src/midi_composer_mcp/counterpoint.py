"""First-species counterpoint: a rule-following counter-melody for a cantus firmus.

The tool contains the *rules*, not the creativity: given a cantus firmus (the
LLM's melody) and a key, it derives a note-against-note counterpoint that obeys
the classical first-species constraints — consonant intervals only, begin and
end on a perfect consonance, contrary/oblique motion preferred, and no parallel
or directly-approached perfect fifths or octaves. The choice at each step is a
deterministic, lowest-penalty rule decision, so the same cantus always yields
the same counterpoint.
"""

from __future__ import annotations

from .melody import _pitch_table
from .notes import parse_notes
from .scales import resolve_scale_type, scale_notes

# Consonant intervals (semitones within an octave): unison/8ve, m3, M3, P5, m6,
# M6. The perfect 4th and the tritone are treated as dissonant in two voices.
_CONSONANT = {0, 3, 4, 7, 8, 9}
_PERFECT = {0, 7}


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def first_species(cantus, root: str, scale_type: str = "major",
                  position: str = "above") -> dict:
    """Write a first-species (note-against-note) counterpoint to a cantus firmus.

    `cantus` is the given melody (notes, ideally with octaves). The counterpoint
    is placed `above` or `below` it, diatonic to `root`/`scale_type`, following
    the first-species rules. Returns the cantus, the counterpoint line, and the
    interval between the voices at each step — render them as two notes tracks
    sharing one rhythm.
    """
    if position not in ("above", "below"):
        raise ValueError(f"position must be 'above' or 'below', got {position!r}")
    scale = resolve_scale_type(scale_type)
    root_note = parse_notes(root)[0].without_octave()
    tonic_pc = root_note.pitch_class

    from .midi_io import assign_octaves  # local import avoids a cycle

    cf = assign_octaves(parse_notes(cantus), 4, "nearest")
    if len(cf) < 2:
        raise ValueError("cantus firmus needs at least 2 notes")
    cf_midis = [n.midi for n in cf]
    table = _pitch_table(scale_notes(scale, root_note)[:-1])  # diatonic pitches with spelling
    last = len(cf) - 1

    def candidates(i: int):
        cantus_m = cf_midis[i]
        cands = []
        for m, note in table:
            if position == "above" and not (cantus_m < m <= cantus_m + 16):
                continue
            if position == "below" and not (cantus_m - 16 <= m < cantus_m):
                continue
            simple = abs(m - cantus_m) % 12
            if simple not in _CONSONANT:
                continue
            # Endpoints: perfect consonance; the final note lands on the tonic.
            if i in (0, last) and simple not in _PERFECT:
                continue
            if i == last and note.pitch_class != tonic_pc:
                continue
            cands.append((m, note, simple))
        if not cands:  # never strand a position — relax to any consonance in range
            for m, note in table:
                in_range = (position == "above" and cantus_m < m <= cantus_m + 16) or \
                           (position == "below" and cantus_m - 16 <= m < cantus_m)
                if in_range and abs(m - cantus_m) % 12 in _CONSONANT:
                    cands.append((m, note, abs(m - cantus_m) % 12))
        return cands

    def node_cost(i: int, cand) -> int:
        m, note, simple = cand
        s = 0
        if 0 < i < last:
            if simple in _PERFECT:
                s += 2            # prefer imperfect consonances in the middle
            if simple == 0:
                s += 2            # unisons mid-phrase are weak
        if i == 0 and note.pitch_class == tonic_pc:
            s -= 1                # a tonic start is strong
        if abs(m - cf_midis[i]) > 16:
            s += 3                # keep the voices within a tenth or so
        return s

    def trans_cost(i: int, prev_m: int, cand) -> int:
        m, _note, simple = cand
        s = 0
        c_dir = _sign(cf_midis[i] - cf_midis[i - 1])
        p_dir = _sign(m - prev_m)
        if c_dir != 0 and p_dir == -c_dir:
            s -= 3                # contrary motion: best
        elif p_dir == 0:
            s -= 1                # oblique: fine
        elif c_dir != 0 and p_dir == c_dir:
            s += 2                # similar motion: discourage
        if simple in _PERFECT and c_dir != 0 and c_dir == p_dir:
            s += 100              # parallel/direct fifth or octave: all but forbidden
        leap = abs(m - prev_m)
        if leap == 0:
            s += 1                # don't repeat the same pitch
        elif leap > 12:
            s += 60               # avoid leaps over an octave
        elif leap > 7:
            s += 4
        elif leap > 2:
            s += 1
        return s

    # Viterbi: globally minimal rule-following line (deterministic; ties -> lower pitch).
    cand_lists = [candidates(i) for i in range(len(cf))]
    dp = [(node_cost(0, c), [c]) for c in cand_lists[0]]
    for i in range(1, len(cf)):
        nxt = []
        for cand in cand_lists[i]:
            best = min(
                ((cost + trans_cost(i, path[-1][0], cand), path) for cost, path in dp),
                key=lambda t: (t[0], t[1][-1][0]),
            )
            nxt.append((best[0] + node_cost(i, cand), best[1] + [cand]))
        dp = nxt
    _, best_path = min(dp, key=lambda t: (t[0], [c[0] for c in t[1]]))
    cp = [c[1] for c in best_path]

    intervals = []
    for cp_note, cf_note in zip(cp, cf):
        simple = abs(cp_note.midi - cf_note.midi) % 12
        intervals.append({0: "P1/P8", 3: "m3", 4: "M3", 7: "P5", 8: "m6", 9: "M6"}.get(simple, f"{simple}st"))

    return {
        "position": position,
        "key": f"{root_note.name} {scale.name}",
        "cantus": [n.name for n in cf],
        "counterpoint": [n.name for n in cp],
        "intervals": intervals,
    }


# ----------------------------------------------- higher species (2nd-5th)

_BIG = 1000  # penalty that all but forbids a move (kept finite so the DP never strands)
_INTERVAL_NAME = {0: "P1/P8", 3: "m3", 4: "M3", 7: "P5", 8: "m6", 9: "M6"}


def _slot_plan(n_bars: int, species: int, bar_beats: float) -> list[dict]:
    """Per-bar slots: their duration, metric strength, and tie/suspension flags."""
    plan: list[dict] = []
    for i in range(n_bars):
        final = i == n_bars - 1
        if final:
            plan.append({"bar": i, "dur": bar_beats, "strong": True, "tie": False, "final": True})
            continue
        if species == 1:
            ks = [(bar_beats, True, False)]
        elif species == 2:
            ks = [(bar_beats / 2, True, False), (bar_beats / 2, False, False)]
        elif species == 3:
            ks = [(bar_beats / 4, s == 0, False) for s in range(4)]
        elif species == 4:  # syncopated: the downbeat is tied over from the previous bar
            ks = [(bar_beats / 2, True, i > 0), (bar_beats / 2, False, False)]
        else:  # species 5 (florid): alternate halves and quarters, suspension before the cadence
            if i == n_bars - 2:
                ks = [(bar_beats / 2, True, True), (bar_beats / 2, False, False)]
            elif i % 2 == 0:
                ks = [(bar_beats / 4, s == 0, False) for s in range(4)]
            else:
                ks = [(bar_beats / 2, True, False), (bar_beats / 2, False, False)]
        for dur, strong, tie in ks:
            plan.append({"bar": i, "dur": dur, "strong": strong, "tie": tie, "final": False})
    plan[0]["first"] = True
    for s in plan:
        s.setdefault("first", False)
    return plan


def species_counterpoint(cantus, root: str, scale_type: str = "major",
                         species: int = 1, position: str = "above") -> dict:
    """Counterpoint in species 1-5 (Fux): note-against-note up to florid, deterministic.

    Species 1 = 1:1, 2 = 2:1 (passing tones on weak beats), 3 = 4:1 (passing and
    neighbour figures), 4 = syncopated suspensions (a tied dissonance resolving
    down by step), 5 = florid (a mix). The counterpoint is diatonic to the key,
    keeps consonances on strong beats, treats every dissonance as a step-wise
    passing/neighbour tone or a properly-resolved suspension, ends on a perfect
    consonance on the tonic, and avoids parallel/direct fifths and octaves. A
    deterministic Viterbi search makes the same cantus always yield the same line.
    Returns the two voices, the per-bar downbeat intervals, and a `render_hint`
    with ready-to-use notes-track specs (cantus as whole notes; the counterpoint
    with its own rhythm).
    """
    if position not in ("above", "below"):
        raise ValueError(f"position must be 'above' or 'below', got {position!r}")
    if species not in (1, 2, 3, 4, 5):
        raise ValueError(f"species must be 1-5, got {species!r}")
    scale = resolve_scale_type(scale_type)
    root_note = parse_notes(root)[0].without_octave()
    tonic_pc = root_note.pitch_class

    from .midi_io import assign_octaves

    cf = assign_octaves(parse_notes(cantus), 4, "nearest")
    if len(cf) < 2:
        raise ValueError("cantus firmus needs at least 2 notes")
    cf_m = [n.midi for n in cf]
    table = _pitch_table(scale_notes(scale, root_note)[:-1])
    bar_beats = 4.0
    slots = _slot_plan(len(cf), species, bar_beats)

    def gen(t: int):
        s = slots[t]
        c = cf_m[s["bar"]]
        out = []
        for m, note in table:
            if position == "above" and not (c < m <= c + 16):
                continue
            if position == "below" and not (c - 16 <= m < c):
                continue
            simple = abs(m - c) % 12
            consonant = simple in _CONSONANT
            if s["first"] or s["final"]:
                if simple not in _PERFECT:
                    continue
                if s["final"] and note.pitch_class != tonic_pc:
                    continue
            elif s["strong"] and not consonant:
                continue
            out.append((m, note, simple, consonant))
        return out

    def node_cost(t: int, cand) -> int:
        m, note, simple, consonant = cand
        s = slots[t]
        cost = 0
        if not s["first"] and not s["final"]:
            if s["strong"] and simple in _PERFECT:
                cost += 2          # imperfect consonances preferred mid-phrase
            if simple == 0 and s["strong"]:
                cost += 2
        if s["first"] and note.pitch_class == tonic_pc:
            cost -= 1
        if abs(m - cf_m[s["bar"]]) > 16:
            cost += 3
        return cost

    def prev_strong(path):
        for cand, s in zip(reversed(path), reversed(slots[: len(path)])):
            if s["strong"]:
                return cand
        return None

    def trans_cost(t: int, a, b, path) -> int:
        s, sp = slots[t], slots[t - 1]
        bm, _bn, b_simple, b_cons = b
        am, _an, a_simple, a_cons = a
        cost = 0
        leap = abs(bm - am)
        if leap > 12:
            cost += _BIG
        elif leap > 7:
            cost += 4
        elif leap > 2:
            cost += 1
        elif leap == 0:
            cost += 1
        # dissonance treatment: passing/neighbour notes are approached and left by step
        if not b_cons and not s["tie"] and leap > 2:
            cost += _BIG
        if not a_cons and leap > 2:
            cost += _BIG
        # a suspension (a tied dissonance) must resolve DOWN by step to a consonance
        if sp["tie"] and not a_cons:
            if not (b_cons and 1 <= am - bm <= 2):
                cost += _BIG
            else:
                cost -= 2          # reward a clean suspension resolution
        # parallel / direct perfect fifths and octaves
        c_dir = _sign(cf_m[s["bar"]] - cf_m[sp["bar"]])
        p_dir = _sign(bm - am)
        if b_simple in _PERFECT and c_dir != 0 and c_dir == p_dir:
            cost += _BIG
        if s["strong"] and b_simple in _PERFECT:
            ps = prev_strong(path)
            if ps is not None:
                psm = ps[0]
                pc_dir = _sign(cf_m[s["bar"]] - cf_m[slots[len(path) - 1]["bar"]])
                if b_simple == abs(psm - cf_m[s["bar"]]) % 12 and _sign(bm - psm) == pc_dir and pc_dir != 0:
                    cost += _BIG
        # motion preference
        if c_dir != 0:
            if p_dir == -c_dir:
                cost -= 3
            elif p_dir == c_dir:
                cost += 2
        elif p_dir == 0:
            cost += 1
        if s["final"] and c_dir != 0 and p_dir == c_dir:
            cost += 4              # approach the final by contrary motion
        return cost

    # Viterbi, pruned to the best path per ending pitch (deterministic tie-breaks).
    frontier: dict[int, tuple] = {}
    for cand in gen(0):
        frontier[cand[0]] = (node_cost(0, cand), [cand])
    for t in range(1, len(slots)):
        nxt: dict[int, tuple] = {}
        tie = slots[t]["tie"]
        for _lm, (cost, path) in frontier.items():
            a = path[-1]
            if tie:  # forced continuation of the previous note
                c = cf_m[slots[t]["bar"]]
                simple = abs(a[0] - c) % 12
                exts = [(a[0], a[1], simple, simple in _CONSONANT)]
            else:
                exts = gen(t)
            for b in exts:
                total = cost + trans_cost(t, a, b, path) + node_cost(t, b)
                key = b[0]
                cand_path = path + [b]
                if key not in nxt or total < nxt[key][0] or (
                    total == nxt[key][0] and [x[0] for x in cand_path] < [x[0] for x in nxt[key][1]]
                ):
                    nxt[key] = (total, cand_path)
        frontier = nxt
    _, best = min(frontier.values(), key=lambda cp: (cp[0], [x[0] for x in cp[1]]))

    # Collapse tied slots into held durations; emit onset notes + a rhythm string.
    onsets: list[list] = []
    for cand, slot in zip(best, slots):
        if slot["tie"] and onsets:
            onsets[-1][1] += slot["dur"]
        else:
            onsets.append([cand[1], slot["dur"]])
    grid = min(s["dur"] for s in slots)
    cp_notes = [o[0].name for o in onsets]
    cp_rhythm = "".join("O" + "." * (int(round(o[1] / grid)) - 1) for o in onsets)

    # Per-bar downbeat interval (the note sounding on each bar's strong beat).
    downbeat = []
    for i in range(len(cf)):
        cand = next(c for c, s in zip(best, slots) if s["bar"] == i and s["strong"])
        downbeat.append(_INTERVAL_NAME.get(abs(cand[0] - cf_m[i]) % 12, f"{abs(cand[0]-cf_m[i])%12}st"))

    return {
        "species": species,
        "ratio": {1: "1:1", 2: "2:1", 3: "4:1", 4: "syncopated", 5: "florid"}[species],
        "position": position,
        "key": f"{root_note.name} {scale.name}",
        "cantus": [n.name for n in cf],
        "cantus_step_beats": bar_beats,
        "counterpoint": cp_notes,
        "counterpoint_rhythm": cp_rhythm,
        "counterpoint_step_beats": grid,
        "downbeat_intervals": downbeat,
        "render_hint": {
            "tracks": [
                {"type": "notes", "name": "cantus", "notes": [n.name for n in cf],
                 "step_beats": bar_beats, "sustain": True},
                {"type": "notes", "name": "counterpoint", "notes": cp_notes,
                 "rhythm": cp_rhythm, "step_beats": grid, "sustain": True},
            ],
        },
    }
