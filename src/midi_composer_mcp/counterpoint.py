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
