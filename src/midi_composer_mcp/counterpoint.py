"""Species counterpoint (Fux, species 1-5): a rule-following line for a cantus firmus.

The tool contains the *rules*, not the creativity: given a cantus firmus (the
LLM's melody) and a key, it derives a counter-melody that obeys the classical
species constraints — consonances on strong beats, every dissonance a stepwise
passing/neighbour tone or a prepared suspension that resolves down, perfect
consonances at the ends with a stepwise cadence on the tonic, no parallel or
direct fifths/octaves, no melodic tritones, sevenths or augmented steps.
Intervals are judged by their *spelling* (letters plus semitones), so a
diminished seventh in harmonic minor is a dissonance even though it spans the
nine semitones of a major sixth. The choice at each step is a deterministic,
lowest-penalty decision (an exact Viterbi search), so the same cantus always
yields the same counterpoint.
"""

from __future__ import annotations

from .melody import _pitch_table, _same_pitch
from .notes import LETTERS, Note, parse_notes
from .scales import resolve_scale_type, scale_notes

# Penalty that all but forbids a move. Kept finite so the search never strands,
# and far above any sum of the small style costs, so best_cost / _BIG counts the
# rules broken.
_BIG = 1_000_000
# Largest distance between the voices, in semitones: a tenth (plus a little), widened
# only when no rule-abiding line fits inside it.
_WIDTHS = (16, 19, 24)
_MAJOR_REF = (0, 2, 4, 5, 7, 9, 11)
_PERFECT_NUMBERS = (1, 4, 5)
# (simple interval number, offset from major/perfect) that count as consonant in two voices
_CONSONANCES = {(1, 0), (5, 0), (3, 0), (3, -1), (6, 0), (6, -1)}


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _spelled(low: Note, high: Note) -> tuple[int, int]:
    """(simple number 1-7, offset) of the interval from `low` up to `high` (concrete notes).

    The offset is measured from the major/perfect interval of that number: 0 is
    major or perfect, -1 minor (or diminished for 1/4/5), +1 augmented...
    """
    steps = (LETTERS.index(high.letter) + 7 * high.octave) - (LETTERS.index(low.letter) + 7 * low.octave)
    octaves, simple = divmod(steps, 7)
    return simple + 1, high.midi - low.midi - 12 * octaves - _MAJOR_REF[simple]


def _interval_name(number: int, offset: int) -> str:
    if (number, offset) == (1, 0):
        return "P1/P8"
    if number in _PERFECT_NUMBERS:
        quality = {0: "P", 1: "A", -1: "d", 2: "AA", -2: "dd"}.get(offset, "?")
    else:
        quality = {0: "M", -1: "m", 1: "A", -2: "d", 2: "AA", -3: "dd"}.get(offset, "?")
    return f"{quality}{number}"


def _melodic_ok(a: Note, b: Note) -> bool:
    """A singable melodic step: no tritone, seventh, augmented/diminished interval or leap over an octave."""
    leap = abs(b.midi - a.midi)
    if leap == 0:
        return True
    if leap > 12 or leap in (6, 10, 11):
        return False
    low, high = (a, b) if a.midi < b.midi else (b, a)
    number, offset = _spelled(low, high)
    if number in _PERFECT_NUMBERS:
        return offset == 0
    return offset in (0, -1)


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
            if i == n_bars - 2 and i > 0:
                ks = [(bar_beats / 2, True, True), (bar_beats / 2, False, False)]
            elif i == n_bars - 3:  # the suspension is prepared by a half note, never a tied quarter
                ks = [(bar_beats / 2, True, False), (bar_beats / 2, False, False)]
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


def _solve(cf: list[Note], table, tonic: Note, species: int, position: str,
           slots: list[dict], width: int):
    """Exact Viterbi search for one voice-distance limit. Returns (cost, path of candidates)."""
    cf_m = [n.midi for n in cf]
    tonic_pc = tonic.pitch_class
    n_bars = len(cf)

    def interval(m: int, note: Note, c: Note) -> tuple[int, int]:
        return _spelled(c, note) if m > c.midi else _spelled(note, c)

    def cand(m: int, note: Note, bar: int):
        number, offset = interval(m, note, cf[bar])
        consonant = (number, offset) in _CONSONANCES
        return (m, note, number, offset, consonant)

    def perfect(x) -> bool:
        return x[4] and x[2] in (1, 5)

    melodic_memo: dict[tuple[int, int], bool] = {}  # each MIDI pitch has one spelling here

    def melodic_ok(a, b) -> bool:
        key = (a[0], b[0])
        ok = melodic_memo.get(key)
        if ok is None:
            ok = melodic_memo[key] = _melodic_ok(a[1], b[1])
        return ok

    def gen(t: int):
        s = slots[t]
        c = cf_m[s["bar"]]
        # a note that will be tied over the barline is the suspension's preparation:
        # it must be struck as a consonance, only the tied continuation may clash
        prepares = t + 1 < len(slots) and slots[t + 1]["tie"]
        out = []
        for m, note in table:
            if position == "above" and not (c < m <= c + width):
                continue
            if position == "below" and not (c - width <= m < c):
                continue
            x = cand(m, note, s["bar"])
            if s["first"] or s["final"]:
                if not perfect(x):
                    continue
                if s["final"] and note.pitch_class != tonic_pc:
                    continue
                if s["first"] and position == "below" and x[2] != 1:
                    continue  # a lower voice opens on the unison/octave: a fifth below implies another key
            elif (s["strong"] or prepares) and not x[4]:
                continue
            out.append(x)
        return out

    def node_cost(t: int, x) -> int:
        m, note, number, _offset, _cons = x
        s = slots[t]
        cost = 0
        if not s["first"] and not s["final"] and s["strong"] and perfect(x):
            cost += 2              # imperfect consonances preferred mid-phrase
            if number == 1:
                cost += 2          # unisons/octaves mid-phrase are weaker still
        if s["first"] and note.pitch_class == tonic_pc:
            cost -= 1
        return cost

    # A search node: (candidate, parent node, latest downbeat (cand, bar), latest struck
    # weak beat (cand, bar)) — parent pointers instead of copied paths keep it fast.
    def parallel_across(ref, b, s) -> bool:
        """Same perfect interval on two successive strong (or weak) beats, both voices moving alike."""
        if ref is None or not perfect(ref[0]) or ref[0][2] != b[2]:
            return False
        c_dir = _sign(cf_m[s["bar"]] - cf_m[ref[1]])
        return c_dir != 0 and _sign(b[0] - ref[0][0]) == c_dir

    def trans_cost(t: int, node, b) -> int:
        a, parent, strong_ref, weak_ref = node
        s, sp = slots[t], slots[t - 1]
        bm, bn, _bnum, _boff, b_cons = b
        am, an, _anum, _aoff, a_cons = a
        c = cf_m[s["bar"]]
        cost = 0
        leap = abs(bm - am)
        if s["tie"]:
            # the held note must still sit on its side of the cantus, within range
            if (position == "above" and not c < bm <= c + width) or \
               (position == "below" and not c - width <= bm < c):
                cost += _BIG
        else:
            if not melodic_ok(a, b):
                cost += _BIG       # tritone, seventh, augmented/diminished step or > octave
            elif leap > 7:
                cost += 4
            elif leap > 2:
                cost += 1
            elif leap == 0:
                cost += 1
            # dissonance treatment: passing/neighbour notes are approached and left by
            # step (a repeated note is not a step), and never touch another dissonance
            if not b_cons and (leap == 0 or leap > 2):
                cost += _BIG
            if not a_cons and not sp["tie"] and (leap == 0 or leap > 2):
                cost += _BIG
            if not a_cons and not b_cons:
                cost += _BIG
            if species == 2 and not a_cons and not sp["tie"] and parent is not None:
                if _sign(am - parent[0][0]) != _sign(bm - am):
                    cost += _BIG   # second species allows passing tones only, no neighbours
            if leap == 0 and s["bar"] == sp["bar"] and s["bar"] != n_bars - 2:
                cost += _BIG if species == 2 else 3  # don't restrike a note inside a bar
        # a suspension (a tied dissonance) must resolve DOWN by step to a consonance
        if sp["tie"] and not a_cons:
            if not (b_cons and 1 <= am - bm <= 2):
                cost += _BIG
            elif position == "below" and b[2] == 1:
                cost += _BIG       # 7-8 in the lower voice resolves into an octave: forbidden
            else:
                cost -= 2          # reward a clean suspension resolution
        # parallel / direct perfect fifths and octaves
        c_dir = _sign(c - cf_m[sp["bar"]])
        p_dir = _sign(bm - am)
        if perfect(b) and c_dir != 0 and c_dir == p_dir:
            cost += _BIG
        # ...also between successive downbeats (hidden by the weak beats in between); in
        # species 4 the struck notes are the weak beats (the downbeats are tied syncopations),
        # so there it is successive weak beats (e.g. a chain of 9-8 or 6-5 suspensions)
        if perfect(b):
            if species != 4 and s["strong"] and not s["tie"] and parallel_across(strong_ref, b, s):
                cost += _BIG
            if species == 4 and not s["strong"] and parallel_across(weak_ref, b, s):
                cost += _BIG
        # motion preference
        if c_dir != 0:
            if p_dir == -c_dir:
                cost -= 3
            elif p_dir == c_dir:
                cost += 2
        elif p_dir == 0:
            cost += 1
        if s["final"]:
            if c_dir != 0 and p_dir == c_dir:
                cost += 4          # approach the final by contrary motion
            if leap > 2:
                cost += _BIG       # the cadence reaches the final by step
        return cost

    def extend(t: int, parent, x):
        s = slots[t]
        # a tied downbeat is a syncopation, not a struck downbeat: it starts no new reference
        if s["tie"]:
            strong_ref = None
        else:
            strong_ref = (x, s["bar"]) if s["strong"] else (parent[2] if parent else None)
        weak_ref = (x, s["bar"]) if not s["strong"] and not s["tie"] else (parent[3] if parent else None)
        return (x, parent, strong_ref, weak_ref)

    def key_of(node) -> tuple:
        """Everything later transitions read from the history — keying the Viterbi on it
        keeps the search exact, and keying on nothing more keeps it small."""
        x, parent, strong_ref, weak_ref = node
        return (
            x[0],
            parent[0][0] if species == 2 and parent is not None and not x[4] else None,
            strong_ref[0][0] if species != 4 and strong_ref else None,
            weak_ref[0][0] if species == 4 and weak_ref else None,
        )

    def pitches(node) -> list[int]:
        out = []
        while node is not None:
            out.append(node[0][0])
            node = node[1]
        return out[::-1]

    cand_lists = [None if s["tie"] else gen(t) for t, s in enumerate(slots)]
    for t, s in enumerate(slots):
        if cand_lists[t] is not None and not cand_lists[t]:
            bar = s["bar"]
            if s["final"]:
                raise ValueError(
                    f"no perfect consonance on the tonic {tonic.name} fits {position} the last cantus"
                    f" note {cf[bar].name}: a cantus firmus should end on the tonic"
                )
            if s["first"]:
                need = "the unison/octave" if position == "below" else "a perfect consonance"
                raise ValueError(
                    f"no diatonic note forms {need} {position} the first cantus note {cf[bar].name}"
                    f" in this key: start the cantus on the tonic (or its fifth)"
                )
            raise ValueError(f"no diatonic note fits {position} the cantus note {cf[bar].name} within MIDI range")

    frontier: dict[tuple, tuple] = {}
    for x in cand_lists[0]:
        node = extend(0, None, x)
        frontier[key_of(node)] = (node_cost(0, x), node)
    for t in range(1, len(slots)):
        nxt: dict[tuple, tuple] = {}
        for cost, node in frontier.values():
            a = node[0]
            exts = [cand(a[0], a[1], slots[t]["bar"])] if slots[t]["tie"] else cand_lists[t]
            for b in exts:
                total = cost + trans_cost(t, node, b) + node_cost(t, b)
                child = extend(t, node, b)
                key = key_of(child)
                old = nxt.get(key)
                # strict improvement only: the first path found keeps a tie, and the
                # exploration order is fixed (sorted candidates), so this is deterministic
                if old is None or total < old[0]:
                    nxt[key] = (total, child)
        frontier = nxt
    cost, node = min(frontier.values(), key=lambda cn: (cn[0], pitches(cn[1])))
    path = []
    while node is not None:
        path.append(node[0])
        node = node[1]
    return cost, path[::-1]


def species_counterpoint(cantus, root: str, scale_type: str = "major",
                         species: int = 1, position: str = "above") -> dict:
    """Counterpoint in species 1-5 (Fux): note-against-note up to florid, deterministic.

    Species 1 = 1:1, 2 = 2:1 (passing tones on weak beats), 3 = 4:1 (passing and
    neighbour figures), 4 = syncopated suspensions (each note is struck as a
    consonance on the weak beat and tied over the barline; if the tie clashes it
    resolves down by step to a consonance — 7-6, 4-3, 9-8 above; 2-3, 4-5, 9-10
    and the diminished-5th-to-6th below; never 7-8 below), 5 = florid (a mix, with a prepared cadential
    suspension). The counterpoint is diatonic to the key and follows the rules:
    consonances on strong beats; every dissonance a stepwise passing/neighbour
    tone between consonances or a prepared, resolved suspension; intervals judged
    by spelling (a diminished 7th in harmonic minor is a dissonance); no melodic
    tritones, sevenths or augmented steps; no parallel/direct fifths or octaves
    (also on successive downbeats, and on successive struck weak beats in
    species 4); a lower voice begins on the unison/octave; the final is the
    tonic, a perfect consonance reached by step. The cantus should begin and end
    on the tonic. A deterministic Viterbi search makes the same cantus always
    yield the same line. Returns the two voices, the per-bar downbeat intervals
    (spelled simple names: 'P1/P8', 'M6', 'm7', 'P4'...), and a `render_hint`
    with ready-to-use notes-track specs (cantus as whole notes; the counterpoint
    with its own rhythm). If no line can satisfy every rule for the given cantus,
    the closest one is returned with `rules_broken` and a `warning`.
    """
    if position not in ("above", "below"):
        raise ValueError(f"position must be 'above' or 'below', got {position!r}")
    if isinstance(species, bool) or species not in (1, 2, 3, 4, 5):
        raise ValueError(f"species must be 1-5, got {species!r}")
    scale = resolve_scale_type(scale_type)
    root_note = parse_notes(root)[0].without_octave()

    from .midi_io import assign_octaves  # local import avoids a cycle

    cf = assign_octaves(parse_notes(cantus), 4, "nearest")
    if len(cf) < 2:
        raise ValueError("cantus firmus needs at least 2 notes")
    spelled = scale_notes(scale, root_note)[:-1]
    table = _pitch_table(spelled)
    # judge an enharmonically written scale tone (D# in Eb major) by the key's spelling (Eb)
    by_pc = {n.pitch_class: n for n in spelled}
    cf = [_same_pitch(by_pc[n.pitch_class], n.midi) if n.pitch_class in by_pc else n for n in cf]
    bar_beats = 4.0
    slots = _slot_plan(len(cf), species, bar_beats)

    # Try the normal voice distance first; widen it only if no rule-abiding line fits.
    best_cost = best = None
    error = None
    for width in _WIDTHS:
        try:
            cost, path = _solve(cf, table, root_note, species, position, slots, width)
        except ValueError as e:
            error = error or e
            continue
        if best is None or round(cost / _BIG) < round(best_cost / _BIG):
            best_cost, best = cost, path
        if cost < _BIG // 2:
            break
    if best is None:
        raise error

    # Collapse tied slots into held durations; emit onset notes + a rhythm string.
    onsets: list[list] = []
    for x, slot in zip(best, slots):
        if slot["tie"] and onsets:
            onsets[-1][1] += slot["dur"]
        else:
            onsets.append([x[1], slot["dur"]])
    grid = min(s["dur"] for s in slots)
    cp_notes = [o[0].name for o in onsets]
    cp_rhythm = "".join("O" + "." * (int(round(o[1] / grid)) - 1) for o in onsets)

    # Per-bar downbeat interval (the note sounding on each bar's strong beat), spelled.
    downbeat = []
    for i in range(len(cf)):
        x = next(x for x, s in zip(best, slots) if s["bar"] == i and s["strong"])
        downbeat.append(_interval_name(x[2], x[3]))

    result = {
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
    broken = round(best_cost / _BIG)
    if broken > 0:
        result["rules_broken"] = broken
        result["warning"] = (
            "No line obeys every rule for this cantus, species and position; this is the"
            " closest one (fewest rules broken). Try the other position or adjust the cantus."
        )
    return result


def first_species(cantus, root: str, scale_type: str = "major",
                  position: str = "above") -> dict:
    """Write a first-species (note-against-note) counterpoint to a cantus firmus.

    The same rule engine as species_counterpoint(species=1): consonances only,
    perfect consonances at both ends with the final on the tonic reached by step,
    contrary/oblique motion preferred, no parallel or direct fifths/octaves, no
    melodic tritones or sevenths. Returns the cantus, the counterpoint line, and
    the spelled interval between the voices at each step — render them as two
    notes tracks sharing one rhythm.
    """
    r = species_counterpoint(cantus, root, scale_type, species=1, position=position)
    out = {
        "position": r["position"],
        "key": r["key"],
        "cantus": r["cantus"],
        "counterpoint": r["counterpoint"],
        "intervals": r["downbeat_intervals"],
    }
    for k in ("rules_broken", "warning"):
        if k in r:
            out[k] = r[k]
    return out
