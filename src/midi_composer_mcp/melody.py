"""Melody-layer generators: deterministic, mechanical building blocks.

These tools turn a scale or chord (the harmonic layer) plus a contour or a
seed into a concrete note sequence. They contain no musical taste — they
resolve degrees, reorder/stack notes, or take a seeded random walk. The
caller decides the shape; these just realize it. Every output is a note list
that feeds straight into a notes track, the rhythm tools, or the renderers.
"""

from __future__ import annotations

import re

from .forms import resolve_form
from .generate import _make_rng, parse_rhythm
from .notes import LETTER_PCS, LETTERS, Note, note_from_midi, parse_note, parse_notes, transpose, transpose_all
from .scales import _label_digit, degree_labels, resolve_scale_type


def _parse_degree_tokens(degrees) -> list[int]:
    if isinstance(degrees, (int,)) and not isinstance(degrees, bool):
        degrees = [degrees]
    elif isinstance(degrees, str):
        degrees = [t for t in re.split(r"[,\s]+", degrees.strip()) if t]
    if not isinstance(degrees, (list, tuple)) or not degrees:
        raise ValueError("degrees must be a non-empty list like [1,3,5,8] or '1 3 5 8'")
    out: list[int] = []
    for tok in degrees:
        if isinstance(tok, bool):
            raise ValueError(f"Invalid scale degree: {tok!r}")
        if isinstance(tok, int):
            d = tok
        elif isinstance(tok, str) and re.fullmatch(r"-?\d+", tok.strip()):
            d = int(tok.strip())
        else:
            raise ValueError(f"Invalid scale degree: {tok!r} (use integers; 1 = root, 8 = octave up, -7 = octave down)")
        if d == 0:
            raise ValueError("Scale degrees are 1-based; 0 is not a degree (1 = root, 8 = octave up)")
        out.append(d)
    return out


# Scale degrees live on a number line with no 0 (...-2, -1, 1, 2, 3...). These
# convert to a contiguous index so transpose/invert never produce degree 0.
def _deg_to_idx(d: int) -> int:
    return d - 1 if d > 0 else d


def _idx_to_deg(i: int) -> int:
    return i + 1 if i >= 0 else i


def notes_from_degrees(root: str, scale_type: str, degrees) -> dict:
    """Resolve a melody written as scale degrees into concrete notes.

    Degree 1 is the root; degrees beyond the scale length wrap into higher
    octaves (8 = root up an octave, 9 = the 2nd up an octave). Negative
    degrees count down from the root on the same no-zero number line that
    motif_grammar uses: -1 = one scale step below the root, -2 = two steps
    below, -7 = the root an octave down (in a 7-note scale). e.g. in C major,
    [1,2,3,5,8] -> C D E G C and from 'C4', [1,-1,-3] -> C4 B3 G3.
    Octave-aware if the root carries one. Deterministic.
    """
    scale = resolve_scale_type(scale_type)
    root_note = parse_notes(root)[0]
    labels = degree_labels(scale)
    n = len(scale.intervals)
    tokens = _parse_degree_tokens(degrees)

    out: list[Note] = []
    for d in tokens:
        octs, pos = divmod(_deg_to_idx(d), n)
        semitones = scale.intervals[pos] + 12 * octs
        letter_steps = (_label_digit(labels[pos]) - 1) + 7 * octs
        out.append(transpose(root_note, semitones, letter_steps))
    return {
        "root": root_note.name,
        "scale_type": scale.name,
        "degrees": tokens,
        "notes": [note.name for note in out],
    }


_ARP_STYLES = ("up", "down", "updown", "downup", "converge", "diverge", "random")


def arpeggiate_notes(notes, style: str = "up", octaves: int = 1,
                     seed: int | None = None) -> dict:
    """Reorder a set of notes (a chord or scale) into an arpeggio sequence.

    Pure reordering — no new pitches except optional octave copies. `style`:
    up, down, updown, downup (no repeated turn note), converge (outside-in),
    diverge (inside-out, the mirror of converge), or random (seeded). `octaves`
    stacks that many octave copies first (requires the notes to carry octaves,
    e.g. a chord from a root like 'C4'); a pitch already present is not doubled,
    so a scale C4..C5 over 2 octaves runs C4..C6 with one C5. Feed a chord's
    `notes` here to build an arp or a broken-chord bass line.
    """
    if style not in _ARP_STYLES:
        raise ValueError(f"style must be one of {_ARP_STYLES}, got {style!r}")
    if not isinstance(octaves, int) or isinstance(octaves, bool) or not 1 <= octaves <= 8:
        raise ValueError(f"octaves must be an integer between 1 and 8, got {octaves!r}")
    parsed = parse_notes(notes)
    if octaves > 1:
        if any(n.octave is None for n in parsed):
            raise ValueError(
                "octaves > 1 needs notes with octaves (e.g. a chord from a root like 'C4')"
            )
        expanded: list[Note] = []
        seen: set[int] = set()
        for k in range(octaves):
            for n in parsed:
                copy = Note(n.letter, n.accidental, n.octave + k)
                if not 0 <= copy.midi <= 127:
                    raise ValueError(f"{octaves} octaves of {n.name} leave the MIDI range 0-127 ({copy.name})")
                if copy.midi not in seen:  # a scale's top root is the next copy's bottom root
                    seen.add(copy.midi)
                    expanded.append(copy)
        parsed = expanded
    names = [n.name for n in parsed]

    if style == "up":
        seq = names
    elif style == "down":
        seq = names[::-1]
    elif style == "updown":
        seq = names + names[-2:0:-1]  # looping it never repeats a turn note
    elif style == "downup":
        seq = names[::-1] + names[1:-1]
    elif style in ("converge", "diverge"):
        seq = []
        lo, hi = 0, len(names) - 1
        while lo <= hi:
            seq.append(names[lo])
            if lo != hi:
                seq.append(names[hi])
            lo, hi = lo + 1, hi - 1
        if style == "diverge":
            seq = seq[::-1]
    else:  # random
        rng, used_seed = _make_rng(seed)
        seq = names[:]
        rng.shuffle(seq)
    result = {"notes": seq, "style": style, "octaves": octaves}
    if style == "random":
        result["seed"] = used_seed
    return result


def melodic_walk(notes, length: int = 8, seed: int | None = None,
                 max_step: int = 2, start: int = 0) -> dict:
    """Generate a stepwise melodic line by a seeded random walk over a note ladder.

    `notes` is an ordered pitch ladder — typically a scale (over one or more
    octaves; pass a wider pool for more range). Each step moves up or down by
    up to `max_step` rungs (so motion is mostly conjunct, like a singable
    melody), staying within the ladder. Deterministic given `seed`, which is
    always returned. Combine with a rhythm pattern in a notes track.
    """
    ladder = parse_notes(notes)
    names = [n.name for n in ladder]
    if not isinstance(length, int) or isinstance(length, bool) or not 1 <= length <= 1000:
        raise ValueError(f"length must be an integer between 1 and 1000, got {length!r}")
    if not isinstance(max_step, int) or isinstance(max_step, bool) or not 1 <= max_step <= 12:
        raise ValueError(f"max_step must be an integer between 1 and 12, got {max_step!r}")
    if not isinstance(start, int) or isinstance(start, bool) or not 0 <= start < len(names):
        raise ValueError(f"start must be an index between 0 and {len(names) - 1}, got {start!r}")
    rng, used_seed = _make_rng(seed)

    idx = start
    out = [names[idx]]
    for _ in range(length - 1):
        step = rng.randint(-max_step, max_step)
        idx = max(0, min(len(names) - 1, idx + step))  # clamp at the ladder ends
        out.append(names[idx])
    return {"notes": out, "length": length, "max_step": max_step, "seed": used_seed}


_GRAMMAR_KINDS = ("notes", "degrees", "rhythm")


def _grammar_literal(spec, kind: str):
    if kind == "rhythm":
        if not isinstance(spec, str):
            raise ValueError("a rhythm motif must be a pattern string like 'O.o.'")
        return parse_rhythm(spec)
    if kind == "degrees":
        return _parse_degree_tokens(spec)
    return [n.name for n in parse_notes(spec)]  # notes


def _grammar_transform(seq, spec: dict, kind: str):
    """Apply variation transforms in a fixed order: retrograde, invert, rotate, transpose."""
    seq = list(seq)
    if spec.get("retrograde"):
        seq = seq[::-1]
    if spec.get("invert"):
        if kind == "notes":
            parsed = [parse_note(n) for n in seq]
            if any(p.octave is None for p in parsed):
                raise ValueError("invert on notes needs octaves (e.g. 'C5'), so it can mirror pitches")
            pivot = parsed[0]
            pivot_pos = LETTERS.index(pivot.letter) + 7 * pivot.octave
            seq = []
            for p in parsed:
                m = 2 * pivot.midi - p.midi
                if not 0 <= m <= 127:
                    raise ValueError(f"inverting {p.name} around {pivot.name} leaves the MIDI range 0-127")
                # mirror the letter too, so the spelled intervals mirror (F G A Bb -> F Eb Db C)
                pos = 2 * pivot_pos - (LETTERS.index(p.letter) + 7 * p.octave)
                octave, letter_idx = divmod(pos, 7)
                letter = LETTERS[letter_idx]
                accidental = m - (LETTER_PCS[letter] + (octave + 1) * 12)
                seq.append((Note(letter, accidental, octave) if abs(accidental) <= 2 else note_from_midi(m)).name)
        elif kind == "degrees":
            pivot = _deg_to_idx(seq[0])
            seq = [_idx_to_deg(2 * pivot - _deg_to_idx(d)) for d in seq]
        else:
            raise ValueError("invert applies to notes or degrees, not rhythm")
    if "rotate" in spec:
        r = spec["rotate"]
        if not isinstance(r, int) or isinstance(r, bool):
            raise ValueError("rotate must be an integer")
        if seq:
            k = r % len(seq)
            seq = seq[k:] + seq[:k]
    if "transpose" in spec:
        t = spec["transpose"]
        if not isinstance(t, int) or isinstance(t, bool):
            raise ValueError("transpose must be an integer (semitones for notes, scale steps for degrees)")
        if kind == "notes":
            seq = [n.name for n in transpose_all([parse_note(n) for n in seq], t)]
        elif kind == "degrees":
            seq = [_idx_to_deg(_deg_to_idx(d) + t) for d in seq]
        else:
            raise ValueError("transpose applies to notes or degrees, not rhythm")
    return "".join(seq) if kind == "rhythm" else seq


def motif_grammar(form, motifs, kind: str = "notes") -> dict:
    """Expand a motif grammar like 'ABAC' into a sequence by substituting labeled motifs.

    Each letter in `form` names a motif in `motifs`. A motif is either a literal
    sequence or a *variation* of another motif. This builds melodies (or
    rhythms) from repetition and variation — AA repeats, AB contrasts, ABA is a
    rounded phrase, ABAC develops. Deterministic.

    `kind` selects what the motifs are:
      - "notes":   note lists/strings (e.g. ["C5","E5","G5"] or "C5 E5 G5") -> a note list
      - "degrees": scale-degree lists (e.g. [1,2,3]) -> a degree list (feed notes_from_degrees)
      - "rhythm":  patterns (e.g. "O.o.") -> one concatenated rhythm string

    A variation references another label and applies transforms (fixed order:
    retrograde, invert, rotate, transpose):
      {"vary": "A", "transpose": 2}   # A up 2 (semitones for notes, scale steps for degrees)
      {"vary": "A", "retrograde": true}
      {"vary": "A", "invert": true}   # mirror pitches/degrees around the first
      {"vary": "A", "rotate": 1}      # rotate the sequence (handy for rhythm)

    Example (a rounded ABAC melody): form="ABAC",
    motifs={"A":"C5 D5 E5 G5", "B":{"vary":"A","transpose":2}, "C":{"vary":"A","retrograde":true}}.
    """
    if kind not in _GRAMMAR_KINDS:
        raise ValueError(f"kind must be one of {_GRAMMAR_KINDS}, got {kind!r}")
    if not isinstance(motifs, dict) or not motifs:
        raise ValueError("motifs must be a non-empty mapping of label to a sequence or variation")
    labels = resolve_form(form, known=motifs)

    resolved: dict[str, object] = {}
    resolving: set[str] = set()

    def resolve(label: str):
        if label in resolved:
            return resolved[label]
        if label in resolving:
            raise ValueError(f"motif {label!r} ultimately varies itself (circular reference)")
        if label not in motifs:
            raise ValueError(f"form uses motif {label!r} not defined in motifs: {', '.join(motifs)}")
        resolving.add(label)
        spec = motifs[label]
        if isinstance(spec, dict):
            base = spec.get("vary", spec.get("from"))
            if not isinstance(base, str):
                raise ValueError(f"motif {label!r} variation needs 'vary': '<other label>' (a string)")
            seq = _grammar_transform(resolve(base), spec, kind)
        else:
            seq = _grammar_literal(spec, kind)
        resolving.discard(label)
        resolved[label] = seq
        return seq

    motif_out = {label: resolve(label) for label in dict.fromkeys(labels)}

    if kind == "rhythm":
        pattern = "".join(resolved[label] for label in labels)
        return {
            "pattern": pattern,
            "form": labels,
            "motifs": motif_out,
            "legend": {"O": "strong beat", "o": "weak beat", ".": "pause"},
        }
    out: list = []
    for label in labels:
        out.extend(resolved[label])
    key = "degrees" if kind == "degrees" else "notes"
    return {key: out, "form": labels, "motifs": motif_out}


def transpose_notes(notes, semitones: int) -> dict:
    """Transpose a list of notes by a number of semitones (octaves are kept).

    Deterministic — useful for key changes, moving a motif, or building a
    sequence by repeating a phrase at a new pitch level. e.g. transposing
    ['C4','E4','G4'] by 5 -> F4 A4 C5. The whole list moves by one spelled
    interval, so it stays in one key: F A C up 1 -> Gb Bb Db (not Gb A# C#).
    """
    if not isinstance(semitones, int) or isinstance(semitones, bool) or not -48 <= semitones <= 48:
        raise ValueError(f"semitones must be an integer between -48 and 48, got {semitones!r}")
    parsed = parse_notes(notes)
    return {
        "semitones": semitones,
        "notes": [n.name for n in transpose_all(parsed, semitones)],
    }


def _pitch_table(spelled: list[Note]) -> list[tuple[int, Note]]:
    """All MIDI pitches (0-127) of a set of pitch classes, with the given spelling."""
    by_pc: dict[int, Note] = {}
    for note in spelled:
        by_pc.setdefault(note.pitch_class, note)
    return [(m, _same_pitch(by_pc[m % 12], m)) for m in range(128) if m % 12 in by_pc]


def _same_pitch(spelling: Note, midi: int) -> Note:
    """`spelling` with the octave that sounds `midi` (B4 respelled as Cb is Cb5, not Cb4)."""
    return Note(spelling.letter, spelling.accidental,
                (midi - LETTER_PCS[spelling.letter] - spelling.accidental) // 12 - 1)


def snap_to_scale(notes, root: str, scale_type: str) -> dict:
    """Snap every note to the nearest note of a scale — guarantees the line fits the key.

    Any melody (hand-written, transposed, or generated) becomes diatonic to
    `root`/`scale_type`, so it stays compatible with chords drawn from that
    scale. Octaves are preserved; ties snap downward. Notes already in the scale
    are unchanged. Deterministic.
    """
    from .scales import resolve_scale_type, scale_notes  # local import avoids cycle at top

    scale = resolve_scale_type(scale_type)
    root_note = parse_notes(root)[0].without_octave()
    spelled = scale_notes(scale, root_note)[:-1]  # canonical octave-less spellings
    pc_to_note = {n.pitch_class: n for n in spelled}
    scale_pcs = sorted(pc_to_note)
    table = _pitch_table(spelled)

    out: list[Note] = []
    changed = 0
    for n in parse_notes(notes):
        if n.pitch_class in pc_to_note:  # already diatonic — respell canonically, same pitch
            canon = pc_to_note[n.pitch_class]
            out.append(canon if n.octave is None else _same_pitch(canon, n.midi))
            continue
        changed += 1
        if n.octave is None:  # nearest scale pitch class (tie -> the one below, even across B/C)
            best = min(scale_pcs, key=lambda pc: (min((n.pitch_class - pc) % 12, (pc - n.pitch_class) % 12),
                                                  (n.pitch_class - pc) % 12 > 6))
            out.append(pc_to_note[best])
        else:  # nearest scale pitch by absolute distance (tie -> lower)
            out.append(min(table, key=lambda t: (abs(t[0] - n.midi), t[0]))[1])
    return {
        "root": root_note.name,
        "scale_type": scale.name,
        "notes": [n.name for n in out],
        "changed": changed,
    }


def melodic_sequence(notes, root: str, scale_type: str, step: int = -1,
                     count: int = 3) -> dict:
    """Repeat a motif as a diatonic sequence — each copy shifted by scale steps.

    A melodic sequence restates a motif at a new pitch level: e.g. step=-1,
    count=4 walks the motif down one scale degree each time (the descending
    sequences of Baroque and pop). The shift stays in `root`/`scale_type`, so a
    diatonic motif stays diatonic and chord-compatible. The first copy is the
    motif itself; a chromatic motif note keeps its offset from the scale note
    below it in every copy (a C# passing tone in C major becomes B# one step
    down). A copy that would leave the MIDI range raises. Deterministic.
    """
    from .scales import resolve_scale_type, scale_notes  # local import avoids cycle

    if not isinstance(step, int) or isinstance(step, bool) or step == 0:
        raise ValueError("step must be a non-zero integer number of scale degrees")
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 64:
        raise ValueError(f"count must be an integer between 1 and 64, got {count!r}")
    scale = resolve_scale_type(scale_type)
    root_note = parse_notes(root)[0].without_octave()
    ladder = _pitch_table(scale_notes(scale, root_note)[:-1])
    ladder_midis = [m for m, _ in ladder]

    seq_notes = assign_octaves_for_sequence(parse_notes(notes))
    anchors = []  # (ladder index of the scale note at or below, semitones and letters above it)
    for n in seq_notes:
        idx = max(i for i, m in enumerate(ladder_midis) if m <= n.midi) if n.midi >= ladder_midis[0] else 0
        letters = (LETTERS.index(n.letter) - LETTERS.index(ladder[idx][1].letter)) % 7
        anchors.append((idx, n.midi - ladder_midis[idx], letters))

    out: list[str] = [n.name for n in seq_notes]
    for k in range(1, count):
        for idx, offset, letters in anchors:
            j = idx + k * step
            if not 0 <= j < len(ladder) or ladder_midis[j] + offset > 127:
                raise ValueError(f"copy {k + 1} of the sequence leaves the MIDI range 0-127; lower count or step")
            base = ladder[j][1]
            out.append((base if offset == 0 else transpose(base, offset, letters)).name)
    return {
        "root": root_note.name,
        "scale_type": scale.name,
        "step": step,
        "count": count,
        "notes": out,
    }


def assign_octaves_for_sequence(parsed):
    from .midi_io import assign_octaves  # local import avoids cycle
    return assign_octaves(parsed, 4, "nearest")


_TINT_POSITIONS = ("superior", "inferior", "alternating")


def _parse_triad(triad) -> list[Note]:
    """A triad given as a chord symbol ('D', 'Am', ['F#m']) or as notes ('A C E', 'A,C,E')."""
    tokens = None
    if isinstance(triad, str):
        tokens = [t for t in re.split(r"[,\s]+", triad.strip()) if t]
    elif isinstance(triad, (list, tuple)) and all(isinstance(t, str) for t in triad):
        tokens = [t for item in triad for t in re.split(r"[,\s]+", item.strip()) if t]
    if tokens is not None and len(tokens) == 1:  # one token = a chord symbol, even "D"
        from .chords import chord_notes, parse_chord_symbol  # local import avoids cycle
        root, chord, _ = parse_chord_symbol(tokens[0])
        return chord_notes(chord, root)
    parsed = parse_notes(triad)
    if len(parsed) < 2:
        raise ValueError("triad must have at least 2 notes (e.g. ['A','C','E'] or 'Am')")
    return parsed


def tintinnabuli_voice(melody, triad, position: str = "superior", rank: int = 1,
                       octave: int = 5) -> dict:
    """Arvo Pärt's tintinnabuli: shadow a melody with the nearest triad notes (deterministic).

    For each note of the melodic voice (M-voice), the tintinnabuli voice
    (T-voice) sounds a note of a fixed `triad` (classically the tonic triad) —
    the nearest triad pitch strictly `superior` (above), `inferior` (below), or
    `alternating` per note. `rank` 1 takes the nearest, 2 the second-nearest
    (Pärt's T1 / T2). The two voices share the same rhythm, so render them as
    two notes tracks. The T-voice is always consonant with the triad, which is
    exactly why the result is compatible by construction. e.g. an A-minor M-voice
    with triad 'Am' yields a shimmering bell-like counter-voice.
    """
    if position not in _TINT_POSITIONS:
        raise ValueError(f"position must be one of {_TINT_POSITIONS}, got {position!r}")
    if not isinstance(rank, int) or isinstance(rank, bool) or not 1 <= rank <= 3:
        raise ValueError(f"rank must be 1, 2 or 3, got {rank!r}")
    if not isinstance(octave, int) or isinstance(octave, bool) or not -1 <= octave <= 9:
        raise ValueError(f"octave must be an integer between -1 and 9, got {octave!r}")

    from .midi_io import assign_octaves  # local import avoids cycle

    m_voice = assign_octaves(parse_notes(melody), octave, "nearest")
    table = _pitch_table(_parse_triad(triad))

    t_voice: list[Note] = []
    for i, mn in enumerate(m_voice):
        m = mn.midi
        pos = position
        if position == "alternating":
            pos = "superior" if i % 2 == 0 else "inferior"
        if pos == "superior":
            cands = [n for (mm, n) in table if mm > m]
        else:
            cands = [n for (mm, n) in reversed(table) if mm < m]
        if len(cands) < rank:
            raise ValueError(
                f"no rank-{rank} triad note {pos} of {mn.name}; move the melody octave,"
                f" lower the rank or change position"
            )
        t_voice.append(cands[rank - 1])

    return {
        "position": position,
        "rank": rank,
        "triad": [n.name for n in _parse_triad(triad)],
        "m_voice": [n.name for n in m_voice],
        "t_voice": [n.name for n in t_voice],
    }
