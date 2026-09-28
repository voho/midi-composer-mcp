"""Arranging-style chord voicings: drop-2/3/2&4, open, shell and Levine rootless.

voice_chords voices a progression in a named arranging style and connects the
chords with voice_leading's own cost, search and tie-breaks. The style rules
are standard jazz-arranging pedagogy (drop-2, drop-3 and drop-2&4, root-3rd-7th
shells, e.g. Mark Levine, The Jazz Piano Book, 1989); the rootless A/B
left-hand voicings are Levine's ("Left-hand voicings"). The menu of styles
follows the voice-grouping profiles of chord tools such as Scaler. Everything
here is deterministic: the connection is greedy (each chord takes the locally
best voicing), exactly like harmony.voice_leading, and style 'close' without
top_notes IS voice_leading, so the two can never disagree.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .chords import ChordType, chord_notes
from .harmony import _root_and_quality, _voicing_cost, voice_leading
from .midi_io import _parse_chord_list, _with_octave, voice_chord
from .notes import Note, parse_notes, spelling_for_pcs, transpose

STYLES = ("close", "drop2", "drop3", "drop24", "open", "shell", "rootless_a", "rootless_b", "rootless")

# style spellings a caller is likely to use, folded (lower case, no spaces/_/-/&)
_STYLE_KEYS = {
    "close": "close", "closeposition": "close",
    "drop2": "drop2", "drop3": "drop3", "drop24": "drop24", "drop2and4": "drop24",
    "open": "open", "shell": "shell", "shells": "shell",
    "rootless": "rootless", "rootlessa": "rootless_a", "rootlessb": "rootless_b",
}

# voices lowered by an octave, numbered from the top (Levine: "drop 2" = 2nd voice from the top)
_DROPS = {"drop2": (2,), "drop3": (3,), "drop24": (2, 4)}
_ROTATING = ("close", "drop2", "drop3", "drop24", "open")   # candidates = rotations x octave shifts
_FIFTHS = ("5", "b5", "#5")


@dataclass(frozen=True)
class _Tone:
    note: Note              # spelled; octave-less except an anchored bottom note
    label: str | None       # chord-member label ('1', 'b3', '5', 'b7', '9', ...); None in a cluster


@dataclass(frozen=True)
class _Chord:
    symbol: str | None
    root: Note | None           # None for a cluster that is no chord type
    ctype: ChordType | None
    tones: tuple[_Tone, ...]    # the chord members in ChordType order (a cluster: as written)
    written: tuple[_Tone, ...]  # the tones as given, in order: what 'close' (voice_leading) stacks
    bass: Note | None           # octave-less slash bass


def _number(label: str) -> int:
    return int(label.lstrip("#b"))


def _style(style) -> str:
    if not isinstance(style, str):
        raise ValueError(f"style must be one of {', '.join(STYLES)}; got {style!r}")
    key = "".join(ch for ch in style.strip().lower() if ch not in " _-&")
    if key not in _STYLE_KEYS:
        raise ValueError(f"Unknown voicing style {style!r}. Use one of: {', '.join(STYLES)}")
    return _STYLE_KEYS[key]


def _read(entry: dict, item) -> _Chord:
    """A chord entry (from midi_io._parse_chord_list) with its members labelled."""
    written = [t.without_octave() for t in entry["tones"]]
    bass = entry["bass"].without_octave() if entry["bass"] is not None else None
    if isinstance(item, str):
        root, ctype, _ = _root_and_quality(item)    # the same parse _parse_chord_list made
        root = root.without_octave()
        tones = tuple(_Tone(n, label) for n, (label, _) in zip(written, ctype.degrees))
        return _Chord(entry["symbol"], root, ctype, tones, tones, bass)
    root, ctype, _ = _root_and_quality(list(item))
    if ctype is None:   # an unnamed cluster: only 'close' can voice it, as written
        tones = tuple(_Tone(n, None) for n in written)
        return _Chord(None, None, None, tones, tones, None)
    root = root.without_octave()
    spelled = spelling_for_pcs(written)     # keep the caller's spelling of every written pitch
    tones = tuple(_Tone(spelled.get(n.pitch_class, n), label)
                  for n, (label, _) in zip(chord_notes(ctype, root), ctype.degrees))
    members = {t.note.pitch_class: t.label for t in tones}
    return _Chord(None, root, ctype, tones, tuple(_Tone(n, members[n.pitch_class]) for n in written), None)


def _name(ch: _Chord) -> str:
    if ch.symbol:
        return ch.symbol
    if ch.ctype is not None:
        return f"{ch.root.pitch_class_name}{ch.ctype.symbol}"
    return " ".join(t.note.name for t in ch.tones)


def _label_of(ch: _Chord, note: Note) -> str | None:
    for t in ch.tones:
        if t.note.pitch_class == note.pitch_class:
            return t.label
    return None


def _need_type(ch: _Chord, style: str) -> None:
    if ch.ctype is None:
        raise ValueError(
            f"{_name(ch)} is not a known chord type, so it has no chord members to voice in"
            f" style {style!r}; only style 'close' voices arbitrary note clusters")


# ------------------------------------------------------------ the styles

def _drop_cycle(ch: _Chord, style: str) -> list[_Tone]:
    """The 4-voice structure a drop voicing is taken from, in close-position order.

    A triad is filled to four voices by doubling the bottom voice on top (R-3-5-R
    in root position). A chord of 5+ tones omits the perfect 5th, then the root,
    until 4 remain (jazz practice). The members are ordered by pitch class
    upward, starting from the first remaining chord member (root, else 3rd).
    """
    _need_type(ch, style)
    tones = list(ch.tones)
    if len(tones) < 3:
        raise ValueError(f"{_name(ch)} has only {len(tones)} tones; {style} voicings need at least 3")
    for drop in ("5", "1"):
        if len(tones) > 4:
            tones = [t for t in tones if t.label != drop]
    if len(tones) > 4:  # no table chord gets here
        raise ValueError(f"{_name(ch)} cannot be reduced to four voices for {style}")
    first = tones[0]
    cycle = sorted(tones, key=lambda t: (t.note.pitch_class - ch.root.pitch_class) % 12)
    start = cycle.index(first)
    return cycle[start:] + cycle[:start]


def _drop(close: list[Note], positions: tuple[int, ...]) -> list[Note]:
    """Lower the given voices (numbered from the top) of a close voicing by an octave."""
    lowered = {len(close) - p for p in positions}
    moved = [_with_octave(n, n.midi - 12) if i in lowered else n for i, n in enumerate(close)]
    return sorted(moved, key=lambda n: n.midi)


def _shell_members(ch: _Chord) -> tuple[_Tone, _Tone]:
    third = next((t for t in ch.tones if _number(t.label) == 3), None)
    if third is None:  # a sus chord's 2nd/4th stands in for the 3rd
        third = next((t for t in ch.tones if _number(t.label) in (2, 4)), None)
    if third is None:
        raise ValueError(f"{_name(ch)} has no 3rd (or sus 2nd/4th) for a shell voicing")
    seventh = (next((t for t in ch.tones if _number(t.label) == 7), None)
               or next((t for t in ch.tones if t.label == "6"), None)
               or next((t for t in ch.tones if t.label in _FIFTHS), None))
    if seventh is None:
        raise ValueError(f"{_name(ch)} has no 7th, 6th or 5th for a shell voicing")
    return third, seventh


def _rootless_forms(ch: _Chord) -> tuple[list[_Tone], list[_Tone]]:
    """Levine's rootless left-hand voicings (A form, B form) for one chord."""
    name = _name(ch)
    by_label = {t.label: t for t in ch.tones}
    third = by_label.get("3") or by_label.get("b3")
    if third is None:
        raise ValueError(f"{name} has no 3rd: rootless (Levine A/B) voicings need a 3rd and a 7th or 6th"
                         f" (use 'shell', 'open' or a drop style for sus chords)")
    if "bb7" in by_label:
        raise ValueError(f"{name}: a diminished 7th chord has no Levine rootless A/B voicing"
                         f" (use a drop style or 'close')")
    seventh = by_label.get("7") or by_label.get("b7")
    sixth = by_label.get("6")
    fifth = next((by_label[f] for f in _FIFTHS if f in by_label), None)
    if seventh is None and sixth is None:
        raise ValueError(f"{name} has no 7th or 6th: rootless (Levine A/B) voicings need one")
    root = ch.root
    ninth = by_label.get("9") or _Tone(transpose(root, 14, 8), "9")
    if third.label == "3" and seventh is not None and seventh.label == "b7":   # dominant
        color = fifth if fifth is not None and fifth.label != "5" else (
            by_label.get("13") or _Tone(transpose(root, 21, 12), "13"))
        tension = by_label.get("b9") or by_label.get("#9") or ninth
        return [third, color, seventh, tension], [seventh, tension, third, color]
    if third.label == "b3" and fifth is not None and fifth.label == "b5":      # half-diminished
        one = _Tone(root, "1")  # the natural 9 lies outside Locrian: the root stays (a documented choice)
        return [third, fifth, seventh, one], [seventh, one, third, fifth]
    if fifth is None:  # no table chord gets here
        raise ValueError(f"{name} has no 5th for a rootless (Levine A/B) voicing")
    top = seventh or sixth                                                     # major / minor
    return [third, fifth, top, ninth], [top, ninth, third, fifth]


def _variants(ch: _Chord, style: str, octave: int, shift: int) -> list[tuple[str, list[_Tone]]]:
    """Every (variant name, tones to stack) of one chord at one octave shift.

    A bottom tone that carries an octave is anchored there; the others stack
    strictly upward (midi_io.voice_chord).
    """
    if style == "close":
        tones = list(ch.written)
        return [(f"rotation {r}", tones[r:] + tones[:r]) for r in range(len(tones))]
    if style in _DROPS:
        cycle = _drop_cycle(ch, style)
        n = len(cycle)
        return [(f"rotation {r}", [cycle[(r + k) % n] for k in range(4)]) for r in range(n)]
    _need_type(ch, style)
    if style in ("open", "shell"):
        bass = ch.bass or ch.root
        low = _Tone(Note(bass.letter, bass.accidental, octave - 1 + shift), _label_of(ch, bass) or "bass")
        if style == "shell":
            third, seventh = _shell_members(ch)
            return [("A", [low, third, seventh]), ("B", [low, seventh, third])]
        fifth = next((t for t in ch.tones if t.label in _FIFTHS and t.note.pitch_class != bass.pitch_class),
                     None)   # a 5th already in the bass ('C/G') is not placed twice
        upper = [t for t in ch.tones if t is not fifth and t.label not in _FIFTHS
                 and t.note.pitch_class != bass.pitch_class]
        head = [low] + ([fifth] if fifth is not None else [])
        return [(f"rotation {r}", head + upper[r:] + upper[:r]) for r in range(max(1, len(upper)))]
    # rootless: the bottom note is the lowest of its pitch class at or above D(octave-1)
    a, b = _rootless_forms(ch)
    forms = {"rootless_a": [("A", a)], "rootless_b": [("B", b)], "rootless": [("A", a), ("B", b)]}[style]
    floor = Note("D", 0, octave - 1 + shift).midi
    out = []
    for name, tones in forms:
        bottom = tones[0]
        midi = floor + (bottom.note.pitch_class - floor) % 12
        out.append((name, [_Tone(_with_octave(bottom.note, midi), bottom.label), *tones[1:]]))
    return out


def _candidates(ch: _Chord, style: str, octave: int, shifts: tuple[int, ...]) -> list[dict]:
    """All voicings to choose from, in voice_leading's search order (shift, rotation, variant)."""
    members = {t.note.pitch_class: t.label for t in ch.tones}
    cands = []
    for shift in shifts:
        for index, (variant, tones) in enumerate(_variants(ch, style, octave, shift)):
            notes = [t.note for t in tones]
            try:  # a register outside MIDI 0-127 is no candidate
                if style in _DROPS:
                    close = voice_chord(notes, octave + shift)
                    voiced = voice_chord(_drop(close, _DROPS[style]), octave, ch.bass)
                elif style == "close":
                    voiced = voice_chord(notes, octave + shift, ch.bass)
                elif style in ("open", "shell"):   # the bass is already their anchored bottom tone
                    voiced = voice_chord(notes, octave)
                else:                              # rootless: a slash bass goes strictly below
                    voiced = voice_chord(notes, octave, ch.bass)
            except ValueError:
                continue
            labels = {**members, **{t.note.pitch_class: t.label for t in tones}}
            rot, var = (index, 0) if style in _ROTATING else (0, index)
            cands.append({"notes": voiced, "labels": [labels.get(n.pitch_class, "bass") for n in voiced],
                          "shift": shift, "rot": rot, "var": var, "variant": variant})
    return cands


# ------------------------------------------------------------ the search

def _parse_top_notes(top_notes, count: int) -> list[Note] | None:
    if top_notes is None:
        return None
    tops = parse_notes(top_notes)
    if len(tops) != count:
        raise ValueError(f"top_notes needs exactly one note per chord: {count} chords, {len(tops)} top notes")
    return tops


def _choose(ch: _Chord, style: str, octave: int, prev: list[int] | None, connect: bool,
            top: Note | None) -> dict:
    exact = top is not None and top.octave is not None
    if style in _ROTATING:
        shifts = (-2, -1, 0, 1, 2) if exact else (-1, 0, 1)
    else:
        shifts = (-1, 0, 1) if exact else (0,)
    cands = _candidates(ch, style, octave, shifts)
    name = _name(ch)
    if not cands:
        raise ValueError(f"cannot voice {name} in style {style!r} at octave {octave} within MIDI 0-127")
    if top is not None:
        tones = {n.pitch_class for c in cands for n in c["notes"]}
        if top.pitch_class not in tones:
            members = []
            for c in cands:
                for n in c["notes"]:
                    if n.pitch_class_name not in members:
                        members.append(n.pitch_class_name)
            raise ValueError(f"top note {top.name} is not a tone of the {style} voicing of {name}"
                             f" ({' '.join(members)})")
        if exact:
            cands = [c for c in cands if c["notes"][-1].midi == top.midi]
        else:
            cands = [c for c in cands if c["notes"][-1].pitch_class == top.pitch_class]
        if not cands:
            raise ValueError(f"no {style} voicing of {name} around octave {octave} has {top.name} as its"
                             f" highest note")
    best = best_key = None
    for c in cands:
        midis = [n.midi for n in c["notes"]]
        if prev is not None and connect:
            held = len(set(prev) & set(midis))
            # voice_leading's tie-breaks: more common tones held, no octave shift, lower rotation
            key = (_voicing_cost(prev, midis), -held, abs(c["shift"]), c["rot"], c["var"])
        else:
            key = (abs(c["shift"]), c["rot"], c["var"])
        if best_key is None or key < best_key:
            best, best_key = c, key
    return best


def _greedy(read: list[_Chord], style: str, octave: int, connect: bool, tops: list[Note] | None) -> list[dict]:
    """voice_leading's greedy chain over any style's candidates (optionally filtered by top notes)."""
    voicings = []
    prev: list[int] | None = None
    for i, ch in enumerate(read):
        best = _choose(ch, style, octave, prev, connect, tops[i] if tops else None)
        voiced = best["notes"]
        voicings.append({"symbol": _name(ch) if ch.ctype else " ".join(n.pitch_class_name for n in voiced),
                         "notes": [n.name for n in voiced], "midi": [n.midi for n in voiced],
                         "variant": best["variant"], "degrees": best["labels"] if ch.ctype else None})
        prev = [n.midi for n in voiced]
    return voicings


def _rotation_of(ch: _Chord, voiced: list[Note]) -> int:
    upper = voiced[1:] if ch.bass is not None else voiced
    first = upper[0].without_octave()
    return next((i for i, t in enumerate(ch.written) if t.note == first), 0)


def voice_chords(chords, style: str = "drop2", octave: int = 4, top_notes=None, connect: bool = True) -> dict:
    """Voice a progression in a named arranging style and connect the chords smoothly.

    Chords are symbols ('Dm7', 'G7/B') or note arrays (read as their chord by
    harmony._root_and_quality; order and octaves ignored, as in voice_leading).
    Styles (octave = the register, 4 = around middle C):
    - 'close': exactly voice_leading(chords, octave) (all tones stacked in chord
      order, nearest inversion). The only style that voices unnamed clusters.
    - 'drop2', 'drop3', 'drop24': four voices in close position, then the 2nd,
      the 3rd, or the 2nd and 4th voice from the top dropped an octave. A
      triad doubles its bottom voice on top (R-3-5-R in root position); a
      chord of 5+ tones omits the perfect 5th, then the root (C13 -> 3 13 b7 9).
      Cmaj7 -> drop2 G3 C4 E4 B4, drop3 E3 C4 G4 B4, drop24 C3 G3 E4 B4.
    - 'open': the bass (slash bass or root) in octave-1, the 5th nearest above
      it (unless the bass is the 5th), then the other members (3rd, 7th,
      extensions, in chord order) stacked upward: Cmaj7 -> C3 G3 E4 B4.
    - 'shell': the bass in octave-1 plus the 3rd and 7th (a 6-chord's 6th, a
      triad's 5th; a sus 2nd/4th stands in for the 3rd): variant A = R-3-7,
      B = R-7-3: Cmaj7 -> C3 E3 B3 / C3 B3 E4.
    - 'rootless_a', 'rootless_b', 'rootless' (A or B, whichever connects
      better): Mark Levine's left-hand rootless voicings (The Jazz Piano Book).
      Major and minor: A = 3-5-7-9, B = 7-9-3-5 (a 6 replaces the 7 in
      6-chords); dominant: A = 3-13-b7-9, B = b7-9-3-13 (the chord's own b9/#9
      replaces the 9, an altered 5th replaces the 13); m7b5: A = b3-b5-b7-1,
      B = b7-1-b3-b5 (the root stays: the natural 9 is outside Locrian). The
      bottom note is the lowest of its pitch class at or above D(octave-1), so
      Levine's register rule picks A or B by key. Triads, sus and dim7 chords
      raise ValueError.
    A slash bass ('G7/B') always sits strictly below the voicing.
    connect=True chains GREEDILY (each chord locally best, like voice_leading,
    with its cost and tie-breaks): each chord takes the voicing (rotation x
    octave shift -1/0/+1; shell/rootless: variant A/B in their fixed register)
    that moves least from the previous one, then keeps more common tones, then
    no shift, then the lower rotation/variant; the first chord takes rotation
    0 / variant A. connect=False takes rotation 0 / variant A for every chord.
    top_notes (one per chord, e.g. a melody)
    keeps only voicings whose highest note is that note: an exact pitch with an
    octave ('E5'; octave shifts then widen to -2..+2, shell/rootless try
    -1/0/+1), else the pitch class; a top note that is not a tone of the
    voicing, or cannot be reached, raises ValueError naming the chord.
    Returns {style, voicings: [{symbol, notes, midi, variant, degrees}],
    chords (voiced note arrays), total_movement (sum of voice movement between
    neighbours), render_hint (a chords track for arrange_to_midi)}; `degrees`
    names each note's chord member, bottom to top. Feed `chords` to
    chords_to_midi, or check them with check_voice_leading(voicings=...).
    Example: voice_chords(['Dm7','G7','Cmaj7'], 'rootless') -> F3 A3 C4 E4 (A),
    F3 A3 B3 E4 (B), E3 G3 B3 D4 (A) — Levine's textbook ii-V-I. Deterministic.
    """
    canonical = _style(style)
    parsed = _parse_chord_list(chords)
    if not isinstance(octave, int) or isinstance(octave, bool) or not 0 <= octave <= 9:
        raise ValueError(f"octave must be an integer between 0 and 9, got {octave!r}")
    if not isinstance(connect, bool):
        raise ValueError(f"connect must be true or false, got {connect!r}")
    items = chords
    if isinstance(items, str):  # split exactly as _parse_chord_list does
        items = [t for t in re.split(r"[,\s]+", items.strip()) if t]
    read = [_read(e, item) for e, item in zip(parsed, items)]
    tops = _parse_top_notes(top_notes, len(read))

    voicings = []
    if canonical == "close" and tops is None and connect:
        # delegated: style 'close' can never disagree with voice_leading
        for ch, v in zip(read, voice_leading(chords, octave=octave)["voicings"]):
            voiced = parse_notes(v["notes"])
            members = {t.note.pitch_class: t.label for t in ch.tones}
            voicings.append({"symbol": _name(ch) if ch.ctype else v["symbol"], "notes": v["notes"],
                             "midi": v["midi"], "variant": f"rotation {_rotation_of(ch, voiced)}",
                             "degrees": [members.get(n.pitch_class, "bass") for n in voiced] if ch.ctype else None})
    else:
        voicings = _greedy(read, canonical, octave, connect, tops)

    notes = [v["notes"] for v in voicings]
    total = sum(_voicing_cost(a["midi"], b["midi"]) for a, b in zip(voicings, voicings[1:]))
    return {
        "style": canonical,
        "voicings": voicings,
        "chords": notes,
        "total_movement": total,
        "render_hint": {"tracks": [{"type": "chords", "name": "keys", "chords": notes, "beats_per_chord": 4}]},
    }
