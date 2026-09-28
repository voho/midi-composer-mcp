"""Historical melody rules: Gregorian modes, Guido's solmization and vowel method, cantus-firmus rules.

Rules for making a melody were written down long before harmony was: the eight
church modes of Gregorian chant (each with a final, a reciting tone and a range),
Guido of Arezzo's hexachords (ut re mi fa sol la, c. 1030) and his method of
deriving a melody from the vowels of a text (Micrologus, ch. 17 — arguably the
oldest written composition algorithm), and the melodic rules for a cantus firmus
as modern species-counterpoint teaching codifies them after Fux (Gradus ad
Parnassum, 1725) and Jeppesen. Every function here is a deterministic
application of those rules.
"""

from __future__ import annotations

import re
import unicodedata

from .harmony import interval_between
from .melody import _pitch_table
from .notes import Note, parse_note, parse_notes, transpose
from .scales import resolve_scale_type, scale_notes

# ------------------------------------------------------------ church modes

# (number, name, Latin name, maneria, authentic, final, tenor, ambitus low, ambitus high)
# Finals and reciting tones (tenors) as in the Liber Usualis; ambitus: an authentic mode
# spans the octave above its final, a plagal one runs from a fourth below the final to
# a fifth above it.
_MODES = [
    (1, "Dorian", "protus authenticus", "protus", True, "D4", "A4", "D4", "D5"),
    (2, "Hypodorian", "protus plagalis", "protus", False, "D4", "F4", "A3", "A4"),
    (3, "Phrygian", "deuterus authenticus", "deuterus", True, "E4", "C5", "E4", "E5"),
    (4, "Hypophrygian", "deuterus plagalis", "deuterus", False, "E4", "A4", "B3", "B4"),
    (5, "Lydian", "tritus authenticus", "tritus", True, "F4", "C5", "F4", "F5"),
    (6, "Hypolydian", "tritus plagalis", "tritus", False, "F4", "A4", "C4", "C5"),
    (7, "Mixolydian", "tetrardus authenticus", "tetrardus", True, "G4", "D5", "G4", "G5"),
    (8, "Hypomixolydian", "tetrardus plagalis", "tetrardus", False, "G4", "C5", "D4", "D5"),
]
_ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8}


def _resolve_mode(mode) -> tuple:
    if isinstance(mode, bool):
        raise ValueError(f"Invalid church mode: {mode!r}")
    if isinstance(mode, int):
        number = mode
    elif isinstance(mode, str):
        text = mode.strip()
        if text.isdigit():
            number = int(text)
        elif text.upper() in _ROMAN:
            number = _ROMAN[text.upper()]
        else:
            by_name = {m[1].lower(): m[0] for m in _MODES}
            number = by_name.get(text.lower().replace("-", ""), 0)
    else:
        raise ValueError(f"Invalid church mode: {mode!r} (use 1-8, I-VIII or a name like 'Hypodorian')")
    if not 1 <= number <= 8:
        raise ValueError(f"Church mode must be 1-8 (I-VIII) or a name such as 'Dorian' / 'Hypodorian', got {mode!r}")
    return _MODES[number - 1]


def _white_keys(low: Note, high: Note) -> list[Note]:
    return [n for m, n in _pitch_table([parse_note(x) for x in "CDEFGAB"]) if low.midi <= m <= high.midi]


def church_mode(mode) -> dict:
    """One of the eight modes of Gregorian chant: its final, reciting tone and range.

    `mode` is 1-8, I-VIII or a name ('Dorian', 'Hypodorian', 'Phrygian',
    'Hypophrygian', 'Lydian', 'Hypolydian', 'Mixolydian', 'Hypomixolydian').
    Authentic modes (odd numbers) span the octave above their final; each plagal
    ('hypo-') mode shares its authentic partner's final but sits a fourth lower.
    The `tenor` (reciting tone) is the pitch a psalm is chanted on; melodies
    end on the `final`. Returns the ambitus as concrete notes (feed them to
    melodic_walk, guido_vowel_melody or a notes track). In practice B was often
    flattened (B♭) to avoid the tritone F–B, and modes V/VI mostly used B♭.
    """
    number, name, latin, maneria, authentic, final, tenor, low, high = _resolve_mode(mode)
    notes = [n.name for n in _white_keys(parse_note(low), parse_note(high))]
    return {
        "mode": number,
        "roman": [k for k, v in _ROMAN.items() if v == number][0],
        "name": name,
        "latin": latin,
        "maneria": maneria,
        "type": "authentic" if authentic else "plagal",
        "final": final,
        "tenor": tenor,
        "ambitus": [low, high],
        "notes": notes,
        "b_flat_note": "B♭ (b molle) was commonly sung to avoid the tritone F–B" + (
            "; modes V and VI used it so often that they sound like F major" if number in (5, 6) else ""),
    }


# ------------------------------------------------------------ solmization

_SYLLABLES = ("ut", "re", "mi", "fa", "sol", "la")


def _hexachord(ut: Note) -> dict[str, str]:
    """Spelled note name -> syllable for the hexachord on `ut` (tone tone semitone tone tone)."""
    return {transpose(ut, semis, steps).without_octave().name: syl
            for semis, steps, syl in zip((0, 2, 4, 5, 7, 9), range(6), _SYLLABLES)}


_GAMUT = {"naturale": _hexachord(parse_note("C")),   # on C
          "durum": _hexachord(parse_note("G")),      # on G (B natural = mi)
          "molle": _hexachord(parse_note("F"))}      # on F (B flat = fa)
_GAMUT_NAMES = set().union(*_GAMUT.values())


def _fictive(note: Note) -> tuple[str, dict[str, str]]:
    """The transposed ('fictive') hexachord of a musica-ficta note: a raised note is mi, a lowered one fa."""
    ut = transpose(note.without_octave(), -4, -2) if note.accidental > 0 else transpose(note.without_octave(), -5, -3)
    return f"ficta on {ut.name}", _hexachord(ut)


def solmization(notes) -> dict:
    """Sing a melody in Guido's hexachord syllables: ut re mi fa sol la, with mutations.

    Three hexachords cover the medieval gamut: naturale on C, durum on G (with
    B natural = mi) and molle on F (with B♭ = fa). The semitone is always mi–fa.
    Notes outside the gamut (F♯, E♭, ...) are musica ficta, sung in a
    transposed ('fictive') hexachord in which a raised note is mi and a lowered
    one fa (D E F♯ G = ut re mi fa). The singer stays in one hexachord as long
    as possible and mutates on a pivot note that belongs to both, by the
    medieval rule 'per re sursum, per la deorsum': rising, the pivot becomes re;
    falling, it becomes la (C D E F G A B♭ c = ut re mi fa re mi fa sol;
    D C B A G F E D = sol fa mi la sol fa mi re). Every diatonic semitone stays
    mi–fa, so a pivot right after a semitone is avoided (F E D C B = fa mi re fa
    mi, not fa la ...), and when no pivot keeps it, the mutation is direct
    (only two semitones in a row, as in C♯ D E♭, cannot both be mi–fa).
    The syllables are chosen together, as the fewest mutations (gamut
    hexachords before fictive ones) that satisfy these rules. Octave-less notes
    are read nearest the previous note. Deterministic.
    """
    from .midi_io import assign_octaves  # local import avoids a cycle

    parsed = parse_notes(notes)
    placed = assign_octaves(parsed, 4, "nearest")
    names = [p.without_octave().name for p in parsed]
    n = len(parsed)
    hexes: dict[str, dict[str, str]] = dict(_GAMUT)
    for p, name in zip(parsed, names):
        if name not in _GAMUT_NAMES:
            label, table = _fictive(p)
            hexes.setdefault(label, table)
    order = list(hexes)

    def semitone(i):  # a diatonic semitone between notes i-1 and i (C-C# is chromatic)
        return abs(placed[i].midi - placed[i - 1].midi) == 1 and parsed[i].letter != parsed[i - 1].letter

    def want(i):  # per re sursum, per la deorsum: judged by where the line goes from the pivot
        nxt, cur = (placed[i + 1], placed[i]) if i + 1 < n else (placed[i], placed[i - 1])
        return "re" if nxt.midi > cur.midi else "la"

    def node(i, h):
        return 300 if (h not in _GAMUT and names[i] in _GAMUT_NAMES) else 0  # prefer the gamut

    # dynamic programming over one hexachord per note; costs are integers
    layer = {h: (node(0, h), None) for h in order if names[0] in hexes[h]}
    back = [layer]
    for i in range(1, n):
        nxt = {}
        for h in order:
            if names[i] not in hexes[h]:
                continue
            best = None
            for hp in order:
                if hp not in back[-1]:
                    continue
                cost = back[-1][hp][0] + node(i, h)
                if h != hp:
                    cost += 1000 + min(n - i, 200)  # a mutation, preferably late
                    if names[i] not in hexes[hp]:
                        cost += 10000                # direct: no pivot note
                    elif hexes[h][names[i]] != want(i):
                        cost += 250
                if semitone(i) and {hexes[hp][names[i - 1]], hexes[h][names[i]]} != {"mi", "fa"}:
                    cost += 1_000_000
                if best is None or cost < best[0]:
                    best = (cost, hp)
            nxt[h] = best
        back.append(nxt)
    h = min(back[-1], key=lambda k: (back[-1][k][0], order.index(k)))
    path = [h]
    for i in range(n - 1, 0, -1):
        h = back[i][h][1]
        path.append(h)
    path.reverse()
    mutations = []
    for i in range(1, n):
        old, new = path[i - 1], path[i]
        if old == new:
            continue
        if names[i] in hexes[old]:
            mutations.append({"index": i, "note": parsed[i].name,
                              "from": f"{hexes[old][names[i]]} ({old})", "to": f"{hexes[new][names[i]]} ({new})"})
        else:
            mutations.append({"index": i, "note": parsed[i].name, "from": old, "to": new,
                              "note_detail": "direct mutation (this note is not in the previous hexachord)"})
    return {
        "notes": [p.name for p in parsed],
        "syllables": [hexes[h][name] for h, name in zip(path, names)],
        "hexachords": path,
        "mutations": mutations,
    }


# ------------------------------------------------------------ Guido's vowels

# Micrologus ch. 17: the vowels a e i o u written under the letters of the gamut, from
# gamma (Γ, G2) upward and repeating; a second row starts on the third letter (B). Every
# vowel of a text may then be sung on any pitch that carries it. Guido's finals sit at
# D3-G3; the modes here use the modern octave (D4-G4), so the table moves up an octave
# with them — gamma is G3 and the protus final D still carries u, as in Guido's table.
_GAMUT_START = parse_note("G3")
_SECOND_ROW_START = parse_note("B3")
_VOWELS = "aeiou"
_DIPHTHONGS = ("ae", "oe", "au")


def _vowel_of(midi: int, start: Note = _GAMUT_START) -> str | None:
    table = _pitch_table([parse_note(x) for x in "CDEFGAB"])
    whites = [m for m, _ in table if m >= start.midi]
    if midi not in whites:
        return None
    return _VOWELS[whites.index(midi) % 5]


def _text_vowels(text: str) -> list[str]:
    """The vowel of each syllable, by basic Latin syllabification.

    'qu' and 'ngu' before a vowel are consonants (que-ant, lin-guam); i before a
    vowel at the start of a word or between vowels is the consonant j (Io-han-nes,
    e-ius); only ae, oe and au are diphthongs (also written æ, œ); a diaeresis
    marks a hiatus (Isra-ël, po-ë-ta); every other vowel is its own syllable
    (De-us, fi-li-i, Ky-ri-e e-le-i-son).
    """
    plain = text.lower().replace("æ", "ae").replace("œ", "oe")  # the ligatures of liturgical editions
    plain = unicodedata.normalize("NFKD", plain)
    plain = plain.replace("e\u0308", "E").replace("i\u0308", "I").replace("u\u0308", "U")  # a diaeresis: hiatus
    plain = "".join(ch for ch in plain if not unicodedata.combining(ch)).replace("y", "i")
    vowels = []
    for word in re.findall(r"[a-zEIU]+", plain):
        w = re.sub(r"qu(?=[aeiou])", "q", word)
        w = re.sub(r"ngu(?=[aeiou])", "ng", w)
        w = re.sub(r"^i(?=[aeiou])", "j", w)
        w = re.sub(r"(?<=[aeiou])i(?=[aeiou])", "j", w)
        k = 0
        while k < len(w):
            if w[k] in _VOWELS:
                vowels.append(w[k])
                k += 2 if w[k:k + 2] in _DIPHTHONGS else 1
            elif w[k] in "EIU":  # a vowel with a diaeresis never joins the one before (Isra-el, po-e-ta)
                vowels.append(w[k].lower())
                k += 1
            else:
                k += 1
    return vowels


def guido_vowel_melody(text: str, mode=1, rows: int = 1) -> dict:
    """Guido of Arezzo's vowel method (Micrologus, c. 1026): derive a chant melody from a text.

    Guido wrote the vowels a e i o u under the notes of the scale, from gamma
    upward, repeating — so each vowel can be sung on several pitches — and then
    a second row starting on the third letter (B) to widen the choice
    (`rows=2` uses both). The table sits in the same octave as the modes
    (Guido's D3 final is D4 here), so D carries u as in his table. Each
    syllable's vowel (by basic Latin syllabification: que-ant, De-us) takes the
    pitch carrying it inside the chosen church mode's range that lies nearest
    the previous note (the first syllable starts nearest the final), and the
    melody closes on the mode's final, as chant does. `mode` is 1-8 or a name
    (see church_mode). Returns the syllable vowels, the notes and the candidate
    pitches per vowel. Deterministic.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string (a Latin text works best)")
    if not isinstance(rows, int) or isinstance(rows, bool) or rows not in (1, 2):
        raise ValueError(f"rows must be 1 or 2 (Guido's second vowel row), got {rows!r}")
    info = church_mode(mode)
    vowels = _text_vowels(text)
    if not vowels:
        raise ValueError("the text has no vowels (a e i o u) to set")
    ambitus = [parse_note(n) for n in info["notes"]]
    starts = (_GAMUT_START, _SECOND_ROW_START)[:rows]
    by_vowel = {v: [n for n in ambitus if any(_vowel_of(n.midi, st) == v for st in starts)] for v in _VOWELS}
    final = parse_note(info["final"])
    melody: list[Note] = []
    prev = final.midi
    for k, v in enumerate(vowels):
        cands = by_vowel[v] or ambitus
        target = final.midi if k == len(vowels) - 1 else prev
        choice = min(cands, key=lambda n: (abs(n.midi - target), abs(n.midi - final.midi), n.midi))
        melody.append(choice)
        prev = choice.midi
    closing = []
    if melody[-1].midi != final.midi:
        closing = [final]
    return {
        "mode": info["mode"],
        "mode_name": info["name"],
        "rows": rows,
        "vowels": vowels,
        "notes": [n.name for n in melody + closing],
        "closing_final_added": bool(closing),
        "vowel_pitches": {v: [n.name for n in by_vowel[v]] for v in _VOWELS},
    }


# ------------------------------------------------------------ cantus firmus rules

def _spelled(a: Note, b: Note) -> dict:
    return interval_between(a.name, b.name)


def _interval_ok(a: Note, b: Note) -> tuple[bool, str]:
    """Fux/Jeppesen melodic intervals: m2 M2 m3 M3 P4 P5, ascending m6, P8. Nothing else."""
    semis = b.midi - a.midi
    r = _spelled(a, b)
    q = r["short"]
    simple = q.lstrip("PMmAd")
    allowed = {"m2", "M2", "m3", "M3", "P4", "P5", "P8"}
    if q in allowed:
        return True, ""
    if q == "m6" and semis > 0:
        return True, ""
    if q == "m6":
        return False, "descending minor sixth"
    if semis == 0:
        return False, "repeated note"
    return False, f"{r['name'] or q} ({'up' if semis > 0 else 'down'})"


# modes whose seventh lies a whole step below the final: at a cadence the singer raises it
# (musica ficta) — the raised 7th only as the penultimate note, the raised 6th only on its way there
_RAISED_SEVENTH = {"natural minor", "dorian", "mixolydian"}
_RAISED_SIXTH = {"natural minor"}


def _leaps_outline_ok(a: Note, c: Note) -> bool:
    """Two leaps in one direction must outline a consonance within an octave (spelled: no augmented fifth)."""
    return interval_between(a.name, c.name)["short"] in ("P5", "m6", "M6", "P8")


def check_melody(notes, root: str, scale_type: str = "major", strict: bool = True) -> dict:
    """Check a melody against the cantus-firmus rules of species-counterpoint teaching (after Fux, Jeppesen).

    Rules (errors): begin and end on the tonic; reach the final by step; stay
    diatonic; no repeated notes; only these melodic intervals — minor/major 2nd,
    minor/major 3rd, perfect 4th and 5th, ascending minor 6th, octave (no
    tritones, sevenths, major sixths, descending sixths, augmented or
    diminished intervals); every leap larger than a third is followed by a step
    in the opposite direction; at most two leaps in a row, and two leaps in one
    direction must outline a consonance within an octave (P5, m6, M6, P8 —
    spelled, so an augmented fifth fails); a single climax (the highest note
    occurs once, not at the ends); a range of at most a tenth; a raised leading
    tone rises to the tonic. In modes whose seventh is a whole step below the
    final (aeolian/natural minor, dorian, mixolydian) the raised seventh (ficta)
    may be the penultimate note, and in minor the raised sixth may lead to it.
    Warnings: fewer than 8 or more than 16 notes, fewer than half the moves by
    step, a tritone outlined between two turning points, more than four steps
    in a row one way, more than one leap of a sixth or octave, a restated
    figure, a note used more than 3 times, a final approached from an unraised
    whole-step seventh, a raised leading tone before the cadence in minor, a
    raised sixth (melodic minor) that does not lead to the raised seventh.
    These are the modern textbook rules (the leading-tone rules are tonal
    pedagogy); Fux's own F- and G-mode cantus firmi break the leap rules — `strict=False` reports leap
    recovery and consecutive leaps as warnings instead. Notes without an octave
    are read nearest the previous note, so write octaves for leaps of a fifth
    or more (C4 G4, not C G); `notes` in the result shows the reading. Returns
    `valid`, `violations` and `warnings` (each with the note index), plus
    statistics. Deterministic.
    """
    scale = resolve_scale_type(scale_type)
    tonic = parse_notes(root)[0].without_octave()
    from .midi_io import assign_octaves  # local import avoids a cycle

    mel = assign_octaves(parse_notes(notes), 4, "nearest")
    if len(mel) < 3:
        raise ValueError("a melody needs at least 3 notes to check")
    scale_pcs = {n.pitch_class for n in scale_notes(scale, tonic)}
    errors, warnings = [], []

    def err(i, rule, msg):
        errors.append({"index": i, "rule": rule, "message": msg})

    def warn(i, rule, msg):
        warnings.append({"index": i, "rule": rule, "message": msg})

    def leap_rule(i, rule, msg):
        (err if strict else warn)(i, rule, msg)

    if mel[0].pitch_class != tonic.pitch_class:
        err(0, "start", f"begins on {mel[0].name}, not on the tonic {tonic.name}")
    if mel[-1].pitch_class != tonic.pitch_class:
        err(len(mel) - 1, "end", f"ends on {mel[-1].name}, not on the tonic {tonic.name}")
    elif abs(mel[-1].midi - mel[-2].midi) > 2:
        err(len(mel) - 1, "final_by_step", "the final is not reached by step")
    raised7 = (tonic.pitch_class - 1) % 12 if scale.name in _RAISED_SEVENTH else None
    raised6 = (tonic.pitch_class - 3) % 12 if scale.name in _RAISED_SIXTH else None
    for i, n in enumerate(mel):
        if n.pitch_class in scale_pcs:
            continue
        if n.pitch_class == raised7 and i == len(mel) - 2 and mel[-1].pitch_class == tonic.pitch_class:
            continue  # the ficta leading tone of the cadence
        if n.pitch_class == raised6 and i == len(mel) - 3 and mel[i + 1].pitch_class == raised7:
            continue  # the raised sixth on its way to the raised seventh
        err(i, "diatonic", f"{n.name} is not in {tonic.name} {scale.name}"
            + (" (a raised 7th only as the penultimate note)" if n.pitch_class in (raised7, raised6) else ""))
    moves = [b.midi - a.midi for a, b in zip(mel, mel[1:])]
    for i, (a, b) in enumerate(zip(mel, mel[1:]), start=1):
        ok, why = _interval_ok(a, b)
        if not ok:
            err(i, "interval", f"{a.name} → {b.name}: {why}")
    for i, m in enumerate(moves):
        if abs(m) > 4:  # a leap larger than a third
            nxt = moves[i + 1] if i + 1 < len(moves) else None
            if nxt is None or not (1 <= abs(nxt) <= 2 and (nxt > 0) != (m > 0)):
                leap_rule(i + 1, "leap_recovery",
                          f"the leap {mel[i].name} → {mel[i + 1].name} is not followed by a step back")
    for i in range(len(moves) - 2):
        if all(abs(x) > 2 for x in moves[i:i + 3]):
            leap_rule(i + 1, "consecutive_leaps", "three leaps in a row")
    for i in range(len(moves) - 1):
        a, b = moves[i], moves[i + 1]
        if abs(a) > 2 and abs(b) > 2 and (a > 0) == (b > 0) and not _leaps_outline_ok(mel[i], mel[i + 2]):
            err(i + 1, "leaps_same_direction", f"two leaps in one direction outline a dissonance "
                f"({interval_between(mel[i].name, mel[i + 2].name)['name'] or abs(a + b)})")
    top = max(n.midi for n in mel)
    peaks = [i for i, n in enumerate(mel) if n.midi == top]
    if len(peaks) > 1:
        err(peaks[1], "climax", "the highest note occurs more than once")
    elif peaks[0] in (0, len(mel) - 1):
        err(peaks[0], "climax", "the climax falls on the first or last note")
    span = top - min(n.midi for n in mel)
    if span > 16:
        err(0, "range", f"range of {span} semitones exceeds a tenth")
    leading = (tonic.pitch_class - 1) % 12
    for i, n in enumerate(mel[:-1]):
        if n.pitch_class == leading and leading in scale_pcs and mel[i + 1].pitch_class != tonic.pitch_class:
            err(i, "leading_tone", f"the leading tone {n.name} does not rise to the tonic")
    if scale.name in ("harmonic minor", "melodic minor"):  # in minor the raised 7th belongs to the cadence
        for i, n in enumerate(mel[:-2]):
            if n.pitch_class == leading:
                warn(i, "leading_tone_position", f"the raised leading tone {n.name} appears before the cadence")
    if scale.name == "melodic minor":  # ... and the raised 6th only leads to it
        for i, n in enumerate(mel):
            if n.pitch_class == (tonic.pitch_class - 3) % 12 and (i + 1 == len(mel) or mel[i + 1].pitch_class != leading):
                warn(i, "raised_sixth", f"the raised sixth {n.name} does not lead to the raised seventh")
    run = 1
    for i in range(1, len(moves)):
        run = run + 1 if (1 <= abs(moves[i]) <= 2 and 1 <= abs(moves[i - 1]) <= 2
                          and (moves[i] > 0) == (moves[i - 1] > 0)) else 1
        if run == 5:
            warn(i, "long_run", "more than four steps in a row in one direction (a scale run, not a melody)")
    big = [i for i, m in enumerate(moves) if abs(m) >= 8]
    if len(big) > 1:
        warn(big[1] + 1, "large_leaps", "more than one leap of a sixth or an octave")
    counts = {}
    for n in mel:
        counts[n.midi] = counts.get(n.midi, 0) + 1
    for m, k in counts.items():
        if k > 3 and m != mel[0].midi:
            warn(0, "repetition", f"{next(n.name for n in mel if n.midi == m)} occurs {k} times")
    mids = [n.midi for n in mel]
    for size in (2, 3, 4):
        for i in range(len(mids) - 2 * size + 1):
            if mids[i:i + size] == mids[i + size:i + 2 * size]:
                warn(i + size, "repeated_figure", f"the figure {' '.join(n.name for n in mel[i:i + size])} is restated")
                break
    if not 8 <= len(mel) <= 16:
        warn(0, "length", f"{len(mel)} notes; a cantus firmus usually has 8-16")
    steps = sum(1 for m in moves if 1 <= abs(m) <= 2)
    if steps < len(moves) / 2:
        warn(0, "stepwise", "fewer than half the moves are steps")
    turns = [0] + [i for i in range(1, len(moves)) if (moves[i] > 0) != (moves[i - 1] > 0)] + [len(moves)]
    for a, b in zip(turns, turns[1:]):  # the span between two turning points
        if abs(mel[b].midi - mel[a].midi) == 6:
            warn(a, "outlined_tritone", f"{mel[a].name} … {mel[b].name} outlines a tritone")
    if (mel[-1].pitch_class == tonic.pitch_class and mel[-1].midi - mel[-2].midi == 2
            and mel[-2].pitch_class in scale_pcs):
        warn(len(mel) - 2, "subtonium", "the final is approached from a whole step below (the unraised 7th): "
             + ("raise it (musica ficta) or " if raised7 is not None else "") + "cadence 2–1 from above")
    return {
        "valid": not errors,
        "notes": [n.name for n in mel],
        "violations": errors,
        "warnings": warnings,
        "stats": {"length": len(mel), "range_semitones": span, "steps": steps, "leaps": len(moves) - steps,
                  "climax": mel[peaks[0]].name},
    }


_MOVES = (1, -1, 2, -2, 3, -3, 4, -4, 7, -7)  # scale steps: steps first, then small leaps, then octaves
_SCHEMES = [(a, b) for b in (3, 7, 1, 9) for a in range(10)]  # move-order rotation a + b*position


def cantus_firmus(root: str, scale_type: str = "major", length: int = 10, variant: int = 0) -> dict:
    """Compose a cantus firmus that obeys every rule of check_melody (deterministic).

    Searches the diatonic notes of `root`/`scale_type` (range within a tenth,
    starting on the tonic in octave 4) for a melody of `length` notes (6-16) that
    begins and ends on the tonic, reaches it by step, has a single climax and
    follows all the leap rules, with none of check_melody's warnings (no restated
    figures, no note more than three times, mostly steps). In aeolian, dorian and
    mixolydian the penultimate note may be the raised seventh (ficta), as at a
    modal cadence. `variant` (0-999) picks among the valid melodies: each search
    rotates the order in which moves are tried, and variant n returns the n-th
    distinct melody found, so 0, 1, 2 ... are all different and reproducible
    (a ValueError says when fewer exist). Feed the result to counterpoint().
    """
    if not isinstance(length, int) or isinstance(length, bool) or not 6 <= length <= 16:
        raise ValueError(f"length must be an integer between 6 and 16, got {length!r}")
    if not isinstance(variant, int) or isinstance(variant, bool) or not 0 <= variant <= 999:
        raise ValueError(f"variant must be an integer between 0 and 999, got {variant!r}")
    scale = resolve_scale_type(scale_type)
    if len(scale.intervals) != 7:
        raise ValueError(f"a cantus firmus needs a seven-note scale (a mode, major or minor); "
                         f"{scale.name} has {len(scale.intervals)} notes")
    tonic = parse_notes(root)[0].without_octave()
    start = Note(tonic.letter, tonic.accidental, 4)
    table = [n for m, n in _pitch_table(scale_notes(scale, tonic)[:-1]) if start.midi - 5 <= m <= start.midi + 12]
    idx0 = next(i for i, n in enumerate(table) if n.midi == start.midi)
    leading = (tonic.pitch_class - 1) % 12
    raised7 = None
    if scale.name in _RAISED_SEVENTH:  # the subtonium a whole step below, raised a semitone
        sub = next((n for n in table if n.midi == start.midi - 2), None)
        raised7 = Note(sub.letter, sub.accidental + 1, sub.octave) if sub is not None else None
    approaches = [n for n in table if n.midi - start.midi in (1, 2, -1) or (n.midi == start.midi - 2 and raised7)]
    if not approaches:
        raise ValueError(f"no cantus firmus in {tonic.name} {scale.name}: its final cannot be reached by step "
                         "(neither a step from above nor a leading tone below)")
    max_leaps = (length - 1) // 2  # check_melody warns when fewer than half the moves are steps
    cadence_only_leading = scale.name in ("harmonic minor", "melodic minor")
    raised_sixth = (tonic.pitch_class - 3) % 12 if scale.name == "melodic minor" else None
    interval_memo: dict[tuple[str, str], bool] = {}
    outline_memo: dict[tuple[str, str], bool] = {}

    table_midi = [n.midi for n in table]
    first_above = {m: next((i for i, x in enumerate(table_midi) if x > m), None) for m in range(128)}
    at_or_below = {m: max((i for i, x in enumerate(table_midi) if x <= m), default=0) for m in range(128)}
    tonic_pc = tonic.pitch_class

    def memo(cache, fn, a, b):
        key = (a.name, b.name)
        if key not in cache:
            cache[key] = fn(a, b)
        return cache[key]

    def ok_so_far(path, mids, m, counts, leaps, bigs) -> bool:
        """Prefix pruning: only rules that, once broken, stay broken in any completion.
        `mids`, `m` (moves), `counts` (per pitch), `leaps` and `bigs` are kept up to date by the caller."""
        if leaps > max_leaps or bigs > 1:
            return False
        if not memo(interval_memo, lambda x, y: _interval_ok(x, y)[0], path[-2], path[-1]):
            return False
        last, n = mids[-1], len(mids)
        if len(m) >= 5 and all(1 <= abs(x) <= 2 for x in m[-5:]) and len({x > 0 for x in m[-5:]}) == 1:
            return False  # more than four steps in a row one way
        if counts[last] > 3 and last != mids[0]:
            return False
        for size in (2, 3, 4):  # a figure restated right away (checked where the path just grew)
            if n >= 2 * size and mids[-size:] == mids[-2 * size:-size]:
                return False
        if len(m) >= 2:
            m1, m2 = m[-1], m[-2]
            if abs(m2) > 4 and not (1 <= abs(m1) <= 2 and (m1 > 0) != (m2 > 0)):
                return False
            if abs(m1) > 2 and abs(m2) > 2:
                if len(m) >= 3 and abs(m[-3]) > 2:
                    return False  # three leaps in a row
                if (m1 > 0) == (m2 > 0) and not memo(outline_memo, _leaps_outline_ok, path[-3], path[-1]):
                    return False
        top, low = max(mids), min(mids)
        if top - low > 16:
            return False
        # a single climax: if the top note so far is doubled (the final tonic counts), the rest of the
        # melody must still climb above it and come back down; one move covers at most an octave
        if counts[top] > 1 or top == start.midi:
            above = first_above[top]
            if above is None:
                return False
            if -(-(above - at_or_below[last]) // 7) + -(-(above - idx0) // 7) > length - n:
                return False
        if mids[-2] % 12 == leading and last % 12 != tonic_pc:
            return False
        if cadence_only_leading and last % 12 == leading and n < length - 1:
            return False  # in minor the raised 7th belongs to the cadence
        if raised_sixth is not None and last % 12 == raised_sixth and n != length - 2:
            return False  # ... and the raised 6th only leads to it
        if len(m) >= 2 and (m[-1] > 0) != (m[-2] > 0):  # a turn completes a segment: no outlined tritone
            j = len(m) - 2
            while j > 0 and (m[j - 1] > 0) == (m[-2] > 0):
                j -= 1
            if abs(mids[-2] - mids[j]) == 6:
                return False
        return True

    total = [1_000_000]  # nodes over all searches, so a scale with no valid melody fails fast

    def search(a, b, want, found, seen) -> bool:
        """Depth-first search with move order rotated by a + b*position; collects up to `want` new
        melodies. Returns True when it walked the whole search space (every order then finds the same set)."""
        budget = [min(200000, total[0])]
        start_budget = budget[0]
        path, mids, moves = [start], [start.midi], []
        counts = {start.midi: 1}

        def walk(idx, leaps, bigs):
            if len(found) >= want or budget[0] <= 0:
                return
            budget[0] -= 1
            pos = len(path)
            if pos == length:
                names = tuple(x.name for x in path)
                if names in seen:
                    return
                report = check_melody(list(names), tonic.name, scale.name)
                if report["valid"] and all(w["rule"] == "length" for w in report["warnings"]):
                    seen.add(names)
                    found.append(list(path))
                return
            shift = (a + b * pos) % len(_MOVES)
            for step in _MOVES[shift:] + _MOVES[:shift]:
                j = idx + step
                if not 0 <= j < len(table):
                    continue
                cand, mid = table[j], table_midi[j]
                if pos == length - 1 and mid != start.midi:
                    continue
                if pos == length - 2:
                    if abs(mid - start.midi) not in (1, 2):
                        continue
                    if mid == start.midi - 2:  # an unraised whole-step 7-1 cadence draws a warning
                        if raised7 is None:
                            continue
                        cand, mid = raised7, raised7.midi  # a modal cadence raises the subtonium
                move = mid - mids[-1]
                path.append(cand)
                mids.append(mid)
                moves.append(move)
                counts[mid] = counts.get(mid, 0) + 1
                big_leap = abs(move) > 2
                if ok_so_far(path, mids, moves, counts, leaps + big_leap, bigs + (abs(move) >= 8)):
                    walk(j, leaps + big_leap, bigs + (abs(move) >= 8))
                counts[mid] -= 1
                moves.pop()
                mids.pop()
                path.pop()

        before = len(found)
        walk(idx0, 0, 0)
        total[0] -= start_budget - budget[0]
        return budget[0] > 0 and len(found) - before < want - before  # stopped because the space ran out

    found: list[list[Note]] = []
    seen: set[tuple[str, ...]] = set()
    exhausted = False
    for a, b in _SCHEMES:  # one melody per move order: the variants differ from the first note on
        exhausted = search(a, b, len(found) + 1, found, seen)
        if len(found) > variant or exhausted or total[0] <= 0:
            break
    if len(found) <= variant and not exhausted and total[0] > 0:  # list further melodies of the first order
        search(*_SCHEMES[0], variant + 1, found, seen)
    if not found:
        raise ValueError(f"no cantus firmus of {length} notes obeys every rule in {tonic.name} {scale.name}")
    if len(found) <= variant:
        raise ValueError(f"only {len(found)} distinct cantus firmi of {length} notes found in {tonic.name} "
                         f"{scale.name}; use a variant below {len(found)}")
    result = found[variant]
    return {
        "root": tonic.name,
        "scale_type": scale.name,
        "length": length,
        "variant": variant,
        "notes": [n.name for n in result],
        "check": check_melody([n.name for n in result], tonic.name, scale.name),
    }
