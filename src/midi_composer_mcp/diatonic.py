"""Diatonic chords of a scale and degree-sequence resolution.

These are the atomic building blocks for chord progressions: the LLM decides
*which* degrees to use (the creative part); these functions only report what
chords live on each scale degree and resolve a chosen degree sequence into
concrete chords.
"""

from __future__ import annotations

import re

from .chords import CHORDS, ChordType, identify_chord_quality, match_chords
from .notes import LETTERS, Note, parse_note, parse_notes, transpose
from .scales import MAJOR_DEGREES, ScaleType, degree_labels, resolve_scale_type, scale_notes

_ROMAN_BASE = ("I", "II", "III", "IV", "V", "VI", "VII")
_ROMAN_VALUES = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7}

_DEGREE_NAMES = ("tonic", "supertonic", "mediant", "subdominant", "dominant", "submediant")
_HARMONIC_FUNCTIONS = ("tonic", "subdominant", "tonic", "subdominant", "dominant", "tonic", "dominant")

# How a chord quality is written after a roman numeral.
_ROMAN_QUALITY = {
    "major": "", "minor": "", "diminished": "°", "augmented": "+",
    "dominant 7": "7", "major 7": "Δ7", "minor 7": "7",
    "half-diminished": "ø7", "diminished 7": "°7",
    "minor major 7": "Δ7", "augmented major 7": "+Δ7",
}


def roman_suffix(quality: ChordType | None, minor_numeral: bool) -> str:
    """How a chord quality is written after a roman numeral ('7', 'ø7', '+', 'sus4'...).

    Qualities without a classic figure fall back to their symbol suffix. A bare
    '6' is written 'add6' so it cannot be misread as a first-inversion figure,
    and a lowercase numeral drops a redundant leading 'm' ('ii9', not 'iim9').
    """
    if quality is None:
        return "?"
    if quality.name in _ROMAN_QUALITY:
        return _ROMAN_QUALITY[quality.name]
    suffix = quality.symbol
    if suffix == "6":
        return "add6"
    if minor_numeral and suffix.startswith("m") and not suffix.startswith(("maj", "m6")):
        suffix = suffix[1:]
    return suffix


# Fallback (spelling-blind) numerals, used only when the letter-based spelling
# would need more than a double accidental.
_REL_TO_ROMAN = {
    0: ("I", ""), 1: ("II", "b"), 2: ("II", ""), 3: ("III", "b"), 4: ("III", ""),
    5: ("IV", ""), 6: ("IV", "#"), 7: ("V", ""), 8: ("VI", "b"), 9: ("VI", ""),
    10: ("VII", "b"), 11: ("VII", ""),
}


def numeral_base(croot: Note, tonic: Note) -> tuple[str, str]:
    """(numeral, accidental prefix) of a chord root, read from its spelling.

    The numeral comes from the letter distance to the tonic and the accidental
    from how far the root sits from the major-scale degree on that letter, so
    Gb in C is bV and F# is #IV (matching diatonic_chords' numerals).
    """
    steps = (LETTERS.index(croot.letter) - LETTERS.index(tonic.letter)) % 7
    rel = (croot.pitch_class - tonic.pitch_class) % 12
    offset = (rel - MAJOR_DEGREES[steps] + 6) % 12 - 6
    if abs(offset) <= 2:
        return _ROMAN_BASE[steps], "#" * offset if offset > 0 else "b" * -offset
    base, accidental = _REL_TO_ROMAN[rel]
    return base, accidental


# ------------------------------------------------ borrowing (modal interchange)

# The parallel modes a chord can be borrowed from, in the fixed order that breaks
# distance ties: the church modes from brightest to darkest, then the minor and
# harmonic variants. Mixture from the parallel minor follows Aldwell & Schachter
# and Kostka & Payne; the other modes follow Berklee modal interchange (Nettles &
# Graf, The Chord Scale Theory & Jazz Harmony, 1997).
PARALLEL_MODES = ("major", "lydian", "mixolydian", "dorian", "natural minor", "phrygian",
                  "locrian", "harmonic minor", "melodic minor", "harmonic major")


def _scale_pcs(scale: ScaleType, tonic_pc: int) -> frozenset[int]:
    return frozenset((tonic_pc + i) % 12 for i in scale.intervals)


def _mode_offset(scale: ScaleType) -> int:
    """Signature offset of a heptatonic mode against major (dorian -2, lydian +1 ...)."""
    if len(scale.intervals) != 7:
        return 0
    return sum(i - m for i, m in zip(scale.intervals, MAJOR_DEGREES))


def _as_pitch_classes(notes) -> set[int]:
    """Pitch classes of notes given as Note objects, note names or integers 0-11."""
    if isinstance(notes, (str, Note)):
        notes = [notes]
    if not isinstance(notes, (list, tuple, set, frozenset)):
        raise ValueError(f"Expected a list of notes or pitch classes, got {type(notes).__name__}")
    pcs: set[int] = set()
    for n in notes:
        if isinstance(n, Note):
            pcs.add(n.pitch_class)
        elif isinstance(n, int) and not isinstance(n, bool):
            pcs.add(n % 12)
        elif isinstance(n, str):
            pcs.add(parse_note(n).pitch_class)
        else:
            raise ValueError(f"Not a note or pitch class: {n!r}")
    return pcs


def borrowing_sources(tonic: str | Note, scale_type: str = "major", modes=None,
                      fifths_steps: int = 0, include_home: bool = False) -> list[dict]:
    """The scales a chord may be borrowed from, in source order, with their distance.

    Sources are (0) the home scale itself, distance 0, only with `include_home`
    (chord_palette's 'sources' lists it; borrowed_from never does), (1) parallel
    modes on the same tonic — `modes` (a name or list),
    by default PARALLEL_MODES — with the home mode itself omitted, then (2) for
    `fifths_steps` n (0-6), the same mode on the keys 1..n steps round the
    circle of fifths, in the order +1, -1, +2, -2 ... (C major: G major, F major,
    D major, Bb major ...). A key's tonic is the home tonic moved 7k semitones
    onto the fifth letter, respelled to a practical signature (circle._practical).

    Each source is {label ('C dorian', 'G major'), tonic (Note), scale (ScaleType),
    distance, pcs}. `distance` is how many of the source's pitch classes lie
    outside the home scale (from C major: lydian, mixolydian, melodic minor and
    harmonic major 1; dorian and harmonic minor 2; natural minor 3; phrygian 4;
    locrian 5); a key k steps away has distance |k|. Pure set arithmetic.
    """
    from .circle import _practical  # circle has no dependency on this module

    home = resolve_scale_type(scale_type)
    home_tonic = (tonic if isinstance(tonic, Note) else parse_notes(tonic)[0]).without_octave()
    home_pcs = _scale_pcs(home, home_tonic.pitch_class)
    if modes is None:
        modes = PARALLEL_MODES
    elif isinstance(modes, str):
        modes = [modes]
    if not isinstance(modes, (list, tuple)):
        raise ValueError(f"modes must be a scale name or a list of scale names, got {modes!r}")
    if not isinstance(fifths_steps, int) or isinstance(fifths_steps, bool) or not 0 <= fifths_steps <= 6:
        raise ValueError(f"fifths_steps must be an integer between 0 and 6, got {fifths_steps!r}")

    sources: list[dict] = []
    if include_home:
        sources.append({"label": f"{home_tonic.pitch_class_name} {home.name}", "tonic": home_tonic,
                        "scale": home, "distance": 0, "pcs": home_pcs})
    seen: set[str] = set()
    for mode in modes:
        scale = resolve_scale_type(mode)
        if scale.name == home.name or scale.name in seen:
            continue
        seen.add(scale.name)
        pcs = _scale_pcs(scale, home_tonic.pitch_class)
        sources.append({"label": f"{home_tonic.pitch_class_name} {scale.name}", "tonic": home_tonic,
                        "scale": scale, "distance": len(pcs - home_pcs), "pcs": pcs})
    for step in range(1, fifths_steps + 1):
        for k in (step, -step):
            key_tonic = transpose(home_tonic, 7 * k, 4 * k)
            key_tonic = _practical(key_tonic, _mode_offset(home)).without_octave()
            label = f"{key_tonic.pitch_class_name} {home.name}"
            if any(s["label"] == label for s in sources):
                continue  # six steps both ways can land on one spelling (F# +-6 -> C)
            sources.append({"label": label, "tonic": key_tonic, "scale": home, "distance": abs(k),
                            "pcs": _scale_pcs(home, key_tonic.pitch_class)})
    return sources


def borrowed_sources(notes, tonic: str | Note, scale_type: str = "major", modes=None,
                     fifths_steps: int = 0, include_home: bool = False) -> list[str]:
    """Label every borrowing source whose pitch set holds all of `notes` ('C natural minor', ...).

    `notes` are the chord tones plus the bass (Note objects, names or pitch
    classes 0-11). The candidates are borrowing_sources(tonic, scale_type,
    modes, fifths_steps) — by default the ten PARALLEL_MODES on the same tonic,
    home mode omitted — and the result is sorted by (distance from the home
    scale, source order). The one rule shared by roman_to_chords' borrowed_from,
    analyze_progression's borrowed_from and chord_palette's sources, so they
    never disagree. e.g. Fm (F Ab C) in C major -> ['C harmonic major',
    'C harmonic minor', 'C natural minor', 'C phrygian', 'C locrian'].
    """
    pcs = _as_pitch_classes(notes)
    sources = borrowing_sources(tonic, scale_type, modes, fifths_steps, include_home)
    hits = [(s["distance"], i, s["label"]) for i, s in enumerate(sources) if pcs <= s["pcs"]]
    return [label for _, _, label in sorted(hits)]


def _roman_numeral(index: int, intervals: tuple[int, ...], quality: ChordType | None,
                   relative_pcs: frozenset[int]) -> str:
    diff = intervals[index] - MAJOR_DEGREES[index]
    prefix = "#" * diff if diff > 0 else "b" * -diff
    numeral = _ROMAN_BASE[index]
    minor_third = 3 in relative_pcs and 4 not in relative_pcs
    if minor_third:
        numeral = numeral.lower()
    return f"{prefix}{numeral}{roman_suffix(quality, minor_third)}"


def _stacked_chord(scale_intervals: tuple[int, ...], labels: list[str],
                   index: int, tone_count: int, root: Note) -> dict:
    """Build the chord on scale degree `index` by stacking scale thirds."""
    n = len(scale_intervals)
    tone_indices = [index + 2 * k for k in range(tone_count)]
    semitones = [scale_intervals[j % n] + 12 * (j // n) for j in tone_indices]
    base = semitones[0]

    tones = [
        transpose(root, semis, int(labels[j % n].lstrip("#b")) - 1 + 7 * (j // n))
        for semis, j in zip(semitones, tone_indices)
    ]
    relative_pcs = frozenset((s - base) % 12 for s in semitones)
    quality = identify_chord_quality(relative_pcs)
    chord_root = tones[0]

    entry: dict = {
        "degree": index + 1,
        "root": chord_root.name,
        "chord_type": quality.name if quality else "unknown",
        "symbol": f"{chord_root.pitch_class_name}{quality.symbol}" if quality else None,
        "notes": [t.name for t in tones],
        "intervals_from_chord_root": [s - base for s in semitones],
    }
    if quality is None:
        # Not a stacked-thirds chord type (common in pentatonic and exotic scales):
        # name it as an inversion if it is one (C E A = Am/C), otherwise hand back the
        # note list itself — either way `symbol` feeds chords_to_midi and friends.
        found = match_chords([t.name for t in tones], include_partial=False, limit=1)["matches"]
        if found:
            entry["symbol"] = found[0]["symbol"]
            entry["chord_type"] = found[0]["chord_type"]
            entry["root"] = found[0]["root"]      # the chord's real root ...
            entry["bass"] = chord_root.name       # ... over the scale degree it was stacked on
            if "inversion" in found[0]:
                entry["inversion"] = found[0]["inversion"]
        else:
            entry["symbol"] = [t.pitch_class_name for t in tones]
    if chord_root.octave is not None:
        entry["midi"] = [t.midi for t in tones]
    if n == 7:
        entry["roman"] = _roman_numeral(index, scale_intervals, quality, relative_pcs)
        entry["degree_name"] = (
            _DEGREE_NAMES[index]
            if index < 6
            else ("leading tone" if scale_intervals[6] == 11 else "subtonic")
        )
        entry["harmonic_function"] = _HARMONIC_FUNCTIONS[index]
    return entry


def diatonic_chords(root: str, scale_type: str, sevenths: bool = False) -> dict:
    """The chord that lives on each degree of a scale (triads or sevenths).

    For 7-note scales each chord also gets a roman numeral, its classical
    degree name and harmonic function (tonic / subdominant / dominant).
    """
    scale = resolve_scale_type(scale_type)
    root_note = parse_notes(root)[0]
    labels = degree_labels(scale)
    degrees = scale_notes(scale, root_note)[:-1]
    tone_count = 4 if sevenths else 3

    chords = [
        _stacked_chord(scale.intervals, labels, i, tone_count, root_note)
        for i in range(len(scale.intervals))
    ]
    return {
        "root": root_note.name,
        "scale_type": scale.name,
        "scale_notes": [n.name for n in degrees],
        "sevenths": sevenths,
        "chords": chords,
    }


# What may follow a roman numeral: the figures diatonic_chords/analyze_progression
# write (°, ø7, +, Δ7, 7, sus4, add6, ?...). They are ignored — `sevenths` decides.
_NUMERAL_SUFFIX = r"(?:[°øo+Δ?]|maj|add|sus|dim|aug|m|b|#|\d)*"


def _parse_degree(token: int | str, intervals: tuple[int, ...]) -> int:
    """Parse a scale degree given as 1-based int, '5', 'V', 'vi', 'vii°' or 'bVII'.

    An accidental prefix is accepted when it names the scale's own degree, so the
    numerals diatonic_chords prints round-trip (bIII, bVI, bVII in natural minor,
    #iv° in lydian); a prefix that alters the scale ('bVII' in major) is rejected.
    """
    count = len(intervals)
    if isinstance(token, bool):
        raise ValueError(f"Invalid scale degree: {token!r}")
    if isinstance(token, int):
        degree = token
    elif isinstance(token, str):
        text = token.strip()
        m = re.match(rf"^([#b]*)(\d+|[ivIV]+){_NUMERAL_SUFFIX}$", text)
        if not m:
            raise ValueError(f"Invalid scale degree: {token!r}. Use 1-{count} or roman numerals I-VII.")
        prefix, core = m.group(1), m.group(2)
        if core.isdigit():
            degree = int(core)
        else:
            degree = _ROMAN_VALUES.get(core.lower(), 0)
            if degree == 0:
                raise ValueError(
                    f"Invalid scale degree: {token!r}. Use 1-{count} or roman numerals I-VII."
                )
        if prefix:
            alteration = prefix.count("#") - prefix.count("b")
            if not (count == 7 and 1 <= degree <= 7
                    and MAJOR_DEGREES[degree - 1] + alteration == intervals[degree - 1]):
                raise ValueError(
                    f"Chromatic alterations are not supported in degree sequences: {token!r}."
                    f" Degrees are positions in the chosen scale (1-{count}); an accidental is"
                    f" only accepted when it names the scale's own degree (bVI in natural minor)."
                    f" For chromatic numerals (bVII in major, V7/V, N6) use roman_to_chords."
                )
    else:
        raise ValueError(f"Invalid scale degree: {token!r} (use an integer or a roman numeral)")
    if not 1 <= degree <= count:
        raise ValueError(f"Scale degree {degree} out of range 1-{count} for this scale")
    return degree


def split_tokens(text: str, bar_lines: bool = False) -> list[str]:
    """Split a degree / numeral string into tokens.

    Separators are commas, whitespace and dashes *between* tokens ('1-5-6-4',
    'ii–V–I'); a leading '-' stays on its number so '-1' is rejected, not read
    as 1, and a free-standing dash ('I - V') is dropped. With `bar_lines`, '|'
    separates too ('I V | vi IV').
    """
    pattern = r"[,\s|]+|(?<=\S)[-–—](?=\S)|[–—]" if bar_lines else r"[,\s]+|(?<=\S)[-–—](?=\S)|[–—]"
    tokens = re.split(pattern, text.strip())
    return [t for t in tokens if t and not re.fullmatch(r"[-–—]+", t)]  # 'I - V - vi - IV'


def _chord_label(entry: dict) -> str:
    symbol = entry["symbol"]
    return symbol if isinstance(symbol, str) else " ".join(symbol)


def _degree_warning(token, entry: dict, root: str, scale: ScaleType, sevenths: bool) -> str | None:
    """A note when a numeral token says more than degrees_to_chords used (case, mark, figure).

    degrees_to_chords reads a numeral as a *position* only. For 7-note scales the
    token is also read the way roman_to_chords reads it; when the two chords
    differ (other notes, or another bass) the caller is told. A bare triad
    numeral with sevenths=true is compared by its triad, so 'V' -> G7 is silent.
    """
    if not isinstance(token, str):
        return None
    text = token.strip()
    m = re.match(r"^([#b]*)(\d+|[ivIV]+)(.*)$", text)
    if not m or m.group(2).isdigit():
        return None  # a plain degree number carries no quality to ignore
    suffix = m.group(3)
    label = _chord_label(entry)
    have = [parse_note(n) for n in entry["notes"]]
    have_bass = parse_note(entry["bass"]) if "bass" in entry else have[0]
    # '?' is diatonic_chords' marker for an unnamed stacked chord: silent only on a degree whose
    # own stacked chord is unnamed (renamed as an inversion, or left as a note array)
    unnamed = "bass" in entry or not isinstance(entry["symbol"], str)
    if "?" in suffix and not unnamed:
        return (f"{text} resolved to {label} (the scale's own chord); its '{suffix}' was ignored ('?'"
                f" marks an unnamed chord, but this degree's chord is {label}: pass the chord's notes,"
                f" or use roman_to_chords for a named quality)")
    if len(scale.intervals) == 7:
        from .roman import roman_to_chords  # roman imports this module at load time

        try:
            reading = roman_to_chords([text], root, scale.name)["chords"][0]
        except ValueError as e:
            if "?" in suffix:
                return None  # the degree's own unnamed stacked chord
            return (f"{text} resolved to {label} (the scale's own chord); its '{suffix}' was ignored"
                    f" (roman_to_chords rejects this numeral: {e})")
        want = {parse_note(n).pitch_class for n in reading["notes"]}
        want_bass = parse_note(reading["bass"]).pitch_class
        compare = have[:3] if sevenths and len(want) == 3 and len(have) == 4 else have
        if {n.pitch_class for n in compare} == want and have_bass.pitch_class == want_bass:
            return None
        wanted = f"{reading['root']} {reading['chord_type']}"
        if reading["inversion"]:
            wanted += f" over {reading['bass']}"
        return f"{text} resolved to {label} (the scale's own chord); for {wanted} use roman_to_chords"
    # other scales: no Roman-numeral dialect, so flag an ignored suffix or a case mismatch
    if suffix.strip("?"):
        return (f"{text} resolved to {label} (the scale's own chord); its '{suffix}' was ignored"
                f" (degrees are positions in the scale)")
    chord_root = parse_note(entry["root"]).pitch_class
    rel = {(n.pitch_class - chord_root) % 12 for n in have}
    lower = m.group(2).islower()
    if (lower and 4 in rel and 3 not in rel) or (not lower and 3 in rel and 4 not in rel):
        return (f"{text} resolved to {label} (the scale's own chord); the numeral's case"
                f" ({'minor' if lower else 'major'}) was ignored")
    return None


def degrees_to_chords(root: str, scale_type: str, degrees, sevenths: bool = False) -> dict:
    """Resolve a degree sequence (e.g. [1, 5, 6, 4] or 'I V vi IV') to chords.

    The caller chooses the sequence; this only maps each degree to the chord
    that the scale defines there. Numerals are *positions*: 'iv' in C major is
    still F, and 'V7' without sevenths=true is still G. `warnings` lists one
    note per token whose case, quality mark, accidental or figure was ignored,
    e.g. "iv resolved to F (the scale's own chord); for F minor use
    roman_to_chords" — roman_to_chords reads numerals literally (iv = Fm, bVII,
    V7/V, N6, inversions).
    """
    scale = resolve_scale_type(scale_type)
    if isinstance(degrees, (int, str)) and not isinstance(degrees, bool):
        degrees = split_tokens(str(degrees))
    if not isinstance(degrees, (list, tuple)) or not degrees:
        raise ValueError("degrees must be a non-empty list like [1, 5, 6, 4] or 'I V vi IV'")
    indices = [_parse_degree(t, scale.intervals) for t in degrees]

    base = diatonic_chords(root, scale_type, sevenths)
    by_degree = {c["degree"]: c for c in base["chords"]}
    progression = [dict(by_degree[d]) for d in indices]
    warnings = [w for t, c in zip(degrees, progression)
                if (w := _degree_warning(t, c, root, scale, sevenths)) is not None]
    return {
        "root": base["root"],
        "scale_type": base["scale_type"],
        "degrees": indices,
        "sevenths": sevenths,
        "chords": progression,
        "symbols": [c["symbol"] for c in progression],
        "warnings": warnings,
    }


# ------------------------------------------------------------ chord palette

_ROMAN_NOTE = ("roman_to_chords reads {roman!r} in {key} as {got}, not {symbol}: palette numerals are"
               " measured from the parallel major, but in this key a numeral without an accidental names"
               " the key's own degree (in minor, ^6/^7 follow the chord's third) and the dialect has no"
               " natural sign. Read the tokens in {tonic} major (roman_to_chords(tokens, '{tonic}',"
               " 'major')) or use the symbol")


def _flag_home_misreadings(entries: list[dict], tonic: Note, home: ScaleType) -> None:
    """Add `roman_note` to every entry whose roman reads back as other notes in the home key.

    Palette numerals are measured from the parallel major (dialect rule a), so they
    always read back in `roman_to_chords(tokens, tonic, 'major')`. In another home
    mode a bare numeral names the home scale's own degree (and in natural, harmonic
    and melodic minor ^6/^7 follow the chord's third), so a borrowed Em in C minor
    ('iii') or F#sus4 in A minor ('VIsus4') cannot be written in that key at all.
    roman_to_chords itself decides; '?' tokens (no table chord) are not numerals it reads.
    """
    from .roman import roman_to_chords  # roman imports this module at load time

    readable = [e for e in entries if "?" not in e["roman"]]
    if not readable:
        return
    readings = roman_to_chords([e["roman"] for e in readable], tonic.name, home.name)["chords"]
    for e, got in zip(readable, readings):
        want = {parse_note(n).pitch_class for n in e["notes"]}
        if {parse_note(n).pitch_class for n in got["notes"]} == want:
            continue
        label = got["symbol"] if isinstance(got["symbol"], str) else " ".join(got["symbol"])
        e["roman_note"] = _ROMAN_NOTE.format(roman=e["roman"], key=f"{tonic.pitch_class_name} {home.name}",
                                             got=label, symbol=e["symbol"], tonic=tonic.pitch_class_name)


def _check_bool(name: str, value) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false, got {value!r}")


def _chord_family(ctype: ChordType | None, root: Note, notes: list[Note]) -> str:
    """Klimper's colour family, read by letters + semitones (never semitones alone).

    'major': a major 3rd, no minor 3rd, a perfect or absent 5th; 'minor': likewise
    with a minor 3rd; anything else (dim, aug, sus, power, b5/#5 chords) 'other'.
    A table chord is read from its own degree labels (so 7#9's #9 is a ninth, not a
    minor third); an unnamed stacked chord from its spelled notes.
    """
    if ctype is not None:
        labels = [label for label, _ in ctype.degrees]
    else:
        labels = []
        for n in notes[1:]:
            steps = (LETTERS.index(n.letter) - LETTERS.index(root.letter)) % 7
            semis = (n.pitch_class - root.pitch_class) % 12
            if steps == 2:
                labels.append({4: "3", 3: "b3"}.get(semis, "x3"))
            elif steps == 4:
                labels.append("5" if semis == 7 else "x5")
    thirds = {label for label in labels if label.lstrip("#bx") == "3"}
    fifths = {label for label in labels if label.lstrip("#bx") == "5"}
    if fifths - {"5"}:
        return "other"
    if thirds == {"3"}:
        return "major"
    if thirds == {"b3"}:
        return "minor"
    return "other"


def _palette_universe(tonic: Note, scale: ScaleType, extended: bool, sevenths: bool,
                      sizes: frozenset[int]) -> list[dict]:
    """U(tonic, scale): the raw palette members of one source scale, in scale order.

    Each member is {root, ctype (None for an unnamed stacked chord), notes (spelled as
    the scale spells them, root first), degree, core, stacked_roman}. `stacked_roman`
    is diatonic_chords' own numeral when it wrote this chord on its root (else None).
    """
    degrees = scale_notes(scale, tonic)[:-1]
    spelled = {n.pitch_class: n for n in degrees}
    position = {n.pitch_class: i + 1 for i, n in enumerate(degrees)}
    members: list[dict] = []
    core: dict[tuple[int, str], str | None] = {}
    for c in diatonic_chords(tonic.name, scale.name, sevenths)["chords"]:
        if not isinstance(c["symbol"], str):  # no table chord: keep the stacked notes as they are
            members.append({"root": parse_note(c["root"]), "ctype": None,
                            "notes": [parse_note(n) for n in c["notes"]],
                            "degree": c["degree"], "core": True, "stacked_roman": c.get("roman")})
            continue
        ctype = CHORDS[c["chord_type"]]
        root = spelled[parse_note(c["root"]).pitch_class]
        key = (root.pitch_class, ctype.name)
        # an inversion (C major pentatonic's C E A = Am/C) is listed in root position, on its root;
        # only a chord stacked on its own root keeps diatonic_chords' numeral
        core[key] = core.get(key) or (c.get("roman") if "bass" not in c else None)
        if not extended:
            members.append({"root": root, "ctype": ctype,
                            "notes": [spelled[(root.pitch_class + s) % 12] for s in ctype.intervals],
                            "degree": position[root.pitch_class], "core": True,
                            "stacked_roman": core[key]})
    if extended:
        for root in degrees:
            for ctype in CHORDS.values():
                if len(ctype.intervals) not in sizes:
                    continue
                pcs = [(root.pitch_class + s) % 12 for s in ctype.intervals]
                if not all(pc in spelled for pc in pcs):
                    continue
                key = (root.pitch_class, ctype.name)
                members.append({"root": root, "ctype": ctype, "notes": [spelled[pc] for pc in pcs],
                                "degree": position[root.pitch_class], "core": key in core,
                                "stacked_roman": core.get(key)})
    return members


def chord_palette(root: str, scale_type: str = "major", extended: bool = False, sevenths: bool = False,
                  max_notes: int = 4, include_dyads: bool = False, borrow: bool = False,
                  source_modes=None, fifths_steps: int = 0, limit: int = 0) -> dict:
    """Every chord that fits a key, plus (optionally) chords borrowed from parallel modes and neighbour keys.

    One rule for both halves, pure set arithmetic (nothing is ranked by taste):

    - UNIVERSE U(tonic, mode): with extended=false the stacked-thirds chords
      diatonic_chords builds (triads, or sevenths with `sevenths`); a stacked
      chord diatonic_chords names as an inversion (C major pentatonic's C E A =
      Am/C) is listed in root position, and a stacked set that is no table chord
      stays a note list. With extended=true every chord type of the chord table
      with 3..max_notes notes (max_notes 3-6; the 2-note power chord only with
      include_dyads), rooted on every scale note, kept iff its pitch classes all
      lie in the scale. Notes are spelled as the scale spells them.
    - IN-KEY entries = U(root, scale_type). Any scale works (pentatonic, symmetric...).
    - BORROWED entries come from `source_modes` (a name or list) if given, else
      with borrow=true from PARALLEL_MODES minus the home mode, plus with
      `fifths_steps` n (0-6) the same mode on the keys 1..n steps round the
      circle (+1, -1, +2, -2 ...) — the sources of borrowing_sources. From each
      source S the U(S) chords with at least one tone outside the home scale
      are kept (so chords the home key already has are dropped). Borrowing
      needs a 7-note home, and every source mode must be a 7-note scale.

    Entry fields: symbol, root, notes, chord_type, size, degree (of the root in
    its source scale), family ('major' | 'minor' | 'other' — Klimper's green /
    purple / grey), core (the stacked chord diatonic_chords builds on that
    degree), roman against the HOME tonic (7-note homes only: a same-tonic
    stacked chord keeps diatonic_chords' numeral, measured from the parallel
    major — 'bVI', 'iv', 'bIII+', '#iv°' — anything else is numeral_base +
    roman_suffix; '?' marks an unnamed stacked chord. Every other roman reads
    back through roman_to_chords(tokens, root, 'major'), and through
    roman_to_chords in the home key too unless the entry carries `roman_note`:
    in a non-major home a numeral without an accidental names the home's own
    degree and the dialect has no natural sign, so e.g. Em borrowed into C
    minor ('iii') would come back as Ebm), degree_function (tonic / subdominant / dominant of that degree in its
    source, 7-note homes only), in_key, source ('C major', 'C dorian', 'G
    major'), distance (0 in key; else how many of the source's pitch classes lie
    outside the home scale, |k| for a key k steps away), non_home_notes,
    same_notes_as (earlier palette symbols with the identical pitch-class set:
    C6 ~ Am7, Csus2 ~ Gsus4) and sources (every considered source, the home
    included, whose scale holds the chord — borrowed_sources, sorted by distance
    then source order; Scaler's 'shared scales').

    Order ('sorted by complexity'): in-key first by (size, degree, harmonize_melody's
    type preference, chord-table order), then borrowed by (distance, source
    order, degree, size, type preference, table order). A repeat (same root pitch
    class + chord type) keeps its first entry. `limit` > 0 truncates the lists;
    count and borrowed_count are the totals. Returns {key, count,
    borrowed_count, chords, symbols, tokens (the romans; empty for a home that
    is not a 7-note scale)}.

    The in-key palette is set membership over the chord table, presented as in
    the Klimper 2 chords manual (families, complexity order). Mixture from the
    parallel minor follows Aldwell & Schachter and Kostka & Payne ('Mode
    Mixture'); borrowing from the other modes is Berklee modal interchange
    (Nettles & Graf, The Chord Scale Theory & Jazz Harmony, 1997); neighbour
    keys follow the circle of fifths. e.g. chord_palette('C') -> C Dm Em F G Am
    Bdim; chord_palette('C', borrow=True) adds D F#dim Bm (lydian), Edim Gm Bb
    (mixolydian), Cm Ebaug Adim (melodic minor), Ddim Fm Abaug (harmonic major),
    Eb (dorian), Ab (harmonic minor), Db Gdim Bbm (phrygian), Cdim Ebm Gb
    (locrian). Deterministic.
    """
    from .harmony import _TYPE_PREF  # harmony imports this module at load time

    if not isinstance(root, str):
        raise ValueError(f"root must be a note name like 'C' or 'F#', got {root!r}")
    parsed = parse_notes(root)
    if len(parsed) != 1:
        raise ValueError(f"root must be a single note name like 'C' or 'F#', got {root!r}")
    tonic = parsed[0].without_octave()
    home = resolve_scale_type(scale_type)
    for name, value in (("extended", extended), ("sevenths", sevenths), ("include_dyads", include_dyads),
                        ("borrow", borrow)):
        _check_bool(name, value)
    if not isinstance(max_notes, int) or isinstance(max_notes, bool) or not 3 <= max_notes <= 6:
        raise ValueError(f"max_notes must be an integer between 3 and 6, got {max_notes!r}")
    if not isinstance(fifths_steps, int) or isinstance(fifths_steps, bool) or not 0 <= fifths_steps <= 6:
        raise ValueError(f"fifths_steps must be an integer between 0 and 6, got {fifths_steps!r}")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
        raise ValueError(f"limit must be a non-negative integer (0 = all), got {limit!r}")

    if source_modes is not None:
        modes = [source_modes] if isinstance(source_modes, str) else source_modes
        if not isinstance(modes, (list, tuple)):
            raise ValueError(f"source_modes must be a scale name or a list of scale names, got {source_modes!r}")
        for mode in modes:
            if not isinstance(mode, str):
                raise ValueError(f"source_modes must be scale names, got {mode!r}")
            scale = resolve_scale_type(mode)
            if len(scale.intervals) != 7:
                raise ValueError(f"Chords can only be borrowed from 7-note modes; {scale.name!r} has"
                                 f" {len(scale.intervals)} notes")
        modes = list(modes)
    elif borrow:
        modes = list(PARALLEL_MODES)
    else:
        modes = []
    heptatonic = len(home.intervals) == 7
    if (borrow or source_modes is not None or fifths_steps > 0) and not heptatonic:
        raise ValueError(f"Borrowing needs a 7-note home scale; {home.name!r} has {len(home.intervals)}"
                         f" notes (use extended=true for its full in-key palette)")

    sizes = frozenset(range(3, max_notes + 1)) | (frozenset({2}) if include_dyads else frozenset())
    sources = borrowing_sources(tonic, home.name, modes, fifths_steps, include_home=True)
    home_pcs = sources[0]["pcs"]
    table_order = {name: i for i, name in enumerate(CHORDS)}

    raw = []
    for index, src in enumerate(sources):
        for m in _palette_universe(src["tonic"], src["scale"], extended, sevenths, sizes):
            pcs = frozenset(n.pitch_class for n in m["notes"])
            if index and pcs <= home_pcs:
                continue  # the home key already has it
            ctype = m["ctype"]
            size = len(m["notes"])
            pref = _TYPE_PREF.get(ctype.name, 99) if ctype else 100
            order = table_order[ctype.name] if ctype else len(table_order)
            sort = ((0, size, m["degree"], pref, order) if index == 0
                    else (1, src["distance"], index, m["degree"], size, pref, order))
            raw.append((sort, index, m, pcs))
    raw.sort(key=lambda r: r[0])

    key_name = f"{tonic.pitch_class_name} {home.name}"
    seen: set = set()
    by_pcs: dict[frozenset[int], list] = {}
    chords: list[dict] = []
    borrowed_count = 0
    for _sort, index, m, pcs in raw:
        croot, ctype, notes = m["root"], m["ctype"], m["notes"]
        dedup = (croot.pitch_class, ctype.name) if ctype else (croot.pitch_class, None, pcs)
        if dedup in seen:
            continue  # the first entry's `sources` already lists every scale that holds it
        seen.add(dedup)
        src = sources[index]
        symbol = f"{croot.pitch_class_name}{ctype.symbol}" if ctype else [n.pitch_class_name for n in notes]
        entry: dict = {
            "symbol": symbol,
            "root": croot.pitch_class_name,
            "notes": [n.pitch_class_name for n in notes],
            "chord_type": ctype.name if ctype else "unknown",
            "size": len(notes),
            "degree": m["degree"],
            "family": _chord_family(ctype, croot, notes),
            "core": m["core"],
        }
        if heptatonic:
            rel = ctype.pitch_classes if ctype else frozenset(
                (n.pitch_class - croot.pitch_class) % 12 for n in notes)
            minor = 3 in rel and 4 not in rel
            if m["stacked_roman"] is not None and src["tonic"] == tonic:
                roman = m["stacked_roman"]
            else:
                base, accidental = numeral_base(croot, tonic)
                roman = f"{accidental}{base.lower() if minor else base}{roman_suffix(ctype, minor)}"
            entry["roman"] = roman
            entry["degree_function"] = _HARMONIC_FUNCTIONS[m["degree"] - 1]
        entry["in_key"] = index == 0
        entry["source"] = src["label"]
        entry["distance"] = src["distance"]
        outside: list[str] = []
        for n in notes:
            if n.pitch_class not in home_pcs and n.pitch_class_name not in outside:
                outside.append(n.pitch_class_name)
        entry["non_home_notes"] = outside
        entry["same_notes_as"] = list(by_pcs.get(pcs, []))
        entry["sources"] = borrowed_sources(notes, tonic, home.name, modes, fifths_steps, include_home=True)
        by_pcs.setdefault(pcs, []).append(symbol)
        borrowed_count += index != 0
        chords.append(entry)

    shown = chords[:limit] if limit else chords
    if heptatonic:
        _flag_home_misreadings(shown, tonic, home)
    return {
        "key": key_name,
        "count": len(chords),
        "borrowed_count": borrowed_count,
        "chords": shown,
        "symbols": [c["symbol"] for c in shown],
        "tokens": [c["roman"] for c in shown] if heptatonic else [],
    }
