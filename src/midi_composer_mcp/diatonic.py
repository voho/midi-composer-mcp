"""Diatonic chords of a scale and degree-sequence resolution.

These are the atomic building blocks for chord progressions: the LLM decides
*which* degrees to use (the creative part); these functions only report what
chords live on each scale degree and resolve a chosen degree sequence into
concrete chords.
"""

from __future__ import annotations

import re

from .chords import ChordType, identify_chord_quality, match_chords
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
    if len(scale.intervals) == 7:
        from .roman import roman_to_chords  # roman imports this module at load time

        try:
            reading = roman_to_chords([text], root, scale.name)["chords"][0]
        except ValueError as e:
            if "?" in suffix:
                return None  # diatonic_chords' own marker for an unnamed stacked chord
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
