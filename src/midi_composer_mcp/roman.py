"""Roman numerals in a key -> concrete chords: the Roman-numeral dialect layer.

roman_to_chords reads numerals literally (iv = F minor in C, bVII, V7/V, N6,
Ger65, inversions), unlike degrees_to_chords, which reads them as positions in
the scale. progression_library is a fixed, cited table of named progressions
written in the same dialect. The helpers read_chord, figure_for_bass,
applied_reading and special_reading are the shared recognition rules the
analysis tools reuse, so a numeral written here and one read back there agree.

THE DIALECT (shared by diatonic_chords, analyze_progression, progression_library,
chord_palette and next_chords):

(a) An accidental prefix is measured from the PARALLEL MAJOR scale on the tonic
    (the repo, Hookpad and jazz convention): 'bVII' in A minor is G, 'bIII' in C
    is Eb. This is NOT music21's convention, where an accidental alters the key's
    own degree (music21's 'bVII' in A minor is Gb).
(b) Quality symbols follow Kostka & Payne and jazz usage: uppercase + 7 is a
    dominant seventh (IV7 in C = F7), Δ7 a major seventh (IVΔ7 = Fmaj7), lowercase
    + 7 a minor seventh, ° diminished (°7 = dim7), ø half-diminished, + augmented.
    Hookpad and music21 write 'IV7' for Fmaj7; here Fmaj7 is IVΔ7.
(c) A bare 'N' means N6, the Neapolitan sixth (music21 reads it the same way).

Figured-bass inversion shorthand (6, 64, 7, 65, 43, 42), recursive applied chords
(V/V/V), the N6, It6/Fr43/Ger65/Sw43 and Cad64 specials and the minor-mode reading
of degrees 6 and 7 (music21's Minor67Default.QUALITY) are ported from music21's
roman.py (Cuthbert & Ariza, BSD-3-Clause). Applied chords, the Neapolitan and the
augmented sixths follow Kostka, Payne & Almén, Tonal Harmony, and Aldwell &
Schachter, Harmony and Voice Leading.

This module imports diatonic, chords, notes and scales only — never harmony — so
harmony, analysis and server can import it without a cycle.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .chords import CHORDS, ChordType, _lookup_suffix, chord_notes, match_chords, parse_chord_symbol
from .diatonic import borrowed_sources, diatonic_chords, roman_suffix, split_tokens
from .notes import LETTERS, Note, parse_note, parse_notes, transpose
from .scales import MAJOR_DEGREES, ScaleType, resolve_scale_type, scale_notes

# ------------------------------------------------------------------ grammar

_NUMERALS = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7}

# Whole-token specials (case-sensitive) and their normalized names.
_SPECIALS = {
    "N": "N6", "N6": "N6",
    "It6": "It6", "It+6": "It6", "It": "It6",
    "Fr43": "Fr43", "Fr+6": "Fr43", "Fr": "Fr43",
    "Ger65": "Ger65", "Ger+6": "Ger65", "Ger": "Ger65",
    "Sw43": "Sw43",
    "Cad64": "Cad64",
}

# A '/' starts a target only before [#b♯♭]* plus a numeral or 'N'; otherwise it is
# part of the chord (I6/9 is the six-nine chord, not I6 of 9).
_TARGET_SLASH = re.compile(r"/(?=[#b♯♭]*(?:[IViv]|N))")
_SEGMENT = re.compile(r"^([#b♯♭]*)([IViv]+)(.*)$", re.S)
# The figure grammar: one mark, an optional major-seventh sign, a figure.
_QUALITY = re.compile(r"^(°|o|ø|h|\+)?(Δ|M|maj)?(|6|64|7|65|43|42|2)$")
_SEVENTH_FIGURES = ("7", "65", "43", "42", "2")
_FIGURE_INDEX = {"": 0, "7": 0, "6": 1, "65": 1, "64": 2, "43": 2, "42": 3}
_INVERSION_FIGURES = ("64", "65", "43", "42", "6", "7", "2")
# music21's full figured-bass forms and the shorthand this grammar writes for them.
_FULL_FIGURES = {"53": "", "63": "6", "753": "7", "653": "65", "643": "43", "642": "42"}
_FULL_FIGURE = re.compile(r"^(°|o|ø|h|\+)?(Δ|M|maj)?(53|63|753|653|643|642)$")

# The chord types the figure grammar can write (and so the only ones with figures).
_TRIAD_TYPES = ("major", "minor", "diminished", "augmented")
_SEVENTH_TYPES = ("dominant 7", "major 7", "minor 7", "minor major 7", "diminished 7",
                  "half-diminished", "augmented 7", "augmented major 7")

# Scales whose degrees 6 and 7 are read by quality (music21 Minor67Default.QUALITY).
_MINOR_MODES = ("natural minor", "harmonic minor", "melodic minor")

_LOOKALIKES = str.maketrans({"∆": "Δ", "º": "°", "Ø": "ø"})


@dataclass
class _Chord:
    """One realized numeral (internal)."""

    roman: str                 # normalized numeral, including '/target'
    root: Note | None          # None for augmented sixths
    ctype: ChordType | None    # None for augmented sixths
    notes: list[Note]          # root first; augmented sixths bass first
    bass: Note
    figure: str
    inversion: int | None
    special: str | None = None     # 'neapolitan' | 'augmented sixth' | 'cadential 6-4'
    target: "_Chord | None" = None
    enharmonic_symbol: str | None = None

    @property
    def symbol(self):
        if self.ctype is None:
            return [n.pitch_class_name for n in self.notes]
        text = f"{self.root.pitch_class_name}{self.ctype.symbol}"
        if self.bass.pitch_class != self.root.pitch_class:
            text += f"/{self.bass.pitch_class_name}"
        return text


def _steps(a: Note, b: Note) -> int:
    """Letter distance from b up to a (0-6)."""
    return (LETTERS.index(a.letter) - LETTERS.index(b.letter)) % 7


def _is_above(a: Note, b: Note, semitones: int, letters: int) -> bool:
    """True when `a` is the spelled interval (semitones, letter steps) above `b`."""
    return _steps(a, b) == letters and (a.pitch_class - b.pitch_class) % 12 == semitones


def _has_minor_third(ctype: ChordType) -> bool:
    pcs = ctype.pitch_classes
    return 3 in pcs and 4 not in pcs


def _quality(rest: str, upper: bool, token: str) -> tuple[ChordType, str, str | None]:
    """(chord type, figure, normalized quality text or None for a suffix chord) of REST."""
    q = _QUALITY.match(rest)
    if q:
        mark, delta, fig = q.groups()
        dim, half, aug = mark in ("°", "o"), mark in ("ø", "h"), mark == "+"
        if fig in ("6", "64") and (half or delta):
            sign = "ø" if half else "Δ"
            raise ValueError(f"{token!r}: {sign} marks a seventh chord: use 65/43/42 for its inversions")
        seventh = fig in _SEVENTH_FIGURES or (fig == "" and bool(half or delta))
        if fig == "2":
            fig = "42"
        if seventh and fig == "":
            fig = "7"
        if seventh:
            if (dim or half) and delta:
                raise ValueError(f"{token!r}: there is no {'diminished' if dim else 'half-diminished'}"
                                 f" chord with a major seventh in the chord table")
            if dim:
                name = "diminished 7"
            elif half:
                name = "half-diminished"
            elif aug:
                name = "augmented major 7" if delta else "augmented 7"
            elif upper:
                name = "major 7" if delta else "dominant 7"
            else:
                name = "minor major 7" if delta else "minor 7"
        else:
            name = "diminished" if dim else "augmented" if aug else ("major" if upper else "minor")
        norm = ("°" if dim else "ø" if half else "+" if aug else "") + ("Δ" if delta else "") + fig
        return CHORDS[name], fig, norm

    # otherwise REST is a chord-symbol suffix (root position only)
    text = rest.replace("♯", "#").replace("♭", "b")
    if text.startswith("Δ"):
        text = "maj" + text[1:]
    if text == "add6":
        text = "6"
    found = _suffix(text, upper)
    if found is None:
        full = _FULL_FIGURE.match(rest)
        if full:  # the complete figured-bass form of a figure the grammar writes in shorthand
            mark, delta, fig = full.groups()
            short = f"{mark or ''}{delta or ''}{_FULL_FIGURES[fig]}"
            written = token.strip().replace(rest, short, 1) if rest in token else short
            raise ValueError(
                f"{token!r}: {fig} is the complete figured-bass form; write the shorthand"
                f" {written!r} (music21's table: 53 -> '', 63 -> 6, 753 -> 7, 653 -> 65,"
                f" 643 -> 43, 642 -> 42)")
        for fig in _INVERSION_FIGURES:
            head = text[: -len(fig)]
            # a head that is itself a figure (the '6' of '643') is no embellished chord
            if (text.endswith(fig) and head and head not in _INVERSION_FIGURES
                    and _suffix(head, upper) is not None):
                raise ValueError(
                    f"{token!r}: an embellished chord ({head}) takes no inversion figure ('{fig}')"
                    f" — Hookpad's rule; write it in root position, or use a triad/seventh figure"
                    f" (6, 64, 65, 43, 42)")
        raise ValueError(
            f"{token!r}: unknown quality or figure {rest!r}. After the numeral write a figure"
            f" (6, 64, 7, 65, 43, 42), a mark (°, ø, +, Δ) or a chord-symbol suffix"
            f" (sus4, add9, 9, 6/9 ...)")
    pcs = found.pitch_classes
    if not upper and 4 in pcs:
        raise ValueError(f"{token!r}: a lowercase numeral means a minor third, but {rest!r} has a"
                         f" major third — write the numeral in uppercase")
    if upper and 3 in pcs and 4 not in pcs:
        fixed = rest[1:] if rest.startswith("m") and not rest.startswith("maj") else rest
        raise ValueError(f"{token!r}: an uppercase numeral means a major third, but {rest!r} is a"
                         f" minor chord — write the numeral in lowercase (e.g. ii{fixed})")
    return found, "", None


def _suffix(text: str, upper: bool) -> ChordType | None:
    """The chord type of a suffix after a numeral; a lowercase numeral supplies the minor 'm'.

    A suffix that already starts with a minor 'm' gets no second one ('iim7' is
    m7, the minor seventh — never 'mm7', which would fold onto minor-major).
    """
    minor_m = text.startswith("m") and not text.startswith(("maj", "ma7"))
    candidates = [text] if upper or minor_m else ["m" + text, text]
    for cand in candidates:
        found = _lookup_suffix(cand)
        if found is not None:
            return found
    return None


def _realize_numeral(segment: str, tonic: Note, scale: ScaleType, token: str) -> _Chord:
    m = _SEGMENT.match(segment)
    if not m:
        raise ValueError(
            f"Invalid Roman numeral {token!r}: expected an optional accidental (b, #), a numeral"
            f" I-VII (uppercase = major, lowercase = minor) and a figure, e.g. 'V7', 'ii65',"
            f" 'bVII', 'vii°7/V', or a special: N6, It6, Fr43, Ger65, Sw43, Cad64")
    acc, numeral, rest = m.groups()
    if len(acc) > 2:
        raise ValueError(f"{token!r}: at most two accidentals before a numeral")
    sharps = sum(c in "#♯" for c in acc)
    flats = len(acc) - sharps
    if sharps and flats:
        raise ValueError(f"{token!r}: mixed sharps and flats before the numeral")
    upper, lower = numeral.isupper(), numeral.islower()
    if not (upper or lower) or numeral.upper() not in _NUMERALS:
        raise ValueError(f"{token!r}: the numeral must be I-VII written in one case"
                         f" (uppercase = major third, lowercase = minor third)")
    n = _NUMERALS[numeral.upper()]
    ctype, figure, norm = _quality(rest, upper, token)

    if acc:  # measured from the parallel major (dialect rule a)
        semis = MAJOR_DEGREES[n - 1] + sharps - flats
    elif scale.name in _MINOR_MODES and n in (6, 7):  # music21 Minor67Default.QUALITY
        raised = _has_minor_third(ctype)
        semis = (9 if raised else 8) if n == 6 else (11 if raised else 10)
    else:
        semis = scale.intervals[n - 1]
    root = transpose(tonic, semis, n - 1)
    tones = chord_notes(ctype, root)
    k = _FIGURE_INDEX[figure]  # seventh figures always come with a seventh chord type

    minor_case = _has_minor_third(ctype)
    if norm is None:  # a suffix chord keeps its written case (sus/power chords have no third)
        numeral_text = numeral
        quality_text = roman_suffix(ctype, lower)
    else:
        numeral_text = numeral.lower() if minor_case else numeral.upper()
        quality_text = norm
    acc_text = "#" * sharps + "b" * flats
    return _Chord(roman=f"{acc_text}{numeral_text}{quality_text}", root=root, ctype=ctype,
                  notes=tones, bass=tones[k], figure=figure_for_bass(ctype, root, tones[k]) or "",
                  inversion=k)


def _realize_special(name: str, tonic: Note, scale: ScaleType) -> _Chord:
    canon = _SPECIALS[name]
    if canon == "N6":  # major triad on the lowered 2nd, in first inversion
        root = transpose(tonic, 1, 1)
        tones = chord_notes(CHORDS["major"], root)
        return _Chord("N6", root, CHORDS["major"], tones, tones[1], "6", 1, special="neapolitan")
    if canon == "Cad64":  # the tonic triad over its fifth
        ctype = CHORDS["major" if scale.intervals[2] == 4 else "minor"]
        tones = chord_notes(ctype, tonic)
        return _Chord("Cad64", tonic, ctype, tones, tones[2], "64", 2, special="cadential 6-4")
    # augmented sixths: b6 in the bass, the tonic, (2 / b3 / #2), #4 an augmented sixth above
    b6 = transpose(tonic, 8, 5)
    sharp4 = transpose(tonic, 6, 3)
    extra = {"It6": None, "Fr43": transpose(tonic, 2, 1), "Ger65": transpose(tonic, 3, 2),
             "Sw43": transpose(tonic, 3, 1)}[canon]
    notes = [b6, tonic] + ([extra] if extra is not None else []) + [sharp4]
    if canon in ("Ger65", "Sw43"):
        enharmonic = f"{b6.pitch_class_name}7"
    elif canon == "Fr43":
        enharmonic = f"{extra.pitch_class_name}7b5/{b6.pitch_class_name}"
    else:
        enharmonic = None
    figure = {"It6": "6", "Fr43": "43", "Ger65": "65", "Sw43": "43"}[canon]
    return _Chord(canon, None, None, notes, b6, figure, None, special="augmented sixth",
                  enharmonic_symbol=enharmonic)


def _tonicized_key(chord: _Chord, token: str) -> tuple[Note, ScaleType]:
    """The temporary key a target chord defines: its root, major or natural minor."""
    label = chord.symbol if isinstance(chord.symbol, str) else " ".join(chord.symbol)
    if chord.ctype is None:
        raise ValueError(f"{token!r}: cannot tonicize {label} (an augmented sixth has no key)")
    pcs = chord.ctype.pitch_classes
    if 7 not in pcs or not (3 in pcs or 4 in pcs):
        why = "no third" if not (3 in pcs or 4 in pcs) else "diminished/augmented: no perfect fifth"
        raise ValueError(f"{token!r}: cannot tonicize {label} ({why}); only a major or minor chord"
                         f" can be a temporary tonic")
    mode = "major" if 4 in pcs else "natural minor"
    return chord.root, resolve_scale_type(mode)


def _split_segments(token: str) -> list[str]:
    segments = []
    rest = token
    while True:
        m = _TARGET_SLASH.search(rest)
        if m is None:
            segments.append(rest)
            return segments
        segments.append(rest[: m.start()])
        rest = rest[m.end():]


def _realize(token: str, tonic: Note, scale: ScaleType) -> _Chord:
    """Resolve one token; targets are right-associative (V/V/V = V of (V of V))."""
    text = token.strip().translate(_LOOKALIKES)
    segments = _split_segments(text)
    current: _Chord | None = None
    for i in range(len(segments) - 1, -1, -1):
        seg = segments[i]
        key_tonic, key_scale = (tonic, scale) if current is None else _tonicized_key(current, token)
        if seg in _SPECIALS:
            if i != len(segments) - 1:
                raise ValueError(f"{token!r}: {seg} is a whole chord name and takes no '/target'")
            chord = _realize_special(seg, key_tonic, key_scale)
        else:
            chord = _realize_numeral(seg, key_tonic, key_scale, token)
        if current is not None:
            chord.target = current
            chord.roman = f"{chord.roman}/{current.roman}"
        current = chord
    return current


def _tokens(numerals) -> list[str]:
    if isinstance(numerals, str):
        tokens = split_tokens(numerals, bar_lines=True)
    elif isinstance(numerals, (list, tuple)):
        tokens = []
        for item in numerals:
            if not isinstance(item, str):
                raise ValueError(f"Each numeral must be a string like 'V7' or 'ii65', got {item!r}")
            if not item.strip():
                raise ValueError("Empty numeral in the list")
            tokens.append(item.strip())
    else:
        raise ValueError(f"numerals must be a string like 'I V6 vi IV' or a list of numerals,"
                         f" got {type(numerals).__name__}")
    if not tokens:
        raise ValueError("numerals must be a non-empty string like 'I V6 vi IV' or a list of numerals")
    return tokens


def _heptatonic_key(root, scale_type) -> tuple[Note, ScaleType]:
    scale = resolve_scale_type(scale_type)
    if len(scale.intervals) != 7:
        raise ValueError(f"Roman numerals need a 7-note scale; {scale.name!r} has"
                         f" {len(scale.intervals)} notes (use degrees_to_chords for other scales)")
    tonic = parse_notes(root)[0].without_octave()
    return tonic, scale


def roman_to_chords(numerals, root: str, scale_type: str = "major") -> dict:
    """Turn Roman numerals in a key into concrete chords (chromatic numerals included).

    Unlike degrees_to_chords (which reads a numeral as a scale position), every
    mark counts: case is the third, accidentals and figures are applied, and
    applied chords, the Neapolitan and augmented sixths are built. The inverse
    of analyze_progression.

    Dialect: an accidental is measured from the PARALLEL MAJOR ('bVII' in A
    minor = G; music21 would say Gb). Uppercase+7 = dominant 7 ('IV7' in C =
    F7), Δ7 = major 7 ('IVΔ7' = Fmaj7), lowercase+7 = minor 7, ° dim (°7 dim7),
    ø half-dim, + aug. Figures invert: 6/64 (triads), 7/65/43/42 (sevenths) —
    the shorthand; a full figure ('V643') is refused with the shorthand to
    write ('V43'); a bare ø or Δ means the seventh chord ('viiø' = viiø7 —
    music21 reads a bare 'viiø' as a triad). Anything else after the numeral
    is a chord-symbol suffix in root position ('V9', 'Vsus4', 'I6/9', 'IΔ9' =
    Cmaj9, 'Iadd6' = C6, 'ii9' = Dm9, 'iim6' = Dm6, 'iim7' = Dm7 — the 'm'
    after a lowercase numeral is redundant, minor-major is 'iiΔ7'); an
    embellished chord takes no inversion
    figure ('V96' is an error). X/Y is X in the key of Y (major, or natural
    minor for a minor Y), right-associative: 'V7/V' = D7, 'vii°7/V' = F#dim7,
    'V65/vi' = E7/G#, 'V/V/V' = A. Specials: 'N'/'N6' = Db/F, 'It6' = Ab C F#,
    'Fr43' = Ab C D F#, 'Ger65' = Ab C Eb F#, 'Sw43' = Ab C D# F#, 'Cad64' = C/G.
    In natural/harmonic/melodic minor, degrees 6 and 7 without an accidental
    follow the chord (music21 Minor67Default): a minor-third chord takes the
    raised degree, anything else the lowered one — in A minor vi = F#m,
    vii°7 = G#dim7, VI = F, VII = G. The key must be a 7-note scale.

    `numerals` is a string ('I V6 vi IV64 | bVII V7/V', separators: spaces,
    commas, dashes between tokens, '|') or a list of tokens. Returns {key,
    chords, symbols, bass}: per chord its token, normalized roman, symbol,
    root, chord_type (null for augmented sixths), spelled notes (root first;
    augmented sixths bass first), bass, figure, inversion, kind ('diatonic' |
    'borrowed' | 'applied' | 'neapolitan' | 'augmented sixth' | 'cadential
    6-4' | 'chromatic'), applied_to (target symbol), in_key, non_scale_notes,
    fit (outside tones, max 2), borrowed_from (parallel modes holding it) and
    enharmonic_symbol (Ger65 = 'Ab7'). `symbols` ('G7/B'; an augmented sixth
    is its bass-first note array) feed voice_leading, bach_chorale_voicing,
    chords_to_midi and sections. e.g. roman_to_chords('I V6 vi IV64 bVII V7/V
    iv N6 Ger65 Cad64 V7 I', 'C')['symbols'] -> C G/B Am F/C Bb D7 Fm Db/F
    [Ab C Eb F#] C/G G7 C. Deterministic.
    """
    tonic, scale = _heptatonic_key(root, scale_type)
    tokens = _tokens(numerals)
    scale_pcs = {(tonic.pitch_class + i) % 12 for i in scale.intervals}

    chords, symbols, basses = [], [], []
    for token in tokens:
        ch = _realize(token, tonic, scale)
        ordered = [ch.bass] + [n for n in ch.notes if n != ch.bass]
        outside: list[str] = []
        for n in ordered:
            if n.pitch_class not in scale_pcs and n.pitch_class_name not in outside:
                outside.append(n.pitch_class_name)
        in_key = not outside
        borrowed: list[str] = []
        if ch.target is not None:
            kind = "applied"
        elif ch.special is not None:
            kind = ch.special
        elif in_key:
            kind = "diatonic"
        else:
            borrowed = borrowed_sources(ordered, tonic, scale.name)
            kind = "borrowed" if borrowed else "chromatic"
        entry: dict = {
            "token": token,
            "roman": ch.roman,
            "symbol": ch.symbol,
            "root": ch.root.pitch_class_name if ch.root is not None else None,
            "chord_type": ch.ctype.name if ch.ctype is not None else None,
            "notes": [n.pitch_class_name for n in ch.notes],
            "bass": ch.bass.pitch_class_name,
            "figure": ch.figure,
            "inversion": ch.inversion,
            "kind": kind,
        }
        if ch.target is not None:
            entry["applied_to"] = ch.target.symbol if isinstance(ch.target.symbol, str) \
                else " ".join(ch.target.symbol)
        entry["in_key"] = in_key
        entry["non_scale_notes"] = outside
        entry["fit"] = min(len(outside), 2)
        if kind == "borrowed":
            entry["borrowed_from"] = borrowed
        if ch.special == "augmented sixth":
            entry["enharmonic_symbol"] = ch.enharmonic_symbol
        chords.append(entry)
        symbols.append(ch.symbol)
        basses.append(ch.bass.pitch_class_name)
    return {"key": f"{tonic.pitch_class_name} {scale.name}", "chords": chords,
            "symbols": symbols, "bass": basses}


# ------------------------------------------------- shared recognition rules

def _omitted_fifth(written: list[Note], bass: Note) -> tuple[Note, ChordType, Note] | None:
    """(root, chord type, restored fifth) of a written set that no table chord matches, or None.

    The omission four-part writing allows (Aldwell & Schachter; Kostka & Payne:
    a seventh chord, or a doubled-root triad, may leave out its fifth — never
    its third), read by letters + semitones: a candidate root (the bass first,
    then the written order) with no note on its fifth's letter gets its perfect
    fifth back, and the set must then SPELL a triad or seventh chord (the types
    the figure grammar writes) on that root exactly — G B F = G7, D F C = Dm7,
    C B E = Cmaj7, C E = C. Ab C F# stays unread (its F# is no seventh of Ab:
    an It6), and so does a rootless jazz voicing such as F A B E (no four-part
    chord). The rule check_voice_leading and find_cadences use, made
    spelling-aware.
    """
    if not 2 <= len(written) <= 3:  # a triad or seventh chord less its fifth
        return None
    for cand in [bass] + [n for n in written if n != bass]:
        fifth_letter = LETTERS[(LETTERS.index(cand.letter) + 4) % 7]
        if any(n.letter == fifth_letter for n in written):
            continue  # a note stands on the fifth's letter already: nothing is omitted
        fifth = transpose(cand, 7, 4)
        names = {n.name for n in written} | {fifth.name}
        for name in _TRIAD_TYPES + _SEVENTH_TYPES:
            if {t.name for t in chord_notes(CHORDS[name], cand)} == names:
                return cand, CHORDS[name], fifth
    return None


def read_chord(item) -> dict:
    """Read a chord symbol or note array into {root, chord_type, bass, notes, written}.

    A symbol ('G7/B') gives its root, type, spelled chord tones and the slash
    bass (the root when there is none). A note array is kept as WRITTEN
    (octave-less, in order, repeats dropped); its bass is the lowest MIDI note
    when every note carries an octave, otherwise the first note (the
    bach_chorale_voicing convention). Its root and type are the exact
    match_chords reading taken over that bass (root position first) whose
    spelling is the written one — or the first exact reading. A set that no
    table chord matches is read with its omitted fifth restored when that
    spells a triad or seventh chord exactly (G2 B3 F3 G4 = G7, D F C = Dm7,
    C E = C — the omission four-part writing allows; the reading then adds
    omitted_fifth, the restored Note, while `notes` stay as written);
    otherwise root and chord_type are None (It6 Ab C F#, a cluster, a
    rootless F A B E). A dict that is already a reading is returned as is.
    Values are Note / ChordType objects (internal helper for the analysis
    tools).
    """
    if isinstance(item, dict) and {"root", "chord_type", "bass", "notes"} <= item.keys():
        return item
    if isinstance(item, str):
        root, ctype, bass = parse_chord_symbol(item)
        root = root.without_octave()
        return {"root": root, "chord_type": ctype, "notes": chord_notes(ctype, root),
                "bass": (bass or root).without_octave(), "written": False}
    if isinstance(item, (list, tuple)):
        parsed = parse_notes(list(item))
        if all(n.octave is not None for n in parsed):
            low = min(parsed, key=lambda n: n.midi)
        else:
            low = parsed[0]
        bass = low.without_octave()
        written: list[Note] = []
        for n in parsed:
            plain = n.without_octave()
            if plain not in written:
                written.append(plain)
        ordered = [bass] + [n for n in written if n != bass]
        matches = match_chords([n.name for n in ordered], include_partial=False, limit=50)["matches"]
        names = {n.name for n in written}
        best = next((m for m in matches if set(m["notes"]) == names), matches[0] if matches else None)
        if best is None:
            restored = _omitted_fifth(written, bass)
            if restored is not None:
                root, ctype, fifth = restored
                return {"root": root, "chord_type": ctype, "notes": written, "bass": bass, "written": True,
                        "omitted_fifth": fifth}
        root = parse_note(best["root"]) if best else None
        ctype = CHORDS[best["chord_type"]] if best else None
        return {"root": root, "chord_type": ctype, "notes": written, "bass": bass, "written": True}
    raise ValueError(f"Invalid chord: {item!r} (use a symbol like 'G7/B' or a note array)")


def figure_for_bass(chord_type, root: Note, bass: Note | None) -> str | None:
    """The figured-bass inversion figure of a chord over a bass note.

    The bass is read as a chord member by its interval above the root: the 3rd
    gives '6' (triad) or '65' (seventh chord), the 5th '64' or '43', the 7th
    '42', the root '' (triad) or '7' (seventh chord). Only the chord types the
    figure grammar can write get inversion figures — the triads (major, minor,
    diminished, augmented) and the seventh chords (7, Δ7, m7, mΔ7, °7, ø7, +7,
    +Δ7). Any other type (sus, add, 6, 9, 11, 13, 7b5 ...) is '' in root
    position and None when inverted; a bass that is no chord tone gives None
    (so Csus4/F is not '6'). music21's figure shorthand (53 -> '', 63 -> 6,
    64, 753 -> 7, 653 -> 65, 643 -> 43, 642 -> 42). `chord_type` is a ChordType
    or a type name; None gives None.
    """
    if chord_type is None:
        return None
    ctype = chord_type if isinstance(chord_type, ChordType) else CHORDS.get(chord_type)
    if ctype is None:
        return None
    triad, seventh = ctype.name in _TRIAD_TYPES, ctype.name in _SEVENTH_TYPES
    if bass is None or bass.pitch_class == root.pitch_class:
        return "7" if seventh else ""
    if not (triad or seventh):
        return None
    rel = (bass.pitch_class - root.pitch_class) % 12
    for label, semis in ctype.degrees:
        if semis % 12 == rel:
            number = int(label.lstrip("#b"))
            return {3: ("6", "65"), 5: ("64", "43"), 7: (None, "42")}.get(number, (None, None))[seventh]
    return None


def _key(root, scale_type) -> tuple[Note, ScaleType, set[int]]:
    scale = resolve_scale_type(scale_type)
    tonic = (root if isinstance(root, Note) else parse_notes(root)[0]).without_octave()
    return tonic, scale, {(tonic.pitch_class + i) % 12 for i in scale.intervals}


def applied_target_row(x: dict) -> bool:
    """True when a diatonic_chords row can be an applied chord's target x.

    x must be a major or minor triad other than the tonic, stacked in thirds on
    its own degree. A row whose stacked notes are no table chord ('?' in its
    roman), which diatonic_chords can only rename as another root's inversion
    (C enigmatic's degree 2 Db F# A# = 'F#/Db'), is skipped: its numeral
    ('bII') names a different chord than the one heard (F#).
    """
    return (x["degree"] != 1 and x["chord_type"] in ("major", "minor")
            and "?" not in x.get("roman", "") and "bass" not in x)


def applied_reading(chord, root, scale_type: str = "major") -> dict | None:
    """Read an out-of-key chord as an applied (secondary) V or vii° chord, or return None.

    The rule shared by analyze_progression and next_chords (Kostka & Payne:
    applied V and vii° chords). Only a chord that is NOT in the key is read (so
    a diatonic I is never 'V/IV'), and only in a 7-note key:
    - a major triad or a dominant-family chord (major 3rd + minor 7th: 7, 9,
      13, 7b9, 7#9, 7#11, 7b5 ...) whose root is a spelled P5 above the root of
      a diatonic major or minor triad x (x not the tonic) is 'V<suffix>/x';
    - a dim, dim7 or m7b5 chord whose root is a spelled m2 below such an x is
      'vii°/x', 'vii°7/x' or 'viiø7/x'.
    The target is x's diatonic numeral without its suffix ('ii', 'V', 'bVII');
    x is a triad stacked on its own degree (applied_target_row), so a scale
    row that only reads as another root's inversion is never a target. In
    natural, harmonic and melodic minor a major/dominant chord on ^5, or a
    dim/dim7/m7b5 chord on the raised ^7, is the minor mode's own dominant:
    {'applied': None, 'function_note': 'harmonic-minor dominant'}.

    Returns {applied ('V7/ii'), numeral ('V7'), target ('ii'), target_symbol
    ('Dm'), target_root, target_degree, type ('dominant' | 'leading-tone')}.
    `chord` is a symbol, a note array or a read_chord reading. e.g.
    applied_reading('A7', 'C') -> applied 'V7/ii'; applied_reading('C', 'C') -> None.
    """
    r = read_chord(chord)
    tonic, scale, scale_pcs = _key(root, scale_type)
    ctype, croot = r["chord_type"], r["root"]
    if len(scale.intervals) != 7 or ctype is None or croot is None:
        return None
    tone_pcs = {n.pitch_class for n in r["notes"]} | {r["bass"].pitch_class}
    if tone_pcs <= scale_pcs:
        return None
    pcs = ctype.pitch_classes
    dominant = ctype.name == "major" or {4, 10} <= pcs
    leading = ctype.name in ("diminished", "diminished 7", "half-diminished")
    if not (dominant or leading):
        return None
    rel = (croot.pitch_class - tonic.pitch_class) % 12
    if scale.name in _MINOR_MODES and ((dominant and rel == 7) or (leading and rel == 11)):
        return {"applied": None, "function_note": "harmonic-minor dominant"}
    for x in diatonic_chords(tonic.name, scale.name)["chords"]:
        if not applied_target_row(x):
            continue
        xroot = parse_note(x["root"])
        if (dominant and _is_above(croot, xroot, 7, 4)) or (leading and _is_above(croot, xroot, 11, 6)):
            target = re.match(r"^[#b]*[IViv]+", x["roman"]).group(0)
            numeral = ("V" + roman_suffix(ctype, False)) if dominant else ("vii" + roman_suffix(ctype, True))
            return {"applied": f"{numeral}/{target}", "numeral": numeral, "target": target,
                    "target_symbol": x["symbol"], "target_root": x["root"],
                    "target_degree": x["degree"], "type": "dominant" if dominant else "leading-tone"}
    return None


_AUG_SIXTHS = (("It6", frozenset({8, 0, 6})), ("Fr43", frozenset({8, 0, 2, 6})),
               ("Ger65", frozenset({8, 0, 3, 6})))


def special_reading(chord, root, scale_type: str = "major", next_chord=None) -> dict | None:
    """Recognize a Neapolitan, an augmented sixth or a cadential 6/4, or return None.

    - Neapolitan: a major triad on the lowered 2nd (spelled: Db in C, not C#):
      {'special': 'Neapolitan', 'roman_figured': 'N6'} in first inversion,
      otherwise 'bII' plus its figure ('bII', 'bII64').
    - Augmented sixths: pitch classes {b6, 1, #4} (It6), plus 2 (Fr43), b3 (Ger65)
      or #2 (Sw43), with b6 in the bass. A WRITTEN note array whose bass is on
      the 6th letter and whose #4 is on the 4th letter (an augmented sixth above
      the bass) is {'special': 'Ger65', 'roman_figured': 'Ger65'}; Sw43 is the
      Ger65 sound with its third spelled as #2 (D# in C). The same sound written
      as a symbol ('Ab7') or misspelled gives {'enharmonic_to': 'Ger65'}.
    - Cad64: a tonic triad with its 5th in the bass, followed by `next_chord`
      on ^5 with a major 3rd: {'special': 'Cad64', 'roman_figured': 'Cad64',
      'function_note': 'dominant (cadential 6/4)'} (Aldwell & Schachter; the
      caller keeps the chord's own function).
    Rules per music21 romanNumeralFromChord and Kostka & Payne. `chord` and
    `next_chord` are symbols, note arrays or read_chord readings.
    """
    r = read_chord(chord)
    tonic, scale, _pcs = _key(root, scale_type)
    bass = r["bass"]
    rel = {(n.pitch_class - tonic.pitch_class) % 12 for n in r["notes"]}
    rel.add((bass.pitch_class - tonic.pitch_class) % 12)
    if (bass.pitch_class - tonic.pitch_class) % 12 == 8:
        for name, pcset in _AUG_SIXTHS:
            if rel != pcset:
                continue
            sharp4 = [n for n in r["notes"] if (n.pitch_class - tonic.pitch_class) % 12 == 6]
            if r["written"] and _steps(bass, tonic) == 5 and sharp4 and all(_steps(n, tonic) == 3 for n in sharp4):
                if name == "Ger65":
                    third = [n for n in r["notes"] if (n.pitch_class - tonic.pitch_class) % 12 == 3]
                    if third and all(_steps(n, tonic) == 1 for n in third):
                        name = "Sw43"
                return {"special": name, "roman_figured": name}
            return {"enharmonic_to": name}
    ctype, croot = r["chord_type"], r["root"]
    if ctype is None or croot is None:
        return None
    if ctype.name == "major" and _is_above(croot, tonic, 1, 1):
        fig = figure_for_bass(ctype, croot, bass)
        if fig is not None:
            return {"special": "Neapolitan", "roman_figured": "N6" if fig == "6" else f"bII{fig}"}
    if (next_chord is not None and ctype.name in ("major", "minor")
            and croot.pitch_class == tonic.pitch_class and figure_for_bass(ctype, croot, bass) == "64"):
        nxt = read_chord(next_chord)
        nroot, ntype = nxt["root"], nxt["chord_type"]
        if (nroot is not None and ntype is not None and 4 in ntype.pitch_classes
                and (nroot.pitch_class - tonic.pitch_class) % 12 == 7):
            return {"special": "Cad64", "roman_figured": "Cad64",
                    "function_note": "dominant (cadential 6/4)"}
    return None


# ------------------------------------------------------ progression library

_GALANT = "Robert Gjerdingen, Music in the Galant Style (2007)"
_LAITZ = "Steven Laitz, The Complete Musician: harmonic sequences"
_LEVINE = "Mark Levine, The Jazz Theory Book (1995)"
_POP = "Common pop idiom, named descriptively"
_GROUND = "Standard Renaissance/Baroque ground bass (Grove Music Online)"

# name: (aliases, mode, category, numerals, description, source, bass_degrees, melody_degrees)
# bass_degrees are scale degrees; a chromatic bass note is a string with its accidental ('#1').
PROGRESSIONS: dict[str, tuple] = {
    # --- pop ----------------------------------------------------------------
    "pop_axis": (("axis", "I-V-vi-IV", "axis progression"), "major", "pop", "I V vi IV",
                 "The 'four-chord' axis loop behind countless pop songs.", _POP, None, None),
    "doo_wop": (("50s", "50s progression", "I-vi-IV-V"), "major", "pop", "I vi IV V",
                "The 1950s doo-wop loop.", _POP, None, None),
    "circle": (("vi-ii-V-I", "circle progression"), "major", "pop", "vi ii V I",
               "Roots falling by fifths into the tonic.", _POP, None, None),
    "royal_road": (("ōdō shinkō", "odo shinko", "royal road progression", "IV-V-iii-vi"), "major", "pop",
                   "IVΔ7 V7 iii7 vi", "The J-pop 'royal road' (ōdō shinkō): subdominant, dominant, then"
                   " a deceptive turn through iii to vi.", _POP, None, None),
    "mario_cadence": (("bVI-bVII-I",), "major", "pop", "bVI bVII I",
                      "Two borrowed major chords stepping up into the tonic (the 'Mario cadence').",
                      _POP, None, None),
    "double_plagal": (("bVII-IV-I",), "major", "pop", "bVII IV I",
                      "Two plagal (down-a-fourth) motions in a row: bVII to IV to I.", _POP, None, None),
    "plagal_amen": (("amen", "amen cadence", "plagal cadence", "IV-I"), "major", "pop", "IV I",
                    "The plagal 'amen' close.", _POP, None, None),
    "andalusian": (("andalusian cadence", "i-VII-VI-V"), "minor", "pop", "i VII VI V",
                   "The descending minor tetrachord i-VII-VI-V ending on the major dominant. VII and VI"
                   " are the lowered degrees in minor (Minor67), the textbook spelling.", _POP, None, None),
    "minor_pop": (("i-VI-III-VII",), "minor", "pop", "i VI III VII",
                  "The minor-key pop loop (the axis loop started on vi).", _POP, None, None),
    # --- jazz -----------------------------------------------------------------
    "ii_V_I": (("ii-V-I", "two-five-one", "2-5-1"), "major", "jazz", "ii7 V7 IΔ7",
               "The major ii-V-I cadence, the core jazz progression.", _LEVINE, None, None),
    "jazz_turnaround": (("turnaround", "I-vi-ii-V"), "major", "jazz", "IΔ7 vi7 ii7 V7",
                        "The I-vi-ii-V turnaround that loops back to the tonic.", _LEVINE, None, None),
    "backdoor": (("backdoor ii-V", "backdoor progression"), "major", "jazz", "iv7 bVII7 IΔ7",
                 "The backdoor ii-V: iv7 to bVII7, resolving up a step into I.", _LEVINE, None, None),
    "ragtime": (("ragtime progression", "III7-VI7-II7-V7"), "major", "jazz", "III7 VI7 II7 V7 I",
                "A chain of secondary dominants round the circle of fifths into the tonic.",
                _LEVINE, None, None),
    "minor_ii_V_i": (("minor ii-V-i", "minor two-five-one"), "minor", "jazz", "iiø7 V7 i",
                     "The minor ii-V-i: half-diminished ii, dominant V7 (raised leading tone), tonic i.",
                     _LEVINE, None, None),
    # --- blues ----------------------------------------------------------------
    "twelve_bar_blues": (("12-bar blues", "twelve-bar blues", "blues"), "major", "blues",
                         "I7 I7 I7 I7 IV7 IV7 I7 I7 V7 IV7 I7 V7",
                         "The 12-bar blues, one chord per bar, with a V7 turnaround in the last bar.",
                         "Common 12-bar blues form", None, None),
    # --- classical / early ------------------------------------------------------
    "lament": (("lament bass", "descending tetrachord", "lamento"), "minor", "classical", "i v6 iv6 V",
               "The lament bass: the descending tetrachord ^1-^7-^6-^5 harmonized i v6 iv6 V.",
               "Ellen Rosand, 'The Descending Tetrachord: An Emblem of Lament', Musical Quarterly 65"
               " (1979)", None, None),
    "folia": (("la folia", "folía"), "minor", "early", "i V i VII III VII i V",
              "La Folia, the Iberian ground varied by Corelli, Vivaldi and many others.", _GROUND, None, None),
    "passamezzo_antico": (("passamezzo",), "minor", "early", "i VII i V III VII i V i",
                          "The passamezzo antico ground of 16th-century dance music.", _GROUND, None, None),
    # --- sequences (Laitz) -------------------------------------------------------
    "descending_fifths": (("diatonic circle", "D2 (-5/+4)", "circle of fifths sequence"), "major",
                          "sequence", "I IV vii° iii vi ii V I",
                          "The descending-fifths sequence (D2, -5/+4): every diatonic root in turn down"
                          " a fifth / up a fourth.", _LAITZ, None, None),
    "pachelbel": (("canon", "D3 (-4/+2)", "pachelbel canon"), "major", "sequence", "I V vi iii IV I IV V",
                  "Pachelbel's Canon in D: the descending 5-6 sequence (D3, -4/+2) closing IV V.",
                  "Pachelbel, Canon in D; " + _LAITZ, None, None),
    "ascending_5_6": (("A2 (-3/+4)", "ascending 5-6"), "major", "sequence", "I vi6 ii vii°6 iii I6 IV",
                      "The ascending 5-6 sequence (A2, -3/+4) over a stepwise rising bass.",
                      _LAITZ, None, None),
    # --- galant schemata (Gjerdingen) ---------------------------------------------
    "prinner": ((), "major", "schema", "IV I6 vii°6 I",
                "The Prinner: a stepwise descent 6-5-4-3 over 4-3-2-1, the galant 'riposte'.",
                _GALANT, (4, 3, 2, 1), (6, 5, 4, 3)),
    "meyer": ((), "major", "schema", "I V43 V65 I",
              "The Meyer: melody 1-7 ... 4-3 over bass 1-2 ... 7-1, an opening gambit.",
              _GALANT, (1, 2, 7, 1), (1, 7, 4, 3)),
    "romanesca": ((), "major", "schema", "I V6 vi I6",
                  "The Romanesca: the bass steps down 1-7-6 and leaps to 3, the melody 1-5-1-1.",
                  _GALANT, (1, 7, 6, 3), (1, 5, 1, 1)),
    "do_re_mi": (("do-re-mi",), "major", "schema", "I V6 I",
                 "The Do-Re-Mi: melody 1-2-3 over bass 1-7-1.", _GALANT, (1, 7, 1), (1, 2, 3)),
    # Gjerdingen, App. A: bass 7-1-2-3 against melody 4-3-7-1, sonorities 6/5/3, 5/3, 6/3, 6/3.
    "fenaroli": ((), "major", "schema", "V65 I vii°6 I6",
                 "The Fenaroli: melody 4-3-7-1 against bass 7-1-2-3, ending on a 6/3 (Gjerdingen: the"
                 " 6/3 ending gives it its lack of finality).",
                 _GALANT, (7, 1, 2, 3), (4, 3, 7, 1)),
    # Gjerdingen, App. A: a local 4-3 in ii, then in I (5-4 | 4-3 in the home key) over #1-2 | 7-1.
    "fonte": ((), "major", "schema", "V65/ii ii V65 I",
              "The Fonte: a sequence down a step, first to ii (minor), then to I (major); melody"
              " 5-4-4-3 over bass #1-2-7-1.",
              _GALANT, ("#1", 2, 7, 1), (5, 4, 4, 3)),
    "monte": ((), "major", "schema", "V65/IV IV V65/V V",
              "The Monte: a sequence up a step, first to IV, then to V.", _GALANT, (3, 4, "#4", 5), None),
}

PROGRESSION_CATEGORIES = ("pop", "jazz", "blues", "classical", "early", "sequence", "schema")


def _lookup_key(text: str) -> str:
    return re.sub(r"[\s_\-–]+", "_", text.strip().lower())


_PROGRESSION_LOOKUP: dict[str, str] = {}
for _name, _spec in PROGRESSIONS.items():
    for _key_text in (_name, *_spec[0]):
        _k = _lookup_key(_key_text)
        if _PROGRESSION_LOOKUP.setdefault(_k, _name) != _name:  # pragma: no cover - table check
            raise RuntimeError(f"progression alias {_key_text!r} is ambiguous")


def _entry(name: str) -> dict:
    aliases, mode, category, numerals, description, source, bass_degrees, melody_degrees = PROGRESSIONS[name]
    entry = {"name": name, "aliases": list(aliases), "mode": mode, "category": category,
             "numerals": numerals, "description": description, "source": source}
    if bass_degrees is not None:
        entry["bass_degrees"] = list(bass_degrees)
    if melody_degrees is not None:
        entry["melody_degrees"] = list(melody_degrees)
    return entry


def _degree_note(degree, tonic: Note, scale: ScaleType) -> Note:
    """A table degree (4, or '#1' with its accidental) read in a key, spelled on its letter."""
    notes = scale_notes(scale, tonic)
    if isinstance(degree, str):
        shift = degree.count("#") - degree.count("b")
        return transpose(notes[int(degree.lstrip("#b")) - 1], shift, 0)
    return notes[degree - 1]


def _label(symbol) -> str:
    return symbol if isinstance(symbol, str) else " ".join(symbol)


def _resolve_entry(entry: dict, root: str, scale_type: str | None) -> dict:
    home = "major" if entry["mode"] == "major" else "natural minor"
    scale_name = scale_type if scale_type is not None else home
    realized = roman_to_chords(entry["numerals"], root, scale_name)
    key = realized["key"]
    entry["key"] = key
    entry["chords"] = realized["symbols"]
    entry["bass"] = realized["bass"]
    tonic = parse_notes(root)[0].without_octave()
    scale = resolve_scale_type(scale_name)
    warnings: list[str] = []
    scale_mode = "major" if scale.intervals[2] == 4 else "minor"
    if scale_mode != entry["mode"]:
        warnings.append(f"{entry['name']} is a {entry['mode']}-mode progression resolved in {key}"
                        f" (a {scale_mode} scale)")
    if scale.name != home:
        # the numerals follow the requested scale's own degrees: name every chord that leaves it
        # although the entry's home mode holds it (the table's own chromatic chords stay silent)
        at_home = roman_to_chords(entry["numerals"], tonic.name, home)["chords"]
        moved = [f"{c['token']} = {_label(c['symbol'])} ({', '.join(c['non_scale_notes'])})"
                 for c, h in zip(realized["chords"], at_home) if h["in_key"] and not c["in_key"]]
        if moved:
            warnings.append(f"these chords leave {key} although {tonic.pitch_class_name} {home} holds them"
                            f" (the numerals follow the scale's own degrees): {'; '.join(moved)}")
    if "bass_degrees" in entry:  # the chords' bass must be the table's bass line read in this key
        want = [_degree_note(d, tonic, scale).pitch_class_name for d in entry["bass_degrees"]]
        if want != realized["bass"]:
            warnings.append(f"the chords' bass {' '.join(realized['bass'])} is not bass_degrees read in"
                            f" {key} ({' '.join(want)})")
    if "melody_degrees" in entry:
        from .melody import notes_from_degrees  # melody imports midi_io; keep it lazy
        melody = notes_from_degrees(tonic.name, scale_name, entry["melody_degrees"])["notes"]
        clashes = [f"{note} against {_label(c['symbol'])}" for note, c in zip(melody, realized["chords"])
                   if parse_note(note).pitch_class not in {parse_note(n).pitch_class for n in c["notes"]}]
        if clashes:  # one texture or none: a melody that contradicts its chords is not handed on
            warnings.append(f"melody withheld: melody_degrees read in {key} give {' '.join(melody)},"
                            f" which clashes with the chords ({'; '.join(clashes)})")
        else:
            entry["melody"] = melody
    if warnings:
        entry["warning"] = "; ".join(warnings)
    return entry


def progression_library(name: str | None = None, root: str | None = None,
                        scale_type: str | None = None, category: str | None = None) -> dict:
    """A fixed, cited table of named progressions in roman_to_chords' dialect.

    Each entry: name, aliases, mode (major | minor), category (pop | jazz |
    blues | classical | early | sequence | schema), numerals, description and
    source; the galant schemata also give bass_degrees, and melody_degrees
    where the melody is diatonic. `name` matches a name or alias ignoring case
    ('Canon', 'D2 (-5/+4)', 'ii-V-I') and returns that entry; without a name
    every entry (optionally of one `category`) is listed, sorted by (category,
    name). With `root`, entries are resolved through roman_to_chords — in
    `scale_type`, by default major for major entries and natural minor for
    minor ones — adding key, chords (symbols), bass (pitch classes, exact
    even for the chromatic Fonte and Monte basses) and, for schemata with
    melody_degrees, melody. Any scale is allowed, and `warning` says what
    changed: a minor entry in a major scale (or the reverse); chords that
    leave the requested scale although the entry's home mode holds them (IV
    in lydian is F#: 'IV = F# (A#, C#)'); a chord bass that is no longer
    bass_degrees read in that scale; and a melody that clashes with its
    chords, which is then withheld (melody, bass and chords always describe
    one texture). No popularity data is stored.
    e.g. progression_library('andalusian', root='E') -> chords Em D C B;
    progression_library('prinner', root='G') -> chords C G/B F#dim/A G, bass
    C B A G, melody E D C B. Deterministic.
    """
    if name is not None and not isinstance(name, str):
        raise ValueError(f"name must be a progression name or alias, got {name!r}")
    if category is not None:
        if not isinstance(category, str) or category.strip().lower() not in PROGRESSION_CATEGORIES:
            raise ValueError(f"Unknown category {category!r}. Known categories:"
                             f" {', '.join(PROGRESSION_CATEGORIES)}")
        category = category.strip().lower()
    if root is not None and not isinstance(root, str):
        raise ValueError(f"root must be a note name like 'C' or 'F#', got {root!r}")
    if scale_type is not None:
        if root is None:
            raise ValueError("scale_type needs a root: pass root to resolve the progressions in a key")
        resolve_scale_type(scale_type)  # validate early, with the scale list in the message

    if name is not None:
        key = _PROGRESSION_LOOKUP.get(_lookup_key(name))
        if key is None:
            raise ValueError(f"Unknown progression {name!r}. Known progressions:"
                             f" {', '.join(sorted(PROGRESSIONS))}")
        entry = _entry(key)
        if category is not None and entry["category"] != category:
            raise ValueError(f"{key} is in category {entry['category']!r}, not {category!r}")
        return _resolve_entry(entry, root, scale_type) if root is not None else entry

    entries = [_entry(n) for n in PROGRESSIONS if category is None or PROGRESSIONS[n][2] == category]
    entries.sort(key=lambda e: (e["category"], e["name"]))
    if root is not None:
        entries = [_resolve_entry(e, root, scale_type) for e in entries]
    return {"count": len(entries), "categories": list(PROGRESSION_CATEGORIES), "progressions": entries}
