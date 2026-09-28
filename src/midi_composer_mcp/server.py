"""MIDI composer MCP server.

Every tool is an atomic, deterministic step (lookups, matching, seeded dice
rolls, rendering). The creative work — choosing scales, progressions,
melodies, rhythms and arrangements — is left to the caller, who chains the
tools: outputs (note arrays, chord symbols, rhythm patterns) feed directly
into other tools' inputs, all the way from an idea to a multi-track MIDI file.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from . import accompany as _accompany
from . import analysis as _analysis
from . import audio as _audio
from . import chant as _chant
from . import chords as _chords
from . import circle as _circle
from . import counterpoint as _counterpoint
from . import diatonic as _diatonic
from . import generate as _generate
from . import harmony as _harmony
from . import masters as _masters
from . import melody as _melody
from . import midi_io as _midi
from . import modern as _modern
from . import roman as _roman
from . import scales as _scales
from . import structure as _structure
from . import voicing as _voicing

mcp = FastMCP(
    "midi-composer",
    instructions=(
        "Atomic, deterministic music-theory and MIDI tools for composing a whole"
        " song from an idea. Layers, each with simple generators you chain:"
        " SCALES (get_scale, match_scales), CHORDS (get_chord, diatonic_chords,"
        " degrees_to_chords, match_chords, chord_palette, roman_to_chords,"
        " progression_library), HARMONY (next_chords, voice_leading, voice_chords),"
        " MELODY (notes_from_degrees,"
        " arpeggiate, melodic_walk, motif_grammar, random_notes, transpose_notes),"
        " ACCOMPANIMENT (chord_pattern, bass_line),"
        " RHYTHM (random_rhythm, euclidean_rhythm), HISTORICAL MELODY (church_mode,"
        " solmization, guido_vowel_melody, check_melody, cantus_firmus), NAMED HARMONY"
        " RULES (rameau_fundamental_bass, schoenberg_progressions, bach_chorale_voicing,"
        " neo_riemannian, bartok_axis, coltrane_changes), MODERN (twelve_tone_matrix,"
        " pitch_class_set, additive_process, phase_shift), ANALYSIS (detect_key,"
        " analyze_progression, find_cadences, check_voice_leading), STRUCTURE (plan_sections,"
        " arrange_song), and rendering (notes/chords/drums/arrange_to_midi,"
        " midi_to_audio). Notes are strings like 'C', 'F#', 'Bb' — add an octave"
        " for concrete pitches ('C5', 'Eb3'; C4 is middle C); octave-less notes"
        " are pitch classes and matching ignores octaves. Every tool's"
        " note/chord/degree/rhythm output feeds other tools — the tools make no"
        " creative choices, you do. Whole-song flow: pick a scale -> build"
        " progressions (degrees_to_chords / roman_to_chords / progression_library,"
        " grown step by step with next_chords) -> voice them (voice_leading /"
        " voice_chords) -> melodies (notes_from_degrees / motif_grammar /"
        " melodic_walk) -> accompaniment (chord_pattern, bass_line) and grooves"
        " (euclidean_rhythm) -> check the draft (check_voice_leading, find_cadences,"
        " detect_key) -> assemble each section's tracks -> arrange_song to sequence"
        " intro/verse/chorus/bridge/outro into one multi-track file -> midi_to_audio"
        " to hear it."
    ),
)


# ------------------------------------------------------------------ lookups

@mcp.tool()
def list_scales() -> dict:
    """List every scale type in the database: intervals, degrees, aliases and a description of each.

    Covers common, modal, jazz, symmetric and exotic/world scales. Use a
    scale type's name (or any alias) with get_scale, match_scales,
    diatonic_chords and degrees_to_chords.
    """
    return _scales.list_scales()


@mcp.tool()
def list_chords() -> dict:
    """List every chord type in the database: intervals, degrees, symbol suffixes, aliases and a description of each.

    Covers triads, sixths, sevenths, extended and altered chords. Use a chord
    type's name or symbol suffix with get_chord, and root+suffix symbols
    (e.g. 'Am', 'G7', 'F#m7b5', 'Cmaj13') anywhere a chord symbol is accepted.
    """
    return _chords.list_chords()


@mcp.tool()
def get_scale(scale_type: str, root: str | None = None) -> dict:
    """Describe a scale type (with a one-line description); with a root note, generate its notes.

    Without `root`: intervals and degree labels only (e.g. major = 0 2 4 5 7 9 11).
    With `root`: the spelled notes, e.g. get_scale('major', 'C') -> C D E F G A B C.
    Give the root an octave for concrete pitches: get_scale('dorian', 'D4') ->
    D4 ... D5 plus MIDI numbers. The returned `notes` array feeds directly into
    match_chords, random_notes, notes_to_midi, arrange_to_midi, etc.
    """
    return _scales.scale_info(scale_type, root)


@mcp.tool()
def get_chord(chord_type: str, root: str | None = None) -> dict:
    """Describe a chord type (with a one-line description); with a root note, generate its notes.

    Without `root`: intervals and degrees only (e.g. minor 7 = 0 3 7 10).
    With `root`: the spelled chord, e.g. get_chord('min', 'F') -> F Ab C.
    Give the root an octave for concrete pitches: get_chord('9', 'C4') ->
    C4 E4 G4 Bb4 D5 plus MIDI numbers. The returned `notes` array feeds
    directly into random_notes, notes_to_midi, match_scales, etc.
    """
    return _chords.chord_info(chord_type, root)


# ----------------------------------------------------------------- matching

@mcp.tool()
def match_scales(notes: str | list[str], exact_only: bool = False, limit: int = 20) -> dict:
    """Find scales that contain all of the given notes (octaves are ignored).

    `notes` is a list like ['C', 'E', 'G'] or a string 'c e g' — notes with
    octaves like ['C5','E5','G5'] work identically, the octave is dropped. A
    match is 'exact' when the input uses every note of the scale; otherwise
    'contains', with the scale's extra notes listed in `added_notes`. Exact
    and tighter (smaller) scales sort first.
    """
    return _scales.match_scales(notes, exact_only=exact_only, limit=limit)


@mcp.tool()
def match_chords(notes: str | list[str], include_partial: bool = True, limit: int = 20) -> dict:
    """Find chords that match the given notes (octaves are ignored).

    `notes` accepts plain or octave-bearing notes (['E','G','C'] or
    ['E4','G4','C5']) — the octave is dropped. 'exact' matches use exactly the
    input pitch classes; when the first input note is not the chord root the
    inversion is reported with slash notation (e.g. 'E G C' -> C/E, first
    inversion). 'partial' matches contain all input notes plus `missing_notes`.
    """
    return _chords.match_chords(notes, include_partial=include_partial, limit=limit)


# ------------------------------------------------------------- harmony rules

@mcp.tool()
def circle_of_fifths(root: str | None = None) -> dict:
    """The circle of fifths: key signatures, relative minors, and related keys.

    Without `root`: the twelve keys with their sharp/flat signatures, relative
    minors and enharmonic spellings. With a key `root` (e.g. 'C', 'Bb', 'F#'):
    its dominant and subdominant, relative and parallel minors, and the closely
    related keys — the natural targets for a modulation or a contrasting bridge.
    """
    return _circle.circle_of_fifths(root)


@mcp.tool()
def interval_between(note_a: str, note_b: str) -> dict:
    """Name the interval from note_a up to note_b (semitones + quality, e.g. C->Eb = minor third)."""
    return _harmony.interval_between(note_a, note_b)


@mcp.tool()
def analyze_progression(chords: str | list, root: str, scale_type: str = "major") -> dict:
    """Analyze a chord progression into Roman numerals in a key — figures, applied chords, specials, mixture.

    The inverse of roman_to_chords and degrees_to_chords. Given chords (symbols
    like ['C','A7','Dm','G7/B'] or note arrays: with octaves the lowest note is
    the bass, without them the first) and a key, each chord gets its roman
    numeral, degree, in_key (every tone in the scale, else non_scale_notes, in
    the written spelling) and harmonic function. A note array of 2-3 distinct
    notes that no table chord matches is read with its omitted fifth restored
    when that spells a triad or seventh chord (bach_chorale_voicing's fifthless
    V7 G2 F3 B3 G4 = V7, the shell D F C = ii7, the dyad C E = I), named in
    omitted_fifth. Plus: figure ('', '6', '64', '7', '65', '43', '42'; null
    with bass_degree when the bass is no writable chord member, e.g. Csus4/F,
    F/G); roman_figured,
    the numeral roman_to_chords reads back ('I6', 'V65', 'V65/ii', 'N6',
    'Ger65', 'Cad64'; textbook minor numerals VI/VII). roman_figured carries no
    bass the figure grammar cannot write: when figure is null the bass is kept
    only in bass_degree (F/G in C -> 'IV', bass_degree 5), so re-add it as a
    slash when moving the draft to another key. It is null when that dialect
    cannot write the root, and for an unnamed cluster (C Db D: roman 'I?',
    roman_figured null). applied, for out-of-key V/x and vii°/x chords (A7 in C
    = 'V7/ii', F#dim7 = 'vii°7/V'; in minor V and vii° get function_note
    'harmonic-minor dominant' instead); special ('Neapolitan';
    'It6'/'Fr43'/'Ger65'/'Sw43' for a written note array with b6 in the bass;
    'Cad64' for a tonic 6/4 before V, function still 'tonic'); enharmonic_to
    (the symbol 'Ab7' sounds like Ger65); borrowed_from for the other
    out-of-key chords: the parallel modes whose own notes spell the chord,
    judged by letters (Fm in C: C harmonic major, C harmonic minor, C natural
    minor, C phrygian, C locrian; G# in C: [] — C natural minor has Ab, not
    G#). e.g. ['C','A7','Dm','G7/B','C'] in C -> roman_figured I V7/ii ii V65
    I. To move a draft to another key, feed its roman_figured list to
    roman_to_chords there (re-adding any bass_degree slash basses).
    """
    return _harmony.analyze_progression(chords, root, scale_type)


@mcp.tool()
def next_chords(chords: str | list, root: str, scale_type: str = "major", sevenths: bool = False,
                include_chromatic: bool = True, sort: str = "rule", octave: int = 4,
                limit: int = 16) -> dict:
    """Rank the chords that may follow the last one, by Piston's Table of Usual Root Progressions (1941) — rules, not probabilities.

    The last chord of `chords` picks a row of the table (VI is followed by II
    or V, sometimes III or IV, less often I; one table for major and minor).
    Tiers: 0 resolution (an applied chord's target: A7 -> Dm; after an
    augmented sixth It6/Fr43/Ger65/Sw43 the major V — V7 with sevenths=true —
    and the cadential 6/4 'Cad64': V first after It6 and Fr43, Cad64 first
    after Ger65 and Sw43, per Kostka & Payne, Aldwell & Schachter), 1 usual, 2
    sometimes, 3 less often (diatonic chords on the listed degrees; triads, or
    sevenths with sevenths=true; natural minor adds the raised V/vii°), 4
    applied (V7/x of the row's usual and sometimes targets), 5 mixture (major
    keys only: ii°, bIII, iv, bVI from the parallel minor, plus bVII), 6
    unlisted (a mixture chord on a degree the row does not list). The last
    chord itself is never offered (a repetition is no root progression: after
    D7 the list has no D7). include_chromatic=false keeps tiers 0-3. Each
    candidate: symbol, token (roman_to_chords dialect; null for a stacked chord
    that dialect cannot write, such as G B Db on V of C double harmonic — the
    `tokens` list stays aligned with `symbols`), roman, notes, tier,
    tier_name, rule (the table row), common_tones, root_motion (Schoenberg:
    ascending (strong) / descending / superstrong), movement (semitones from
    the last chord's voice_leading voicing to the candidate's best inversion)
    and in_key. sort='rule' (by tier) or 'movement' (smoothest first); limit
    1-64. The key must have 7 notes. e.g. next_chords(['C','Am'], 'C') -> Dm G
    (usual), F Em (sometimes), C (less often), A7 D7 B7 C7 (applied), Ddim Fm
    Eb (mixture), Ab Bb (unlisted); next_chords(['C','D7'], 'C',
    sort='movement') starts F, C7, E7. Pick one, append it and call again to
    grow a progression; symbols feed voice_leading and chords_to_midi.
    """
    return _harmony.next_chords(chords, root, scale_type, sevenths, include_chromatic, sort, octave, limit)


@mcp.tool()
def voice_leading(chords: str | list, octave: int = 4) -> dict:
    """Voice a chord progression smoothly — each chord takes the inversion nearest the last.

    Minimizes movement between chords and keeps common tones, like a real
    keyboard comp. Returns voiced note lists (with octaves) per chord; feed the
    `chords` (arrays of notes) into chords_to_midi or an arrange/song chords
    track for natural-sounding pads instead of parallel root-position blocks.
    """
    return _harmony.voice_leading(chords, octave=octave)


@mcp.tool()
def voice_chords(chords: str | list, style: str = "drop2", octave: int = 4,
                 top_notes: str | list[str] | None = None, connect: bool = True) -> dict:
    """Voice a progression in an arranging style — drop-2/3/2&4, open, shell or Levine rootless — connected smoothly.

    Styles: 'close' (exactly voice_leading), 'drop2' / 'drop3' / 'drop24' (four
    voices in close position, then the 2nd / 3rd / 2nd+4th voice from the top
    dropped an octave; a triad doubles its bottom voice, R-3-5-R; a chord of 5+
    tones omits the 5th, then the root), 'open' (bass in octave-1, the 5th above
    it, the other members stacked up), 'shell' (bass + 3rd + 7th: A = R-3-7,
    B = R-7-3; a 6-chord uses its 6th, a triad its 5th; a slash bass goes under
    the whole shell: C/E -> E3 C4 G4), 'rootless_a' / 'rootless_b' / 'rootless'
    (Mark Levine's left-hand A/B voicings, The Jazz Piano Book: major/minor
    3-5-7-9 / 7-9-3-5, a named #11/11/13 in the 5th's slot; dominant 3-13-b7-9
    / b7-9-3-13 with the chord's own b9/#9, and its altered 5th, #11 or b13 in
    the 13's slot; m7b5 b3-b5-b7-1; bottom note at or above D(octave-1);
    triads, sus chords (a dominant 11th too) and dim7 raise). 'rootless'
    chains from each first form and keeps the smaller total_movement (A on a
    tie), so a ii-V-I alternates A-B-A or B-A-B. connect=true chains greedily
    with voice_leading's cost and tie-breaks (least movement, common tones
    held); connect=false takes rotation 0 / variant A. top_notes (one per
    chord, e.g. a melody: 'E5' exact, 'E' any octave) forces each voicing's
    highest note. A slash bass stays lowest. Returns voicings [{symbol, notes,
    midi, variant, degrees}], `chords` for chords_to_midi or
    check_voice_leading(voicings=...), total_movement and a render_hint chords track.
    e.g. voice_chords(['Dm7','G7','Cmaj7'], 'rootless') -> F3 A3 C4 E4 | F3 A3 B3 E4 | E3 G3 B3 D4.
    """
    return _voicing.voice_chords(chords, style=style, octave=octave, top_notes=top_notes, connect=connect)


@mcp.tool()
def secondary_dominant(target: str, chord_type: str = "7") -> dict:
    """The secondary dominant (V/x) of a target chord — e.g. secondary_dominant('Dm') -> A7.

    Returns the dominant chord a fifth above the target, the standard way to
    tonicize a non-tonic chord. Pairs well with analyze_progression for reharm.
    """
    return _harmony.secondary_dominant(target, chord_type=chord_type)


@mcp.tool()
def tritone_substitute(symbol: str) -> dict:
    """The tritone substitution of a dominant chord — e.g. tritone_substitute('G7') -> Db7.

    A dominant a tritone away shares the same guide tones, giving a chromatic
    bass descent (G7->C becomes Db7->C). A staple jazz reharmonization.
    """
    return _harmony.tritone_substitute(symbol)


@mcp.tool()
def negative_harmony(notes: str | list[str], tonic: str) -> dict:
    """Reflect notes through a key's negative-harmony axis (major <-> minor shadow).

    Ernst Levy's mirror (popularized by Jacob Collier): each pitch is reflected
    around the axis between the tonic and its fifth, flipping a progression's
    colour while preserving its function. Works on a melody or a chord's notes.
    """
    return _harmony.negative_harmony(notes, tonic)


@mcp.tool()
def harmonize_melody(notes: str | list[str], root: str | None = None, scale_type: str = "major",
                     in_scale: bool = False, max_chord_notes: int = 4, allow_repeats: bool = True,
                     options: int = 4, octave: int = 4, step_beats: float = 2.0) -> dict:
    """Put chords under a melody, each chord reusing as many notes from the previous one as possible.

    For every melody note it searches the WHOLE chord database (all chord types,
    all roots) for chords that contain that note, ranks them by how many notes
    they share with the previous chord (most-shared first = smoothest voice
    leading), auto-picks the top one, and runs voice_leading for the inversions.
    Each note also lists the top `options` ranked alternatives. Deterministic.

    Pass `root`/`scale_type` with `in_scale=true` to keep the harmony in a key
    (any chord whose notes fit the scale — not just the seven diatonic chords);
    otherwise every chord is fair game. `max_chord_notes` caps complexity (4 =
    triads and sevenths). `allow_repeats=false` forces a different chord on each
    note. Returns the chosen progression with voiced notes + per-note options,
    and a `render_hint` (harmony track + melody on top) for arrange_to_midi.
    """
    return _harmony.harmonize_melody(notes, root=root, scale_type=scale_type, in_scale=in_scale,
                                     max_chord_notes=max_chord_notes, allow_repeats=allow_repeats,
                                     options=options, octave=octave, step_beats=step_beats)


# ----------------------------------------------------------------- diatonic

@mcp.tool()
def diatonic_chords(root: str, scale_type: str, sevenths: bool = False) -> dict:
    """List the chord built on each degree of a scale (triads, or sevenths).

    E.g. diatonic_chords('C', 'major') -> I=C, ii=Dm, iii=Em, IV=F, V=G,
    vi=Am, vii°=Bdim. Seven-note scales also get roman numerals, degree names
    and harmonic functions (tonic/subdominant/dominant) — the raw material for
    designing a progression yourself; then resolve it with degrees_to_chords.
    A root with an octave (e.g. 'C4') yields concrete pitches with MIDI numbers.
    """
    return _diatonic.diatonic_chords(root, scale_type, sevenths)


@mcp.tool()
def degrees_to_chords(root: str, scale_type: str, degrees: str | list[int | str],
                      sevenths: bool = False) -> dict:
    """Resolve a chord-degree sequence you chose into concrete chords of a scale.

    `degrees` is your sequence as numbers or roman numerals: [1, 5, 6, 4],
    'I V vi IV' or '1-5-6-4'. Returns the chord (symbol + notes) on each
    chosen degree, in order — e.g. in C major: C, G, Am, F. The `symbols`
    array feeds directly into chords_to_midi / song_to_midi / arrange_to_midi.
    Numerals are *positions*: case, quality marks and figures are ignored, so
    'iv' in C major is still F and 'V7' without sevenths=true is still G (an
    accidental must name the scale's own degree, like bVI in natural minor; a
    chromatic one such as bVII in major is a ValueError). The result's
    `warnings` list has one string per token whose mark was ignored, e.g.
    degrees_to_chords('C', 'major', 'IV iv I') -> F F C with "iv resolved to F
    (the scale's own chord); for F minor use roman_to_chords". To read
    numerals literally (iv = Fm, bVII, V7/V, N6, inversions) use
    roman_to_chords. This tool only maps degrees to chords; choosing the
    degrees is up to you.
    """
    return _diatonic.degrees_to_chords(root, scale_type, degrees, sevenths)


@mcp.tool()
def chord_palette(root: str, scale_type: str = "major", extended: bool = False, sevenths: bool = False,
                  max_notes: int = 4, include_dyads: bool = False, borrow: bool = False,
                  source_modes: str | list[str] | None = None, fifths_steps: int = 0,
                  limit: int = 0) -> dict:
    """Every chord that fits a key, simplest first (a Klimper-style palette), plus optional borrowed chords.

    Pure set arithmetic, any scale. Default: the scale's stacked triads
    (sevenths=true: seventh chords) — chord_palette('C') -> C Dm Em F G Am Bdim.
    extended=true: every chord type of the table (sus2, sus4, 6, add9, 7sus4 ...
    up to max_notes 3-6; include_dyads adds power chords) rooted on a scale note
    whose notes all lie in the scale — 42 chords in C major, 17 with
    max_notes=3. Judged by letters + semitones: in a 7-note scale a chord
    counts only when the scale's own notes spell it (C harmonic minor has Ebaug
    but no Abm, which would need Cb); other scale sizes go by pitch class and
    spell such a chord from its own root (whole tone's Daug = D F# A#).
    borrow=true adds modal interchange from the parallel modes (lydian,
    mixolydian, dorian, natural minor, phrygian, locrian, harmonic/melodic minor,
    harmonic major), or only from `source_modes`; fifths_steps=n (0-6) borrows
    from the same mode n keys round the circle, each neighbour key spelled on
    the home's letters so its tokens read alike in every key (from Cb major the
    key a fifth down gives Gbm 'v', not F#m; only its source label uses the
    practical name, 'E major' for Fb major) and the tritone key listed once. A
    borrowed chord has a note outside the key: chord_palette('C', borrow=True)
    adds D F#dim Bm (lydian), Edim Gm Bb (mixolydian), Cm Ebaug Adim (melodic
    minor), Ddim Fm Abaug (harmonic major), Eb (dorian), Ab (harmonic minor),
    Db Gdim Bbm (phrygian), Cdim Ebm Gb (locrian); fifths_steps=1 in C adds Bm
    D F#dim (G major) and Gm Bb Edim (F major). Per chord: symbol, notes,
    chord_type, size, degree, family (major/minor/other), core (the stacked
    chord), roman against the home tonic (7-note keys: 'bVI', 'iv', '#iv°',
    'Isus4'), degree_function, in_key, source ('C dorian'), distance (the
    source's notes outside the key), non_home_notes, same_notes_as (earlier
    entries with the same notes: Am7 -> C6) and sources (every considered scale, the home included, whose own notes
    spell it, e.g. Fm: C harmonic major, harmonic minor, natural minor,
    phrygian, locrian). In-key chords come first by (size, degree), borrowed
    ones by distance. `symbols` feed voice_leading / chords_to_midi, `tokens`
    feed roman_to_chords (in a non-major key an entry with `roman_note` reads
    back only in the parallel major). limit>0 truncates; count is the total.
    """
    return _diatonic.chord_palette(root, scale_type=scale_type, extended=extended, sevenths=sevenths,
                                   max_notes=max_notes, include_dyads=include_dyads, borrow=borrow,
                                   source_modes=source_modes, fifths_steps=fifths_steps, limit=limit)


# ----------------------------------------------------------- roman numerals

@mcp.tool()
def roman_to_chords(numerals: str | list[str], root: str, scale_type: str = "major") -> dict:
    """Turn Roman numerals in a key into concrete chords — chromatic, applied, Neapolitan, augmented sixths, inversions.

    Unlike degrees_to_chords (a numeral = a scale position), every mark counts:
    case is the third (iv in C = Fm), an accidental is measured from the
    PARALLEL MAJOR (bVII = Bb in C and G in A minor), figures invert (V6 = G/B,
    I64 = C/G, ii65 = Dm7/F, V42 = G7/F), uppercase+7 is a dominant 7 (IV7 = F7)
    and Δ7 a major 7 (IVΔ7 = Fmaj7), a bare ø/Δ is the seventh chord (viiø =
    Bm7b5), other suffixes are root-position chords (V9, Vsus4, I6/9, IΔ9 =
    Cmaj9, Iadd6 = C6). X/Y is X in the key of Y: V7/V = D7, vii°7/V = F#dim7,
    V65/vi = E7/G#, V/V/V = A. Specials: N (= N6, Db/F), It6, Fr43, Ger65, Sw43
    (note arrays, bass first) and Cad64 (C/G). In minor keys vi/vii° take the
    raised degree and VI/VII the lowered one (A minor: vi = F#m, vii°7 = G#dim7,
    VII = G). Needs a 7-note scale. Separators: spaces, commas, dashes, '|'.
    e.g. roman_to_chords('I V6 vi IV64 bVII V7/V iv N6 Ger65 Cad64 V7 I', 'C')
    -> symbols C G/B Am F/C Bb D7 Fm Db/F [Ab C Eb F#] C/G G7 C, with per chord
    its kind (diatonic/borrowed/applied/neapolitan/augmented sixth/cadential
    6-4/chromatic), borrowed_from (the parallel modes whose own notes spell it,
    judged by letters: Fm in C comes from C harmonic major, harmonic minor,
    natural minor, phrygian, locrian; III7 = E7 in C is kind 'chromatic', since
    no parallel mode spells G#), non_scale_notes, fit, figure and bass.
    Figures are the shorthand ('V43'; the full 'V643' is refused, naming the
    shorthand), and 'iim7' = 'ii7' = Dm7. `symbols` feed voice_leading,
    bach_chorale_voicing, chords_to_midi and song sections; slash basses stay
    in the bass.
    """
    return _roman.roman_to_chords(numerals, root, scale_type)


@mcp.tool()
def progression_library(name: str | None = None, root: str | None = None,
                        scale_type: str | None = None, category: str | None = None) -> dict:
    """Named chord progressions from a fixed, cited table: pop, jazz, blues, classical, early, sequences, galant schemata.

    28 presets in roman_to_chords' dialect — pop_axis (I V vi IV), doo_wop,
    circle, royal_road, mario_cadence, double_plagal, plagal_amen, andalusian
    (i VII VI V), minor_pop; ii_V_I, jazz_turnaround, backdoor, ragtime,
    minor_ii_V_i; twelve_bar_blues; lament; folia, passamezzo_antico; Laitz's
    sequences descending_fifths, pachelbel, ascending_5_6; Gjerdingen's
    schemata prinner, meyer, romanesca, do_re_mi, fenaroli, fonte, monte (with
    bass_degrees, and melody_degrees for all but the Monte). `name` matches a
    name or alias ignoring case ('Canon', 'ii-V-I', 'D2 (-5/+4)'); without it
    every entry (of one `category`) is listed. Give `root` (and optionally
    `scale_type`; default major / natural minor by the entry's mode) to resolve
    the chords, the bass line and a schema's melody in a key. Any scale is
    allowed; `warning` then says what changed: a minor entry in a major scale
    (or the reverse), chords that leave the scale although the home mode holds
    them (andalusian in D dorian: 'VI = B (D#, F#)'), a bass that is no longer
    bass_degrees read there, and 'melody withheld: ...' when the melody would
    clash with its chords (the `melody` key is then absent). e.g.
    progression_library('andalusian', root='E') -> chords Em D C B;
    progression_library('prinner', root='G') -> chords C G/B F#dim/A G, bass C
    B A G, melody E D C B; progression_library('fenaroli', root='C') -> V65 I
    vii°6 I6 = G7/B C Bdim/D C/E, bass B C D E, melody F E B C. Feed `chords`
    to voice_leading / chords_to_midi, `numerals` to roman_to_chords in any
    key, a schema's bass/melody to notes tracks or
    bach_chorale_voicing(melody=...).
    """
    return _roman.progression_library(name=name, root=root, scale_type=scale_type, category=category)


# --------------------------------------------------------------- randomness

@mcp.tool()
def random_notes(notes: str | list[str], count: int = 4, allow_repeats: bool = True,
                 seed: int | None = None) -> dict:
    """Pick `count` uniformly random notes from a pool of notes (a pure dice roll).

    The pool is any notes array — typically from get_scale or get_chord, e.g.
    random notes from A minor pentatonic. Octaves in the pool are kept.
    Reproducible via `seed`; the seed used is always returned.
    """
    return _generate.random_notes(notes, count=count, allow_repeats=allow_repeats, seed=seed)


@mcp.tool()
def random_rhythm(length: int = 8, density: float = 0.65,
                  accent_probability: float = 0.35, seed: int | None = None) -> dict:
    """Roll a random rhythm pattern of `length` steps (a pure dice roll).

    Returns a pattern string like 'O...Oo..' where O = strong beat, o = weak
    beat, . = pause. `density` is the chance a step holds a note;
    `accent_probability` the chance a note is strong. The pattern feeds the
    `rhythm` argument of notes_to_midi / arrange_to_midi and drum lanes; you
    can also edit it by hand. Reproducible via `seed`, which is always returned.
    """
    return _generate.random_rhythm(length=length, density=density,
                                   accent_probability=accent_probability, seed=seed)


@mcp.tool()
def euclidean_rhythm(pulses: int, steps: int = 16, rotation: int = 0) -> dict:
    """Build a Euclidean rhythm: `pulses` onsets spread as evenly as possible over `steps`.

    Euclidean rhythms underlie countless grooves worldwide — e.g.
    euclidean_rhythm(3, 8) is the tresillo 'O..o..o.', euclidean_rhythm(5, 8)
    the cinquillo. The downbeat onset is 'O', other onsets 'o', gaps '.';
    `rotation` shifts the pattern. Deterministic. The pattern feeds the
    `rhythm` argument of notes_to_midi / arrange_to_midi or any drum lane —
    great for basslines and percussion.
    """
    return _generate.euclidean_rhythm(pulses=pulses, steps=steps, rotation=rotation)


@mcp.tool()
def groove(name: str) -> dict:
    """Return a named rhythm preset (four_on_floor, backbeat, tresillo, son_clave_32, bossa_nova...).

    A library of idiomatic rhythm cells as O/o/. patterns from world and popular
    music. The pattern feeds the `rhythm` of a notes track or a drum lane (repeat
    it to fill more bars). Use list_grooves to browse them all.
    """
    return _generate.groove(name)


@mcp.tool()
def list_grooves() -> dict:
    """List every named rhythm preset (clave, bossa, tresillo, dembow, four-on-the-floor...) with descriptions."""
    return _generate.list_grooves()


# ------------------------------------------------------------------ melody

@mcp.tool()
def notes_from_degrees(root: str, scale_type: str, degrees: str | list[int | str]) -> dict:
    """Write a melody as scale degrees and get back concrete notes (deterministic).

    Degree 1 is the root; degrees past the scale length wrap up an octave
    (8 = root +8ve, 9 = 2nd +8ve), negatives go below. e.g.
    notes_from_degrees('C', 'major', [1,2,3,5,8]) -> C D E G C;
    notes_from_degrees('A', 'minor pentatonic', '1 3 4 5 7'). Lets you design a
    melodic contour once and transpose it to any key/scale. The `notes` output
    feeds a notes track, the rhythm tools, or transpose_notes.
    """
    return _melody.notes_from_degrees(root, scale_type, degrees)


@mcp.tool()
def arpeggiate(notes: str | list[str], style: str = "up", octaves: int = 1,
               seed: int | None = None) -> dict:
    """Reorder a chord or scale into an arpeggio/broken-chord sequence (deterministic).

    `style`: up, down, updown, downup, converge (outside-in), diverge
    (inside-out), or random (seeded). `octaves` stacks octave copies first
    (needs notes with octaves, e.g. a chord from get_chord('min','A4')). Feed a
    chord's notes to build an arpeggio line or a broken-chord bass; the output
    is a note list for a notes track.
    """
    return _melody.arpeggiate_notes(notes, style=style, octaves=octaves, seed=seed)


@mcp.tool()
def melodic_walk(notes: str | list[str], length: int = 8, seed: int | None = None,
                 max_step: int = 2, start: int = 0) -> dict:
    """Generate a singable, stepwise melody by a seeded random walk over a note ladder.

    `notes` is an ordered pitch ladder — usually a scale (pass it over two
    octaves for more range, e.g. via notes_from_degrees with degrees 1..15).
    Each step moves up/down by at most `max_step` rungs, so the line is mostly
    conjunct. Unlike random_notes (uniform jumps), this produces melodic
    contour. Reproducible via `seed`, always returned. Pair with a rhythm.
    """
    return _melody.melodic_walk(notes, length=length, seed=seed, max_step=max_step, start=start)


@mcp.tool()
def melodic_sequence(notes: str | list[str], root: str, scale_type: str,
                     step: int = -1, count: int = 3) -> dict:
    """Repeat a motif as a diatonic sequence, shifting it by scale steps each time.

    Restates a motif at successive pitch levels within `root`/`scale_type` — e.g.
    step=-1, count=4 walks it down one scale degree per repeat (the classic
    descending sequence). Diatonic motif notes stay in key; a chromatic note keeps
    its offset from the scale note below it. The first copy is the motif itself.
    """
    return _melody.melodic_sequence(notes, root, scale_type, step=step, count=count)


@mcp.tool()
def transpose_notes(notes: str | list[str], semitones: int) -> dict:
    """Transpose a note list by semitones, keeping octaves (deterministic).

    Useful for key changes, moving a motif, or building a sequence by repeating
    a phrase a step higher. e.g. transpose_notes(['C4','E4','G4'], 5) -> F4 A4 C5.
    """
    return _melody.transpose_notes(notes, semitones)


@mcp.tool()
def motif_grammar(form: str | list[str], motifs: dict, kind: str = "notes") -> dict:
    """Build a melody or rhythm from a motif grammar like 'ABAC' (repetition + variation).

    Each letter of `form` names a motif in `motifs`; a motif is a literal
    sequence or a variation of another. This is how phrases are made: AA repeats,
    AB contrasts, ABA rounds off, ABAC develops. Deterministic.

    `kind`: 'notes' (motifs are note lists/strings -> a note list), 'degrees'
    (scale-degree lists -> a degree list for notes_from_degrees), or 'rhythm'
    (patterns -> one concatenated rhythm string). A variation references another
    label and applies transforms in order retrograde -> invert -> rotate ->
    transpose, e.g. {"vary":"A","transpose":2} (semitones for notes, scale steps
    for degrees), {"vary":"A","retrograde":true}, {"vary":"A","invert":true},
    {"vary":"A","rotate":1}.

    Example: form='ABAC', motifs={"A":"C5 D5 E5 G5", "B":{"vary":"A","transpose":2},
    "C":{"vary":"A","retrograde":true}} -> a rounded, developing 16-note phrase.
    """
    return _melody.motif_grammar(form, motifs, kind=kind)


@mcp.tool()
def snap_to_scale(notes: str | list[str], root: str, scale_type: str) -> dict:
    """Snap every note to the nearest note of a scale, so a melody fits the key/chords.

    Guarantees compatibility: any line — hand-written, transposed, or generated
    — becomes diatonic to `root`/`scale_type`, staying consonant with chords
    drawn from that scale. Octaves are kept, ties snap down, in-scale notes are
    untouched. Deterministic. Use it as a safety net after editing or
    transposing a melody, or to fit borrowed material into the current key.
    """
    return _melody.snap_to_scale(notes, root, scale_type)


@mcp.tool()
def tintinnabuli_voice(melody: str | list[str], triad: str | list[str],
                       position: str = "superior", rank: int = 1, octave: int = 5) -> dict:
    """Arvo Pärt's tintinnabuli: derive a triad-note counter-voice that shadows a melody.

    Pärt pairs a stepwise melodic voice (M-voice) with a tintinnabuli voice
    (T-voice) that always sounds a note of a fixed `triad` (classically the
    tonic) — the nearest triad pitch `superior` (above), `inferior` (below) or
    `alternating` per note; `rank` 1 = nearest, 2 = second-nearest (T1/T2). Pass
    the melody (e.g. from melodic_walk or notes_from_degrees over the scale) and
    a triad ('Am' or ['A','C','E']). Returns aligned `m_voice` and `t_voice`
    lists — render them as two notes tracks sharing one rhythm (a bell-like
    program such as 8/9/11/14 suits the T-voice). The T-voice is consonant by
    construction, so it is always compatible with the harmony.
    """
    return _melody.tintinnabuli_voice(melody, triad, position=position, rank=rank, octave=octave)


@mcp.tool()
def counterpoint(cantus: str | list[str], root: str, scale_type: str = "major",
                 species: int = 1, position: str = "above") -> dict:
    """Write counterpoint to a cantus firmus in any of the five species (Fux), deterministic.

    Given your melody (`cantus`) and a key, derives a counter-melody `above` or
    `below` it that follows the classical rules. `species`: 1 = note-against-note
    (1:1), 2 = 2:1 with passing tones, 3 = 4:1 with passing/neighbour figures,
    4 = syncopated suspensions (a tied dissonance resolving down by step),
    5 = florid (a mix). Consonances on strong beats, every dissonance treated as
    a stepwise passing/neighbour tone or a resolved suspension, perfect-consonance
    ending on the tonic, and no parallel/direct fifths or octaves. Returns both
    voices, the per-bar downbeat intervals, and a `render_hint` with ready
    notes-track specs (cantus as whole notes; the counterpoint with its rhythm)
    you can drop straight into arrange_to_midi.
    """
    return _counterpoint.species_counterpoint(cantus, root, scale_type,
                                              species=species, position=position)


# ------------------------------------------------ historical melody rules

@mcp.tool()
def church_mode(mode: int | str) -> dict:
    """One of the eight Gregorian church modes: final, reciting tone (tenor) and range (ambitus).

    `mode` is 1-8, I-VIII or a name: Dorian, Hypodorian, Phrygian, Hypophrygian,
    Lydian, Hypolydian, Mixolydian, Hypomixolydian. Authentic modes span the
    octave above the final; plagal ('hypo-') modes sit a fourth lower with the
    same final. Chant melodies circle the tenor and end on the final. Returns the
    ambitus as notes (feed them to melodic_walk or guido_vowel_melody).
    """
    return _chant.church_mode(mode)


@mcp.tool()
def solmization(notes: str | list[str]) -> dict:
    """Sing a melody in Guido of Arezzo's hexachord syllables (ut re mi fa sol la) with mutations.

    Hexachords naturale (on C), durum (on G, B = mi) and molle (on F, B♭ = fa);
    mi–fa is always the semitone. The melody stays in one hexachord while it can
    and mutates on a shared pivot note by the rule 'per re sursum, per la
    deorsum' — rising, the pivot becomes re; falling, la (C D E F G A B♭ c = ut
    re mi fa re mi fa sol). Notes outside the gamut are musica ficta, sung in a
    transposed hexachord (raised = mi, lowered = fa: D E F♯ G = ut re mi fa).
    """
    return _chant.solmization(notes)


@mcp.tool()
def guido_vowel_melody(text: str, mode: int | str = 1, rows: int = 1) -> dict:
    """Guido of Arezzo's vowel method (Micrologus, c. 1026): turn a text into a chant melody.

    Guido wrote the vowels a e i o u under the notes of the scale from gamma
    upward, repeating (the table sits in the modes' octave, so the Dorian final
    D carries u, as in Guido); `rows=2` adds his second row, starting on B.
    Each syllable (Latin syllabification: que-ant, De-us) is sung on the pitch
    carrying its vowel nearest the previous note within the church mode's
    range, and the melody closes on the mode's final. E.g.
    guido_vowel_melody('Ut queant laxis resonare fibris', 'Dorian').
    """
    return _chant.guido_vowel_melody(text, mode, rows)


@mcp.tool()
def check_melody(notes: str | list[str], root: str, scale_type: str = "major", strict: bool = True) -> dict:
    """Check a melody against the cantus-firmus rules of species-counterpoint teaching (after Fux, Jeppesen).

    Errors: start/end on the tonic, final by step, diatonic (a raised 7th may be
    the penultimate note in aeolian/dorian/mixolydian), no repeated notes, only
    2nds/3rds/4ths/5ths/ascending minor 6th/octave (no tritones, 7ths, major
    6ths), leaps over a third recovered by a step back, at most two leaps in a
    row (same-direction leaps outlining a consonance), a single climax, range
    within a tenth, leading tone to tonic. Warnings: length, too few steps,
    tritones outlined between turning points, long scale runs, repeated
    figures, an unraised whole-step 7–1 cadence. `strict=False` turns the leap
    rules into warnings (Fux's own F- and G-mode cantus firmi break them).
    Write octaves for leaps of a 5th or more; octave-less notes are read nearest
    the previous note.
    """
    return _chant.check_melody(notes, root, scale_type, strict)


@mcp.tool()
def cantus_firmus(root: str, scale_type: str = "major", length: int = 10, variant: int = 0) -> dict:
    """Compose a cantus firmus (6-16 notes) that obeys every check_melody rule — deterministic.

    `variant` 0-999: each variant is a different reproducible melody (an error
    says when fewer exist). Feed the notes to counterpoint() for species
    counterpoint, or harmonize them.
    """
    return _chant.cantus_firmus(root, scale_type=scale_type, length=length, variant=variant)


# ----------------------------------------------- harmony rules of the masters

@mcp.tool()
def rameau_fundamental_bass(chords: str | list, root: str | None = None, scale_type: str = "major") -> dict:
    """Rameau's fundamental bass (1722): the chord roots under a progression and his cadences.

    Returns the root line (render it as a bass track), each root motion (by
    fifth, third or step — Rameau preferred fifths) and, with a key, the
    cadences: cadence parfaite (V7→I), cadence irrégulière (bass rises a fifth:
    IV→I, I→V), cadence rompue (V→vi).
    """
    return _masters.rameau_fundamental_bass(chords, root=root, scale_type=scale_type)


@mcp.tool()
def schoenberg_progressions(chords: str | list) -> dict:
    """Classify root progressions as Schoenberg did: ascending (strong), descending, superstrong.

    Ascending: root up a fourth or down a third. Descending: root up a fifth or
    up a third. Superstrong: root by step. Root motion is read from the sounding
    interval, so spelling does not matter. Schoenberg built harmony mostly on
    ascending progressions — use the summary to judge a progression's drive.
    """
    return _masters.schoenberg_progressions(chords)


@mcp.tool()
def bach_chorale_voicing(chords: str | list, root: str | None = None, scale_type: str = "major",
                         melody: str | list[str] | None = None) -> dict:
    """Voice a progression in four parts (SATB) by the rules of Bach-chorale writing.

    Voice ranges and spacing, complete chords (a 7th chord may drop its fifth),
    the bass on the root or slash bass, no doubled leading tone, seventh or
    augmented-sixth tone, no parallel or outer-voice direct fifths/octaves, no
    overlap, no melodic augmented interval (A2, A4 — the b6->#7 step in minor;
    judged by spelling), leading tone up, sevenths (spelled as sevenths) down,
    an augmented sixth (It6/Fr43/Ger65/Sw43) expanding outward, #4 up and b6
    down a semitone — then the smoothest voicing (exact search). Returns the four
    voices, per-chord SATB notes, any unavoidable rule breaks and a four-track
    render_hint for arrange_to_midi. Give `melody` (one note per chord, e.g.
    from harmonize_melody) to keep a chorale tune in the soprano.
    """
    return _masters.bach_chorale_voicing(chords, root=root, scale_type=scale_type, melody=melody)


@mcp.tool()
def neo_riemannian(chord: str | list[str], operations: str) -> dict:
    """Neo-Riemannian P, L, R transformations of a major/minor triad (Riemann, Lewin, Cohn).

    P: C ↔ Cm, R: C ↔ Am, L: C ↔ Em; each keeps two notes and moves one by step.
    `operations` like 'PLR' or 'LRLR' are applied in order — the smooth,
    chromatic triad chains of film scores. Returns each chord and the moving note.
    """
    return _masters.neo_riemannian(chord, operations)


@mcp.tool()
def bartok_axis(key: str) -> dict:
    """Bartók's axis system (after Lendvai): the tonic, subdominant and dominant axes of a key.

    Each function is a minor-third cycle of four keys with a pole and a
    tritone counterpole (C: tonic axis C–F♯ with A–E♭). Keys on one axis can
    substitute for each other.
    """
    return _masters.bartok_axis(key)


@mcp.tool()
def coltrane_changes(key: str) -> dict:
    """Coltrane changes ('Giant Steps'): replace a ii–V–I with three tonal centres a major third apart.

    In C: Dm7 E♭7 A♭maj7 B7 Emaj7 G7 Cmaj7. Returns original and substituted
    symbols (ready for voice_leading / chords_to_midi) and the tonal centres.
    """
    return _masters.coltrane_changes(key)


# ------------------------------------------------------ modern techniques

@mcp.tool()
def twelve_tone_matrix(row: str | list) -> dict:
    """Schoenberg's twelve-tone matrix: all 48 forms (P, I, R, RI) of a 12-note row.

    `row`: the 12 pitch classes as notes or numbers 0-11 ('0 11 3 4 ...' or a
    list). Forms are labelled by
    their first pitch class (P0 starts on C). Each form is a note list you can
    use as a melody or slice into chords.
    """
    return _modern.twelve_tone_matrix(row)


@mcp.tool()
def pitch_class_set(notes: str | list) -> dict:
    """Pitch-class set analysis: normal form, prime form, interval vector, complement, symmetry.

    `notes`: notes or numbers 0-11 ('0 4 7' or a list). Prime form (Rahn's
    algorithm, as in Straus) names the set class (every major/minor triad is
    [0,3,7]; the whole-tone scale [0,2,4,6,8,10]). Useful for comparing
    post-tonal sonorities.
    """
    return _modern.pitch_class_set(notes)


@mcp.tool()
def additive_process(notes: str | list[str], mode: str = "additive", repeats: int = 1, octave: int = 4) -> dict:
    """Philip Glass's additive/subtractive process: 1 2, 1 2 3, 1 2 3 4 … over a figure.

    `mode`: additive, subtractive or both; `repeats` repeats each stage. Notes
    without an octave are placed once from `octave`, so every stage repeats the
    same pitches. Returns the note sequence and stage boundaries for a notes track.
    """
    return _modern.additive_process(notes, mode=mode, repeats=repeats, octave=octave)


@mcp.tool()
def phase_shift(notes: str | list[str], repeats_per_stage: int = 2, step_beats: float = 0.25,
                octave: int = 4) -> dict:
    """Steve Reich's phasing: a pattern against a copy that slips one step ahead per stage.

    The pattern may contain rests ('.') that rotate with it — 'C C C . C C . C .
    C C .' is Clapping Music; Piano Phase's gradual drift is shown at its locked
    positions. Returns both voices (plus rhythm strings when there are rests)
    and a two-track render_hint for arrange_to_midi.
    """
    return _modern.phase_shift(notes, repeats_per_stage=repeats_per_stage, step_beats=step_beats,
                               octave=octave)


# ------------------------------------------- analysis (read a draft back)

@mcp.tool()
def detect_key(notes: str | list | None = None, chords: str | list | None = None, tracks: list[dict] | None = None,
               durations: list[float] | None = None, rhythm: str | None = None, step_beats: float = 1.0,
               beats_per_chord: float = 4.0, profile: str = "krumhansl", window_beats: float = 0.0,
               hop_beats: float = 0.0) -> dict:
    """Find the key of a melody, a progression or a whole arrangement (Krumhansl–Schmuckler key finding).

    Correlates a duration-weighted pitch-class histogram with a key profile in
    all 24 major/minor keys, exactly as music21 does. Give `notes` (an inner
    list = simultaneous notes; timed by `durations`, or a `rhythm` as in a
    notes track, or `step_beats` each) and/or `chords` (each chord
    `beats_per_chord`), OR `tracks` alone (render_hint / arrange tracks, timed
    as they render; drums ignored). `profile`: krumhansl (default), temperley,
    bellman, aarden (its minor weights are of uncertain origin, per
    music21), simple. Returns root and
    scale_type ('major' / 'natural minor' — ready for diatonic_chords,
    analyze_progression, snap_to_scale, check_voice_leading), correlation,
    certainty, the 24-key ranking and the histogram; the tonic keeps your
    spelling (Gb stays Gb). `window_beats` > 0 adds key `regions` (hop_beats
    defaults to half a window) — where a song modulates.
    e.g. detect_key(notes='C4 D4 E4 F4 G4 A4 B4 C5') -> C major, r 0.9014;
    detect_key(chords='C Am F G C Am F G G Em C D G Em C D', window_beats=16, hop_beats=16)
    -> regions C major (0-32), G major (32-64).
    """
    return _analysis.detect_key(notes=notes, chords=chords, tracks=tracks, durations=durations, rhythm=rhythm,
                                step_beats=step_beats, beats_per_chord=beats_per_chord, profile=profile,
                                window_beats=window_beats, hop_beats=hop_beats)


@mcp.tool()
def check_voice_leading(voices: list | None = None, voicings: list | None = None, root: str | None = None,
                        scale_type: str = "major") -> dict:
    """Lint your own parts for voice-leading faults — parallels, direct fifths, crossing, overlap, resolutions.

    Give `voices` (2+ parts: note lists with octaves, one note per slot, listed
    highest first; or notes tracks such as bach_chorale_voicing's or
    counterpoint's render_hint tracks, aligned on every attack) OR `voicings`
    (voice_leading's or voice_chords' `chords`, 2+ notes each; sizes may
    differ, as in I V7 I — a smaller chord leaves a part silent, null in that
    slice). Violations: parallel and contrary fifths/octaves (spelled, compound
    too), direct fifths/octaves in the outer voices with a leaping upper voice
    (also from a differently spelled interval: d6 -> P5), voice crossing and
    overlap, augmented melodic intervals, leaps over an octave; with `root` (+
    scale_type, a 7-note key; each slice read as a triad or seventh chord, a
    missing fifth restored): the leading tone of a chord on ^5 or a diminished
    chord on ^7 rising to I/vi in the outer voices (not into a chord that holds
    ^7, nor as the passing ^7 of a bass stepping 1-7-6), chord sevenths
    (spelled as sevenths) falling by step, augmented_sixth (It6/Fr43/Ger65/Sw43:
    b6 in the bass, #4 an augmented sixth above it; #4 rises and b6 falls a
    semitone when the harmony changes, a held note is a warning), no doubled
    leading tone. Warnings: unequal fifths, spacing, SATB range, melodic
    sevenths, unrecovered leaps, enharmonic fifths. Also music21 motion counts
    per voice pair and the dissonances above the bass. Fifths/octaves on
    successive downbeats (Fux species 2-5) are not checked. Run it before
    arrange_to_midi.
    e.g. check_voice_leading(voices=[['C5','D5'],['F4','G4']]) -> parallel_fifths at index 1, voices [0, 1].
    """
    return _analysis.check_voice_leading(voices=voices, voicings=voicings, root=root, scale_type=scale_type)


@mcp.tool()
def find_cadences(chords: str | list, root: str, scale_type: str = "major", soprano: str | list | None = None,
                  phrase_ends: list[int] | str | None = None, phrase_length: int = 0) -> dict:
    """Label the cadence at each phrase end — authentic (perfect/imperfect), half, Phrygian, plagal, deceptive or none.

    Textbook rules (Kostka, Payne & Almén, Tonal Harmony ch. 10; Caplin,
    Classical Form; Aldwell & Schachter) — a codification, not music21, which
    has no cadence classifier. Give `chords` (symbols or note arrays;
    bach_chorale_voicing's SATB rows written as [s, a, t, b] arrays read
    correctly, even a V7 without its fifth) and a seven-note key. `soprano`:
    one note per chord (bach_chorale_voicing(...)['voices']['soprano'] or your
    melody); note arrays with octaves give their top note. Phrase ends:
    `phrase_ends` (0-based chord indices, e.g. the last chord of each
    plan_sections section), 'all' (every authentic, plagal and deceptive
    cadence; half cadences only at the end), or `phrase_length` (every L
    chords); default the last chord. Rules: authentic = V/vii° -> I; 'perfect'
    needs root-position V–I AND ^1 in the soprano, and is never claimed without
    it (unknown soprano: subtype null, 'give the soprano to decide PAC vs IAC');
    otherwise 'imperfect' with the reason ('inversion', 'soprano on ^3',
    'leading-tone chord'). plagal = IV/iv -> root-position I. deceptive = V ->
    vi (VI in minor), bVI or IV6. half = ends on V ('phrygian' for iv6 -> V in
    minor; a V7 or inverted V gets Caplin's root-position-triad caveat). 'none'
    says why ('ends on vi'); ask next_chords for a stronger close. Each cadence
    also gives its span, romans (ready for roman_to_chords; a Picardy tonic is
    'I'), soprano and bass degrees, cadential_64, picardy and caveats;
    `summary` counts the types.
    e.g. find_cadences('C F C/G G7 C', 'C', soprano='E5 F5 E5 D5 C5') ->
    authentic perfect, romans Cad64 V7 I, cadential_64 true;
    find_cadences(['Am','Dm/F','E'], 'A', 'natural minor') -> half, phrygian.
    """
    return _analysis.find_cadences(chords, root, scale_type=scale_type, soprano=soprano,
                                   phrase_ends=phrase_ends, phrase_length=phrase_length)


# ------------------------------------------------------------ accompaniment

@mcp.tool()
def chord_pattern(chords: str | list, pattern: str = "alberti", beats_per_chord: float | list[float] = 4.0,
                  step_beats: float = 0.5, mode: str = "chord", phase: str = "restart",
                  root: str | None = None, scale_type: str = "major", octave: int = 4,
                  smooth: bool = True, sustain: bool = False) -> dict:
    """Play a chord-relative figure over a whole progression — Alberti bass, broken chords, arpeggios, scale runs.

    `pattern` is a preset — alberti '^1 3 2 3', up '^1 2 3 4', down '^4 3 2 1',
    updown '^1 2 3 4 3 2', murky "^1, 1" (broken octaves) — or your own steps:
    an index (1 = the voicing's bottom note), '^' accents it (O in the rhythm),
    each ' / , moves it an octave up / down, '.' is a rest (with `sustain` a
    hold of the previous note, only within its chord: a sustained pattern that
    would hold a note over a rest opening the next chord, like '. 1 2 3', is a
    ValueError). mode='chord': index k is the k-th voicing tone, wrapping up an
    octave (4 of a triad = the bottom note an octave up). mode='scale' (needs
    `root`/`scale_type`): index k is the k-th scale note from the chord root
    (spelled in the key; a chord whose root is off the scale falls back to chord
    mode). Voicings come from voice_leading (smooth=true) or root position with
    note arrays kept as written (smooth=false, so voice_chords voicings drive
    it). Each chord lasts `beats_per_chord` (one number or one per chord) =
    a whole number of `step_beats` steps; phase 'restart' restarts the figure on
    every chord, 'continue' runs it on across the changes.
    e.g. chord_pattern(['C','Am','F','G'], 'alberti', beats_per_chord=2) ->
    C4 G4 E4 G4 | C4 A4 E4 A4 | C4 A4 F4 A4 | D4 B4 G4 B4, rhythm 'Oooo' per chord.
    Returns the notes track, per_chord notes and a render_hint (chords + pattern
    tracks) for arrange_to_midi or an arrange_song section; pair with bass_line.
    """
    return _accompany.chord_pattern(chords, pattern=pattern, beats_per_chord=beats_per_chord,
                                    step_beats=step_beats, mode=mode, phase=phase, root=root,
                                    scale_type=scale_type, octave=octave, smooth=smooth, sustain=sustain)


@mcp.tool()
def bass_line(chords: str | list, style: str = "root", beats_per_chord: float | list[float] = 4.0,
              step_beats: float = 1.0, rhythm: str | None = None, octave: int = 2,
              pedal: str | None = None, ending: str = "loop", sustain: bool | None = None) -> dict:
    """Write a bass part from chords by fixed rules — root, root-fifth, root-octave, chromatic approach, walking, pedal.

    The bass of a chord is its slash bass, else its root (a note array: its
    lowest note). Every note is folded into E1-G3 (MIDI 28-55); each chord's
    bass goes to the pitch nearest the previous note. Styles (default rhythm per
    chord): root (held), root_fifth (bass + the chord's fifth below, country
    two-beat 'O.o.'; over a 6/4 such as C/G the alternate is the root),
    root_octave ('Oo'; a bass from Ab2 up is played an octave lower, so its
    octave fits: A D E A -> A1 A2 A1 A2 | D2 D3 D2 D3 | …), approach (the bass,
    then a chromatic approach into the next chord on the last onset: from below
    when it lies above, G -> F#, else from above, C -> Db), walking (chord
    tones 3rd-5th-7th or octave on the beats — turning down once they would
    pass G3 — and the approach on the last beat; a simplified codification of
    walking-bass pedagogy), pedal (the `pedal` note, default the first bass,
    under every chord). `rhythm` (O/o/., one chord span long) replaces the
    default on every chord. ending='loop' approaches the line's actual first
    note from the last chord by a semitone (so the approach may sit an octave
    away from the note before it); 'root' ends without an approach. `sustain`
    (default null = the style's own: root, approach and pedal hold each note
    over the rests after it, the others play one step per note) forces held or
    detached notes; a hold never crosses a chord change, so a held style whose
    rhythm opens with '.' plays detached over 2+ chords, and sustain=true with
    such a rhythm is a ValueError.
    e.g. bass_line(['C','Am','F','G'], 'walking') -> C2 E2 G2 G#2 | A2 C3 E3 Gb3 |
    F3 C3 A2 Ab2 | G2 B2 D3 Db2 (Db2 leads back into the first C2);
    bass_line(['C','G/B','Am','F']) -> C2 B1 A1 F1.
    Returns a notes track (program 33, finger bass), per_chord notes and a
    render_hint for arrange_to_midi / arrange_song, next to chord_pattern.
    """
    return _accompany.bass_line(chords, style=style, beats_per_chord=beats_per_chord,
                                step_beats=step_beats, rhythm=rhythm, octave=octave, pedal=pedal,
                                ending=ending, sustain=sustain)


# ----------------------------------------------------------- song structure

@mcp.tool()
def plan_sections(form: str | list[str], bars: int | dict[str, int] = 8, beats_per_bar: int = 4,
                  tempo: int | None = None) -> dict:
    """Lay out a song form on the timeline: where each section starts and how long it lasts.

    `form` is the running order — a list, a string ('intro verse chorus verse
    chorus outro'), or a letter form ('AABA'). `bars` is one number for every
    section, or a mapping of section name to bars (intro/verse/chorus/bridge/
    outro fall back to sensible defaults). Returns each section's start bar,
    start beat, length, and (with `tempo`) start time in seconds — use it to
    place material with `start_beat` in arrange_to_midi, or as the blueprint for
    arrange_song.
    """
    return _structure.plan_sections(form, bars=bars, beats_per_bar=beats_per_bar, tempo=tempo)


@mcp.tool()
def arrange_song(sections: dict, form: str | list[str] | None = None, tempo: int = 120,
                 beats_per_bar: int = 4, step_beats: float = 0.5, swing: float = 0.5,
                 swing_unit: float = 0.5, file_name: str | None = None,
                 output_dir: str | None = None) -> dict:
    """Assemble named sections (intro/verse/chorus/bridge/outro) into one whole-song MIDI.

    The capstone "build a whole song" tool. `sections` maps a section name to
    ``{"bars": N, "tracks": [ ...arrange_to_midi-style tracks... ]}`` — each
    section is a little arrangement (chords, bass, melody, drums) with timing
    relative to its own start. `form` is the running order (list/string/letters,
    repeats allowed; omitted = each section once in given order). Sections are
    placed end to end, and tracks sharing a `name` across sections are stitched
    into one continuous MIDI track (so "bass" is a single track for the whole
    song; a part used only in the chorus simply rests elsewhere; a different
    `program` in a later section switches the instrument there). Name your
    tracks; unnamed ones become 'notes' (or 'notes_1', 'notes_2'...). Chords
    tracks take the same comping fields as in arrange_to_midi (durations,
    rhythm, sustain, strum; an 'alternate' strum counts its down/up steps from
    the section start). `swing` (0.5 straight .. 2/3 triplet .. 0.75) and
    `swing_unit` (0.25/0.5/1.0 beats) swing the whole song on its bar grid:
    pairs of steps count from each downbeat, so every section (and every repeat)
    swings alike and bar lines never move — in an odd meter with swing_unit 1.0
    (3/4, 5/4) the bar's unpaired last quarter stays straight. A track's own
    "swing"/"swing_unit" overrides them.
    Compose each layer with the scale/chord/melody/rhythm tools, drop them into
    sections, and sequence — then midi_to_audio to hear it. Returns the file plus
    a section timeline and per-track summary.
    Example: sections={"verse": {"bars": 4, "tracks": [{"type": "chords", "name": "keys",
    "chords": ["Am", "F", "C", "G"], "rhythm": "O.o.O.o.", "step_beats": 0.5}]}},
    form="verse verse", swing=0.6667.
    """
    return _structure.render_song_structure(sections, form=form, tempo=tempo,
                                             beats_per_bar=beats_per_bar, step_beats=step_beats,
                                             swing=swing, swing_unit=swing_unit,
                                             file_name=file_name, output_dir=output_dir)


# -------------------------------------------------------------------- MIDI

@mcp.tool()
def notes_to_midi(notes: str | list[str], rhythm: str | None = None,
                  step_beats: float = 0.5, tempo: int = 120, octave: int = 4,
                  octave_policy: str = "nearest", velocity: int = 90,
                  accent_velocity: int = 110, sustain: bool = False,
                  program: int = 0, swing: float = 0.5, swing_unit: float = 0.5,
                  file_name: str | None = None, output_dir: str | None = None) -> dict:
    """Write a note sequence (scale, arpeggio or melody) to a single-track MIDI file.

    Plays the notes in order, one per `step_beats`. With `rhythm` (a pattern
    like 'O.oo.O..' from random_rhythm/euclidean_rhythm or hand-written), each
    step follows the pattern: O = accented note, o = soft note, . = pause
    (notes are consumed in order and wrap around; with sustain=true pauses
    extend the previous note). Octave-less notes are placed by `octave_policy`:
    'nearest' for melodies, 'ascending' for scale runs. `program` is a General
    MIDI instrument (0 piano, 24 guitar, 32 bass...). `swing` is the DAW/MPC
    swing ratio (0.5 straight, 0.6667 triplet swing, at most 0.75) applied to
    pairs of `swing_unit`-beat steps (0.25, 0.5 or 1.0); on-beats never move.
    E.g. eighths at 0, 0.5, 1, 1.5 with swing=0.6667 play at 0, 0.667, 1, 1.667.
    Returns the file path, base64 and the exact note events written (swung times).
    """
    return _midi.render_notes(notes, rhythm=rhythm, step_beats=step_beats, tempo=tempo,
                              octave=octave, octave_policy=octave_policy,
                              velocity=velocity, accent_velocity=accent_velocity,
                              sustain=sustain, program=program, swing=swing,
                              swing_unit=swing_unit, file_name=file_name,
                              output_dir=output_dir)


@mcp.tool()
def chords_to_midi(chords: str | list[str | list[str]], beats_per_chord: float = 4.0,
                   tempo: int = 120, octave: int = 4, arpeggiate: bool = False,
                   velocity: int = 80, program: int = 0, durations: list[float] | None = None,
                   rhythm: str | None = None, step_beats: float = 0.5, sustain: bool = False,
                   accent_velocity: int = 100, strum: float = 0.0, strum_direction: str = "down",
                   swing: float = 0.5, swing_unit: float = 0.5, file_name: str | None = None,
                   output_dir: str | None = None) -> dict:
    """Write a chord sequence to a single-track MIDI file (block chords, comped rhythm, strummed or arpeggiated).

    `chords` items are chord symbols ('C', 'Am7', 'F#dim', 'C/E', 'C4maj7' —
    e.g. the `symbols` output of degrees_to_chords) and/or explicit note
    arrays (['C','E','G'] or ['C4','E4','G4']). Octave-less chords are voiced
    upward from `octave`. Each chord lasts `beats_per_chord`, or its entry in
    `durations` (one number of beats per chord, 0.25-64: variable harmonic rhythm).
    `rhythm` comps each chord with an O/o/. pattern on a `step_beats` grid: O strikes
    the voicing at `accent_velocity`, o at `velocity`, . rests (with sustain=true it
    holds the previous strike, never past the chord change). The pattern is either
    one chord long (repeated for every chord) or the whole progression long.
    `strum` (0-0.25 beats) delays each voice of a strike: 'down' from the lowest note,
    'up' from the highest, 'alternate' down on even steps of the absolute
    `step_beats` grid (counted from beat 0) and up on odd ones (without a rhythm:
    down on even chords, up on odd); voices end together, and a strum too wide
    for its strike at MIDI resolution is a ValueError naming the widest that
    fits. rhythm/strum cannot be combined with arpeggiate. `swing`
    (0.5 straight, 0.6667 triplet, max 0.75) swings pairs of `swing_unit`-beat steps.
    Example: chords=['Am','F','C','G'], beats_per_chord=2, step_beats=0.25,
    rhythm='O..o..o.', strum=0.03 strikes each chord at beats 0, 0.75 and 1.5 of its
    span (a tresillo stab); durations=[2,2,4,8] gives a 2+2+4+8-beat harmonic rhythm.
    Returns the file path, base64 and each chord's voiced notes, MIDI numbers and span.
    """
    return _midi.render_chords(chords, beats_per_chord=beats_per_chord, tempo=tempo,
                               octave=octave, arpeggiate=arpeggiate, velocity=velocity,
                               program=program, durations=durations, rhythm=rhythm,
                               step_beats=step_beats, sustain=sustain,
                               accent_velocity=accent_velocity, strum=strum,
                               strum_direction=strum_direction, swing=swing,
                               swing_unit=swing_unit, file_name=file_name,
                               output_dir=output_dir)


@mcp.tool()
def drums_to_midi(lanes: dict[str, str], step_beats: float = 0.5, tempo: int = 120,
                  velocity: int = 100, accent_velocity: int = 120, swing: float = 0.5,
                  swing_unit: float = 0.5, file_name: str | None = None,
                  output_dir: str | None = None) -> dict:
    """Write a drum pattern to a single-track General MIDI percussion file.

    `lanes` maps a drum name to a rhythm pattern, e.g.
    {"kick": "O...O...", "snare": "..O...O.", "hat": "oooooooo"} — each
    pattern uses O (accented hit), o (soft hit), . (rest), one step per
    `step_beats`. Lanes can be any length and play simultaneously. Drum names
    include kick, snare, side_stick, clap, closed_hat/open_hat/pedal_hat,
    low_tom/mid_tom/high_tom, crash, ride, tambourine, cowbell, clave, shaker,
    conga, bongo... (or a raw GM note number). Patterns from random_rhythm /
    euclidean_rhythm work directly as lanes. `swing` (0.5 straight, 0.6667 triplet
    shuffle, max 0.75) delays every second `swing_unit`-beat step (0.25 swung
    sixteenths, 0.5 swung eighths, 1.0 swung quarters); on-beats never move.
    Example: lanes={"hat": "oooooooo"}, swing=0.6667 gives a shuffled eighth hat.
    """
    return _midi.render_drums(lanes, step_beats=step_beats, tempo=tempo, velocity=velocity,
                              accent_velocity=accent_velocity, swing=swing,
                              swing_unit=swing_unit, file_name=file_name,
                              output_dir=output_dir)


@mcp.tool()
def song_to_midi(melody_notes: str | list[str], chords: str | list[str | list[str]],
                 melody_rhythm: str | None = None, step_beats: float = 0.5,
                 beats_per_chord: float = 4.0, tempo: int = 120,
                 melody_octave: int = 5, chord_octave: int = 4,
                 octave_policy: str = "nearest", melody_velocity: int = 95,
                 accent_velocity: int = 115, chord_velocity: int = 70,
                 sustain: bool = False, arpeggiate_chords: bool = False,
                 melody_program: int = 0, chord_program: int = 0,
                 swing: float = 0.5, swing_unit: float = 0.5,
                 file_name: str | None = None, output_dir: str | None = None) -> dict:
    """Write a melody plus chord accompaniment into one two-track MIDI file.

    A convenient shortcut for the common melody+chords case; for bass, drums,
    comped/strummed chords or more tracks use arrange_to_midi. Track 1 plays
    `melody_notes` (optionally shaped by `melody_rhythm`, same rules as
    notes_to_midi); track 2 plays `chords` (same formats as chords_to_midi), one
    every `beats_per_chord`. Align lengths yourself: a melody over 4 chords of 4
    beats at 0.5-beat steps needs a 32-step rhythm. `*_program` numbers pick GM
    instruments. `swing` (0.5 straight, 0.6667 triplet, max 0.75) swings both tracks
    on pairs of `swing_unit`-beat steps (0.25, 0.5 or 1.0).
    Example: melody_notes='E5 D5 C5 D5 E5 E5 E5 D5', chords=['C','G'],
    beats_per_chord=2, swing=0.6667 gives a swung-eighths tune over C and G.
    """
    return _midi.render_song(melody_notes, chords, melody_rhythm=melody_rhythm,
                             step_beats=step_beats, beats_per_chord=beats_per_chord,
                             tempo=tempo, melody_octave=melody_octave,
                             chord_octave=chord_octave, octave_policy=octave_policy,
                             melody_velocity=melody_velocity, accent_velocity=accent_velocity,
                             chord_velocity=chord_velocity, sustain=sustain,
                             arpeggiate_chords=arpeggiate_chords,
                             melody_program=melody_program, chord_program=chord_program,
                             swing=swing, swing_unit=swing_unit,
                             file_name=file_name, output_dir=output_dir)


@mcp.tool()
def arrange_to_midi(tracks: list[dict], tempo: int = 120, step_beats: float = 0.5,
                    beats_per_chord: float = 4.0, swing: float = 0.5, swing_unit: float = 0.5,
                    file_name: str | None = None, output_dir: str | None = None) -> dict:
    """Render any number of fitting tracks into one multi-track MIDI file — the full arrangement.

    This is the capstone "idea -> song" tool. You assemble the parts (the
    creative part) and it renders them together. `tracks` is a list of track
    objects, each of one of three types:

    - notes:  {"type":"notes", "notes":[...], "rhythm":"O.o.O.o.", "octave":5,
               "program":0, "octave_policy":"nearest", "sustain":false}  (melody/bass/arp)
    - chords: {"type":"chords", "chords":["Am","F","C","G"], "beats_per_chord":4,
               "octave":4, "arpeggiate":false, "program":0}              (pads/comping)
              comping fields: "durations":[2,2,4] (beats per chord, replaces
              beats_per_chord), "rhythm":"O..o..o." (strikes on a "step_beats" grid,
              one chord long or the whole track long; O at "accent_velocity" (100),
              o at "velocity"), "sustain":true (rests hold the strike within its
              chord), "strum":0.03 (beats between voices, max 0.25) with
              "strum_direction":"down"|"up"|"alternate" (alternate: down on even
              steps of the absolute step_beats grid, counted from beat 0 with
              start_beat included, up on odd ones)
    - drums:  {"type":"drums", "lanes":{"kick":"O...O...","snare":"..O...O.","hat":"oooooooo"}}

    Shared per-track options: "name", "velocity", "start_beat" (beat offset for
    intros/drops), "step_beats", "swing"/"swing_unit" (override the renderer's),
    "channel" (auto-assigned; drums forced to the GM percussion channel). `swing`
    (0.5 straight, 0.6667 triplet swing, max 0.75) swings pairs of `swing_unit`-beat
    steps (0.25/0.5/1.0) on the absolute timeline, so offset tracks swing together.
    Align track lengths via start_beat and step counts. Typical full arrangement:
    a chords track, a bass notes track (chord roots, low octave, program 33), a
    melody notes track (program 0/80), and a drums track. Render hints from
    harmonize_melody/counterpoint/bach_chorale_voicing/phase_shift go straight in.
    Example (reggae skank on the off-beats over a one-drop): tracks=[{"type":"chords",
    "chords":["C","F"],"beats_per_chord":4,"step_beats":0.5,"rhythm":".o.o.o.o"},
    {"type":"drums","step_beats":0.25,"lanes":{"kick":"........O...............O......."}}].
    Returns the file path, base64 and a per-track summary (swung times reported).
    """
    return _midi.render_arrangement(tracks, tempo=tempo, step_beats=step_beats,
                                    beats_per_chord=beats_per_chord, swing=swing,
                                    swing_unit=swing_unit, file_name=file_name,
                                    output_dir=output_dir)


@mcp.tool()
def midi_to_audio(midi_file: str, wav_file: str | None = None,
                  sample_rate: int = 44100, include_base64: bool = False) -> dict:
    """Synthesize a generated MIDI file into a playable WAV audio file.

    A .mid file needs a synthesizer/soundfont to be heard; this renders one to
    a self-contained 16-bit PCM WAV that plays on any device or browser, using
    a simple built-in synth (additive tones for pitched parts, percussive
    synthesis for General MIDI drums) — no soundfont required. Pass the `file`
    path returned by notes_to_midi / chords_to_midi / drums_to_midi /
    song_to_midi / arrange_to_midi. The WAV is written next to the .mid unless
    `wav_file` names another path. Returns the WAV path, size and duration;
    `include_base64=true` also returns the audio base64-encoded (large — a few
    seconds is already hundreds of KB). Renders at most the first 5 minutes.
    This is a preview render, not a production mix.
    """
    return _audio.render_midi_to_wav(midi_file, wav_path=wav_file, sample_rate=sample_rate,
                                     include_base64=include_base64)


def main() -> None:
    """Run the MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
