"""MIDI composer MCP server.

Every tool is an atomic, deterministic step (lookups, matching, seeded dice
rolls, rendering). The creative work — choosing scales, progressions,
melodies and rhythms — is left to the caller, who chains the tools:
outputs (note arrays, chord symbols, rhythm patterns) feed directly into
other tools' inputs.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from . import chords as _chords
from . import diatonic as _diatonic
from . import generate as _generate
from . import midi_io as _midi
from . import scales as _scales

mcp = FastMCP(
    "midi-composer",
    instructions=(
        "Atomic music-theory and MIDI tools for composing. Notes are strings like"
        " 'C', 'F#', 'Bb' — add an octave for concrete pitches ('C5', 'Eb3'; C4 is"
        " middle C). Octave-less notes are abstract pitch classes. Every tool's"
        " note/chord/rhythm output can be passed to every other tool, so compose"
        " creatively by chaining: e.g. get_scale -> pick degrees -> degrees_to_chords"
        " -> chords_to_midi, or get_chord + random_rhythm -> notes_to_midi. The tools"
        " make no creative choices — you do."
    ),
)


# ------------------------------------------------------------------ lookups

@mcp.tool()
def list_scales() -> dict:
    """List every scale type in the database with intervals, degree labels and aliases.

    Use a scale type's name (or any alias) with get_scale, match_scales,
    diatonic_chords and degrees_to_chords.
    """
    return _scales.list_scales()


@mcp.tool()
def list_chords() -> dict:
    """List every chord type in the database with intervals, degree labels, symbol suffixes and aliases.

    Use a chord type's name or symbol suffix with get_chord, and root+suffix
    symbols (e.g. 'Am', 'G7', 'F#m7b5') anywhere a chord symbol is accepted.
    """
    return _chords.list_chords()


@mcp.tool()
def get_scale(scale_type: str, root: str | None = None) -> dict:
    """Describe a scale type; with a root note, generate its notes.

    Without `root`: intervals and degree labels only (e.g. major = 0 2 4 5 7 9 11).
    With `root`: the spelled notes, e.g. get_scale('major', 'C') -> C D E F G A B C.
    Give the root an octave for concrete pitches: get_scale('major', 'C5') ->
    C5 ... C6 plus MIDI numbers. The returned `notes` array feeds directly into
    match_chords, random_notes, notes_to_midi, etc.
    """
    return _scales.scale_info(scale_type, root)


@mcp.tool()
def get_chord(chord_type: str, root: str | None = None) -> dict:
    """Describe a chord type; with a root note, generate its notes.

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

    `notes` is a list like ['C', 'E', 'G'] or a string 'c e g' — any other
    tool's notes output works as-is. A match is 'exact' when the input uses
    every note of the scale; otherwise 'contains', with the scale's extra
    notes listed in `added_notes`. Exact and tighter (smaller) scales sort first.
    """
    return _scales.match_scales(notes, exact_only=exact_only, limit=limit)


@mcp.tool()
def match_chords(notes: str | list[str], include_partial: bool = True, limit: int = 20) -> dict:
    """Find chords that match the given notes (octaves are ignored).

    'exact' matches use exactly the input pitch classes; when the first input
    note is not the chord root the inversion is reported with slash notation
    (e.g. 'E G C' -> C/E, first inversion). 'partial' matches are chords that
    contain all input notes plus the listed `missing_notes`.
    """
    return _chords.match_chords(notes, include_partial=include_partial, limit=limit)


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
    array feeds directly into chords_to_midi / song_to_midi. This tool only
    maps degrees to chords; choosing and ordering the degrees is up to you.
    """
    return _diatonic.degrees_to_chords(root, scale_type, degrees, sevenths)


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
    `rhythm` argument of notes_to_midi / song_to_midi; you can also edit it
    by hand first. Reproducible via `seed`; the seed used is always returned.
    """
    return _generate.random_rhythm(length=length, density=density,
                                   accent_probability=accent_probability, seed=seed)


# -------------------------------------------------------------------- MIDI

@mcp.tool()
def notes_to_midi(notes: str | list[str], rhythm: str | None = None,
                  step_beats: float = 0.5, tempo: int = 120, octave: int = 4,
                  octave_policy: str = "nearest", velocity: int = 90,
                  accent_velocity: int = 110, sustain: bool = False,
                  program: int = 0, file_name: str | None = None,
                  output_dir: str | None = None) -> dict:
    """Write a note sequence (scale, arpeggio or melody) to a MIDI file.

    Plays the notes in order, one per `step_beats`. With `rhythm` (a pattern
    like 'O.oo.O..' from random_rhythm or hand-written), each step follows the
    pattern: O = accented note, o = soft note, . = pause (notes are consumed
    in order and wrap around if the pattern needs more; with sustain=true
    pauses extend the previous note instead). Octave-less notes are placed by
    `octave_policy`: 'nearest' for melodies, 'ascending' for scale runs.
    Returns the file path and base64 plus the exact note events written.
    """
    return _midi.render_notes(notes, rhythm=rhythm, step_beats=step_beats, tempo=tempo,
                              octave=octave, octave_policy=octave_policy,
                              velocity=velocity, accent_velocity=accent_velocity,
                              sustain=sustain, program=program,
                              file_name=file_name, output_dir=output_dir)


@mcp.tool()
def chords_to_midi(chords: str | list[str | list[str]], beats_per_chord: float = 4.0,
                   tempo: int = 120, octave: int = 4, arpeggiate: bool = False,
                   velocity: int = 80, program: int = 0, file_name: str | None = None,
                   output_dir: str | None = None) -> dict:
    """Write a chord sequence to a MIDI file (block chords, or arpeggiated).

    `chords` items are chord symbols ('C', 'Am7', 'F#dim', 'C/E', 'C4maj7' —
    e.g. the `symbols` output of degrees_to_chords) and/or explicit note
    arrays (['C','E','G'] or ['C4','E4','G4']). Octave-less chords are voiced
    upward from `octave`. Each chord lasts `beats_per_chord`. Returns the file
    path and base64 plus each chord's voiced notes and MIDI numbers.
    """
    return _midi.render_chords(chords, beats_per_chord=beats_per_chord, tempo=tempo,
                               octave=octave, arpeggiate=arpeggiate, velocity=velocity,
                               program=program, file_name=file_name, output_dir=output_dir)


@mcp.tool()
def song_to_midi(melody_notes: str | list[str], chords: str | list[str | list[str]],
                 melody_rhythm: str | None = None, step_beats: float = 0.5,
                 beats_per_chord: float = 4.0, tempo: int = 120,
                 melody_octave: int = 5, chord_octave: int = 4,
                 octave_policy: str = "nearest", melody_velocity: int = 95,
                 accent_velocity: int = 115, chord_velocity: int = 70,
                 sustain: bool = False, arpeggiate_chords: bool = False,
                 melody_program: int = 0, chord_program: int = 0,
                 file_name: str | None = None, output_dir: str | None = None) -> dict:
    """Write a melody plus chord accompaniment into one two-track MIDI file.

    Track 1 plays `melody_notes` (optionally shaped by `melody_rhythm`,
    same rules as notes_to_midi); track 2 plays `chords` (same formats as
    chords_to_midi), one every `beats_per_chord`. You decide which melody
    notes go over which chords — align them by length: a melody over 4 chords
    of 4 beats with 0.5-beat steps needs a 32-step rhythm/notes. General MIDI
    `*_program` numbers pick instruments (0 piano, 32 bass, 48 strings...).
    """
    return _midi.render_song(melody_notes, chords, melody_rhythm=melody_rhythm,
                             step_beats=step_beats, beats_per_chord=beats_per_chord,
                             tempo=tempo, melody_octave=melody_octave,
                             chord_octave=chord_octave, octave_policy=octave_policy,
                             melody_velocity=melody_velocity, accent_velocity=accent_velocity,
                             chord_velocity=chord_velocity, sustain=sustain,
                             arpeggiate_chords=arpeggiate_chords,
                             melody_program=melody_program, chord_program=chord_program,
                             file_name=file_name, output_dir=output_dir)


def main() -> None:
    """Run the MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
