"""MIDI composer MCP server.

Every tool is an atomic, deterministic step (lookups, matching, seeded dice
rolls, rendering). The creative work — choosing scales, progressions,
melodies, rhythms and arrangements — is left to the caller, who chains the
tools: outputs (note arrays, chord symbols, rhythm patterns) feed directly
into other tools' inputs, all the way from an idea to a multi-track MIDI file.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from . import audio as _audio
from . import chords as _chords
from . import diatonic as _diatonic
from . import generate as _generate
from . import midi_io as _midi
from . import scales as _scales

mcp = FastMCP(
    "midi-composer",
    instructions=(
        "Atomic music-theory and MIDI tools for composing, from an idea to a"
        " multi-track MIDI file. Notes are strings like 'C', 'F#', 'Bb' — add an"
        " octave for concrete pitches ('C5', 'Eb3'; C4 is middle C). Octave-less"
        " notes are abstract pitch classes; matching ignores octaves entirely."
        " Every tool's note/chord/rhythm output can be passed to every other"
        " tool, so compose by chaining — the tools make no creative choices, you"
        " do. A typical flow: get_scale -> diatonic_chords -> (pick degrees)"
        " degrees_to_chords -> build a melody and bass from the scale/chord notes"
        " -> random_rhythm or euclidean_rhythm for grooves -> arrange_to_midi to"
        " render melody, chords, bass and drums as fitting tracks in one file."
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
    This tool only maps degrees to chords; choosing the degrees is up to you.
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


# -------------------------------------------------------------------- MIDI

@mcp.tool()
def notes_to_midi(notes: str | list[str], rhythm: str | None = None,
                  step_beats: float = 0.5, tempo: int = 120, octave: int = 4,
                  octave_policy: str = "nearest", velocity: int = 90,
                  accent_velocity: int = 110, sustain: bool = False,
                  program: int = 0, file_name: str | None = None,
                  output_dir: str | None = None) -> dict:
    """Write a note sequence (scale, arpeggio or melody) to a single-track MIDI file.

    Plays the notes in order, one per `step_beats`. With `rhythm` (a pattern
    like 'O.oo.O..' from random_rhythm/euclidean_rhythm or hand-written), each
    step follows the pattern: O = accented note, o = soft note, . = pause
    (notes are consumed in order and wrap around; with sustain=true pauses
    extend the previous note). Octave-less notes are placed by `octave_policy`:
    'nearest' for melodies, 'ascending' for scale runs. `program` is a General
    MIDI instrument (0 piano, 24 guitar, 32 bass...). Returns the file path,
    base64 and the exact note events written.
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
    """Write a chord sequence to a single-track MIDI file (block chords, or arpeggiated).

    `chords` items are chord symbols ('C', 'Am7', 'F#dim', 'C/E', 'C4maj7' —
    e.g. the `symbols` output of degrees_to_chords) and/or explicit note
    arrays (['C','E','G'] or ['C4','E4','G4']). Octave-less chords are voiced
    upward from `octave`. Each chord lasts `beats_per_chord`. Returns the file
    path, base64 and each chord's voiced notes and MIDI numbers.
    """
    return _midi.render_chords(chords, beats_per_chord=beats_per_chord, tempo=tempo,
                               octave=octave, arpeggiate=arpeggiate, velocity=velocity,
                               program=program, file_name=file_name, output_dir=output_dir)


@mcp.tool()
def drums_to_midi(lanes: dict[str, str], step_beats: float = 0.5, tempo: int = 120,
                  velocity: int = 100, accent_velocity: int = 120,
                  file_name: str | None = None, output_dir: str | None = None) -> dict:
    """Write a drum pattern to a single-track General MIDI percussion file.

    `lanes` maps a drum name to a rhythm pattern, e.g.
    {"kick": "O...O...", "snare": "..O...O.", "hat": "oooooooo"} — each
    pattern uses O (accented hit), o (soft hit), . (rest), one step per
    `step_beats`. Lanes can be any length and play simultaneously. Drum names
    include kick, snare, side_stick, clap, closed_hat/open_hat/pedal_hat,
    low_tom/mid_tom/high_tom, crash, ride, tambourine, cowbell, clave, shaker,
    conga, bongo... (or a raw GM note number). Patterns from random_rhythm /
    euclidean_rhythm work directly as lanes.
    """
    return _midi.render_drums(lanes, step_beats=step_beats, tempo=tempo, velocity=velocity,
                              accent_velocity=accent_velocity, file_name=file_name,
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
                 file_name: str | None = None, output_dir: str | None = None) -> dict:
    """Write a melody plus chord accompaniment into one two-track MIDI file.

    A convenient shortcut for the common melody+chords case; for bass, drums
    or more tracks use arrange_to_midi. Track 1 plays `melody_notes` (optionally
    shaped by `melody_rhythm`, same rules as notes_to_midi); track 2 plays
    `chords` (same formats as chords_to_midi), one every `beats_per_chord`.
    Align lengths yourself: a melody over 4 chords of 4 beats at 0.5-beat steps
    needs a 32-step rhythm. `*_program` numbers pick GM instruments.
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


@mcp.tool()
def arrange_to_midi(tracks: list[dict], tempo: int = 120, step_beats: float = 0.5,
                    beats_per_chord: float = 4.0, file_name: str | None = None,
                    output_dir: str | None = None) -> dict:
    """Render any number of fitting tracks into one multi-track MIDI file — the full arrangement.

    This is the capstone "idea -> song" tool. You assemble the parts (the
    creative part) and it renders them together. `tracks` is a list of track
    objects, each of one of three types:

    - notes:  {"type":"notes", "notes":[...], "rhythm":"O.o.O.o.", "octave":5,
               "program":0, "octave_policy":"nearest", "sustain":false}  (melody/bass/arp)
    - chords: {"type":"chords", "chords":["Am","F","C","G"], "beats_per_chord":4,
               "octave":4, "arpeggiate":false, "program":0}              (pads/comping)
    - drums:  {"type":"drums", "lanes":{"kick":"O...O...","snare":"..O...O.","hat":"oooooooo"}}

    Shared per-track options: "name", "velocity", "start_beat" (beat offset for
    intros/drops), "step_beats", "channel" (auto-assigned; drums forced to the
    GM percussion channel). Align track lengths via start_beat and step counts.
    Typical full arrangement: a chords track, a bass notes track (chord roots,
    low octave, program 33), a melody notes track (program 0/80), and a drums
    track. Returns the file path, base64 and a per-track summary.
    """
    return _midi.render_arrangement(tracks, tempo=tempo, step_beats=step_beats,
                                    beats_per_chord=beats_per_chord,
                                    file_name=file_name, output_dir=output_dir)


@mcp.tool()
def midi_to_audio(midi_file: str, wav_file: str | None = None,
                  sample_rate: int = 44100) -> dict:
    """Synthesize a generated MIDI file into a playable WAV audio file.

    A .mid file needs a synthesizer/soundfont to be heard; this renders one to
    a self-contained 16-bit PCM WAV that plays on any device or browser, using
    a simple built-in synth (additive tones for pitched parts, percussive
    synthesis for General MIDI drums) — no soundfont required. Pass the `file`
    path returned by notes_to_midi / chords_to_midi / drums_to_midi /
    song_to_midi / arrange_to_midi. Returns the WAV path, duration and base64.
    This is a preview render, not a production mix.
    """
    return _audio.render_midi_to_wav(midi_file, wav_path=wav_file, sample_rate=sample_rate)


def main() -> None:
    """Run the MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
