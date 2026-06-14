# midi-composer-mcp

An [MCP](https://modelcontextprotocol.io) server that gives an LLM **atomic music-theory and MIDI tools** for composing — from an idea all the way to a multi-track MIDI file you can actually play.

The design principle: **the tools contain no creativity**. Every tool is a small, deterministic (or seeded-random) step — looking up scales and chords, matching notes, resolving degree sequences, rolling dice, rendering MIDI and audio. The LLM is the composer: it chains the tools, makes every musical choice, and the tools do the mechanical work correctly.

All tools are **compatible with each other**: notes, chord symbols and rhythm patterns returned by one tool are valid inputs to every other tool.

## Note format

- Notes are strings: `C`, `F#`, `Bb`, `Ebb` (case-insensitive, unicode `♯`/`♭` accepted).
- A note **without an octave** is an abstract pitch class.
- A note **with an octave** is a concrete pitch: `C4` is middle C (MIDI 60), `Eb3`, `A5`...
  Generation respects it: `get_scale("major", "C5")` → `C5 D5 E5 F5 G5 A5 B5 C6` with MIDI numbers; `get_chord("9", "C4")` → `C4 E4 G4 Bb4 D5`.
- **Matching ignores octaves**: `match_chords(["E3","G4","C5"])` → `C/E` (first inversion), exactly as `["E","G","C"]` would.
- Note lists may be JSON arrays (`["C", "E", "G"]`) or plain strings (`"c e g"`, `"C, E, G"`).
- Spelling is proper: F major has a `Bb` (not `A#`), Cdim7 has a `Bbb`.

## Tools

The toolset is organized by **layer** — scales, chords, harmony rules, melody, rhythm, song structure, and rendering — so the LLM can go from an idea to a finished multi-track song. Everything is deterministic (seeded where random).

### Scales & chords

| Tool | What it does |
|---|---|
| `list_scales` / `get_scale` | 40+ scale types (common, modal, jazz, symmetric, world/exotic), each with a description; generate notes from a root. `maj + C → C D E F G A B C`. |
| `list_chords` / `get_chord` | 35+ chord types (triads → 13ths and altered), each with a description; generate notes. `min + F → F Ab C`. |
| `match_scales` / `match_chords` | Find scales/chords containing given notes (**octaves ignored**); inversions detected (`e g c → C/E`), partials list missing notes. |
| `diatonic_chords` | The chord on each scale degree, with roman numerals, degree names and harmonic functions. |
| `degrees_to_chords` | Resolve a chosen degree sequence (`[1,5,6,4]`, `"I V vi IV"`) into concrete chords. |

### Harmony rules

| Tool | What it does |
|---|---|
| `circle_of_fifths` | Key signatures, relative/parallel minors, and closely related keys (for modulations and bridges). |
| `interval_between` | Name the interval between two notes (`C→Eb = m3`, `C→F# = A4` vs `C→Gb = d5`). |
| `analyze_progression` | The inverse of `degrees_to_chords`: chords → roman numerals + functions, chromatic chords flagged. |
| `voice_leading` | Voice a progression smoothly (nearest inversion, common tones held) — natural pads instead of parallel blocks. |
| `secondary_dominant` / `tritone_substitute` | Classic reharmonizations (`V/ii of Dm → A7`; `G7 → Db7`). |
| `negative_harmony` | Reflect notes through a key's negative-harmony axis (major ↔ minor shadow). |

### Melody

| Tool | What it does |
|---|---|
| `notes_from_degrees` | Write a melody as scale degrees → notes; transposable to any key/scale. `[1,2,3,5,8]` in C → `C D E G C`. |
| `motif_grammar` | Build a phrase from a form like `ABAC` over labeled motifs; a variant can `transpose`/`invert`/`retrograde`/`rotate` another. Works on notes, degrees, or rhythm. |
| `melodic_walk` | A singable line by a seeded random walk over a scale ladder (mostly stepwise). |
| `melodic_sequence` | Repeat a motif as a diatonic sequence (e.g. down a step each time). |
| `arpeggiate` | Reorder a chord/scale into an arpeggio (up/down/updown/converge/…, multi-octave). |
| `tintinnabuli_voice` | **Arvo Pärt's tintinnabuli:** shadow a melody with the nearest notes of a fixed triad (T1/T2, above/below/alternating). |
| `counterpoint` | **First-species counterpoint:** a rule-following counter-melody to a cantus firmus (consonances only, contrary motion, no parallel fifths/octaves). |
| `snap_to_scale` | Snap any line to the nearest scale notes — guarantees a melody fits the key/chords. |
| `transpose_notes` | Transpose a note list by semitones. |
| `random_notes` | 🎲 Uniform random picks from any note pool (seeded). |

### Rhythm

| Tool | What it does |
|---|---|
| `random_rhythm` | 🎲 Random pattern `O...Oo..` — `O` strong, `o` weak, `.` rest (seeded). |
| `euclidean_rhythm` | Evenly-spread Bjorklund rhythm; `euclidean_rhythm(3,8) → O..o..o.` (tresillo). |
| `groove` / `list_grooves` | Named presets: four-on-the-floor, backbeat, tresillo, son/rumba clave, bossa nova, dembow… |

### Song structure & rendering

| Tool | What it does |
|---|---|
| `plan_sections` | Lay out a form (`"intro verse chorus … outro"` / `"AABA"`) on the timeline — start bars, beats, seconds. |
| `arrange_song` | **The capstone:** assemble named sections (intro/verse/chorus/bridge/outro) into one whole-song MIDI; like-named tracks stitch into continuous parts. |
| `notes_to_midi` / `chords_to_midi` / `drums_to_midi` | Render a single track (melody/scale, chords block-or-arpeggiated, GM drum lanes). |
| `arrange_to_midi` | Render any number of fitting tracks (chords, bass, melody, drums) into one multi-track `.mid`. |
| `song_to_midi` | Melody + chords as a two-track file (shortcut for the common case). |
| `midi_to_audio` | Render any generated `.mid` into a **playable WAV** with a built-in synth (no soundfont needed). |

MIDI/audio tools write to `./midi_output` (override per call with `output_dir` or globally with `MIDI_COMPOSER_OUTPUT_DIR`) and also return the file base64-encoded.

See `examples/generate_examples.py` for worked pieces (an Arvo Pärt tintinnabuli study and a full verse/chorus/bridge song) built entirely from these tools.

## Playable output

A bare `.mid` is a valid Standard MIDI File (Format 1, tempo map, General MIDI programs, drums on channel 10) that plays in any DAW or synth — but it needs a soundfont to be *heard*. `midi_to_audio` solves that: it synthesizes the MIDI into a 16-bit PCM **WAV** using only the Python standard library (additive tones for pitched parts, percussive synthesis for drums), so every result is playable anywhere — no soundfont, no external synth. It's a faithful preview, not a production mix.

## A composing session looks like this

The LLM drives; each tool call is one mechanical step:

1. `get_scale("harmonic minor", "C")` → `C D Eb F G Ab B` (+ a description of the scale's character)
2. `diatonic_chords("C", "harmonic minor", sevenths=true)` → the 7th chord on each degree, with roman numerals and functions
3. *LLM decides on* `i–iv–V–i` → `degrees_to_chords("C", "harmonic minor", "i iv V i", sevenths=true)` → `CmMaj7 Fm7 G7 CmMaj7`
4. `euclidean_rhythm(5, 16)` → `O..o..o..o..o...` for a bass groove
5. `random_notes` / hand-written melody from the scale notes
6. `arrange_to_midi([...pad, bass, lead, drums...])` → a four-track `.mid`
7. `midi_to_audio(file)` → a `.wav` you can play immediately

Every intermediate result is plain data the LLM can inspect, edit by hand (tweak a rhythm string, swap a chord), or feed into another tool.

### Arrangement track shapes (for `arrange_to_midi`)

```jsonc
[
  {"type":"chords","name":"pad",   "chords":["Am","F","C","G"], "beats_per_chord":4, "octave":4, "program":89},
  {"type":"notes", "name":"bass",  "notes":["A","F","C","G"],   "rhythm":"O..o..o..o..o...", "octave":2, "program":33, "step_beats":0.25},
  {"type":"notes", "name":"lead",  "notes":["A4","C5","E5","D5"], "octave":5, "program":0},
  {"type":"drums", "name":"drums", "lanes":{"kick":"O...O...","snare":"..O...O.","hat":"oooooooo"}}
]
```
Shared per-track options: `name`, `velocity`, `start_beat` (beat offset for intros/drops), `step_beats`, `channel` (auto-assigned; drums forced to the GM percussion channel).

## Installation

Requires Python ≥ 3.10.

```bash
# with uv (recommended)
uv pip install .          # or: uv sync && uv run midi-composer-mcp

# or with pip
pip install .
```

Run the server (stdio transport):

```bash
midi-composer-mcp
# or without installing:
uv run --with mcp --with mido python -m midi_composer_mcp.server
```

### Claude Code

```bash
claude mcp add midi-composer -- uv run --directory /path/to/midi-composer-mcp midi-composer-mcp
```

### Claude Desktop

```json
{
  "mcpServers": {
    "midi-composer": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/midi-composer-mcp", "midi-composer-mcp"],
      "env": { "MIDI_COMPOSER_OUTPUT_DIR": "/path/to/your/midi/files" }
    }
  }
}
```

## Development

```bash
uv venv && uv pip install -e ".[dev]"
.venv/bin/python -m pytest
```

Layout:

```
src/midi_composer_mcp/
  notes.py        # note parsing, proper spelling, octaves, MIDI numbers
  scales.py       # scale database (40+, described), generation, matching
  chords.py       # chord database (35+, described), symbols, generation, matching
  diatonic.py     # chords per scale degree, degree-sequence resolution
  circle.py       # circle of fifths: key signatures and related keys
  harmony.py      # intervals, roman-numeral analysis, voice leading, reharmonization
  melody.py       # degrees, arpeggios, walks, motif grammar, sequence, snap, tintinnabuli
  counterpoint.py # first-species counterpoint (deterministic, rule-following)
  generate.py     # seeded dice + euclidean rhythm + groove presets
  structure.py    # song structure: plan sections, assemble a whole song
  midi_io.py      # deterministic MIDI rendering: notes, chords, drums, multi-track (mido)
  audio.py        # MIDI -> playable WAV preview, pure standard library
  server.py       # the MCP server (FastMCP) — thin wrappers over the above
```

## Roadmap ideas

- Rhythmic chord comping (a `rhythm` on chord tracks, for stabs/funk/reggae)
- Swing/shuffle and humanize (timing/velocity jitter as a seeded, mechanical step)
- Higher-species counterpoint
- Reading MIDI files back into note/chord data
