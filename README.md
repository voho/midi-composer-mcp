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

| Tool | What it does (deterministic unless noted) |
|---|---|
| `list_scales` | Scale-type database: intervals, degree labels, aliases, and a **one-line description** of each (40+ scales: common, modal, jazz, symmetric, world/exotic). |
| `list_chords` | Chord-type database: intervals, degrees, symbol suffixes, aliases, **descriptions** (35+ chords: triads → 13ths and altered). |
| `get_scale` | Describe a scale type; with a root, generate its notes. `maj + C → C D E F G A B C`. |
| `get_chord` | Describe a chord type; with a root, generate its notes. `min + F → F Ab C`. |
| `match_scales` | Scales containing the given notes (**octaves ignored**), exact matches flagged. |
| `match_chords` | Chords matching the given notes (**octaves ignored**); `c e g → C`, `e g c → C/E` (first inversion), partial matches list missing notes. |
| `diatonic_chords` | The chord on each scale degree, with roman numerals, degree names and harmonic functions (tonic/subdominant/dominant). |
| `degrees_to_chords` | Resolve a degree sequence *you* chose (`[1,5,6,4]`, `"I V vi IV"`, `"1-5-6-4"`) into concrete chords. |
| `random_notes` | 🎲 Uniform random picks from any note pool (seeded, reproducible). |
| `random_rhythm` | 🎲 Random rhythm pattern like `O...Oo..` — `O` strong beat, `o` weak beat, `.` pause (seeded, reproducible). |
| `euclidean_rhythm` | Evenly-spread rhythm (Bjorklund); `euclidean_rhythm(3,8)` → `O..o..o.` (tresillo). Great for basslines and drums. |
| `notes_to_midi` | Render a note sequence (scale, arpeggio, melody) to a `.mid` file, optionally shaped by a rhythm pattern. |
| `chords_to_midi` | Render chord symbols and/or note arrays to a `.mid` file (block or arpeggiated). |
| `drums_to_midi` | Render named drum lanes (`{"kick":"O...","snare":"..O.","hat":"oooo"}`) to a General MIDI percussion file. |
| `song_to_midi` | Render melody + chord accompaniment into one two-track `.mid` file (shortcut for the common case). |
| `arrange_to_midi` | **The capstone:** render any number of fitting tracks — chords, bass, melody, drums — into one multi-track `.mid` file. |
| `midi_to_audio` | Render any generated `.mid` into a **playable WAV** with a built-in synth (no soundfont needed) so the output is audible on any device. |

MIDI/audio tools write to `./midi_output` (override per call with `output_dir` or globally with the `MIDI_COMPOSER_OUTPUT_DIR` environment variable) and also return the file base64-encoded.

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
  notes.py      # note parsing, proper spelling, octaves, MIDI numbers
  scales.py     # scale database (40+, described), generation, matching
  chords.py     # chord database (35+, described), symbols, generation, matching
  diatonic.py   # chords per scale degree, degree-sequence resolution
  generate.py   # seeded dice: random notes, random rhythm, euclidean rhythm
  midi_io.py    # deterministic MIDI rendering: notes, chords, drums, multi-track arrangements (mido)
  audio.py      # MIDI -> playable WAV preview, pure standard library
  server.py     # the MCP server (FastMCP) — thin wrappers over the above
```

## Roadmap ideas

- Transposition and inversion helpers
- Humanize options (timing/velocity jitter as a seeded, mechanical step)
- Reading MIDI files back into note/chord data
