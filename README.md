# midi-composer-mcp

An [MCP](https://modelcontextprotocol.io) server that gives an LLM **atomic music-theory and MIDI tools** for composing.

The design principle: **the tools contain no creativity**. Every tool is a small, deterministic (or seeded-random) step — looking up scales and chords, matching notes, resolving degree sequences, rolling dice, rendering MIDI. The LLM is the composer: it chains the tools, makes every musical choice, and the tools do the mechanical work correctly.

All tools are **compatible with each other**: notes, chord symbols and rhythm patterns returned by one tool are valid inputs to every other tool.

## Note format

- Notes are strings: `C`, `F#`, `Bb`, `Ebb` (case-insensitive, unicode `♯`/`♭` accepted).
- A note **without an octave** is an abstract pitch class.
- A note **with an octave** is a concrete pitch: `C4` is middle C (MIDI 60), `Eb3`, `A5`...
  Generation respects it: `get_scale("major", "C5")` → `C5 D5 E5 F5 G5 A5 B5 C6` with MIDI numbers; `get_chord("9", "C4")` → `C4 E4 G4 Bb4 D5`.
- Note lists may be JSON arrays (`["C", "E", "G"]`) or plain strings (`"c e g"`, `"C, E, G"`).
- Spelling is proper: F major has a `Bb` (not `A#`), Cdim7 has a `Bbb`.

## Tools

| Tool | What it does (deterministic unless noted) |
|---|---|
| `list_scales` | Scale-type database: intervals, degree labels, aliases. |
| `list_chords` | Chord-type database: intervals, degrees, symbol suffixes, aliases. |
| `get_scale` | Describe a scale type; with a root, generate its notes. `maj + C → C D E F G A B C`. |
| `get_chord` | Describe a chord type; with a root, generate its notes. `min + F → F Ab C`. |
| `match_scales` | Scales containing the given notes (octave-insensitive), exact matches flagged. |
| `match_chords` | Chords matching the given notes; `c e g → C`, `e g c → C/E` (first inversion), partial matches list missing notes. |
| `diatonic_chords` | The chord on each scale degree, with roman numerals, degree names and harmonic functions (tonic/subdominant/dominant). |
| `degrees_to_chords` | Resolve a degree sequence *you* chose (`[1,5,6,4]`, `"I V vi IV"`, `"1-5-6-4"`) into concrete chords. |
| `random_notes` | 🎲 Uniform random picks from any note pool (seeded, reproducible). |
| `random_rhythm` | 🎲 Random rhythm pattern like `O...Oo..` — `O` strong beat, `o` weak beat, `.` pause (seeded, reproducible). |
| `notes_to_midi` | Render a note sequence (scale, arpeggio, melody) to a `.mid` file, optionally shaped by a rhythm pattern. |
| `chords_to_midi` | Render chord symbols and/or note arrays to a `.mid` file (block or arpeggiated). |
| `song_to_midi` | Render melody + chord accompaniment into one two-track `.mid` file. |

MIDI tools write to `./midi_output` (override per call with `output_dir` or globally with the `MIDI_COMPOSER_OUTPUT_DIR` environment variable) and also return the file base64-encoded, plus the exact events written.

## A composing session looks like this

The LLM drives; each tool call is one mechanical step:

1. `get_scale("minor", "A")` → `A B C D E F G A`
2. `diatonic_chords("A", "minor")` → `i=Am, ii°=Bdim, bIII=C, iv=Dm, v=Em, bVI=F, bVII=G` *(+ harmonic functions)*
3. *LLM decides on* `i–VI–III–VII` → `degrees_to_chords("A", "minor", "i VI III VII")` → `Am F C G`
4. `random_rhythm(length=32, seed=42)` → `.Oo..oOOoOoo.O.Oo...oo...oOOOooO`
5. `random_notes(notes=<scale notes>, count=21, seed=42)` → melody material *(or the LLM writes the melody itself)*
6. `song_to_midi(melody_notes=..., melody_rhythm=..., chords=["Am","F","C","G"])` → `song.mid`

Every intermediate result is plain data the LLM can inspect, edit by hand (e.g. tweak the rhythm string), or feed into another tool — e.g. `get_chord("m7","A")` → `match_scales(...)` to find scales for a solo.

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
  scales.py     # scale database, generation, matching
  chords.py     # chord database, symbols, generation, matching
  diatonic.py   # chords per scale degree, degree-sequence resolution
  generate.py   # seeded dice: random notes, random rhythm
  midi_io.py    # deterministic MIDI rendering (mido)
  server.py     # the MCP server (FastMCP) — thin wrappers over the above
```

## Roadmap ideas

- Transposition and inversion helpers
- Humanize options (timing/velocity jitter as a seeded, mechanical step)
- Reading MIDI files back into note/chord data
