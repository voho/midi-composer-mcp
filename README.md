# midi-composer-mcp

An [MCP](https://modelcontextprotocol.io) server that gives an LLM a **large palette of deterministic music-theory and composition tools**, so a composer can state a goal and the LLM finds the best way to achieve it by linking the tools into a composition draft — from "give me the notes of this scale" to a full multi-track song you can actually play.

The guiding split: **the tools contain the rules, the LLM contains the creativity.** Every tool is a small, deterministic step — scales and chords, diatonic harmony, intervals, voice leading, reharmonization, the circle of fifths, motif grammars, sequences, tintinnabuli, species counterpoint, song structure, MIDI and audio rendering. A tool never decides what is "good"; it mechanically applies a rule. The LLM decides *which* rules to invoke and *how* to combine them, so the music follows real theory and is not random.

Two more invariants: all tools are **compatible** (the note/chord/degree/rhythm output of one is valid input to another), and **randomness is contained** in a few clearly-named, seeded tools (`random_notes`, `random_rhythm`) — everything else is deterministic. See [`CLAUDE.md`](CLAUDE.md) for the full design principles.

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
| `harmonize_melody` | Put a chord under each melody note — searching the **whole** chord database — that reuses as many notes from the previous chord as possible, then voice-leads it. Returns ranked options + a `render_hint`. |

### Melody

| Tool | What it does |
|---|---|
| `notes_from_degrees` | Write a melody as scale degrees → notes; transposable to any key/scale. `[1,2,3,5,8]` in C → `C D E G C`. |
| `motif_grammar` | Build a phrase from a form like `ABAC` over labeled motifs; a variant can `transpose`/`invert`/`retrograde`/`rotate` another. Works on notes, degrees, or rhythm. |
| `melodic_walk` | A singable line by a seeded random walk over a scale ladder (mostly stepwise). |
| `melodic_sequence` | Repeat a motif as a diatonic sequence (e.g. down a step each time). |
| `arpeggiate` | Reorder a chord/scale into an arpeggio (up/down/updown/converge/…, multi-octave). |
| `tintinnabuli_voice` | **Arvo Pärt's tintinnabuli:** shadow a melody with the nearest notes of a fixed triad (T1/T2, above/below/alternating). |
| `counterpoint` | **Species counterpoint (1–5):** a rule-following counter-melody to a cantus firmus — note-against-note through florid, with passing tones and resolving suspensions, no parallel fifths/octaves. Returns a `render_hint` of ready tracks. |
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

## Examples

Each example is a sequence of tool calls. The composer states a goal; the LLM chains tools to reach it. Outputs feed the next call — that's the whole idea.

### Simple

**"Give me the notes of E Dorian."**
```
get_scale("dorian", "E")            → E F# G A B C# D E
```

**"What chord do the notes C, E, G make? And what scales fit them?"**
```
match_chords(["C", "E", "G"])       → C (exact);  "E G C" → C/E (first inversion)
match_scales(["C", "E", "G"])       → C major pentatonic, C major, A minor, …
```
(Octaves are ignored, so `["C5","E5","G5"]` gives the same answer.)

**"A ii–V–I in F, with sevenths."**
```
degrees_to_chords("F", "major", "ii V I", sevenths=True)   → Gm7  C7  Fmaj7
```

**"A random melody from A minor pentatonic, then save it as MIDI."**
```
get_scale("minor pentatonic", "A5")            → A5 C6 D6 E6 G6 A6   (octave-aware)
random_notes(<those notes>, count=8, seed=1)   → a reproducible 8-note line  [contained randomness]
notes_to_midi(<the notes>, tempo=120)          → a .mid file (+ base64)
midi_to_audio(<that file>)                     → a playable .wav
```

### Intermediate

**"Build a pop loop: I–V–vi–IV in C with a bass, a hook, and a backbeat."**
```
voice_leading(["C","G","Am","F"])                          → smooth pad voicings
notes_from_degrees("C5","major",[5,5,6,5,3,2,1,1])         → a diatonic hook
groove("backbeat"); groove("four_on_floor")               → drum patterns
arrange_to_midi([                                          → one 4-track .mid
  {"type":"chords","name":"pad","chords":<voicings>,"beats_per_chord":4},
  {"type":"notes","name":"bass","notes":["C","G","A","F"],"step_beats":4,"octave":2,"program":33},
  {"type":"notes","name":"lead","notes":<hook>,"octave":5,"program":80},
  {"type":"drums","name":"drums","step_beats":0.25,"lanes":{"kick":"O...O...O...O...","snare":"....O.......O...","hat":"o.o.o.o.o.o.o.o."}},
])
```

**"Reharmonize G7→C and analyze it."**
```
tritone_substitute("G7")                  → Db7   (chromatic bass G→Db→C)
secondary_dominant("Dm")                  → A7    (V7 of ii)
analyze_progression(["C","A7","Dm","G7","C"], "C", "major")
                                          → I, V7/ii (chromatic), ii, V7, I
```

**"Where can I modulate from C major?"**
```
circle_of_fifths("C")    → dominant G, subdominant F, relative A minor,
                           closely related: A minor, G major, E minor, F major, D minor
```

**"Put chords under this melody, reusing as many notes as possible between chords."**
```
harmonize_melody(["C5","E5","F5","A5","G5"], root="C", scale_type="major", in_scale=True)
   → searches the whole chord DB for chords containing each note, ranks them by shared
     notes with the previous chord, picks the smoothest, and voice-leads:
     C → C → Cadd4 → Am7 → C6 …   (each chord keeps 3 notes from the last)
   → plus ranked `options` per note and a render_hint (harmony + melody) for arrange_to_midi
```

### Advanced

**"Write a third-species counterpoint to a cantus firmus."**
```
counterpoint(["C5","D5","E5","F5","E5","D5","C5"], "C", "major", species=3)
   → cantus + a 4:1 counter-line (passing tones, perfect-consonance cadence, no parallel 5ths/8ves)
   → plus render_hint.tracks  →  arrange_to_midi(<render_hint tracks>)  →  midi_to_audio(…)
```

**"Develop a melody by motif grammar (ABAC), kept in key."**
```
motif_grammar("ABAC", {                                    # kind="degrees" stays diatonic
  "A":[1,2,3,5], "B":{"vary":"A","transpose":1}, "C":{"vary":"A","retrograde":true}}, kind="degrees")
notes_from_degrees("C5","major", <those degrees>)          → the realized, in-key phrase
```

**"Compose with tintinnabuli rules over a few maj7 chords, with two verses and a chorus."**
```
# Verse M-voice (A minor) + its tintinnabuli T-voice, over voice-led maj7/m7 pads:
m = notes_from_degrees("A4","natural minor",
       motif_grammar("ABAC", {"A":[1,2,3,2],"B":{"vary":"A","transpose":1},"C":[3,2,1,1]}, kind="degrees")["degrees"])
t = tintinnabuli_voice(m, "Am", position="inferior", rank=1)          # nearest A-minor triad note below each M note
verse_pads  = voice_leading(["Am7","Dm7","Fmaj7","Cmaj7"])["chords"]
chorus_pads = voice_leading(["Fmaj7","Cmaj7","Dm7","Em7"])["chords"]

arrange_song({                                                        # sequence sections into a song
  "verse":  {"bars":4, "tracks":[
     {"type":"chords","name":"pads","chords":verse_pads,"beats_per_chord":4,"program":89},
     {"type":"notes","name":"M-voice","notes":m,"step_beats":2,"octave":5,"program":48,"sustain":true},
     {"type":"notes","name":"T-voice","notes":t,"step_beats":2,"octave":4,"program":9,"sustain":true}]},
  "chorus": {"bars":4, "tracks":[
     {"type":"chords","name":"pads","chords":chorus_pads,"beats_per_chord":4,"program":89},
     {"type":"notes","name":"M-voice","notes":notes_from_degrees("C5","major",[5,6,8,6,5,3,2,1])["notes"],"step_beats":2,"octave":5,"program":48,"sustain":true},
     {"type":"notes","name":"bass","notes":["F","C","D","E"],"step_beats":4,"octave":2,"program":33}]},
}, form="verse verse chorus", tempo=72)   →   midi_to_audio(<the song>)
```

These advanced examples (a Pärt tintinnabuli study, a species-3 counterpoint, the tintinnabuli verse/chorus song, and a full verse/chorus/bridge song) are runnable in **`examples/generate_examples.py`**:

```bash
python examples/generate_examples.py            # writes .mid + .wav for each
```

### Demo gallery

The [**`demos/`**](demos/) folder is a gallery of finished pieces, each paired with the plain-language **prompt** it answers — from a Pärt-style tintinnabuli study to a modulating pop anthem, a jazz reharmonization, all five counterpoint species, a flamenco piece in Phrygian dominant, and a negative-harmony before/after. The `.mid` files are committed (open them in a DAW); regenerate everything with:

```bash
python demos/generate.py                        # rewrites demos/*.mid and *.wav
```

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
