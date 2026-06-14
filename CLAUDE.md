# CLAUDE.md

## What this project is

An MCP server that gives an LLM a **large palette of deterministic music-theory
and composition tools**, so that a composer can state a goal in plain language
and the LLM can find the best way to achieve it by **linking these tools
together into a composition draft**. The tools supply the *techniques and
rules* (harmony, melody, counterpoint, rhythm, structure); the LLM supplies the
*creativity and the linking*. The result follows real musical rules — it is not
random noise.

The guiding split: **the tools contain the rules, the LLM contains the
creativity.** A tool never decides what is "good"; it mechanically applies a
rule (build this scale, voice these chords smoothly, derive the tintinnabuli
voice, write a first-species counterpoint, lay out these sections). The LLM
decides *which* rules to invoke and *how* to combine them to serve the goal.

## Design principles (read before adding or changing tools)

1. **Tools provide harmony, melody and composition rules — deterministically.**
   Given the same inputs, a tool always returns the same output. Scales, chords,
   diatonic harmony, intervals, voice leading, reharmonization, the circle of
   fifths, motif grammars, sequences, tintinnabuli, species counterpoint, song
   structure — all are mechanical rule applications, not heuristics that vary
   run to run.

2. **Tools are compatible — the output of one is valid input to another.**
   There is a small shared vocabulary that every tool speaks:
   - **notes** as strings: `"C"`, `"F#"`, `"Bb"` (pitch classes) or with an
     octave `"C4"`, `"Eb3"` (concrete pitches; C4 = middle C = MIDI 60). Note
     lists may be arrays (`["C","E","G"]`) or strings (`"c e g"`).
   - **chords** as symbols (`"Am7"`, `"G7"`, `"Cmaj13"`, `"C/E"`) or as note
     arrays.
   - **scale degrees** as integers (`[1,2,3,5]`, octave-wrapping, 1-based).
   - **rhythm** as patterns: `O` strong beat, `o` weak beat, `.` rest.
   - **tracks** and **sections** as plain objects for the renderers/assembler.
   So `get_scale` → `notes` feeds `match_chords`, `random_notes`,
   `notes_to_midi`; `degrees_to_chords` → `symbols` feeds `voice_leading`,
   `chords_to_midi`, a section; `tintinnabuli_voice` → two note lists feed two
   notes tracks; `counterpoint` returns a `render_hint` of ready tracks. When
   adding a tool, take and return these same shapes so it chains.

3. **Randomness is clearly marked and contained in separate tools.**
   The only stochastic tools are the ones named for it — `random_notes`,
   `random_rhythm` — and the explicitly stochastic `arpeggiate(style="random")`.
   They always accept a `seed` and always return the seed used, so any result is
   reproducible. Everything else is deterministic. Do not sprinkle randomness
   into rule tools; if a new generator needs chance, make it a separate,
   clearly-named, seeded tool.

4. **Some tools are purely technical** (no musical opinion): `notes_to_midi`,
   `chords_to_midi`, `drums_to_midi`, `song_to_midi`, `arrange_to_midi`,
   `arrange_song`, and `midi_to_audio` (renders a `.mid` to a playable WAV with
   a built-in synth, no soundfont needed). They render exactly what they are
   given.

5. **Keep `README.md` up to date.** It must always describe **every** tool and
   carry **enough examples across the difficulty range** — from the simple
   ("give me the notes of this scale", "what chord do these notes make", "a
   ii–V–I in F") to the advanced ("compose with tintinnabuli rules over a few
   maj7 chords, with two verses and a chorus", "a third-species counterpoint to
   this cantus"). Update the README and this file in the same change as any tool
   addition.

## Layer map

```
src/midi_composer_mcp/
  notes.py        # note parsing, spelling, octaves, MIDI numbers (the vocabulary)
  scales.py       # scale database (40+), generation, matching
  chords.py       # chord database (35+), symbols, generation, matching
  diatonic.py     # chords per scale degree, degree-sequence resolution
  circle.py       # circle of fifths: key signatures and related keys
  harmony.py      # intervals, roman-numeral analysis, voice leading, reharmonization
  melody.py       # degrees, arpeggios, walks, motif grammar, sequence, snap, tintinnabuli
  counterpoint.py # species counterpoint 1-5 (deterministic, rule-following)
  generate.py     # the contained randomness + euclidean rhythm + groove presets
  structure.py    # song structure: plan sections, assemble a whole song
  midi_io.py      # technical: deterministic MIDI rendering (mido)
  audio.py        # technical: MIDI -> playable WAV, pure standard library
  server.py       # FastMCP server: thin wrappers over the above
```

## Conventions for new tools

- Atomic and single-purpose; the docstring is the LLM-facing spec — say what it
  does, the input/output shapes, and one concrete example.
- Deterministic, or clearly seeded (principle 3).
- Take and return the shared vocabulary (principle 2).
- Validate inputs and raise clear `ValueError`s.
- Add tests (`tests/`), and update `README.md` tables **and** examples.

## Development

```bash
uv venv && uv pip install -e ".[dev]"
.venv/bin/python -m pytest        # full suite
```

`examples/generate_examples.py` renders worked pieces end to end (style demos,
an Arvo Pärt tintinnabuli study, a full verse/chorus/bridge song) using only
these tools — a good reference for how the layers link together.
