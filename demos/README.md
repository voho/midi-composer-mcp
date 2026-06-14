# Demos

A gallery of pieces composed **only from this server's tools** — each one starts
from a composer's plain-language goal (the *prompt*), which the LLM realizes by
linking deterministic rule-tools together. The result follows real music theory;
it is not random.

Every piece is reproducible: `python demos/generate.py` rewrites the `.mid` and
`.wav` files from the prompts below. The `.mid` files are the canonical output
(open them in any DAW for real instrument sounds); the `.wav` files are quick
previews from the built-in sine synth (a couple of the longer ones are MIDI-only
here to keep the repo light — run the generator to render their audio).

---

### 1. Tintinnabuli — Arvo Pärt style

> *"Write a slow, meditative Arvo Pärt-style tintinnabuli piece in A minor — a melody that grows in descending phrases out of the tonic, shadowed by the notes of the A-minor triad, with a tolling bell, a drone, and a few soft maj7/m7 colours."*

**Showcases:** the `tintinnabuli_voice` rule (the T-voice is always a triad note, consonant by construction), additive melodic phrases via `notes_from_degrees`, and `voice_leading` over maj7/m7 pads.
🎵 [01_tintinnabuli_cantus.mid](01_tintinnabuli_cantus.mid) · 🔊 [.wav](01_tintinnabuli_cantus.wav)

### 2. Anthem — song structure + modulation

> *"Write an uplifting pop anthem in C major: intro, verse, pre-chorus, a big chorus, a bridge, and a final chorus that modulates up a whole step."*

**Showcases:** `arrange_song` sequencing intro/verse/pre/chorus/bridge/outro into continuous tracks, `voice_leading` pads, a `motif_grammar` melody, groove drums, and `transpose_notes` for the key change.
🎵 [02_anthem.mid](02_anthem.mid) · 🔊 *render with `python demos/generate.py`*

### 3. Jazz reharmonization

> *"Take a ii-V-I turnaround in C and make it jazzier — add secondary dominants and a tritone substitution, then comp it with smooth voicings and a walking bass."*

**Showcases:** `secondary_dominant`, `tritone_substitute`, `analyze_progression` (Roman numerals), `voice_leading` comping, and a root-fifth walking bass with a voice-led guide-tone head.
🎵 [03_jazz_reharm.mid](03_jazz_reharm.mid) · 🔊 [.wav](03_jazz_reharm.wav)

### 4. Counterpoint — all five species

> *"Write a counterpoint to this cantus firmus and show me all five species, from note-against-note to florid."*

**Showcases:** the `counterpoint` engine across species 1–5 (1:1, 2:1, 4:1, syncopated suspensions, florid) over one cantus firmus, stitched into one piece with `arrange_song`.
🎵 [04_counterpoint_species.mid](04_counterpoint_species.mid) · 🔊 *render with `python demos/generate.py`*

### 5. Flamenco — exotic scale + clave

> *"Give me a flamenco piece in E Phrygian dominant — an improvised-sounding guitar line over a rumba clave with hand percussion."*

**Showcases:** the exotic Phrygian-dominant scale, a `melodic_walk` line (seeded), the `groove("rumba_clave_32")` preset, and named drum lanes (clave, conga, palmas).
🎵 [05_flamenco.mid](05_flamenco.mid) · 🔊 [.wav](05_flamenco.wav)

### 6. Negative harmony — before / after

> *"Play a I-vi-IV-V with a melody in C, then play its negative-harmony mirror so I can hear major flip to its minor shadow."*

**Showcases:** the `negative_harmony` transformation applied to chords, bass, and melody — the original section followed by its mirror, so the flip is audible.
🎵 [06_negative_harmony.mid](06_negative_harmony.mid) · 🔊 [.wav](06_negative_harmony.wav)

---

For the source that builds these, see [`generate.py`](generate.py); for library-level
worked examples, see [`../examples/generate_examples.py`](../examples/generate_examples.py).
