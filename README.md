# midi-composer-mcp

An [MCP](https://modelcontextprotocol.io) server that gives an LLM a **large palette of deterministic music-theory and composition tools**, so a composer can state a goal and the LLM finds the best way to achieve it by linking the tools into a composition draft — from "give me the notes of this scale" to a full multi-track song you can actually play.

The guiding split: **the tools contain the rules, the LLM contains the creativity.** Every tool is a small, deterministic step — scales and chords, diatonic harmony, intervals, voice leading, reharmonization, the circle of fifths, motif grammars, sequences, tintinnabuli, species counterpoint, song structure, MIDI and audio rendering. A tool never decides what is "good"; it mechanically applies a rule. The LLM decides *which* rules to invoke and *how* to combine them, so the music follows real theory and is not random.

Two more invariants: all tools are **compatible** (the note/chord/degree/rhythm output of one is valid input to another), and **randomness is contained** in a few clearly-marked, seeded tools (`random_notes`, `random_rhythm`, `melodic_walk`, and `arpeggiate(style="random")`) — each takes a `seed` and returns the seed it used; everything else is deterministic. See [`CLAUDE.md`](CLAUDE.md) for the full design principles.

## Note format

- Notes are strings: `C`, `F#`, `Bb`, `Ebb` (case-insensitive, unicode `♯`/`♭` accepted).
- A note **without an octave** is an abstract pitch class.
- A note **with an octave** is a concrete pitch: `C4` is middle C (MIDI 60), `Eb3`, `A5`...
  Generation respects it: `get_scale("major", "C5")` → `C5 D5 E5 F5 G5 A5 B5 C6` with MIDI numbers; `get_chord("9", "C4")` → `C4 E4 G4 Bb4 D5`.
- **Matching ignores octaves**: `match_chords(["E3","G4","C5"])` → `C/E` (first inversion), exactly as `["E","G","C"]` would.
- Note lists may be JSON arrays (`["C", "E", "G"]`) or plain strings (`"c e g"`, `"C, E, G"`).
- Spelling is proper: F major has a `Bb` (not `A#`), Cdim7 has a `Bbb`.

## Tools

The toolset is organized by **layer** — scales, chords, harmony rules, melody, accompaniment, analysis, rhythm, song structure, and rendering — so the LLM can go from an idea to a finished multi-track song. Everything is deterministic (seeded where random).

Several tools port the best ideas of composition software as **cited rules** rather than data or AI: Hookpad's Roman-numeral entry and borrowed/applied chords (`roman_to_chords`), Scaler's chord sets, voicings, motions and comping (`progression_library`, `voice_chords`, `chord_pattern`, `bass_line`, chords-track `rhythm`/`strum`/`swing`), Klimper's chord palette (`chord_palette`), Hookpad Magic Chord / Scaler Suggest rebuilt on Piston's 1941 table (`next_chords`), and music21's key finding, Roman-numeral analysis, cadence and voice-leading checks (`detect_key`, `analyze_progression`, `find_cadences`, `check_voice_leading`).

### Scales & chords

| Tool | What it does |
|---|---|
| `list_scales` / `get_scale` | 50 scale types (common, modal, jazz, symmetric, world/exotic), each with a description; generate notes from a root. `maj + C → C D E F G A B C`. |
| `list_chords` / `get_chord` | 37 chord types (triads → 13ths and altered), each with a description; generate notes. `min + F → F Ab C`. |
| `match_scales` / `match_chords` | Find scales/chords containing given notes (**octaves ignored**); inversions detected (`e g c → C/E`), partials list missing notes. |
| `diatonic_chords` | The chord on each scale degree, with roman numerals, degree names and harmonic functions. |
| `degrees_to_chords` | Resolve a chosen degree sequence (`[1,5,6,4]`, `"I V vi IV"`) into concrete chords. Numerals are *positions* (`iv` in C is still F); a `warnings` list says when a numeral's case, mark or figure was ignored — use `roman_to_chords` to read numerals literally. |
| `roman_to_chords` | Roman numerals read **literally** (the inverse of `analyze_progression`; Hookpad-style entry): case is the third (`iv` in C = Fm), accidentals count from the parallel major (`bVII` = Bb; in A minor = G), figures invert (`V6` = G/B, `ii65` = Dm7/F), `IV7` = F7 but `IVΔ7` = Fmaj7, applied chords (`V7/V` = D7, `vii°7/V` = F#dim7, `V/V/V` = A), `N6`, `It6`/`Fr43`/`Ger65`/`Sw43`, `Cad64`. Each chord comes with its kind (diatonic/borrowed/applied/…), `borrowed_from` (the parallel modes whose own notes spell it, judged by letters — so `III7` = E7 in C, whose G# no parallel mode has, is `chromatic`), `non_scale_notes`, fit and bass. |
| `progression_library` | 28 named, cited progressions in that dialect: pop (axis, doo-wop, royal road, Andalusian…), jazz (ii–V–I, backdoor, ragtime…), 12-bar blues, lament bass, La Folia, passamezzo, Laitz's sequences (descending fifths, Pachelbel, ascending 5-6) and Gjerdingen's galant schemata (Prinner, Meyer, Romanesca, Do-Re-Mi, Fenaroli, Fonte, Monte) with bass degrees and — all but the Monte — melody degrees. Resolve any of them in any key, or any scale: a `warning` then names chords that leave it (`andalusian` in D dorian: `VI = B (D#, F#)`), and a schema melody that would clash with its chords is withheld. |
| `chord_palette` | Every chord that fits a key, simplest first (a Klimper-style palette), in any scale: the stacked triads/sevenths, or with `extended` every chord type (sus, 6, add9, 7sus4 … up to `max_notes`) whose notes all lie in the scale. `borrow` / `source_modes` / `fifths_steps` add modal-interchange and neighbour-key chords, each with its source, distance, roman numeral (`bVI`, `iv`, `#iv°`), the scales whose own notes spell it and `same_notes_as` (earlier entries with the same notes: `Am7` → `C6`). Chords are judged by letters + semitones: C harmonic minor has `Ebaug` but no `Abm` (that would need Cb), and a neighbour key keeps the home's letters (from Cb major the key a fifth down gives `Gbm` = `v`, not F#m). |

### Harmony rules

| Tool | What it does |
|---|---|
| `circle_of_fifths` | Key signatures, relative/parallel minors, and closely related keys (for modulations and bridges). |
| `interval_between` | Name the interval between two notes (`C→Eb = m3`, `C→F# = A4` vs `C→Gb = d5`). With octaves it names the real interval, compound or descending (`C4→C5 = P8`, `C4→E5 = M10`, `G4→C4 = P5 descending`). |
| `analyze_progression` | The inverse of `degrees_to_chords` / `roman_to_chords`: chords → roman numerals + functions. A chord is in key only if **every** tone is (E7 in A natural minor is flagged, with `non_scale_notes: ["G#"]`); numerals follow spelling (`Gb` in C = `bV`, `F#` = `#IV`). It reads inversions (`figure`: `G7/B` → `65`), gives a `roman_figured` numeral that round-trips through `roman_to_chords` (`V65`, `V7/ii`, `N6`, `Ger65`, `Cad64`) — with two exceptions: a slash bass the figure grammar cannot write (`F/G`, `Dm/G`, `Csus4/F`: `figure` null) is kept only in `bass_degree`, so re-add it when moving a draft to another key, and an unnamed cluster (`C Db D`) gets `roman_figured` null while its `roman` keeps a `?`. It labels applied chords (`A7` in C = `V7/ii`), the Neapolitan, augmented sixths and the cadential 6/4, and lists `borrowed_from` for mixture chords — the parallel modes whose own notes spell them (`Fm` in C ← C harmonic major, harmonic/natural minor, phrygian, locrian; `G#` in C ← none, as C minor has Ab). In a note array the lowest note (or the first, without octaves) is the bass, and a 2–3 note array missing its fifth (a shell `D F C`, a fifthless V7) is read as that chord, with `omitted_fifth`. |
| `next_chords` | Rank what may follow the last chord by **Piston's Table of Usual Root Progressions (1941)** — a rule table, never probabilities (the deterministic answer to Hookpad's Magic Chord / Scaler's Suggest). Tiers: an applied chord's resolution (after an augmented sixth: V and the cadential 6/4, `Cad64`); usual / sometimes / less-often diatonic chords; applied V7/x; parallel-minor mixture (major keys); unlisted. The last chord itself is never offered. Each candidate carries a `roman_to_chords` token (null for a stacked chord that dialect cannot write), the table rule, common tones, Schoenberg root motion and `movement` (voice_leading's semitone cost); `sort="movement"` puts the smoothest first. |
| `voice_leading` | Voice a progression smoothly (nearest inversion, common tones held) — natural pads instead of parallel blocks. |
| `voice_chords` | Voice a progression in an arranging style (Scaler's voicings): `drop2` / `drop3` / `drop24`, `open`, `shell` (R-3-7 / R-7-3), Mark Levine's rootless A/B left-hand voicings (`rootless_a`, `rootless_b`, or `rootless`, which starts from whichever form gives the smaller total movement — A-B-A or B-A-B), or `close` (= `voice_leading`). Connected greedily with voice_leading's cost; a slash bass stays at the bottom; `top_notes` puts a melody on top. `Cmaj7` drop2 → `G3 C4 E4 B4`; `Dm7 G7 Cmaj7` rootless → `F3 A3 C4 E4 · F3 A3 B3 E4 · E3 G3 B3 D4`. |
| `secondary_dominant` / `tritone_substitute` | Classic reharmonizations (`V/ii of Dm → A7`; `G7 → Db7`). |
| `negative_harmony` | Reflect notes through a key's negative-harmony axis (major ↔ minor shadow), in any key: `D F# A` in D → `A F D` (D minor). |
| `harmonize_melody` | Put a chord under each melody note — searching the **whole** chord database — that reuses as many notes from the previous chord as possible, then voice-leads it. Returns ranked options + a `render_hint`. |

### Melody

| Tool | What it does |
|---|---|
| `notes_from_degrees` | Write a melody as scale degrees → notes; transposable to any key/scale. `[1,2,3,5,8]` in C → `C D E G C`. Negative degrees count down on the same no-zero line `motif_grammar` uses: from `C4`, `-1 → B3`, `-7 → C3`. |
| `motif_grammar` | Build a phrase from a form like `ABAC` over labeled motifs; a variant can `transpose`/`invert`/`retrograde`/`rotate` another. Works on notes, degrees, or rhythm. |
| `melodic_walk` | 🎲 A singable line by a seeded random walk over a scale ladder (mostly stepwise). |
| `melodic_sequence` | Repeat a motif as a diatonic sequence (e.g. down a step each time); the first copy is the motif as given, chromatic notes keep their offset. |
| `arpeggiate` | Reorder a chord/scale into an arpeggio (up/down/updown/converge/…, multi-octave). |
| `tintinnabuli_voice` | **Arvo Pärt's tintinnabuli:** shadow a melody with the nearest notes of a fixed triad (T1/T2, above/below/alternating). |
| `counterpoint` | **Species counterpoint (1–5):** a rule-following counter-melody to a cantus firmus — note-against-note through florid, with passing tones and prepared, resolving suspensions (7-6, 4-3, 9-8 above; 2-3, 9-10, 4-5 and a diminished 5th resolving to a 6th below), no parallel fifths/octaves. Returns a `render_hint` of ready tracks; if no line can obey every rule for a cantus, the closest one comes with a `warning`. |
| `snap_to_scale` | Snap any line to the nearest scale notes — guarantees a melody fits the key/chords. |
| `transpose_notes` | Transpose a note list by semitones, spelled as one interval so it stays in one key (`F A C` +1 → `Gb Bb Db`). |
| `random_notes` | 🎲 Uniform random picks from any note pool (seeded); repeated pool notes count once. |

### Accompaniment

| Tool | What it does |
|---|---|
| `chord_pattern` | Play a chord-relative figure over a whole progression (like Scaler's Motions). Presets: `alberti` (`^1 3 2 3`), `up`, `down`, `updown`, `murky` (broken octaves) — or your own steps: an index into the voicing (`4` of a triad is the bottom note an octave up), `^` for an accent, `'`/`,` for an octave up/down, `.` for a rest. `mode="scale"` plays the key's scale from each chord root instead (an off-scale chord falls back to chord mode); `phase="continue"` carries the figure across chord changes (Scaler's Follow). Voicings come from `voice_leading`, or `smooth=false` keeps note arrays as written (so `voice_chords` can drive it). Returns a notes track and a `render_hint` (voicings plus pattern). |
| `bass_line` | A bass part by fixed rules, folded into the walking register E1–G3: `root`, `root_fifth` (country two-beat; over a 6/4 like `C/G` the alternate is the root), `root_octave` (a bass from Ab2 up drops an octave so its octave fits), `approach` (chromatic approach into the next chord: `G → F#`, `C → Db`), `walking` (chord tones on beats 1–3, a chromatic approach on beat 4 — a simplified form of walking-bass teaching, after Friedland), `pedal`. Your own `rhythm`, one `beats_per_chord` per chord, `ending` loop (the last note leads by a semitone into the line's actual first note, even an octave away) or root, and `sustain` (unset = the style's own: root, approach and pedal hold, the others play detached; a hold never crosses a chord change). GM finger bass (program 33). |

### Historical melody rules

| Tool | What it does |
|---|---|
| `church_mode` | The eight **Gregorian modes** (Dorian … Hypomixolydian): final, reciting tone (tenor), ambitus. |
| `solmization` | **Guido of Arezzo's hexachords**: sing any line as ut re mi fa sol la, mutating *per re sursum, per la deorsum*; musica ficta in transposed hexachords (mi–fa is always the semitone). |
| `guido_vowel_melody` | **Guido's vowel method** (*Micrologus*, c. 1026): a chant melody derived from the vowels of a text (Latin syllabification, one or both of Guido's vowel rows) — arguably the oldest composition algorithm. |
| `check_melody` | Lint a line against the **cantus-firmus rules** of species-counterpoint teaching, after Fux and Jeppesen (leap recovery, single climax, forbidden intervals, modal cadences…); every violation with its index. `strict=False` relaxes the leap rules that Fux's own cantus firmi break. |
| `cantus_firmus` | Compose a cantus firmus that passes all those rules — deterministic; every `variant` 0, 1, 2… is a different melody. |

### Harmony rules of the masters

| Tool | What it does |
|---|---|
| `rameau_fundamental_bass` | **Rameau** (1722): the root line under any progression; root motion by fifth/third/step; cadence parfaite, irrégulière, rompue. |
| `schoenberg_progressions` | **Schoenberg**'s root-progression classes: ascending (strong), descending, superstrong — read from the sounding interval, so spelling never matters. |
| `bach_chorale_voicing` | **Bach-chorale** four-part (SATB) voicing: ranges, spacing, doubling, no parallels/overlaps, no melodic augmented intervals (the A2 of b6→#7 in minor), leading tone up, sevenths down, an augmented sixth expanding outward (#4 up, b6 down a semitone) — plus a four-track `render_hint`. |
| `neo_riemannian` | **Riemann / Lewin / Cohn** P, L, R triad transformations (C → Em → G → Bm …). |
| `bartok_axis` | **Bartók**'s axis system (Lendvai): tonic, subdominant and dominant axes of minor-third-related keys. |
| `coltrane_changes` | **Coltrane** changes: a ii–V–I through three tonal centres a major third apart (*Giant Steps*). |

### Modern techniques

| Tool | What it does |
|---|---|
| `twelve_tone_matrix` | **Schoenberg**'s twelve-tone method: the 12×12 matrix and all 48 row forms (P, I, R, RI). |
| `pitch_class_set` | **Forte/Rahn** set theory: normal form, prime form (Rahn's algorithm, as in Straus), interval vector, complement, symmetry. Numbers work too: `"0 4 7"`. |
| `additive_process` | **Philip Glass**'s additive/subtractive process (1 2, 1 2 3, 1 2 3 4 …); the figure's register is fixed once, so every stage repeats the same pitches. |
| `phase_shift` | **Steve Reich**'s phasing: a pattern against a copy slipping one step per stage, as two tracks. Rests (`.`) rotate with it, so `"C C C . C C . C . C C ."` is *Clapping Music*. |

**Messiaen's modes of limited transposition** are in the scale database (`get_scale("messiaen mode 3", "C")`; modes 1 and 2 are the whole-tone and half-whole octatonic scales).

### Analysis: read a draft back

| Tool | What it does |
|---|---|
| `detect_key` | **Krumhansl–Schmuckler key finding** (the five profiles and the tonal-certainty measure exactly as in music21): notes, chords or whole tracks → the best-correlating major/minor key, its certainty and the full 24-key ranking, ready for `diatonic_chords` / `analyze_progression`. The tonic keeps your spelling (`Gb` stays `Gb`). `window_beats` gives key **regions** — where a song modulates. |
| `check_voice_leading` | Lint **your own parts** the way `check_melody` lints a cantus: note lists, `voice_leading` / `voice_chords` voicings, or notes tracks such as `bach_chorale_voicing` / `counterpoint` render hints. Parallel, contrary and direct fifths/octaves (spelled, compound too), crossing, overlap, augmented melodic intervals; with a key also the leading tone, chord sevenths, the augmented sixth's resolution (#4 up, b6 down a semitone: `augmented_sixth`) and doubled leading tones. Voicings may differ in size (I V7 I). Plus music21 motion counts per voice pair and the dissonances above the bass. |
| `find_cadences` | Label the cadence at each phrase end: **authentic** perfect/imperfect, **half** (including the Phrygian iv6–V), **plagal**, **deceptive** (V–vi, bVI, V–IV6) or **none**, each with its reason, span, romans, soprano/bass degrees and cadential-6/4 and Picardy flags (Kostka–Payne–Almén, Caplin, Aldwell & Schachter). A PAC needs root-position V–I **and ^1 in the soprano**; without a soprano it says so (`subtype: null`) instead of guessing. Phrase ends: a list, `'all'`, or `phrase_length`. |

### Rhythm

| Tool | What it does |
|---|---|
| `random_rhythm` | 🎲 Random pattern `O...Oo..` — `O` strong, `o` weak, `.` rest (seeded). |
| `euclidean_rhythm` | Evenly-spread Bjorklund rhythm; `euclidean_rhythm(3,8) → O..o..o.` (tresillo). |
| `groove` / `list_grooves` | Named presets: four-on-the-floor, backbeat, tresillo, son/rumba clave, bossa nova, dembow… |

### Song structure & rendering

| Tool | What it does |
|---|---|
| `plan_sections` | Lay out a form (`"intro verse chorus … outro"` / an uppercase letter form `"AABA"`; a single word like `"verse"` is one section) on the timeline — start bars, beats, seconds. |
| `arrange_song` | **The capstone:** assemble named sections (intro/verse/chorus/bridge/outro) into one whole-song MIDI; like-named tracks stitch into continuous parts; `swing`/`swing_unit` swing every section on the song's bar grid (pairs counted from each downbeat; in an odd meter with `swing_unit` 1 the bar's last quarter stays straight). |
| `notes_to_midi` / `chords_to_midi` / `drums_to_midi` | Render a single track (melody/scale; chords as blocks, arpeggios or **comped** — per-chord `durations`, an `O/o/.` strike `rhythm` with `sustain`, a guitar `strum` down/up/alternate; GM drum lanes). All take `swing` (0.5 straight … 2/3 triplet … 0.75) and `swing_unit` (0.25/0.5/1 beat). |
| `arrange_to_midi` | Render any number of fitting tracks (chords, bass, melody, drums) into one multi-track `.mid`; chords tracks take the comping fields, and every track can swing (a track's own `swing` overrides the renderer's). |
| `song_to_midi` | Melody + chords as a two-track file (shortcut for the common case), optionally swung. |
| `midi_to_audio` | Render any generated `.mid` into a **playable WAV** with a built-in synth (no soundfont needed). |

MIDI tools write to `./midi_output` (override per call with `output_dir` or globally with `MIDI_COMPOSER_OUTPUT_DIR`) and also return the small `.mid` file base64-encoded; the file lasts exactly the reported `duration_seconds` (trailing rests included). `midi_to_audio` writes the WAV next to the `.mid` (or to `wav_file`) and returns base64 only when asked (`include_base64=true`) — audio is too large for most tool-result limits.

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
match_scales(["C", "E", "G"])       → C major pentatonic, E balinese pelog, D egyptian, … A minor pentatonic, …
```
(Octaves are ignored, so `["C5","E5","G5"]` gives the same answer. Exact matches and smaller scales sort first, so 7-note scales such as C major come further down — raise `limit` or use `exact_only`.)

**"A ii–V–I in F, with sevenths."**
```
degrees_to_chords("F", "major", "ii V I", sevenths=True)   → Gm7  C7  Fmaj7
degrees_to_chords("C", "natural minor", "i bVI bIII bVII") → Cm  Ab  Eb  Bb   (numerals as diatonic_chords prints them)
```

**"How far is C4 from E5? And which keys neighbour G♭?"**
```
interval_between("C4", "E5")        → major tenth (M10), 16 semitones, ascending
circle_of_fifths("Gb")              → Gb major (6 flats), dominant Db, subdominant Cb, relative Eb minor
```

**"A random melody from A minor pentatonic, then save it as MIDI."**
```
get_scale("minor pentatonic", "A5")            → A5 C6 D6 E6 G6 A6   (octave-aware)
random_notes(<those notes>, count=8, seed=1)   → a reproducible 8-note line  [contained randomness]
notes_to_midi(<the notes>, tempo=120)          → a .mid file (+ base64)
midi_to_audio(<that file>)                     → a playable .wav
```

**"A ii–V–I in F with a V/V — and an Andalusian cadence in E."**
```
roman_to_chords("ii7 V7/V V7 I", "F")          → Gm7  G7  C7  F    (V7/V = G7: kind "applied", applied_to "C", non_scale_notes ["B"])
progression_library("andalusian", root="E")    → i VII VI V = Em  D  C  B   (bass E D C B)
degrees_to_chords("C", "major", "IV iv I")     → F  F  C  + warning "iv resolved to F …; for F minor use roman_to_chords"
```

**"Which chords fit A minor, including sus and add9? What could follow Am in C?"**
```
chord_palette("A", "natural minor", extended=True)
   → 42 chords, simplest first: Am Asus4 Asus2 Bdim C Csus4 Csus2 Dm … Gsus2, then Am7 Amadd9 A7sus4 … —
     each with its roman, family (major / minor / other) and same_notes_as (C6 lists Am7, which has the same notes)
next_chords(["C","Am"], "C")
   → usual: Dm (ii), G (V) · sometimes: F (IV), Em (iii) · less often: C (I) · applied: A7 D7 B7 C7
   → mixture: Ddim (ii°), Fm (iv), Eb (bIII) · unlisted: Ab (bVI), Bb (bVII) — each with its rule,
     e.g. "Piston 1941: VI is usually followed by II or V"
```

**"What key is this melody in?"**
```
detect_key(notes="E4 G4 A4 B4 A4 G4 E4 D4 E4")    → E natural minor, r 0.8225 (runner-up A major 0.5927)
detect_key(chords="Am Dm E7 Am")                  → A natural minor  →  diatonic_chords("A", "natural minor")
```

**"Drop-2 voicings for a ii–V–I in F, and a root bass for C–G/B–Am–F."**
```
voice_chords(["Gm7","C7","Fmaj7"], "drop2")   → D4 G4 Bb4 F5 | C4 G4 Bb4 E5 | C4 F4 A4 E5
bass_line(["C","G/B","Am","F"])               → C2 B1 A1 F1   (slash basses in the bass, each nearest the last note)
```

**"Give C–G–Am–F a 2+2+4+8-beat harmonic rhythm; play a scale in swung eighths."**
```
chords_to_midi(["C","G","Am","F"], durations=[2,2,4,8])   → chords at beats 0, 2, 4, 8 — 16 beats in all
notes_to_midi("C4 D4 E4 F4 G4 A4 B4 C5", swing=0.6667)   → eighths at 0, 0.667, 1, 1.667 … (on-beats stay put)
```

**"Does my progression end with a real cadence?"**
```
find_cadences("C F G7 C", "C")                        → authentic, subtype null — "give the soprano to decide PAC vs IAC"
find_cadences("C F G7 C", "C", soprano="E5 F5 D5 C5") → authentic perfect (root-position V7 → I, ^1 on top)
find_cadences("C G Am", "C")                          → deceptive: "V -> vi instead of I"
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
                                          → I, VI7 (applied "V7/ii", roman_figured "V7/ii"), ii, V7, I
```

**"Is E7 in A minor? And what is the negative-harmony mirror of a D major chord?"**
```
analyze_progression(["Am","E7","Am"], "A", "natural minor")
                                          → E7 = V7, in_key false, non_scale_notes ["G#"], function_note "harmonic-minor dominant"
analyze_progression(["Am","E7","Am"], "A", "harmonic minor")   → E7 = V7, in key, dominant
negative_harmony(["D","F#","A"], "D")     → A F D   (D major ↔ D minor, in any key)
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

**"Which Gregorian mode is this, and how would a medieval singer solmize it?"**
```
church_mode("Hypodorian")                     → final D, tenor F, ambitus A3–A4
solmization("C D E F G A B C5")               → ut re mi fa sol re mi fa  (mutation la→re on A: per re sursum)
solmization("D5 C5 B4 A4 G4 F4 E4 D4")        → sol fa mi la sol fa mi re  (re→la on A: per la deorsum)
solmization("D E F# G")                       → ut re mi fa  (F# is musica ficta: mi of a hexachord on D)
```

**"Is my melody a good cantus firmus? Write me one in D Dorian."**
```
check_melody(["C4","A4","B4","G4","F4","E4","D4","C4"], "C")  → interval: C4 → A4 is a major sixth; leap_recovery: it is not followed by a step back
cantus_firmus("D", "dorian", length=11, variant=2)            → an 11-note line passing every cantus-firmus rule
```

**"Check my SATB for parallels."**
```
check_voice_leading(voices=[["E4","F4"],["C4","D4"],["G3","A3"],["C3","D3"]], root="C")
   → valid: false — parallel_octaves alto/bass (C4/C3 → D4/D3), parallel_fifths tenor/bass (G3/C3 → A3/D3)
bach_chorale_voicing("C Dm G7 C", root="C")                       → a four-part voicing by the chorale rules
check_voice_leading(voices=<its render_hint tracks>, root="C")    → valid: true
```

**"Levine's rootless left-hand voicings for a ii–V–I, with shells for comparison."**
```
voice_chords(["Dm7","G7","Cmaj7"], "rootless") → F3 A3 C4 E4 (A) | F3 A3 B3 E4 (B) | E3 G3 B3 D4 (A)
voice_chords(["Dm7","G7","Cmaj7"], "shell")    → D3 F3 C4 (R-3-7) | G3 B3 F4 (R-3-7) | C3 B3 E4 (R-7-3)
```

**"An Alberti accompaniment over I–vi–IV–V, with a walking bass under it."**
```
roman_to_chords("I vi IV V", "C")               → C Am F G
chord_pattern(<symbols>, "alberti")             → C4 G4 E4 G4 ×2 | C4 A4 E4 A4 ×2 | C4 A4 F4 A4 ×2 | D4 B4 G4 B4 ×2
bass_line(<symbols>, "walking")                 → C2 E2 G2 G#2 | A2 C3 E3 Gb3 | F3 C3 A2 Ab2 | G2 B2 D3 Db2   (Db2 leads back into C2)
arrange_to_midi(<chord_pattern render_hint tracks> + <bass_line render_hint tracks>)   → chords, pattern and bass
```

**"Seventh-chord colours from C minor and C dorian for a C-major tune."**
```
chord_palette("C", sevenths=True, source_modes=["natural minor", "dorian"])
   → Cmaj7 Dm7 Em7 Fmaj7 G7 Am7 Bm7b5, then Cm7 Ebmaj7 F7 Gm7 Am7b5 Bbmaj7 (C dorian, distance 2)
     and Dm7b5 Fm7 Abmaj7 Bb7 (C natural minor, distance 3) — tokens i7 bIIIΔ7 IV7 v7 … iv7 bVIΔ7 bVII7
```

**"A reggae skank on the off-beats, and a strummed acoustic pattern."**
```
arrange_to_midi([
  {"type":"chords","name":"skank","chords":["Am","D"],"step_beats":0.5,"rhythm":".o.o.o.o","program":27},
  {"type":"notes","name":"bass","notes":"A2 A2 E3 D3 D3 A2","rhythm":"O..o..o.O..o..o.","step_beats":0.5,"sustain":true,"program":33},
  {"type":"drums","name":"drums","step_beats":0.25,"lanes":{"kick":"........O...............O.......","side_stick":"........O...............O.......","hat":"o.o.o.o.o.o.o.o.o.o.o.o.o.o.o.o."}},
], tempo=74)                                  → chord stabs on every "and" (beats 0.5, 1.5, 2.5 …), bass A A E | D D A,
                                                the one-drop (kick + side stick) on beat 3 of both bars
chords_to_midi(["G","D/F#","Em","C"], step_beats=0.5, rhythm="O.oo.oo.", sustain=True,
               strum=0.02, strum_direction="alternate", program=25)
   → strikes at beats 0, 1, 1.5, 2.5, 3 of each chord (down, down, up, up, down), each held to the next
```

**"What cadences do my verse and chorus end with?"**
```
find_cadences(["C","Am","F","G", "C","F","G7","C"], "C", phrase_ends=[3, 7], soprano="E5 E5 F5 D5 E5 F5 D5 C5")
  → 3: half ("ends on V") — the verse stays open · 7: authentic perfect (soprano ^2 → ^1)
    (phrase_length=4 gives the same two ends; end the tune on E and it is imperfect, "soprano on ^3")
```

**"Reharmonize a ii–V–I the Coltrane way, then voice it; and give me a neo-Riemannian chain."**
```
coltrane_changes("C")          → Dm7 Eb7 Abmaj7 B7 Emaj7 G7 Cmaj7
voice_leading(<those symbols>) → smooth pads
neo_riemannian("C", "LRLR")    → C Em G Bm D   (each step keeps two notes)
```

### Advanced

**"Write a third-species counterpoint to a cantus firmus."**
```
counterpoint(["C5","D5","E5","F5","E5","D5","C5"], "C", "major", species=3)
   → cantus + a 4:1 counter-line (passing/neighbour tones between consonances, no parallel 5ths/8ves,
     the final reached by step: … G5 A5 B5 C6)
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
       motif_grammar("ABAC", {"A":[1,2,3,2],"B":{"vary":"A","transpose":1},"C":[3,2,1,1]}, kind="degrees")["degrees"])["notes"]
t = tintinnabuli_voice(m, "Am", position="inferior", rank=1)["t_voice"]   # nearest A-minor triad note below each M note
verse_pads  = voice_leading(["Am7","Dm7","Fmaj7","Cmaj7"])["chords"]
chorus_pads = voice_leading(["Fmaj7","Cmaj7","Dm7","Em7"])["chords"]

arrange_song({                                                        # sequence sections into a song
  "verse":  {"bars":4, "tracks":[
     {"type":"chords","name":"pads","chords":verse_pads,"beats_per_chord":4,"program":89},
     {"type":"notes","name":"M-voice","notes":m,"step_beats":1,"octave":5,"program":48,"sustain":true},
     {"type":"notes","name":"T-voice","notes":t,"step_beats":1,"octave":4,"program":9,"sustain":true}]},
  "chorus": {"bars":4, "tracks":[
     {"type":"chords","name":"pads","chords":chorus_pads,"beats_per_chord":4,"program":89},
     {"type":"notes","name":"M-voice","notes":notes_from_degrees("C5","major",[5,6,8,6,5,3,2,1])["notes"],"step_beats":2,"octave":5,"program":48,"sustain":true},
     {"type":"notes","name":"bass","notes":["F","C","D","E"],"step_beats":4,"octave":2,"program":33}]},
}, form="verse verse chorus", tempo=72)   →   midi_to_audio(<the song>)
```

**"Set 'Ut queant laxis' as Guido would, harmonize the result in four parts like a Bach chorale, and check its root progressions against Rameau and Schoenberg."**
```
guido_vowel_melody("Ut queant laxis resonare fibris", "Dorian")    → a D-Dorian chant closing on D
harmonize_melody(<the notes>, root="D", scale_type="dorian", in_scale=True, max_chord_notes=3)
                                                                    → Dm Dm Dsus2 Dsus2 Gsus2 Dm Dm F Dm Gsus2 Gsus2 Dsus4
bach_chorale_voicing(<the progression>, root="D", scale_type="dorian", melody=<the notes>)
                                                                    → SATB with the chant in the soprano (rule_breaks lists
                                                                      the few parallels the fixed tune forces)
rameau_fundamental_bass(<progression>, "D", "dorian"); schoenberg_progressions(<progression>)
arrange_to_midi(<bach_chorale_voicing render_hint tracks>)         → a four-part chorale .mid
```

**"A twelve-tone piece: the row, its inversion as a counter-line, and a Reich-style phase canon on a fragment."**
```
m = twelve_tone_matrix("E F G C# F# D# D B C A Bb G#")
arrange_to_midi([{"type":"notes","notes":m["forms"]["P4"],"octave":5},
                 {"type":"notes","notes":m["forms"]["I9"],"octave":3}])
phase_shift(m["forms"]["P4"][:6], repeats_per_stage=4)  → two voices drifting out of and back into phase
pitch_class_set(m["forms"]["P4"][:3])                   → the set class of the opening trichord
```

**"An N6 and a German sixth into a cadential 6/4, in D minor, voiced as a chorale."**
```
roman_to_chords("i iv N6 Ger65 Cad64 V7 i", "D", "natural minor")
   → Dm  Gm  Eb/G  [Bb D F G#]  Dm/A  A7  Dm
     (N6 = Neapolitan in first inversion; Ger65 is a bass-first note array, enharmonic_symbol "Bb7";
      A7 is kind "borrowed", borrowed_from ["D harmonic minor", …])
bach_chorale_voicing(<symbols>, root="D", scale_type="harmonic minor")   → SATB, slash basses and the Bb under the Ger65 kept
```

**"A Prinner answering a Meyer in G, realised as a chorale."**
```
m = progression_library("meyer", root="G")     → G  D7/A  D7/F#  G    melody G F# C B
p = progression_library("prinner", root="G")   → C  G/B  F#dim/A  G   melody E D C B, bass C B A G
bach_chorale_voicing(m["chords"] + p["chords"], root="G", melody=m["melody"] + p["melody"])
   → four parts with the schema melodies in the soprano
```

**"Label the secondary dominants, borrowed chords and inversions in my song — then move it to Eb."**
```
analyze_progression(["C","E7/G#","Am","C7","F","Fm","C/G","G7","C"], "C")
   → roman_figured: I  V65/vi  vi  V7/IV  IV  iv  Cad64  V7  I
     Fm: borrowed_from [C harmonic major, C harmonic minor, C natural minor, C phrygian, C locrian]
     C/G: special "Cad64", function still "tonic", function_note "dominant (cadential 6/4)"
roman_to_chords(<the roman_figured list>, "Eb")   → Eb G7/B Cm Eb7 Ab Abm Eb/Bb Bb7 Eb
```
(Every bass here is a chord member, so the move is exact. A slash bass the figure grammar cannot write — `F/G`, `Dm/G` — comes back as plain `IV` / `ii` with `figure` null and `bass_degree` 5: re-add the slash in the new key, e.g. `Ab/Bb`.)

**"Grow a progression step by step, preferring smooth movement; borrow colour chords for the chorus."**
```
next_chords(["C","Am"], "C", sort="movement")   → F (1 semitone, sometimes), C (2), Fm (2, mixture), Ab (2), Dm (3, usual) …
next_chords(["C","Am","F"], "C")                → G (usual: IV → V), Dm, C (sometimes), Am, Em (less often), D7 = V7/V …
chord_palette("C", borrow=True)
   → the 7 in-key triads, then 20 borrowed chords, nearest mode first: D F#dim Bm (lydian) · Edim Gm Bb (mixolydian)
     · Cm Ebaug Adim (melodic minor) · Ddim Fm Abaug (harmonic major) · Eb (dorian) · Ab (harmonic minor) · …
roman_to_chords("I V vi iv bVI bVII I", "C")    → C G Am Fm Ab Bb C
```

**"Where does my song modulate?"**
```
detect_key(chords="C Am F G C Am F G G Em C D G Em C D", window_beats=16, hop_beats=16)
   → regions: C major (beats 0–32), G major (32–64)
detect_key(tracks=<any render_hint or arrange_song section tracks>, window_beats=16, hop_beats=16)
   → the same reading on a whole arrangement (drums ignored)
```

**"Autumn-Leaves-style changes: rootless voicings in the left hand, drop-2 block chords under my melody in the right."**
```
chords = ["Cm7","F7","Bbmaj7","Ebmaj7","Am7b5","D7b9","Gm6"]
lh = voice_chords(chords, "rootless")          → Bb3 D4 Eb4 G4 (B) | A3 D4 Eb4 G4 (A) | A3 C4 D4 F4 (B) | …   (B-A-B…: 25 semitones in all)
rh = voice_chords(chords, "drop2", octave=5, top_notes=["Eb6","Eb6","D6","D6","C6","C6","Bb5"])   # melody on top
   → C5 G5 Bb5 Eb6 | C5 F5 A5 Eb6 | Bb4 F5 A5 D6 | …
arrange_to_midi([{**lh["render_hint"]["tracks"][0], "name":"left hand"},
                 {**rh["render_hint"]["tracks"][0], "name":"right hand"},
                 {"type":"notes","name":"bass","notes":["C","F","Bb","Eb","A","D","G"],"step_beats":4,"octave":2,"program":32}])
```

**"A Dorian scale-run figure that keeps running across the chord changes, and a swung walking bass under a jazz turnaround."**
```
chord_pattern(["Dm7","G7","Dm7","Cmaj7"], "^1 2 3 4 5 6", beats_per_chord=2, step_beats=0.25,
              mode="scale", root="D", scale_type="dorian", phase="continue")
   → 6-step runs up D Dorian from each chord root, carried over the bar lines (Scaler's Follow)
progression_library("jazz_turnaround", root="F")   → Fmaj7 Dm7 Gm7 C7
bass_line(<chords>, "walking")                     → F2 A2 C3 C#3 | D3 F3 D3 F#3 | G3 F3 D3 Db3 | C3 E3 G3 Gb2
arrange_to_midi([{"type":"chords","chords":voice_leading(<chords>)["chords"],"rhythm":"O..o..o.","step_beats":0.5},
                 <bass_line render_hint track>], swing=2/3)            → a swung comp over the walking line
```

**"A minor-key chorale close with a cadential 6/4: is it a PAC? And a Phrygian half cadence."**
```
roman_to_chords("i iv Cad64 V7 i", "A", "harmonic minor")              → Am Dm Am/E E7 Am
bach_chorale_voicing(<symbols>, root="A", scale_type="harmonic minor", melody="C5 D5 C5 B4 A4")
find_cadences(<symbols>, "A", "harmonic minor", soprano=<its soprano>)   → authentic perfect (^3 ^2 ^1), span [2, 4], cadential_64 true
progression_library("lament", root="D")                                 → Dm Am/C Gm/Bb A
find_cadences(<chords>, "D", "natural minor")                            → half, phrygian: iv6 → V, the bass falls a semitone
```

These advanced examples (a Pärt tintinnabuli study, a species-3 counterpoint, the tintinnabuli verse/chorus song, and a full verse/chorus/bridge song) are runnable in **`examples/generate_examples.py`**:

```bash
python examples/generate_examples.py            # writes .mid + .wav for each
```

### Demo gallery

The [**`demos/`**](demos/) folder is a gallery of finished pieces, each paired with the plain-language **prompt** it answers — from a Pärt-style tintinnabuli study to a modulating pop anthem, a jazz reharmonization, all five counterpoint species, a flamenco piece in Phrygian dominant, a negative-harmony before/after, and a swung jazz combo built from the ported tools (rootless comping, walking bass, cadence check). The `.mid` files are committed (open them in a DAW); regenerate everything with:

```bash
python demos/generate.py                        # rewrites demos/*.mid and *.wav
```

### The book and the carousels

[**`docs/book/the-rules-of-harmony.pdf`**](docs/book/the-rules-of-harmony.pdf) is a 44-page musician's guide generated from the same rule tables the tools run: notes and spelling, intervals, all 50 scales and 37 chord types (each with a keyboard diagram), harmony in a key, the circle of fifths, voice leading and reharmonization, melody craft and tintinnabuli, species counterpoint, rhythm and form, the rules of the old masters (church modes, Guido's hexachords and vowel method, the cantus firmus, Rameau, Schoenberg, Bach chorales) and modern approaches (neo-Riemannian transformations, Bartók's axes, Coltrane changes, twelve-tone rows, pitch-class sets, Messiaen, Glass, Reich). [**`docs/carousels/`**](docs/carousels/) holds *Hudební teorie v kostce*, eight Czech 2:3 slide decks for LinkedIn (major scales in all 12 keys, church modes, the T–S–D chords of every key, chord symbols and inversions, the circle of fifths key by key, negative harmony in every key, Fux counterpoint as piano rolls, Euclidean rhythms as necklaces), set in Avenir Next Condensed with Czech note names and terminology (H = B, B = B♭, dur/moll, sextakord…). Rebuild both after changing a rule:

```bash
pip install -e ".[book]"
python docs/book/build_book.py && python docs/carousels/build_carousels.py
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
  {"type":"chords","name":"comp",  "chords":["Am","F","C","G"], "durations":[4,4,2,6], "rhythm":"O..o..o.O..o..o.O..oO..o..o.o...", "step_beats":0.5, "sustain":true, "strum":0.03, "strum_direction":"alternate"},
  {"type":"notes", "name":"bass",  "notes":["A","F","C","G"],   "rhythm":"O..o..o..o..o...", "octave":2, "program":33, "step_beats":0.25},
  {"type":"notes", "name":"lead",  "notes":["A4","C5","E5","D5"], "octave":5, "program":0},
  {"type":"drums", "name":"drums", "lanes":{"kick":"O...O...","snare":"..O...O.","hat":"oooooooo"}}
]
```
Shared per-track options: `name`, `velocity`, `start_beat` (beat offset for intros/drops), `step_beats`, `swing`/`swing_unit`, `channel` (auto-assigned around any channels you set explicitly; drums always go to the GM percussion channel 10). MIDI has 15 melodic channels, so more than 15 melodic tracks is an error — in `arrange_to_midi` give parts that share an instrument the same `channel`; in `arrange_song`, reuse a track `name` across sections. `arrange_song` writes a time signature matching its `beats_per_bar`.

Chords tracks can be **comped**: `durations` (one beat length per chord, replacing `beats_per_chord`) gives a variable harmonic rhythm. `rhythm` is an `O/o/.` pattern on the track's `step_beats` grid, either one chord long (repeated for every chord; only when every chord lasts the same) or the whole track long — the comp above has mixed durations, so its rhythm spans all 16 beats (32 steps: 8 + 8 + 4 + 12). `O` strikes the whole voicing at `accent_velocity` (100), `o` at `velocity`, and `.` rests; with `sustain` a rest holds the strike, but never past the chord change. `strum` (0–0.25 beats) staggers the voices low to high (`"down"`), high to low (`"up"`) or both (`"alternate"`: down on even steps of the absolute `step_beats` grid, counted from beat 0 with `start_beat` included, up on odd ones — in `arrange_song` counted from the section start), and all voices end together. `rhythm` and `strum` cannot be combined with `arpeggiate`. **Swing** (`swing` 0.5 = straight, 0.6667 = triplet swing, max 0.75; `swing_unit` 0.25/0.5/1.0 beat) delays every second step, and on-beats never move. `arrange_to_midi` and the single-track renderers swing on the absolute timeline, so offset tracks swing together; `arrange_song` swings on the song's bar grid, counting pairs from each downbeat, so every section swings alike and bar lines never move (in an odd meter with `swing_unit` 1.0, such as 3/4, the bar's unpaired last quarter stays straight). With the defaults every renderer writes exactly the file it always wrote (golden files in `tests/golden/` guard this).

## Installation

Requires Python ≥ 3.10. The server is built on the MCP Python SDK 1.x (`FastMCP`); `mcp` 2.x renamed that API, so the dependency is pinned to `mcp>=1.2,<2`.

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
# from a clone, using the project's own virtualenv (code edits are picked up on restart):
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
claude mcp add midi-composer -e MIDI_COMPOSER_OUTPUT_DIR="$PWD/midi_output" -- "$PWD/.venv/bin/midi-composer-mcp"

# or let uv manage the environment:
claude mcp add midi-composer -- uv run --directory /path/to/midi-composer-mcp midi-composer-mcp
```

`claude mcp add` defaults to the *local* scope (only this project directory); add `--scope user` to use the composer from any directory. Check it with `claude mcp get midi-composer`.

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
uv venv && uv pip install -e ".[dev]"      # or: python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest
```

Besides per-tool unit tests, `tests/test_invariants.py` holds **property sweeps** that guard whole classes of bugs: every key-relative tool must give the same result in all 12 keys (transposed), every chord spelling one tool accepts must work in the others, `analyze_progression` must round-trip `diatonic_chords` and judge key membership by every chord tone, the degree number line must have no gaps, an independent Fux rule checker sweeps species 1–5 over many cantus firmi, keys and both positions, and every MCP wrapper in `server.py` must forward all of its function's parameters with the same defaults. When you add a tool, add it to the relevant sweep.

Layout:

```
src/midi_composer_mcp/
  notes.py        # note parsing, proper spelling, octaves, MIDI numbers
  scales.py       # scale database (50, described), generation, matching
  chords.py       # chord database (37, described), symbols, generation, matching
  diatonic.py     # chords per scale degree, degree sequences, borrowing sources, chord palette
  roman.py        # the Roman-numeral dialect: roman_to_chords, progression_library, chord-reading helpers
  circle.py       # circle of fifths: key signatures and related keys
  forms.py        # form strings ('AABA', 'intro verse chorus') -> ordered labels
  chant.py        # Gregorian modes, Guido's solmization and vowel method, cantus-firmus rules
  masters.py      # Rameau, Schoenberg, Bach chorales, neo-Riemannian, Bartók axes, Coltrane changes
  modern.py       # twelve-tone matrix, pitch-class sets, Glass additive process, Reich phasing
  harmony.py      # intervals, roman-numeral analysis, voice leading, next-chord ranking, reharmonization
  voicing.py      # arranging voicings: drop-2/3/2&4, open, shell, Levine rootless
  melody.py       # degrees, arpeggios, walks, motif grammar, sequence, snap, tintinnabuli
  accompany.py    # accompaniment: chord patterns (Alberti, runs) and bass lines (walking, approach …)
  counterpoint.py # species counterpoint 1-5 (deterministic, rule-following)
  generate.py     # seeded dice + euclidean rhythm + groove presets
  analysis.py     # reading a draft back: key finding, cadences, voice-leading lint
  structure.py    # song structure: plan sections, assemble a whole song
  midi_io.py      # deterministic MIDI rendering: notes, chords, drums, multi-track (mido)
  audio.py        # MIDI -> playable WAV preview, pure standard library
  server.py       # the MCP server (FastMCP) — thin wrappers over the above
```

## Roadmap ideas

- Humanize (timing/velocity jitter as a separate, seeded tool; swing is already on every renderer)
- Pivot-chord modulation planning (`modulation_path`: pivot chords, closely related keys, direct modulation)
- Reading MIDI files back into note/chord data (so `detect_key`, `analyze_progression` and `check_voice_leading` run on your own MIDI); MusicXML export for notation
- Non-chord-tone labelling (passing, neighbour, suspension, appoggiatura …) and its generative inverse, melody embellishment
- Figured-bass realization and Campion's rule of the octave (needs a bass pin in `bach_chorale_voicing`)
- Harmonizing a melody at a harmonic rhythm (one chord per N beats, tolerating non-chord tones); guide-tone lines
- An open chord-symbol grammar (7alt, 7b13, 9sus4, add/omit, parenthesised alterations); chord-scale tables (Levine/Berklee)
- Psalm-tone recitation formulas for the eight modes; Palestrina-style (Jeppesen) melodic rules beyond the cantus
