"""Generate the demo gallery under ``demos/`` — one piece per feature cluster.

    python demos/generate.py            # writes .mid + .wav into demos/

Every piece is composed only from the toolset's deterministic rule-tools (with
the two seeded random tools used reproducibly), then rendered to MIDI and a
playable WAV. See demos/README.md for what each one showcases.
"""

from __future__ import annotations

import os
import struct
import sys
import wave

from midi_composer_mcp.chords import chord_info, chord_notes, parse_chord_symbol
from midi_composer_mcp.counterpoint import species_counterpoint
from midi_composer_mcp.generate import groove
from midi_composer_mcp.harmony import (
    analyze_progression, negative_harmony, secondary_dominant,
    tritone_substitute, voice_leading,
)
from midi_composer_mcp.melody import (
    melodic_walk, motif_grammar, notes_from_degrees, tintinnabuli_voice, transpose_notes,
)
from midi_composer_mcp.midi_io import render_arrangement
from midi_composer_mcp.structure import render_song_structure
from midi_composer_mcp.audio import render_midi_to_wav
from midi_composer_mcp.accompany import bass_line, chord_pattern
from midi_composer_mcp.analysis import detect_key, find_cadences
from midi_composer_mcp.roman import progression_library, roman_to_chords
from midi_composer_mcp.voicing import voice_chords

HERE = os.path.dirname(os.path.abspath(__file__))


def _drums(bars, big=False):
    lanes = {"kick": "O...O...O...O..." * bars, "snare": "....O.......O..." * bars,
             "hat": "o.o.o.o.o.o.o.o." * bars}
    if big:
        lanes["open_hat"] = "..O...O...O...O." * bars
    return {"type": "drums", "name": "drums", "step_beats": 0.25, "lanes": lanes}


# ===================================== 1. TINTINNABULI (Arvo Pärt "Cantus" style)
def tintinnabuli_cantus(out):
    """A descending additive cantus in A minor, shadowed by an A-minor tintinnabuli
    voice, with a tolling bell, a tonic drone, and soft voice-led maj7/m7 pads."""
    degrees = []
    for length in range(1, 7):                 # phrases 1..6 notes, each descending to the tonic
        degrees += list(range(length, 0, -1))  # e.g. length 3 -> [3, 2, 1]
    m_voice = notes_from_degrees("A4", "natural minor", degrees)["notes"]
    t_voice = tintinnabuli_voice(m_voice, "Am", position="alternating", rank=1, octave=4)["t_voice"]
    step, n = 1.5, len(m_voice)
    total = n * step
    pads = voice_leading(["Am7", "Fmaj7", "Cmaj7", "Em7", "Dm7", "Fmaj7", "Em7", "Am7"], octave=3)["chords"]
    return render_arrangement([
        {"type": "chords", "name": "pads", "chords": pads, "beats_per_chord": total / len(pads),
         "program": 89, "velocity": 38},
        {"type": "notes", "name": "M-voice", "notes": m_voice, "step_beats": step,
         "octave": 4, "program": 48, "velocity": 80, "sustain": True},
        {"type": "notes", "name": "T-voice", "notes": t_voice, "step_beats": step,
         "octave": 4, "program": 9, "velocity": 58, "sustain": True},
        {"type": "notes", "name": "bell", "notes": ["A5"] * max(1, int(total // 4)),
         "step_beats": 4.0, "octave": 5, "program": 14, "velocity": 64},
        {"type": "notes", "name": "drone", "notes": ["A2"] * max(1, int(total // 8)),
         "step_beats": 8.0, "octave": 2, "program": 48, "velocity": 40, "sustain": True},
    ], tempo=60, output_dir=out, file_name="01_tintinnabuli_cantus.mid")


# ====================================== 2. ANTHEM (song structure + modulation)
def anthem(out):
    def pads(chords, prog=0, vel=66, arp=False, up=0):
        v = voice_leading(chords, octave=4)["chords"]
        if up:
            v = [transpose_notes(c, up)["notes"] for c in v]
        return {"type": "chords", "name": "pads", "chords": v, "beats_per_chord": 4,
                "program": prog, "velocity": vel, "arpeggiate": arp}

    def bass(roots, up=0):
        return {"type": "notes", "name": "bass", "notes": transpose_notes(roots, up)["notes"] if up else roots,
                "step_beats": 4.0, "octave": 2, "program": 33, "velocity": 95}

    def lead(notes, prog=80, vel=100, up=0):
        return {"type": "notes", "name": "lead", "notes": transpose_notes(notes, up)["notes"] if up else notes,
                "octave": 5, "program": prog, "velocity": vel}

    verse_mel = notes_from_degrees("C5", "major", motif_grammar(
        "ABAC", {"A": [5, 5, 6, 5], "B": {"vary": "A", "transpose": 1},
                 "C": {"vary": "A", "retrograde": True}}, kind="degrees")["degrees"])["notes"]
    chorus_mel = notes_from_degrees("C5", "major", motif_grammar(
        "ABAB", {"A": [8, 8, 7, 5], "B": [6, 5, 6, 8]}, kind="degrees")["degrees"])["notes"]
    bridge_mel = notes_from_degrees("C5", "major",
                                    [6, 5, 4, 3, 4, 5, 6, 5, 2, 3, 4, 5, 4, 2, 7, 1])["notes"]
    sections = {
        "intro": {"bars": 2, "tracks": [pads(["Am", "F"], prog=89, vel=55, arp=True)]},
        "verse": {"bars": 4, "tracks": [pads(["C", "G", "Am", "F"]), bass(["C", "G", "A", "F"]), lead(verse_mel), _drums(4)]},
        "pre": {"bars": 2, "tracks": [pads(["F", "G", "Em", "Am"], vel=70), bass(["F", "G", "E", "A"]), _drums(2)]},
        "chorus": {"bars": 4, "tracks": [pads(["C", "G", "F", "Am"], vel=74), bass(["C", "G", "F", "A"]),
                                         lead(chorus_mel, prog=81, vel=108), _drums(4, big=True)]},
        "bridge": {"bars": 4, "tracks": [pads(["Am", "F", "C", "G"], prog=48, vel=62), bass(["A", "F", "C", "G"]),
                                         lead(bridge_mel, vel=88), _drums(4)]},
        "chorus_up": {"bars": 4, "tracks": [pads(["C", "G", "F", "Am"], vel=78, up=2), bass(["C", "G", "F", "A"], up=2),
                                            lead(chorus_mel, prog=81, vel=112, up=2), _drums(4, big=True)]},
        "outro": {"bars": 2, "tracks": [pads(["F", "C"], prog=89, vel=55, arp=True)]},
    }
    return render_song_structure(sections, form="intro verse pre chorus verse pre chorus bridge chorus_up outro",
                                 tempo=120, output_dir=out, file_name="02_anthem.mid")


# ============================== 3. JAZZ REHARM (analysis + reharmonization + voice leading)
def jazz_reharm(out):
    chords = ["Cmaj7", "A7", "Dm7", tritone_substitute("G7")["symbol"],
              "Cmaj7", "E7", "Am7", secondary_dominant("G")["symbol"],
              "Dm7", "G7", "Cmaj7", "Cmaj7"]
    roots = [parse_chord_symbol(c)[0].pitch_class_name for c in chords]
    voiced = voice_leading(chords, octave=4)["chords"]
    walk = []
    for r in roots:
        fifth = transpose_notes([r], 7)["notes"][0]
        walk += [f"{r}2", f"{fifth}2", f"{r}3", f"{fifth}2"]
    guide = []
    for c in chords:
        root, ctype, _ = parse_chord_symbol(c)
        tones = chord_info(ctype.name, root.pitch_class_name)["notes"]
        guide += [tones[1], tones[3]]
    numerals = [c["roman"] for c in analyze_progression(chords, "C", "major")["chords"]]
    result = render_arrangement([
        {"type": "chords", "name": "comp", "chords": voiced, "beats_per_chord": 4, "program": 0, "velocity": 64},
        {"type": "notes", "name": "walk", "notes": walk, "step_beats": 1.0, "octave": 2, "program": 32, "velocity": 88},
        {"type": "notes", "name": "head", "notes": guide, "step_beats": 2.0, "octave": 4,
         "octave_policy": "nearest", "program": 56, "velocity": 84},
        {"type": "drums", "name": "swing", "step_beats": 0.5,
         "lanes": {"ride": "o.O.o.O." * 12, "pedal_hat": "..O...O." * 12}},
    ], tempo=146, output_dir=out, file_name="03_jazz_reharm.mid")
    result["analysis"] = " ".join(f"{c}={n}" for c, n in zip(chords, numerals))
    return result


# ============================== 4. COUNTERPOINT SUITE (all five species, one cantus)
def counterpoint_species(out):
    cf = ["C5", "D5", "E5", "F5", "E5", "D5", "C5"]
    sections = {}
    for s in (1, 2, 3, 4, 5):
        t = species_counterpoint(cf, "C", "major", species=s, position="above")["render_hint"]["tracks"]
        t[0] = {**t[0], "program": 48, "velocity": 70}
        t[1] = {**t[1], "program": 6, "velocity": 88}
        sections[f"sp{s}"] = {"bars": len(cf), "tracks": t}
    return render_song_structure(sections, form="sp1 sp2 sp3 sp4 sp5", tempo=88,
                                 output_dir=out, file_name="04_counterpoint_species.mid")


# ============================== 5. FLAMENCO (exotic scale + clave + generative melody)
def flamenco(out):
    ladder = notes_from_degrees("E4", "phrygian dominant", list(range(1, 15)))["notes"]
    mel = melodic_walk(ladder, length=24, seed=5, max_step=2, start=2)["notes"]
    clave = groove("rumba_clave_32")["pattern"]
    return render_arrangement([
        {"type": "chords", "name": "guitar", "chords": ["E", "E", "F", "F"] * 2, "beats_per_chord": 4,
         "octave": 3, "program": 25, "arpeggiate": True, "velocity": 70},
        {"type": "notes", "name": "lead", "notes": mel, "rhythm": clave * 8, "step_beats": 0.25,
         "octave": 4, "program": 24, "velocity": 92},
        {"type": "notes", "name": "bass", "notes": ["E", "E", "F", "F", "E", "E", "F", "F"],
         "step_beats": 4.0, "octave": 2, "program": 33, "velocity": 96},
        {"type": "drums", "name": "perc", "step_beats": 0.25,
         "lanes": {"clave": clave * 8, "conga": "..o.o...o.o.o..." * 8, "clap": "....O.......O..." * 8}},
    ], tempo=124, output_dir=out, file_name="05_flamenco.mid")


# ============================== 6. NEGATIVE HARMONY (the idea, then its mirror)
def negative_harmony_demo(out):
    prog = ["C", "Am", "F", "G"]
    mel = notes_from_degrees("C5", "major", [5, 3, 1, 5, 6, 5, 3, 2])["notes"]

    def neg_notes(sym):
        root, ctype, _ = parse_chord_symbol(sym)
        return negative_harmony([t.name for t in chord_notes(ctype, root)], "C")["notes"]

    sections = {
        "original": {"bars": 4, "tracks": [
            {"type": "chords", "name": "keys", "chords": voice_leading(prog, octave=4)["chords"],
             "beats_per_chord": 4, "program": 0, "velocity": 70},
            {"type": "notes", "name": "bass", "notes": ["C", "A", "F", "G"], "step_beats": 4.0, "octave": 2, "program": 33},
            {"type": "notes", "name": "lead", "notes": mel, "step_beats": 2.0, "octave": 5, "program": 73, "velocity": 90},
        ]},
        "mirror": {"bars": 4, "tracks": [
            {"type": "chords", "name": "keys", "chords": [neg_notes(s) for s in prog],
             "beats_per_chord": 4, "octave": 4, "program": 0, "velocity": 70},
            {"type": "notes", "name": "bass",
             "notes": [negative_harmony([parse_chord_symbol(s)[0].pitch_class_name], "C")["notes"][0] for s in prog],
             "step_beats": 4.0, "octave": 2, "program": 33},
            {"type": "notes", "name": "lead", "notes": negative_harmony(mel, "C")["notes"],
             "step_beats": 2.0, "octave": 5, "program": 73, "velocity": 90},
        ]},
    }
    return render_song_structure(sections, form="original mirror", tempo=100,
                                 output_dir=out, file_name="06_negative_harmony.mid")


# ============================== 7. JAZZ COMBO (ideas ported from Scaler / Hookpad / music21)
def jazz_combo(out):
    """A swung AABA-ish combo tune in F: a named turnaround, a backdoor bridge written in
    Roman numerals, Levine rootless piano comping, a vibes figure over the voicings, a walking
    bass, and swing on the song's grid — then read back with detect_key and find_cadences."""
    a = progression_library("jazz_turnaround", root="F")["chords"] * 2       # Fmaj7 Dm7 Gm7 C7 x2
    b = roman_to_chords("iv7 bVII7 IΔ7 IΔ7 ii7 V7 IΔ7 IΔ7", "F")["symbols"]  # backdoor, then ii-V-I
    coda = roman_to_chords("ii7 V7 IΔ7 IΔ7", "F")["symbols"]

    def section(chords):
        lh = voice_chords(chords, "rootless", octave=4)
        vibes = chord_pattern(lh["chords"], "^4 . 3 2 . 1 3 .", beats_per_chord=4, step_beats=0.5, smooth=False)
        bass = bass_line(chords, "walking")
        return {"bars": len(chords), "tracks": [
            {"type": "chords", "name": "piano", "chords": lh["chords"], "rhythm": "O..o....", "step_beats": 0.5,
             "program": 0, "velocity": 64, "accent_velocity": 80},                # Charleston comping
            {**vibes["track"], "name": "vibes", "program": 11, "velocity": 78},
            {**bass["track"], "program": 32, "velocity": 96},
            {"type": "drums", "name": "drums", "step_beats": 0.5,
             "lanes": {"ride": "O.oOO.oO" * len(chords), "pedal_hat": "..o...o." * len(chords)}},
        ]}

    sections = {"A": section(a), "B": section(b), "coda": section(coda)}
    result = render_song_structure(sections, form="A B A coda", tempo=132, swing=2 / 3,
                                   output_dir=out, file_name="07_jazz_combo.mid")
    tune = a + b + a + coda
    key = detect_key(chords=tune)
    cad = find_cadences(tune, key["root"], key["scale_type"], phrase_ends=[7, 14, 23, 26])  # where each phrase arrives
    result["analysis"] = f"key {key['root']} {key['scale_type']}; cadences " + ", ".join(
        f"{c['index']}:{c['type']}" for c in cad["cadences"])
    return result


DEMOS = [
    (tintinnabuli_cantus,
     "Write a slow, meditative Arvo Part-style tintinnabuli piece in A minor — a melody that "
     "grows in descending phrases out of the tonic, shadowed by the notes of the A-minor triad, "
     "with a tolling bell, a drone, and a few soft maj7/m7 colours."),
    (anthem,
     "Write an uplifting pop anthem in C major: intro, verse, pre-chorus, a big chorus, a bridge, "
     "and a final chorus that modulates up a whole step."),
    (jazz_reharm,
     "Take a ii-V-I turnaround in C and make it jazzier — add secondary dominants and a tritone "
     "substitution, then comp it with smooth voicings and a walking bass."),
    (counterpoint_species,
     "Write a counterpoint to this cantus firmus and show me all five species, from "
     "note-against-note to florid."),
    (flamenco,
     "Give me a flamenco piece in E Phrygian dominant — an improvised-sounding guitar line over a "
     "rumba clave with hand percussion."),
    (negative_harmony_demo,
     "Play a I-vi-IV-V with a melody in C, then play its negative-harmony mirror so I can hear "
     "major flip to its minor shadow."),
    (jazz_combo,
     "Write a swung jazz combo tune in F — a turnaround, a backdoor bridge, Bill Evans-style "
     "rootless piano comping, a vibes line, a walking bass and ride cymbal — and tell me its key "
     "and cadences."),
]


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else HERE
    os.makedirs(out, exist_ok=True)
    for builder, prompt in DEMOS:
        midi = builder(out)
        wav = render_midi_to_wav(midi["file"], sample_rate=22050)  # lighter previews for the repo
        with wave.open(wav["file"], "rb") as w:
            frames = w.readframes(w.getnframes())
        peak = max(abs(s) for s in struct.unpack("<%dh" % (len(frames) // 2), frames))
        print(f"\n{midi['file_name']}  ({midi['duration_seconds']:.1f}s, peak {peak})")
        print(f'  prompt: "{prompt}"')
        if "analysis" in midi:
            print(f"  analysis: {midi['analysis']}")


if __name__ == "__main__":
    main()
