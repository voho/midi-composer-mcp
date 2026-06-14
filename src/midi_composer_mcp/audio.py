"""Render a generated MIDI file to a self-contained, universally playable WAV.

A bare .mid file needs a synthesizer or soundfont to be heard. This module
turns any MIDI file the server produced into a 16-bit PCM WAV using only the
Python standard library — no soundfont, no external synth — so the output is
playable on any device or browser. The synthesis is intentionally simple
(additive tones for pitched notes, percussive noise/sine for General MIDI
drums); it is a faithful preview, not a production-quality render.
"""

from __future__ import annotations

import array
import base64
import math
import os
import random
import wave

import mido

_TABLE_BITS = 12
_TABLE_SIZE = 1 << _TABLE_BITS
_TABLE_MASK = _TABLE_SIZE - 1
_SINE = [math.sin(2 * math.pi * i / _TABLE_SIZE) for i in range(_TABLE_SIZE)]

# Harmonic recipes (multiple, amplitude) by instrument family, chosen from the
# General MIDI program number so bass/lead/pad timbres differ a little.
_TIMBRES = {
    "bass": [(1, 1.0), (2, 0.4), (3, 0.15)],
    "lead": [(1, 1.0), (2, 0.5), (3, 0.3), (4, 0.12)],
    "pad": [(1, 1.0), (2, 0.25), (3, 0.12), (5, 0.05)],
    "default": [(1, 1.0), (2, 0.35), (3, 0.15)],
}

# General MIDI percussion note -> a coarse drum category for synthesis.
_KICKS = {35, 36}
_SNARES = {37, 38, 39, 40}
_CLOSED_HATS = {42, 44}
_OPEN_HATS = {46}
_CYMBALS = {49, 51, 52, 53, 55, 57, 59}
_TOMS = {41, 43, 45, 47, 48, 50}


def _timbre_for_program(program: int) -> list[tuple[int, float]]:
    if 32 <= program <= 39:
        return _TIMBRES["bass"]
    if 88 <= program <= 103 or 48 <= program <= 55:
        return _TIMBRES["pad"]
    if 80 <= program <= 87 or program <= 7:
        return _TIMBRES["lead"]
    return _TIMBRES["default"]


def _freq(note: int) -> float:
    return 440.0 * 2.0 ** ((note - 69) / 12.0)


def _add_tone(buf: array.array, sample_rate: int, start: float, duration: float,
              note: int, velocity: int, harmonics: list[tuple[int, float]]) -> None:
    n = len(buf)
    start_i = int(start * sample_rate)
    total = max(1, int(duration * sample_rate))
    attack = min(max(1, int(0.006 * sample_rate)), total)
    release = min(max(1, int(0.05 * sample_rate)), total)
    amp = (velocity / 127.0) * 0.28
    base = _freq(note) / sample_rate * _TABLE_SIZE
    phases = [0.0] * len(harmonics)
    increments = [base * mult for mult, _ in harmonics]
    for i in range(total):
        idx = start_i + i
        if idx >= n:
            break
        if i < attack:
            env = i / attack
        elif i > total - release:
            env = (total - i) / release
        else:
            env = 1.0
        env *= 1.0 - 0.35 * (i / total)  # gentle decay across the note
        sample = 0.0
        for h, (_, h_amp) in enumerate(harmonics):
            phase = phases[h]
            sample += h_amp * _SINE[int(phase) & _TABLE_MASK]
            phases[h] = phase + increments[h]
        buf[idx] += amp * env * sample


def _add_noise(buf: array.array, sample_rate: int, start: float, duration: float,
               amp: float, rng: random.Random, lowpass: float = 0.0) -> None:
    n = len(buf)
    start_i = int(start * sample_rate)
    total = max(1, int(duration * sample_rate))
    prev = 0.0
    for i in range(total):
        idx = start_i + i
        if idx >= n:
            break
        env = (1.0 - i / total) ** 2
        white = rng.uniform(-1.0, 1.0)
        if lowpass:
            prev = prev + lowpass * (white - prev)
            white = prev
        buf[idx] += amp * env * white


def _add_drum(buf: array.array, sample_rate: int, start: float, note: int,
              velocity: int, rng) -> None:
    amp = (velocity / 127.0) * 0.5
    if note in _KICKS:
        total = int(0.18 * sample_rate)
        start_i = int(start * sample_rate)
        for i in range(total):
            idx = start_i + i
            if idx >= len(buf):
                break
            t = i / sample_rate
            freq = 110.0 * math.exp(-30.0 * t) + 45.0  # pitch drop
            env = math.exp(-18.0 * t)
            buf[idx] += amp * env * math.sin(2 * math.pi * freq * t)
    elif note in _SNARES:
        _add_noise(buf, sample_rate, start, 0.16, amp * 0.8, rng)
        start_i = int(start * sample_rate)
        for i in range(int(0.12 * sample_rate)):
            idx = start_i + i
            if idx >= len(buf):
                break
            t = i / sample_rate
            buf[idx] += amp * 0.4 * math.exp(-24.0 * t) * math.sin(2 * math.pi * 185.0 * t)
    elif note in _CLOSED_HATS:
        _add_noise(buf, sample_rate, start, 0.04, amp * 0.5, rng, lowpass=0.0)
    elif note in _OPEN_HATS:
        _add_noise(buf, sample_rate, start, 0.22, amp * 0.45, rng)
    elif note in _CYMBALS:
        _add_noise(buf, sample_rate, start, 0.5, amp * 0.4, rng)
    elif note in _TOMS:
        total = int(0.2 * sample_rate)
        start_i = int(start * sample_rate)
        freq = _freq(note)
        for i in range(total):
            idx = start_i + i
            if idx >= len(buf):
                break
            t = i / sample_rate
            env = math.exp(-12.0 * t)
            buf[idx] += amp * env * math.sin(2 * math.pi * freq * t)
    else:
        _add_noise(buf, sample_rate, start, 0.1, amp * 0.5, rng)


def render_midi_to_wav(midi_path: str, wav_path: str | None = None,
                       sample_rate: int = 44100, max_seconds: float = 300.0) -> dict:
    """Synthesize `midi_path` into a 16-bit mono WAV and return its path + base64."""
    if not os.path.isfile(midi_path):
        raise ValueError(f"MIDI file not found: {midi_path}")
    if not 8000 <= sample_rate <= 48000:
        raise ValueError(f"sample_rate must be between 8000 and 48000, got {sample_rate}")
    mid = mido.MidiFile(midi_path)

    # Collect note voices with absolute start/duration in seconds.
    programs: dict[int, int] = {}
    active: dict[tuple[int, int], tuple[float, int, int]] = {}
    voices: list[tuple] = []  # (start, duration, channel, note, velocity, program)
    drums: list[tuple[float, int, int]] = []  # (start, note, velocity)
    now = 0.0
    duration_limited = False
    for msg in mid:
        now += msg.time
        if now > max_seconds:
            duration_limited = True
            break
        if msg.type == "program_change":
            programs[msg.channel] = msg.program
        elif msg.type == "note_on" and msg.velocity > 0:
            active[(msg.channel, msg.note)] = (now, msg.velocity, programs.get(msg.channel, 0))
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            key = (msg.channel, msg.note)
            started = active.pop(key, None)
            if started is None:
                continue
            start, velocity, program = started
            if msg.channel == 9:
                drums.append((start, msg.note, velocity))
            else:
                voices.append((start, max(0.02, now - start), msg.channel, msg.note, velocity, program))
    # Notes still held when the file ends (or was truncated): give them a tail.
    for (channel, note), (start, velocity, program) in active.items():
        if channel == 9:
            drums.append((start, note, velocity))
        else:
            voices.append((start, 0.3, channel, note, velocity, program))

    end = 0.0
    for start, dur, *_ in voices:
        end = max(end, start + dur)
    for start, *_ in drums:
        end = max(end, start + 0.5)
    end = min(end, max_seconds) + 0.2
    total_samples = max(1, int(end * sample_rate))
    buf = array.array("d", bytes(8 * total_samples))  # zero-filled float accumulator

    for start, dur, _channel, note, velocity, program in voices:
        _add_tone(buf, sample_rate, start, dur, note, velocity, _timbre_for_program(program))
    rng = random.Random(1234)
    for start, note, velocity in drums:
        _add_drum(buf, sample_rate, start, note, velocity, rng)

    # Normalize to avoid clipping, then quantize to int16.
    peak = max((abs(s) for s in buf), default=0.0)
    scale = (0.95 * 32767.0 / peak) if peak > 1e-9 else 0.0
    pcm = array.array("h", bytes(2 * total_samples))
    for i in range(total_samples):
        v = int(buf[i] * scale)
        pcm[i] = -32768 if v < -32768 else 32767 if v > 32767 else v

    if wav_path is None:
        root, _ = os.path.splitext(midi_path)
        wav_path = root + ".wav"
    wav_path = os.path.abspath(wav_path)
    with wave.open(wav_path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())

    with open(wav_path, "rb") as fh:
        data = fh.read()
    return {
        "file": wav_path,
        "file_name": os.path.basename(wav_path),
        "size_bytes": len(data),
        "format": "WAV (16-bit PCM, mono)",
        "sample_rate": sample_rate,
        "duration_seconds": round(total_samples / sample_rate, 3),
        "note_count": len(voices),
        "drum_count": len(drums),
        "truncated": duration_limited,
        "base64": base64.b64encode(data).decode("ascii"),
    }
