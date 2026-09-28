"""Song-structure layer: arrange named sections into a whole song.

The LLM composes each section (intro, verse, chorus, bridge, outro...) as a
small multi-track arrangement, then sequences the sections by a form like
"intro verse chorus verse chorus outro". These tools are mechanical: they lay
sections end to end on the timeline and stitch like-named instrument tracks
into continuous MIDI tracks. No musical choices are made here — the caller
decides the sections and their order.
"""

from __future__ import annotations

import math

from .forms import resolve_form as _resolve_form
from .midi_io import (
    _build_file,
    _channel_allocator,
    _check_range,
    _check_swing,
    _common_meta,
    _swing_events,
    _swing_warp,
    _write_file,
    build_track_events,
    DRUM_CHANNEL,
)

# A few conventional default lengths (in bars) by section name.
_DEFAULT_BARS = {
    "intro": 4, "verse": 8, "prechorus": 4, "pre-chorus": 4, "chorus": 8,
    "bridge": 8, "outro": 4, "drop": 8, "break": 4, "fill": 1, "hook": 8,
}


def _bars_for(label: str, bars, default: int) -> int:
    if isinstance(bars, dict):
        value = bars[label] if label in bars else _DEFAULT_BARS.get(label.lower(), default)
    else:
        value = bars
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, str) and value.strip().isdigit():  # MCP clients may send "8"
        value = int(value.strip())
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"section {label!r} bars must be a positive whole number, got {value!r}")
    return value


def plan_sections(form, bars=8, beats_per_bar: int = 4, tempo: int | None = None) -> dict:
    """Lay out a song form on the timeline: where each section starts and how long it is.

    `form` is the running order — a list, a string ("intro verse chorus verse
    chorus outro"), or a letter form ("AABA"). `bars` is either one number for
    every section or a mapping of section name to bars (names like intro/verse/
    chorus/bridge/outro fall back to sensible defaults). Returns each section
    with its start bar, start beat, length, and (if `tempo` is given) start time
    in seconds — so you can place material with `start_beat`, or feed the same
    sections to arrange_song.
    """
    if not isinstance(beats_per_bar, int) or isinstance(beats_per_bar, bool) or not 1 <= beats_per_bar <= 32:
        raise ValueError(f"beats_per_bar must be an integer between 1 and 32, got {beats_per_bar!r}")
    if tempo is not None:
        _check_range("tempo", tempo, 10, 400)
    known = set(_DEFAULT_BARS) | (set(bars) if isinstance(bars, dict) else set())
    labels = _resolve_form(form, known=known)
    sections = []
    bar_cursor = 0
    for i, label in enumerate(labels):
        n_bars = _bars_for(label, bars, 8)
        entry = {
            "index": i,
            "section": label,
            "start_bar": bar_cursor,
            "bars": n_bars,
            "start_beat": bar_cursor * beats_per_bar,
            "length_beats": n_bars * beats_per_bar,
        }
        if tempo is not None:
            entry["start_seconds"] = round(bar_cursor * beats_per_bar * 60 / tempo, 3)
        sections.append(entry)
        bar_cursor += n_bars
    result = {
        "form": labels,
        "beats_per_bar": beats_per_bar,
        "total_bars": bar_cursor,
        "total_beats": bar_cursor * beats_per_bar,
        "sections": sections,
    }
    if tempo is not None:
        result["total_seconds"] = round(bar_cursor * beats_per_bar * 60 / tempo, 3)
    return result


def render_song_structure(sections, form=None, tempo: int = 120, beats_per_bar: int = 4,
                          step_beats: float = 0.5, swing: float = 0.5, swing_unit: float = 0.5,
                          file_name: str | None = None, output_dir: str | None = None) -> dict:
    """Assemble named sections into one multi-track song MIDI file.

    `sections` maps a section name to a section spec
    ``{"bars": N, "tracks": [ ...track objects... ]}`` where each track is the
    same shape as an arrange_to_midi track (notes/chords/drums) with timing
    relative to the section start. `form` is the running order (list/string/
    letters); omitted, the sections play once in given order. Sections are laid
    end to end; tracks with the same `name` across sections become one
    continuous MIDI track (so the "bass" line is one track for the whole song),
    and a name present in only some sections simply rests elsewhere. If a later
    section gives that track another `program`, a program change is written at
    the section start (the verse's piano becomes the bridge's strings). Unnamed
    tracks are named by type ('notes', or 'notes_1', 'notes_2' when a section has
    several), so separate parts never merge. A name cannot be drums in one section
    and pitched in another. An explicit `channel` is honoured (drums always use
    channel 10). The file's time signature follows `beats_per_bar`. `swing` /
    `swing_unit` (or a track's own) swing every section on the song's bar grid:
    pairs of steps count from each downbeat, so every section (and every repeat of
    it) swings alike and bar lines never move — in an odd meter with swung quarters
    (3/4, 5/4 at swing_unit 1.0) the bar's unpaired last quarter stays straight.
    """
    if not isinstance(sections, dict) or not sections:
        raise ValueError("sections must be a non-empty mapping of section name to {bars, tracks}")
    if not isinstance(beats_per_bar, int) or isinstance(beats_per_bar, bool) or not 1 <= beats_per_bar <= 32:
        raise ValueError(f"beats_per_bar must be an integer between 1 and 32, got {beats_per_bar!r}")
    _check_range("tempo", tempo, 10, 400, integer=True)
    _check_range("step_beats", step_beats, 0.0625, 16)
    swing, swing_unit = _check_swing(swing, swing_unit)

    order = _resolve_form(form, known=sections) if form is not None else list(sections)
    for label in order:
        if label not in sections:
            raise ValueError(
                f"form references section {label!r} which is not defined; "
                f"known sections: {', '.join(sections)}"
            )

    bpb = beats_per_bar

    explicit = {t.get("name") for spec in sections.values() if isinstance(spec, dict)
                for t in (spec.get("tracks") or []) if isinstance(t, dict) and t.get("name")}

    def default_names(tracks) -> list:
        """Unnamed tracks are named by type — 'notes', or 'notes_1', 'notes_2' when a section
        has several — skipping names the caller uses, so separate parts never merge."""
        counts: dict[str, int] = {}
        for t in tracks:
            if isinstance(t, dict) and not t.get("name"):
                counts[t.get("type")] = counts.get(t.get("type"), 0) + 1
        taken = set(explicit)
        names = []
        for t in tracks:
            if not isinstance(t, dict) or t.get("name"):
                names.append(None)
                continue
            ttype = str(t.get("type"))
            if counts[t.get("type")] == 1 and ttype not in taken:
                name = ttype
            else:
                k = 1
                while f"{ttype}_{k}" in taken:
                    k += 1
                name = f"{ttype}_{k}"
            taken.add(name)
            names.append(name)
        return names

    # Build each distinct section once (events relative to its own start).
    built_sections: dict[str, dict] = {}
    for label, spec in sections.items():
        if not isinstance(spec, dict) or "tracks" not in spec:
            raise ValueError(f"section {label!r} must be an object with a 'tracks' list")
        tracks = spec["tracks"]
        if not isinstance(tracks, (list, tuple)) or not tracks:
            raise ValueError(f"section {label!r} has no tracks")
        built = []
        for i, (t, default) in enumerate(zip(tracks, default_names(tracks))):
            # swing is applied below, once the section sits at its place in the song
            b = build_track_events(t, i, step_beats, bpb, swing=swing, swing_unit=swing_unit,
                                   apply_swing=False)
            if default:
                b["name"] = default
            b["channel"] = t.get("channel")
            if b["channel"] is not None and not b["is_drums"]:
                _check_range(f"section {label!r} track {i} channel", b["channel"], 0, 15, integer=True)
            built.append(b)
        content = max((b["rel_end"] for b in built), default=0.0)
        declared = spec.get("bars")
        if declared is not None:
            if not isinstance(declared, int) or isinstance(declared, bool) or declared < 1:
                raise ValueError(f"section {label!r} bars must be a positive integer")
            length = max(content, declared * bpb)
        else:
            length = content
        # round up to whole bars, ignoring float noise (0.4 + 58 * 0.2 is 12 beats, not 12.000...02)
        length = max(bpb, math.ceil(round(length / bpb, 6)) * bpb)
        built_sections[label] = {"built": built, "length": length, "bars": int(length // bpb)}

    # Sequence sections, accumulating events per track name. A name keeps one MIDI
    # track; if a later section plays it with another program, a program change is
    # written at that section's start.
    name_events: dict[str, list] = {}
    name_meta: dict[str, dict] = {}  # name -> first program, drums flag, explicit channel
    name_program: dict[str, int] = {}  # the program currently sounding on that track
    name_order: list[str] = []
    timeline = []
    offset = 0.0
    bar_cursor = 0
    for occ, label in enumerate(order):
        sec = built_sections[label]
        for b in sec["built"]:
            tname = b["name"]
            if tname not in name_meta:
                name_meta[tname] = {"program": b["program"], "is_drums": b["is_drums"], "channel": b["channel"]}
                name_program[tname] = b["program"]
                name_order.append(tname)
            elif name_meta[tname]["is_drums"] != b["is_drums"]:
                raise ValueError(
                    f"track {tname!r} is a drums track in one section and a pitched track in another;"
                    f" give them different names"
                )
            elif not b["is_drums"] and b["program"] != name_program[tname]:
                change = _swing_warp(b["start"] + offset, *b["swing"], bpb)
                name_events[tname].append({"start": change, "program": b["program"]})
                name_program[tname] = b["program"]
            shifted = [dict(e, start=e["start"] + offset) for e in b["events"]]
            # swung on the song's bar grid, which never moves a bar line
            shifted = _swing_events(shifted, *b["swing"], bpb)
            name_events.setdefault(tname, []).extend(shifted)
        timeline.append({
            "index": occ,
            "section": label,
            "start_bar": bar_cursor,
            "bars": sec["bars"],
            "start_beat": offset,
            "length_beats": sec["length"],
        })
        offset += sec["length"]
        bar_cursor += sec["bars"]

    # One channel per track name: drums share the percussion channel, explicit channels
    # are honoured (first occurrence wins), the rest are allocated around them.
    reserved = {m["channel"] for m in name_meta.values() if m["channel"] is not None and not m["is_drums"]}
    channels = _channel_allocator(reserved)
    parts = []
    track_summary = []
    for tname in name_order:
        meta = name_meta[tname]
        program, is_drums = meta["program"], meta["is_drums"]
        if is_drums:
            channel = DRUM_CHANNEL
        elif meta["channel"] is not None:
            channel = meta["channel"]
        else:
            channel = next(channels)
        events = name_events[tname]
        parts.append({"events": events, "channel": channel, "program": program, "name": tname})
        summary = {
            "name": tname,
            "channel": channel,
            "program": program,
            "is_drums": is_drums,
            "event_count": sum(1 for e in events if "program" not in e),
        }
        changes = [e["program"] for e in events if "program" in e]
        if changes:
            summary["program_changes"] = changes
        track_summary.append(summary)

    total_beats = offset   # swing on the bar grid ends every note by the final bar line
    mid = _build_file(parts, tempo, bpb, total_beats=total_beats)
    result = _write_file(mid, file_name, output_dir, "song")
    result.update(_common_meta(tempo, total_beats))
    result["form"] = order
    result["beats_per_bar"] = bpb
    result["total_bars"] = bar_cursor
    result["section_count"] = len(order)
    result["track_count"] = len(parts)
    result["sections"] = timeline
    result["tracks"] = track_summary
    return result
