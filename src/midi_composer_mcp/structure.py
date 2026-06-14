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
    _common_meta,
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
        if label in bars:
            return int(bars[label])
        return int(_DEFAULT_BARS.get(label.lower(), default))
    return int(bars)


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
    labels = _resolve_form(form)
    sections = []
    bar_cursor = 0
    for i, label in enumerate(labels):
        n_bars = _bars_for(label, bars, 8)
        if n_bars < 1:
            raise ValueError(f"section {label!r} must be at least 1 bar")
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
                          step_beats: float = 0.5, file_name: str | None = None,
                          output_dir: str | None = None) -> dict:
    """Assemble named sections into one multi-track song MIDI file.

    `sections` maps a section name to a section spec
    ``{"bars": N, "tracks": [ ...track objects... ]}`` where each track is the
    same shape as an arrange_to_midi track (notes/chords/drums) with timing
    relative to the section start. `form` is the running order (list/string/
    letters); omitted, the sections play once in given order. Sections are laid
    end to end; tracks with the same `name` across sections become one
    continuous MIDI track (so the "bass" line is one track for the whole song),
    and a name present in only some sections simply rests elsewhere.
    """
    if not isinstance(sections, dict) or not sections:
        raise ValueError("sections must be a non-empty mapping of section name to {bars, tracks}")
    if not isinstance(beats_per_bar, int) or isinstance(beats_per_bar, bool) or not 1 <= beats_per_bar <= 32:
        raise ValueError(f"beats_per_bar must be an integer between 1 and 32, got {beats_per_bar!r}")

    order = _resolve_form(form) if form is not None else list(sections)
    for label in order:
        if label not in sections:
            raise ValueError(
                f"form references section {label!r} which is not defined; "
                f"known sections: {', '.join(sections)}"
            )

    bpb = beats_per_bar

    # Build each distinct section once (events relative to its own start).
    built_sections: dict[str, dict] = {}
    for label, spec in sections.items():
        if not isinstance(spec, dict) or "tracks" not in spec:
            raise ValueError(f"section {label!r} must be an object with a 'tracks' list")
        tracks = spec["tracks"]
        if not isinstance(tracks, (list, tuple)) or not tracks:
            raise ValueError(f"section {label!r} has no tracks")
        built = [build_track_events(t, i, step_beats, bpb) for i, t in enumerate(tracks)]
        content = max((b["rel_end"] for b in built), default=0.0)
        declared = spec.get("bars")
        if declared is not None:
            if not isinstance(declared, int) or isinstance(declared, bool) or declared < 1:
                raise ValueError(f"section {label!r} bars must be a positive integer")
            length = max(content, declared * bpb)
        else:
            length = content
        length = max(bpb, math.ceil(length / bpb) * bpb)  # round up to whole bars
        built_sections[label] = {"built": built, "length": length, "bars": int(length // bpb)}

    # Sequence sections, accumulating events per track name.
    name_events: dict[str, list] = {}
    name_meta: dict[str, tuple] = {}  # name -> (program, is_drums)
    name_order: list[str] = []
    timeline = []
    offset = 0.0
    bar_cursor = 0
    for occ, label in enumerate(order):
        sec = built_sections[label]
        for b in sec["built"]:
            tname = b["name"]
            if tname not in name_meta:
                name_meta[tname] = (b["program"], b["is_drums"])
                name_order.append(tname)
            shifted = [dict(e, start=e["start"] + offset) for e in b["events"]]
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

    # Assign one channel per track name (drums share the percussion channel).
    channels = _channel_allocator()
    parts = []
    track_summary = []
    for tname in name_order:
        program, is_drums = name_meta[tname]
        channel = DRUM_CHANNEL if is_drums else next(channels)
        events = name_events[tname]
        parts.append({"events": events, "channel": channel, "program": program, "name": tname})
        track_summary.append({
            "name": tname,
            "channel": channel,
            "program": program,
            "is_drums": is_drums,
            "event_count": len(events),
        })

    mid = _build_file(parts, tempo)
    result = _write_file(mid, file_name, output_dir, "song")
    result.update(_common_meta(tempo, offset))
    result["form"] = order
    result["beats_per_bar"] = bpb
    result["total_bars"] = bar_cursor
    result["section_count"] = len(order)
    result["track_count"] = len(parts)
    result["sections"] = timeline
    result["tracks"] = track_summary
    return result
