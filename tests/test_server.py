"""Smoke tests: the MCP server exposes all tools and they are callable."""

import asyncio
import json

from midi_composer_mcp.server import mcp

EXPECTED_TOOLS = {
    "list_scales",
    "list_chords",
    "get_scale",
    "get_chord",
    "match_scales",
    "match_chords",
    "diatonic_chords",
    "degrees_to_chords",
    "random_notes",
    "random_rhythm",
    "notes_to_midi",
    "chords_to_midi",
    "song_to_midi",
}


def _call(name, arguments):
    return asyncio.run(mcp.call_tool(name, arguments))


def test_all_tools_registered():
    tools = asyncio.run(mcp.list_tools())
    names = {t.name for t in tools}
    assert EXPECTED_TOOLS <= names
    for tool in tools:
        assert tool.description, f"tool {tool.name} has no description"


def test_call_get_scale_via_server():
    result = _call("get_scale", {"scale_type": "major", "root": "C"})
    text = json.dumps([getattr(c, "text", "") for c in result[0]] if isinstance(result, tuple) else str(result))
    for note in ["C", "D", "E", "F", "G", "A", "B"]:
        assert note in text


def test_call_chain_scale_to_match(tmp_path):
    # output of one tool is valid input for another
    from midi_composer_mcp.chords import chord_info
    from midi_composer_mcp.scales import match_scales, scale_info

    chord = chord_info("m", "A")
    scales = match_scales(chord["notes"])
    assert any(
        m["root"] == "A" and m["scale_type"] == "natural minor" for m in scales["matches"]
    )

    pool = scale_info("blues", "A")["notes"]
    result = _call(
        "notes_to_midi",
        {"notes": pool, "octave_policy": "ascending", "output_dir": str(tmp_path)},
    )
    assert result is not None
