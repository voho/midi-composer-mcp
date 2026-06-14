import pytest

from midi_composer_mcp.generate import parse_rhythm, random_notes, random_rhythm
from midi_composer_mcp.scales import scale_info


def test_random_notes_from_pool():
    result = random_notes(["C", "E", "G"], count=10, seed=42)
    assert len(result["notes"]) == 10
    assert set(result["notes"]) <= {"C", "E", "G"}
    assert result["seed"] == 42


def test_random_notes_reproducible():
    a = random_notes("c d e f g a b", count=8, seed=7)
    b = random_notes("c d e f g a b", count=8, seed=7)
    assert a["notes"] == b["notes"]


def test_random_notes_reports_generated_seed():
    result = random_notes(["C", "E", "G"], count=4)
    again = random_notes(["C", "E", "G"], count=4, seed=result["seed"])
    assert result["notes"] == again["notes"]


def test_random_notes_composes_with_scale_output():
    pool = scale_info("minor pentatonic", "A")["notes"]
    result = random_notes(pool, count=6, seed=1)
    assert set(result["notes"]) <= set(pool)


def test_random_notes_keeps_octaves():
    pool = scale_info("major", "C5")["notes"]
    result = random_notes(pool, count=5, seed=3)
    assert all(any(ch.isdigit() for ch in n) for n in result["notes"])


def test_random_notes_no_repeats():
    result = random_notes(["C", "D", "E", "F"], count=4, allow_repeats=False, seed=5)
    assert sorted(result["notes"]) == ["C", "D", "E", "F"]
    with pytest.raises(ValueError, match="distinct"):
        random_notes(["C", "D"], count=3, allow_repeats=False)


def test_random_notes_validation():
    with pytest.raises(ValueError):
        random_notes(["C"], count=0)
    with pytest.raises(ValueError):
        random_notes([], count=1)


def test_random_rhythm_shape():
    result = random_rhythm(length=8, seed=42)
    assert len(result["pattern"]) == 8
    assert set(result["pattern"]) <= {"O", "o", "."}
    assert result["note_count"] == result["strong_count"] + result["weak_count"]
    assert result["note_count"] + result["rest_count"] == 8
    assert result["seed"] == 42


def test_random_rhythm_reproducible():
    assert random_rhythm(16, seed=9)["pattern"] == random_rhythm(16, seed=9)["pattern"]
    generated = random_rhythm(16)
    assert random_rhythm(16, seed=generated["seed"])["pattern"] == generated["pattern"]


def test_random_rhythm_density_extremes():
    assert random_rhythm(32, density=0.0, seed=1)["pattern"] == "." * 32
    full = random_rhythm(32, density=1.0, accent_probability=1.0, seed=1)
    assert full["pattern"] == "O" * 32
    soft = random_rhythm(32, density=1.0, accent_probability=0.0, seed=1)
    assert soft["pattern"] == "o" * 32


def test_random_rhythm_validation():
    with pytest.raises(ValueError):
        random_rhythm(0)
    with pytest.raises(ValueError):
        random_rhythm(8, density=1.5)
    with pytest.raises(ValueError):
        random_rhythm(8, accent_probability=-0.1)


def test_parse_rhythm():
    assert parse_rhythm("O.o. O..o") == "O.o.O..o"
    with pytest.raises(ValueError, match="Invalid rhythm"):
        parse_rhythm("O.x.")
    with pytest.raises(ValueError):
        parse_rhythm("   ")
