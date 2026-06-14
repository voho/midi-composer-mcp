import pytest

from midi_composer_mcp.melody import (
    arpeggiate_notes,
    melodic_sequence,
    melodic_walk,
    motif_grammar,
    notes_from_degrees,
    snap_to_scale,
    tintinnabuli_voice,
    transpose_notes,
)


def test_notes_from_degrees():
    assert notes_from_degrees("C", "major", [1, 2, 3, 5, 8])["notes"] == ["C", "D", "E", "G", "C"]
    assert notes_from_degrees("C", "major", "1 3 5")["notes"] == ["C", "E", "G"]
    # octave wrap up and down
    assert notes_from_degrees("C", "major", [1, 8, 9])["notes"] == ["C", "C", "D"]
    assert notes_from_degrees("C", "major", [-7, 1])["notes"] == ["B", "C"]
    # transposable contour: same degrees, different key/scale
    assert notes_from_degrees("A", "minor pentatonic", [1, 2, 3])["notes"] == ["A", "C", "D"]
    with pytest.raises(ValueError):
        notes_from_degrees("C", "major", [0])


def test_arpeggiate_styles():
    assert arpeggiate_notes(["C4", "E4", "G4"], "up")["notes"] == ["C4", "E4", "G4"]
    assert arpeggiate_notes(["C4", "E4", "G4"], "down")["notes"] == ["G4", "E4", "C4"]
    assert arpeggiate_notes(["C4", "E4", "G4", "B4"], "updown")["notes"] == ["C4", "E4", "G4", "B4", "G4", "E4"]
    assert arpeggiate_notes(["C4", "E4", "G4"], "converge")["notes"] == ["C4", "G4", "E4"]
    two = arpeggiate_notes(["C4", "E4", "G4"], "up", octaves=2)["notes"]
    assert two == ["C4", "E4", "G4", "C5", "E5", "G5"]
    r = arpeggiate_notes(["C4", "E4", "G4"], "random", seed=5)
    assert sorted(r["notes"]) == ["C4", "E4", "G4"] and r["seed"] == 5
    with pytest.raises(ValueError):
        arpeggiate_notes(["C", "E", "G"], "up", octaves=2)  # needs octaves


def test_melodic_walk_is_stepwise_and_reproducible():
    ladder = notes_from_degrees("C", "major", list(range(1, 15)))["notes"]
    a = melodic_walk(ladder, length=12, seed=11, max_step=2)
    b = melodic_walk(ladder, length=12, seed=11, max_step=2)
    assert a["notes"] == b["notes"] and a["seed"] == 11
    assert len(a["notes"]) == 12
    # every note is from the ladder (in scale)
    assert set(a["notes"]) <= set(ladder)


def test_transpose_notes():
    assert transpose_notes(["C4", "E4", "G4"], 5)["notes"] == ["F4", "A4", "C5"]
    assert transpose_notes(["C", "E", "G"], 2)["notes"] == ["D", "F#", "A"]


def test_snap_to_scale_guarantees_diatonic():
    assert snap_to_scale(["C5", "C#5", "F#5"], "C", "major")["notes"] == ["C5", "C5", "F5"]
    # octave-less snapping by pitch class
    assert snap_to_scale(["Db", "Gb"], "C", "major")["notes"] == ["C", "F"]  # ties snap down
    # already diatonic notes unchanged in count
    res = snap_to_scale(["C5", "D5", "E5"], "C", "major")
    assert res["changed"] == 0 and res["notes"] == ["C5", "D5", "E5"]
    # output is fully in the scale
    scale_pcs = {0, 2, 4, 5, 7, 9, 11}
    from midi_composer_mcp.notes import parse_note
    out = snap_to_scale(["C5", "C#5", "Eb5", "F#5", "Ab5", "Bb5"], "C", "major")["notes"]
    assert all(parse_note(n).pitch_class in scale_pcs for n in out)


def test_motif_grammar_notes():
    g = motif_grammar("ABAC", {"A": "C5 D5 E5 G5", "B": {"vary": "A", "transpose": 2},
                               "C": {"vary": "A", "retrograde": True}}, kind="notes")
    assert g["motifs"]["B"] == ["D5", "E5", "F#5", "A5"]
    assert g["motifs"]["C"] == ["G5", "E5", "D5", "C5"]
    assert g["notes"][:4] == ["C5", "D5", "E5", "G5"]
    assert len(g["notes"]) == 16


def test_motif_grammar_degrees_invert_no_zero():
    g = motif_grammar("AB", {"A": [1, 2, 3, 5], "B": {"vary": "A", "invert": True}}, kind="degrees")
    # inversion mirrors around the first degree on the 0-less degree line
    assert 0 not in g["degrees"]
    assert g["motifs"]["B"][0] == 1
    # transposing degrees stays valid (no 0)
    t = motif_grammar("A", {"A": {"vary": "Z", "transpose": 1}, "Z": [1, 7]}, kind="degrees")
    assert 0 not in t["degrees"]


def test_motif_grammar_rhythm():
    g = motif_grammar("ABAB", {"A": "O.o.", "B": {"vary": "A", "rotate": 1}}, kind="rhythm")
    assert isinstance(g["pattern"], str)
    assert set(g["pattern"]) <= {"O", "o", "."}
    assert g["pattern"][:4] == "O.o."
    # AABB literal concatenation
    lit = motif_grammar("AB", {"A": "O...", "B": "..O."}, kind="rhythm")
    assert lit["pattern"] == "O.....O."


def test_motif_grammar_errors():
    with pytest.raises(ValueError, match="not defined"):
        motif_grammar("AB", {"A": "C5"}, kind="notes")
    with pytest.raises(ValueError, match="circular"):
        motif_grammar("A", {"A": {"vary": "B"}, "B": {"vary": "A"}}, kind="notes")


def test_melodic_sequence_descends_in_key():
    seq = melodic_sequence(["C5", "E5", "D5"], "C", "major", step=-1, count=3)
    assert seq["notes"][:3] == ["C5", "E5", "D5"]
    assert len(seq["notes"]) == 9
    from midi_composer_mcp.notes import parse_note
    assert all(parse_note(n).pitch_class in {0, 2, 4, 5, 7, 9, 11} for n in seq["notes"])


def test_tintinnabuli_picks_nearest_triad_note():
    # T-voice inferior, rank 1: nearest A-minor triad note below each M note
    t = tintinnabuli_voice(["A4", "B4", "C5", "D5", "E5"], "Am", position="inferior", rank=1)
    assert t["t_voice"] == ["E4", "A4", "A4", "C5", "C5"]
    # superior is strictly above and a triad member
    t2 = tintinnabuli_voice(["A4", "C5"], "Am", position="superior", rank=1)
    assert t2["t_voice"] == ["C5", "E5"]
    # every T-voice note belongs to the triad
    from midi_composer_mcp.notes import parse_note
    triad_pcs = {parse_note(n).pitch_class for n in ["A", "C", "E"]}
    assert all(parse_note(n).pitch_class in triad_pcs for n in t["t_voice"])


def test_tintinnabuli_rank_two():
    t1 = tintinnabuli_voice(["A4"], "Am", position="superior", rank=1)["t_voice"]
    t2 = tintinnabuli_voice(["A4"], "Am", position="superior", rank=2)["t_voice"]
    assert t1 == ["C5"] and t2 == ["E5"]  # 1st and 2nd nearest above
