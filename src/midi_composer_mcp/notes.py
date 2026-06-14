"""Note parsing, spelling and MIDI-number utilities.

Notes travel between tools as plain strings: ``"C"``, ``"F#"``, ``"Bb"``,
optionally with an octave suffix (``"C4"``, ``"Eb3"``). A note without an
octave is an abstract pitch class; a note with an octave is a concrete pitch
(middle C = ``C4`` = MIDI 60).

Internally a note is a (letter, accidental, octave) triple so that proper
spelling (``Bb`` vs ``A#``, ``Bbb`` in a diminished 7th, ...) survives all the
way to the output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

LETTERS = "CDEFGAB"
LETTER_PCS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

SHARP_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
FLAT_NAMES = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")

# Natural-letter roots whose conventional spelling uses flats (only F among
# the naturals). Roots written with an accidental keep their own preference.
_FLAT_NATURAL_PCS = {5}

_NOTE_RE = re.compile(r"^([A-Ga-g])([#♯b♭]*)(-?\d+)?$")


@dataclass(frozen=True)
class Note:
    """A spelled note: letter A-G, accidental offset, optional octave."""

    letter: str
    accidental: int  # negative = flats, positive = sharps
    octave: int | None = None

    @property
    def pitch_class(self) -> int:
        return (LETTER_PCS[self.letter] + self.accidental) % 12

    @property
    def midi(self) -> int | None:
        """MIDI number (C4 = 60), or None for an abstract pitch class.

        The octave digit follows the letter, so Cb4 is MIDI 59 and B#3 is
        MIDI 60, matching standard music notation.
        """
        if self.octave is None:
            return None
        return LETTER_PCS[self.letter] + self.accidental + (self.octave + 1) * 12

    @property
    def name(self) -> str:
        acc = "#" * self.accidental if self.accidental >= 0 else "b" * -self.accidental
        base = f"{self.letter}{acc}"
        return base if self.octave is None else f"{base}{self.octave}"

    @property
    def pitch_class_name(self) -> str:
        """Name without the octave suffix."""
        acc = "#" * self.accidental if self.accidental >= 0 else "b" * -self.accidental
        return f"{self.letter}{acc}"

    def without_octave(self) -> "Note":
        return Note(self.letter, self.accidental, None)


def parse_note(text: str) -> Note:
    """Parse ``"C"``, ``"f#"``, ``"Bb3"``, ``"Ebb"``, ``"A♭5"`` ... into a Note."""
    if not isinstance(text, str):
        raise ValueError(f"Not a note name: {text!r} (expected a string like 'C', 'F#' or 'Bb3')")
    m = _NOTE_RE.match(text.strip())
    if not m:
        raise ValueError(
            f"Invalid note name: {text!r}. Use a letter A-G, optional accidentals"
            f" (# or b) and an optional octave, e.g. 'C', 'F#', 'Bb3', 'C5'."
        )
    letter, accidentals, octave = m.groups()
    if any(c in "#♯" for c in accidentals) and any(c in "b♭" for c in accidentals):
        raise ValueError(f"Invalid note name: {text!r} (mixed sharps and flats)")
    accidental = sum(1 if ch in "#♯" else -1 for ch in accidentals)
    if not -2 <= accidental <= 2:
        raise ValueError(f"Invalid note name: {text!r} (at most a double sharp/flat)")
    return Note(letter.upper(), accidental, int(octave) if octave is not None else None)


def parse_notes(notes: str | list) -> list[Note]:
    """Parse a note list given as a list of strings or one separated string.

    Accepts ``["C", "E", "G"]``, ``"c e g"`` and ``"C, E, G"`` alike, so the
    output of any tool can be fed back into any other tool.
    """
    if isinstance(notes, str):
        tokens = [t for t in re.split(r"[,\s]+", notes.strip()) if t]
    elif isinstance(notes, (list, tuple)):
        tokens = []
        for item in notes:
            if not isinstance(item, str):
                raise ValueError(f"Not a note name: {item!r} (expected a string like 'C' or 'Bb3')")
            tokens.extend(t for t in re.split(r"[,\s]+", item.strip()) if t)
    else:
        raise ValueError(f"Expected a list of note names or a string, got {type(notes).__name__}")
    if not tokens:
        raise ValueError("No notes given")
    return [parse_note(t) for t in tokens]


def prefers_flats(root: Note) -> bool:
    if root.accidental < 0:
        return True
    if root.accidental > 0:
        return False
    return root.pitch_class in _FLAT_NATURAL_PCS


def spell_pitch_class(pc: int, prefer_flats: bool = False) -> Note:
    """Spell a bare pitch class (0-11) using the sharp or flat name table."""
    name = (FLAT_NAMES if prefer_flats else SHARP_NAMES)[pc % 12]
    return parse_note(name)


def note_from_midi(midi: int, prefer_flats: bool = False) -> Note:
    """Concrete Note for a MIDI number (0-127), table-spelled."""
    if not 0 <= midi <= 127:
        raise ValueError(f"MIDI number out of range 0-127: {midi}")
    base = spell_pitch_class(midi % 12, prefer_flats)
    octave = (midi - LETTER_PCS[base.letter] - base.accidental) // 12 - 1
    return Note(base.letter, base.accidental, octave)


def transpose(root: Note, semitones: int, letter_steps: int | None = None) -> Note:
    """Return the note `semitones` above `root`.

    When `letter_steps` is given the result is spelled on the staff letter
    that many letters above the root (proper interval spelling: a minor third
    above Bb is Db, not C#). Falls back to plain table spelling when no letter
    distance is known or the spelling would need more than a double
    sharp/flat. Octaves carry over when the root has one.
    """
    if letter_steps is not None:
        root_index = LETTERS.index(root.letter)
        octave_carry, letter_index = divmod(root_index + letter_steps, 7)
        letter = LETTERS[letter_index]
        target_offset = LETTER_PCS[root.letter] + root.accidental + semitones
        accidental = target_offset - (LETTER_PCS[letter] + 12 * octave_carry)
        if -2 <= accidental <= 2:
            octave = None if root.octave is None else root.octave + octave_carry
            return Note(letter, accidental, octave)
    flats = prefers_flats(root)
    if root.octave is None:
        return spell_pitch_class((root.pitch_class + semitones) % 12, flats)
    midi = root.midi + semitones
    if not 0 <= midi <= 127:
        raise ValueError(
            f"{root.name} transposed by {semitones} semitones falls outside the MIDI range 0-127"
        )
    return note_from_midi(midi, flats)


def spelling_for_pcs(parsed: list[Note]) -> dict[int, Note]:
    """Map pitch class -> the spelling the user used for it (first wins)."""
    spelling: dict[int, Note] = {}
    for n in parsed:
        spelling.setdefault(n.pitch_class, n.without_octave())
    return spelling
