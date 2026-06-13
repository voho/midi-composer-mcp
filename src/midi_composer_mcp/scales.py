"""Scale database: scale types with intervals, descriptions, generation and matching."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .notes import Note, parse_notes, spell_pitch_class, spelling_for_pcs, transpose

MAJOR_DEGREES = (0, 2, 4, 5, 7, 9, 11)

# Default degree label for each chromatic interval (used for non-heptatonic
# scales; heptatonic scales get sequential degree numbers 1-7).
_CHROMATIC_LABELS = {
    0: "1", 1: "b2", 2: "2", 3: "b3", 4: "3", 5: "4",
    6: "b5", 7: "5", 8: "b6", 9: "6", 10: "b7", 11: "7",
}


@dataclass(frozen=True)
class ScaleType:
    name: str
    intervals: tuple[int, ...]
    aliases: tuple[str, ...] = ()
    matchable: bool = True  # chromatic matches everything, so it is excluded
    description: str = ""


SCALES: dict[str, ScaleType] = {
    s.name: s
    for s in [
        # --- major scale and its modes ----------------------------------
        ScaleType("major", (0, 2, 4, 5, 7, 9, 11), ("ionian", "maj"),
                  description="The standard major scale (do-re-mi); bright, stable, and the bedrock of Western tonal music."),
        ScaleType("dorian", (0, 2, 3, 5, 7, 9, 10), (),
                  description="A minor mode with a bright raised 6th; the hopeful, jazzy minor of folk and modal jazz."),
        ScaleType("phrygian", (0, 1, 3, 5, 7, 8, 10), (),
                  description="A minor mode with a flat 2nd; dark and tense, with a Spanish/flamenco undercurrent."),
        ScaleType("lydian", (0, 2, 4, 6, 7, 9, 11), (),
                  description="A major mode with a raised 4th; dreamy, floating and luminous, a film-score favorite."),
        ScaleType("mixolydian", (0, 2, 4, 5, 7, 9, 10), ("dominant scale",),
                  description="A major mode with a flat 7th; the dominant-7th sound of blues, rock, funk and Celtic music."),
        ScaleType("natural minor", (0, 2, 3, 5, 7, 8, 10), ("minor", "min", "aeolian"),
                  description="The standard minor scale and relative minor of the major scale; wistful and melancholic."),
        ScaleType("locrian", (0, 1, 3, 5, 6, 8, 10), (),
                  description="A diminished mode with flat 2nd and flat 5th; unstable and rarely a tonic, home of the m7b5 chord."),
        # --- minor variants ---------------------------------------------
        ScaleType("harmonic minor", (0, 2, 3, 5, 7, 8, 11), (),
                  description="Natural minor with a raised 7th, adding a strong leading tone and a dramatic augmented-2nd step."),
        ScaleType("melodic minor", (0, 2, 3, 5, 7, 9, 11), ("jazz minor", "melodic minor ascending"),
                  description="Natural minor with raised 6th and 7th; smooths melodies to the tonic and is the parent scale of much jazz."),
        # --- modes of melodic minor -------------------------------------
        ScaleType("dorian b2", (0, 1, 3, 5, 7, 9, 10), ("phrygian #6", "assyrian"),
                  description="The 2nd mode of melodic minor; dorian brightness over a phrygian flat-2nd, restless and exotic."),
        ScaleType("lydian augmented", (0, 2, 4, 6, 8, 9, 11), (),
                  description="The 3rd mode of melodic minor; a lydian scale with a raised 5th, shimmering and unresolved."),
        ScaleType("lydian dominant", (0, 2, 4, 6, 7, 9, 10), ("lydian b7", "mixolydian #4", "overtone", "acoustic"),
                  description="The 4th mode of melodic minor: a dominant scale with a raised 4th, matching the natural overtone series."),
        ScaleType("mixolydian b6", (0, 2, 4, 5, 7, 8, 10), ("melodic major", "hindu"),
                  description="Mixolydian with a flat 6th; a dominant scale that leans melancholy, good moving to minor."),
        ScaleType("locrian #2", (0, 2, 3, 5, 6, 8, 10), ("half-diminished scale", "aeolian b5"),
                  description="The 6th mode of melodic minor; locrian with a natural 2nd, the smoothest scale for m7b5 chords."),
        ScaleType("altered", (0, 1, 3, 4, 6, 8, 10), ("super locrian", "altered dominant", "diminished whole tone"),
                  description="The 7th mode of melodic minor; packed with every altered tension (b9 #9 #11 b13) for maximally tense dominants."),
        # --- harmonic major and harmonic minor relatives ----------------
        ScaleType("harmonic major", (0, 2, 4, 5, 7, 8, 11), (),
                  description="Major with a flat 6th; blends major brightness with a touch of minor shadow and an augmented-2nd step."),
        ScaleType("phrygian dominant", (0, 1, 4, 5, 7, 8, 10), ("spanish phrygian", "freygish", "ahava raba"),
                  description="The 5th mode of harmonic minor; a major 3rd over a phrygian b2, the sound of flamenco and Klezmer."),
        ScaleType("double harmonic", (0, 1, 4, 5, 7, 8, 11), ("byzantine", "arabic", "gypsy major"),
                  description="Two augmented 2nds give this scale a vivid Middle-Eastern/Byzantine character."),
        ScaleType("hungarian minor", (0, 2, 3, 6, 7, 8, 11), ("gypsy minor",),
                  description="Harmonic minor with a raised 4th; the gypsy-minor scale with two dramatic augmented 2nds."),
        ScaleType("hungarian major", (0, 3, 4, 6, 7, 9, 10), (),
                  description="A bright, fiery scale opening on a raised 2nd and carrying a flat 7th; rooted in Hungarian folk music."),
        ScaleType("neapolitan minor", (0, 1, 3, 5, 7, 8, 11), (),
                  description="Harmonic minor with a flat 2nd; somber and exotic, resolving with a strong leading tone."),
        ScaleType("neapolitan major", (0, 1, 3, 5, 7, 9, 11), (),
                  description="Melodic minor with a flat 2nd; smooth and slightly unusual, major on top of a phrygian start."),
        ScaleType("persian", (0, 1, 4, 5, 6, 8, 11), (),
                  description="A flat-2nd, flat-5th scale with two augmented 2nds; intensely ornamental and Middle-Eastern."),
        ScaleType("ukrainian dorian", (0, 2, 3, 6, 7, 9, 10), ("romanian minor", "altered dorian"),
                  description="Dorian with a raised 4th; plaintive and folk-like across Eastern European and Klezmer music."),
        ScaleType("enigmatic", (0, 1, 4, 6, 8, 10, 11), (),
                  description="Verdi's invented scale mixing whole tones with leading tones; strange, bright and unresolved."),
        # --- pentatonics and blues --------------------------------------
        ScaleType("major pentatonic", (0, 2, 4, 7, 9), ("pentatonic", "pentatonic major"),
                  description="A five-note major scale with no semitones; open, singable and consonant in any combination."),
        ScaleType("minor pentatonic", (0, 3, 5, 7, 10), ("pentatonic minor",),
                  description="The five-note minor scale at the core of blues, rock and pop soloing."),
        ScaleType("blues", (0, 3, 5, 6, 7, 10), ("minor blues", "blues minor"),
                  description="Minor pentatonic plus a chromatic 'blue note' (b5); the gritty, vocal sound of the blues."),
        ScaleType("major blues", (0, 2, 3, 4, 7, 9), ("blues major",),
                  description="Major pentatonic with an added b3 blue note; bright yet bluesy, common in country and rock."),
        ScaleType("egyptian", (0, 2, 5, 7, 10), ("suspended pentatonic",),
                  description="A suspended-sounding mode of the major pentatonic built on stacked 4ths and 5ths; open and unresolved."),
        # --- symmetric scales -------------------------------------------
        ScaleType("whole tone", (0, 2, 4, 6, 8, 10), ("wholetone",),
                  description="Six notes a whole step apart; dreamlike and rootless with no leading tone, beloved by Debussy."),
        ScaleType("augmented", (0, 3, 4, 7, 8, 11), ("symmetric augmented",),
                  description="A six-note symmetric scale alternating minor 3rds and half steps; stark and built from two augmented triads."),
        ScaleType("diminished whole-half", (0, 2, 3, 5, 6, 8, 9, 11), ("diminished", "octatonic", "whole-half diminished"),
                  description="An eight-note symmetric scale alternating whole and half steps; played over diminished-7th chords."),
        ScaleType("diminished half-whole", (0, 1, 3, 4, 6, 7, 9, 10), ("dominant diminished", "half-whole diminished"),
                  description="The octatonic scale starting with a half step; the go-to choice over dominant 7b9/#9 chords."),
        ScaleType("prometheus", (0, 2, 4, 6, 9, 10), ("mystic",),
                  description="Scriabin's six-note 'mystic' scale built from the mystic chord; hovering, ambiguous and tense."),
        # --- bebop ------------------------------------------------------
        ScaleType("bebop dominant", (0, 2, 4, 5, 7, 9, 10, 11), ("bebop",),
                  description="Mixolydian plus a passing major 7th; the extra note keeps chord tones on the beat in fast bebop lines."),
        ScaleType("bebop major", (0, 2, 4, 5, 7, 8, 9, 11), (),
                  description="The major scale with a passing #5; aligns chord tones to strong beats over major harmony."),
        ScaleType("spanish 8-tone", (0, 1, 3, 4, 5, 6, 8, 10), ("spanish gypsy", "jewish 8-tone"),
                  description="An eight-note flamenco scale extending phrygian dominant with chromatic passing tones for both 3rds."),
        # --- Japanese and world pentatonics -----------------------------
        ScaleType("hirajoshi", (0, 2, 3, 7, 8), (),
                  description="A Japanese pentatonic with two semitone steps; sparse, dark and contemplative on koto and shakuhachi."),
        ScaleType("in sen", (0, 1, 5, 7, 10), ("in", "insen"),
                  description="A dark Japanese pentatonic built on the flat 2nd; haunting and traditional."),
        ScaleType("iwato", (0, 1, 5, 6, 10), (),
                  description="A somber Japanese pentatonic with two tritone-flavored tensions; tense and shadowy."),
        ScaleType("kumoi", (0, 2, 3, 7, 9), ("kumoijoshi",),
                  description="A Japanese pentatonic with a bright opening and minor color; delicate and evocative."),
        ScaleType("yo", (0, 2, 5, 7, 9), (),
                  description="A bright, semitone-free Japanese pentatonic used in folk songs and Buddhist chant."),
        ScaleType("balinese pelog", (0, 1, 3, 7, 8), ("pelog",),
                  description="A five-note scale approximating Indonesian gamelan pelog tuning in 12-tone equal temperament; exotic and uneven."),
        # --- complete chromatic -----------------------------------------
        ScaleType("chromatic", tuple(range(12)), (), False,
                  description="All twelve semitones; not a tonal scale but the complete palette for passing tones and atonal writing."),
    ]
}


def _normalize(name: str) -> str:
    return re.sub(r"[\s_\-]+", "", name.strip().lower())


_SCALE_LOOKUP: dict[str, ScaleType] = {}
for _scale in SCALES.values():
    for _key in (_scale.name, *_scale.aliases):
        _SCALE_LOOKUP[_normalize(_key)] = _scale


def resolve_scale_type(scale_type: str) -> ScaleType:
    if not isinstance(scale_type, str):
        raise ValueError(f"Scale type must be a string, got {type(scale_type).__name__}")
    found = _SCALE_LOOKUP.get(_normalize(scale_type))
    if found is None:
        raise ValueError(
            f"Unknown scale type: {scale_type!r}. Known types: "
            + ", ".join(sorted(SCALES))
        )
    return found


def degree_labels(intervals: tuple[int, ...]) -> list[str]:
    """Degree labels such as ['1', '2', 'b3', ...] for a scale's intervals."""
    if len(intervals) == 7:
        labels = []
        for i, semis in enumerate(intervals):
            diff = semis - MAJOR_DEGREES[i]
            prefix = "#" * diff if diff > 0 else "b" * -diff
            labels.append(f"{prefix}{i + 1}")
        return labels
    return [_CHROMATIC_LABELS.get(s % 12, str(s)) for s in intervals]


def _label_digit(label: str) -> int:
    return int(label.lstrip("#b"))


def scale_notes(scale: ScaleType, root: Note) -> list[Note]:
    """Spelled notes of `scale` from `root`, including the octave root on top.

    Spelling follows the degree labels (a b3 lands on the third letter), so
    F major yields Bb rather than A#. If the root carries an octave, every
    note does too.
    """
    labels = degree_labels(scale.intervals)
    notes = [
        transpose(root, semis, _label_digit(label) - 1)
        for semis, label in zip(scale.intervals, labels)
    ]
    top_octave = None if root.octave is None else root.octave + 1
    if top_octave is not None and root.midi + 12 > 127:
        raise ValueError(f"Scale from {root.name} exceeds the MIDI range 0-127")
    notes.append(Note(root.letter, root.accidental, top_octave))
    return notes


def scale_info(scale_type: str, root: str | None = None) -> dict:
    """Describe a scale type; with a root, also generate its notes."""
    scale = resolve_scale_type(scale_type)
    result: dict = {
        "scale_type": scale.name,
        "description": scale.description,
        "aliases": list(scale.aliases),
        "intervals": list(scale.intervals),
        "degrees": degree_labels(scale.intervals),
        "note_count": len(scale.intervals),
    }
    if root is not None:
        root_note = parse_notes(root)[0]
        notes = scale_notes(scale, root_note)
        result["root"] = root_note.name
        result["name"] = f"{root_note.pitch_class_name} {scale.name}"
        result["notes"] = [n.name for n in notes]
        if root_note.octave is not None:
            result["midi"] = [n.midi for n in notes]
    return result


def list_scales() -> dict:
    return {
        "count": len(SCALES),
        "scales": [
            {
                "scale_type": s.name,
                "description": s.description,
                "aliases": list(s.aliases),
                "intervals": list(s.intervals),
                "degrees": degree_labels(s.intervals),
                "note_count": len(s.intervals),
            }
            for s in SCALES.values()
        ],
    }


def match_scales(notes: str | list, exact_only: bool = False, limit: int = 20) -> dict:
    """Find scales containing all the given notes (octaves are ignored).

    A match is "exact" when the input covers every note of the scale.
    Results are sorted: exact matches first, then smaller (tighter) scales,
    then scales rooted on the first input note.
    """
    parsed = parse_notes(notes)
    pcs = {n.pitch_class for n in parsed}
    spelling = spelling_for_pcs(parsed)
    first_pc = parsed[0].pitch_class

    matches = []
    for scale in SCALES.values():
        if not scale.matchable:
            continue
        for root_pc in range(12):
            scale_pcs = {(root_pc + i) % 12 for i in scale.intervals}
            if not pcs <= scale_pcs:
                continue
            exact = pcs == scale_pcs
            if exact_only and not exact:
                continue
            root = spelling.get(root_pc) or spell_pitch_class(root_pc)
            generated = scale_notes(scale, root)
            matches.append(
                {
                    "match": "exact" if exact else "contains",
                    "root": root.name,
                    "scale_type": scale.name,
                    "name": f"{root.name} {scale.name}",
                    "notes": [n.name for n in generated],
                    "added_notes": sorted(
                        {n.name for n in generated[:-1] if n.pitch_class not in pcs}
                    ),
                    "_sort": (
                        not exact,
                        len(scale.intervals),
                        root.pitch_class != first_pc,
                        scale.name,
                        root.name,
                    ),
                }
            )

    matches.sort(key=lambda m: m.pop("_sort"))
    return {
        "input_notes": [n.name for n in parsed],
        "match_count": len(matches),
        "matches": matches[: max(1, limit)],
    }
