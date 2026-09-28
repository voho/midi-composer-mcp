"""Build "The Rules of Harmony" — a musician's guide generated from the rule tables.

    python docs/book/build_book.py [output.pdf]     # needs: pip install -e ".[book]"

Every catalogue entry, table and worked example in the book is produced by the
server's own deterministic tools (scales.py, chords.py, diatonic.py, harmony.py,
counterpoint.py, ...), so the book always matches what the tools do. The prose
around them is written for musicians: approachable for a beginner, precise
enough for a professional.
"""

from __future__ import annotations

import math
import os
import re
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, CondPageBreak, Flowable, Frame, KeepTogether, NextPageTemplate, PageBreak,
    PageTemplate, Paragraph, Spacer, Table, TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

from midi_composer_mcp.chords import CHORDS, chord_notes, parse_chord_symbol
from midi_composer_mcp.circle import _CIRCLE
from midi_composer_mcp.counterpoint import species_counterpoint
from midi_composer_mcp.diatonic import degrees_to_chords, diatonic_chords
from midi_composer_mcp.generate import GROOVES, euclidean_rhythm
from midi_composer_mcp.harmony import (
    analyze_progression, interval_between, negative_harmony, secondary_dominant,
    tritone_substitute, voice_leading,
)
from midi_composer_mcp.melody import motif_grammar, notes_from_degrees, tintinnabuli_voice
from midi_composer_mcp.notes import parse_note
from midi_composer_mcp.scales import SCALES, degree_labels, scale_notes
from midi_composer_mcp.structure import _DEFAULT_BARS

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = (7 * 72, 10 * 72)  # a 7 x 10 inch book page
MARGIN = 18 * mm
WIDTH = PAGE[0] - 2 * MARGIN

# ------------------------------------------------------------------ fonts

_FONT_DIR = "/System/Library/Fonts/Supplemental"


def _register_fonts() -> tuple[str, str, str, str]:
    """Georgia for the text and Arial Unicode for the ♭/♯ signs, else the built-in fonts."""
    try:
        pdfmetrics.registerFont(TTFont("Body", f"{_FONT_DIR}/Georgia.ttf"))
        pdfmetrics.registerFont(TTFont("Body-Bold", f"{_FONT_DIR}/Georgia Bold.ttf"))
        pdfmetrics.registerFont(TTFont("Body-Italic", f"{_FONT_DIR}/Georgia Italic.ttf"))
        pdfmetrics.registerFont(TTFont("Sym", f"{_FONT_DIR}/Arial Unicode.ttf"))
        pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic",
                                      boldItalic="Body-Bold")
        return "Body", "Body-Bold", "Body-Italic", "Sym"
    except Exception:  # pragma: no cover - fallback on systems without these fonts
        return "Times-Roman", "Times-Bold", "Times-Italic", "Helvetica"


BODY, BOLD, ITALIC, SYM = _register_fonts()
UNICODE = SYM == "Sym"
INK = colors.HexColor("#1f2430")
ACCENT = colors.HexColor("#8a3b12")
SOFT = colors.HexColor("#f4efe6")
RULE = colors.HexColor("#c9bfae")
KEY_HI = colors.HexColor("#d9822b")
KEY_ROOT = colors.HexColor("#8a3b12")


_ROMAN = r"(?:VII|VI|IV|V|III|II|I|vii|vi|iv|v|iii|ii|i)(?![a-z])"


def acc(text: str) -> str:
    """Accidentals as ♭/♯ in the symbol font, in paragraph markup (tags are left alone).

    Converts note names (Bb -> B♭), scale degrees (b3 -> ♭3), chord alterations
    (7#11 -> 7♯11), roman numerals (bVII -> ♭VII) and literal ♭ ♯ → ↔ signs.
    """
    if not UNICODE:
        return text

    def sym(signs: str) -> str:
        return f'<font name="{SYM}">{signs.replace("#", "♯").replace("b", "♭")}</font>'

    def convert(seg: str) -> str:
        seg = re.sub(r"[♭♯→↔]+", lambda m: f'<font name="{SYM}">{m.group(0)}</font>', seg)
        seg = re.sub(r"\b([A-G])([#b]+)(?![a-z])", lambda m: m.group(1) + sym(m.group(2)), seg)
        seg = re.sub(r"(?<![A-Za-z0-9#])([#b]{1,2})(?=\d)", lambda m: sym(m.group(1)), seg)
        seg = re.sub(r"(?<=\d)([#b])(?=\d)", lambda m: sym(m.group(1)), seg)
        seg = re.sub(rf"(?<![A-Za-z0-9])([#b]{{1,2}})(?={_ROMAN})", lambda m: sym(m.group(1)), seg)
        return seg

    parts = re.split(r"(<[^>]+>)", text)
    return "".join(p if p.startswith("<") else convert(p) for p in parts)


def title(name: str) -> str:
    """'dorian b2' -> 'Dorian b2' (degree tokens keep their lowercase flat)."""
    return " ".join(w if re.match(r"[#b]*\d", w) else w[:1].upper() + w[1:] for w in name.split())


def note(name: str) -> str:
    return acc(name)


def notes(names) -> str:
    return " ".join(note(n) for n in names)


def plain_acc(text: str) -> str:
    """♭/♯ for canvas-drawn labels (no markup)."""
    if not UNICODE:
        return text
    return re.sub(r"(?<=[A-G])([#b]+)", lambda m: m.group(1).replace("#", "♯").replace("b", "♭"), text)


# ------------------------------------------------------------------ styles

def S(name, **kw) -> ParagraphStyle:
    base = dict(fontName=BODY, fontSize=10, leading=14.2, textColor=INK, spaceAfter=6)
    base.update(kw)
    return ParagraphStyle(name, **base)


STYLES = {
    "body": S("body"),
    "lead": S("lead", fontSize=11.2, leading=16, spaceAfter=10, fontName=ITALIC),
    "h1": S("h1", fontName=BOLD, fontSize=21, leading=25, spaceBefore=0, spaceAfter=6, textColor=ACCENT),
    "kicker": S("kicker", fontName=BOLD, fontSize=8.5, leading=11, textColor=ACCENT, spaceAfter=2),
    "h2": S("h2", fontName=BOLD, fontSize=13.5, leading=17, spaceBefore=12, spaceAfter=5),
    "h3": S("h3", fontName=BOLD, fontSize=11, leading=14, spaceBefore=8, spaceAfter=3),
    "entry": S("entry", fontName=BOLD, fontSize=10.5, leading=13, spaceAfter=1),
    "small": S("small", fontSize=8.6, leading=11.6, spaceAfter=3),
    "cell": S("cell", fontSize=8.6, leading=10.8, spaceAfter=0),
    "cellb": S("cellb", fontName=BOLD, fontSize=8.6, leading=10.8, spaceAfter=0),
    "note": S("note", fontSize=9.2, leading=13, spaceAfter=0),
    "title": S("title", fontName=BOLD, fontSize=34, leading=40, alignment=TA_CENTER, textColor=ACCENT),
    "subtitle": S("subtitle", fontName=ITALIC, fontSize=14, leading=19, alignment=TA_CENTER),
    "center": S("center", alignment=TA_CENTER, fontSize=9.5),
    "toc0": S("toc0", fontName=BOLD, fontSize=10.5, leading=15, leftIndent=0),
    "toc1": S("toc1", fontSize=9.5, leading=13, leftIndent=14),
    "bullet": S("bullet", leftIndent=12, bulletIndent=2, spaceAfter=3),
}


def P(text, style="body") -> Paragraph:
    return Paragraph(acc(text), STYLES[style])


def bullets(items) -> list:
    return [Paragraph(acc(t), STYLES["bullet"], bulletText="•") for t in items]


class Box(Flowable):
    """A tinted side box ('Pro note', 'Try it') holding paragraphs."""

    def __init__(self, title: str, paragraphs: list[str]):
        super().__init__()
        self.title = title
        self.paras = [Paragraph(f'<font name="{BOLD}" color="#8a3b12">{title}.</font> ' + acc(paragraphs[0]),
                                STYLES["note"])] + [Paragraph(acc(p), STYLES["note"]) for p in paragraphs[1:]]

    def wrap(self, aw, ah):
        self.width = aw
        self.heights = [p.wrap(aw - 16, ah)[1] for p in self.paras]
        self.height = sum(self.heights) + 12 + 4 * (len(self.paras) - 1)
        return aw, self.height

    def draw(self):
        c = self.canv
        c.setFillColor(SOFT)
        c.setStrokeColor(SOFT)
        c.roundRect(0, 0, self.width, self.height, 4, fill=1, stroke=0)
        c.setFillColor(ACCENT)
        c.rect(0, 0, 2.5, self.height, fill=1, stroke=0)
        y = self.height - 6
        for p, h in zip(self.paras, self.heights):
            y -= h
            p.drawOn(c, 9, y)
            y -= 4


def box(title, *paras):
    return [Spacer(1, 4), Box(title, list(paras)), Spacer(1, 8)]


def table(rows, widths, header=True, font=8.6, zebra=True, align_left=True) -> Table:
    data = []
    for r, row in enumerate(rows):
        style = STYLES["cellb"] if header and r == 0 else STYLES["cell"]
        data.append([Paragraph(acc(str(c)), style) if not isinstance(c, Flowable) else c for c in row])
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    cmds = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LINEBELOW", (0, -1), (-1, -1), 0.6, RULE),
    ]
    if header:
        cmds += [("LINEBELOW", (0, 0), (-1, 0), 0.8, ACCENT), ("LINEABOVE", (0, 0), (-1, 0), 0.8, ACCENT)]
    if zebra:
        for r in range(1 if header else 0, len(rows)):
            if (r % 2 == 0) == header:
                cmds.append(("BACKGROUND", (0, r), (-1, r), SOFT))
    t.setStyle(TableStyle(cmds))
    return t


# ------------------------------------------------------------ diagrams

class Keyboard(Flowable):
    """Two octaves of keys from C, with the given MIDI-relative keys marked and labelled."""

    WHITE = [0, 2, 4, 5, 7, 9, 11]

    def __init__(self, marks: dict[int, str], root: int | None = 0, width: float = 150, octaves: int = 2):
        super().__init__()
        self.marks, self.root, self.w, self.octaves = marks, root, width, octaves
        self.n_white = 7 * octaves + 1
        self.kw = width / self.n_white
        self.h = self.kw * 3.6

    def wrap(self, aw, ah):
        return self.w, self.h + 20

    def draw(self):
        c = self.canv
        kw, h = self.kw, self.h
        base = 10
        whites = [o * 12 + s for o in range(self.octaves) for s in self.WHITE] + [12 * self.octaves]
        for i, semi in enumerate(whites):
            fill = KEY_ROOT if semi == self.root else KEY_HI if semi in self.marks else colors.white
            c.setFillColor(fill)
            c.setStrokeColor(colors.HexColor("#555555"))
            c.setLineWidth(0.5)
            c.rect(i * kw, base, kw, h, fill=1, stroke=1)
        for o in range(self.octaves):
            for pos, semi in ((0, 1), (1, 3), (3, 6), (4, 8), (5, 10)):
                s = o * 12 + semi
                x = (o * 7 + pos + 1) * kw - kw * 0.3
                fill = KEY_ROOT if s == self.root else KEY_HI if s in self.marks else INK
                c.setFillColor(fill)
                c.setStrokeColor(INK)
                c.rect(x, base + h * 0.38, kw * 0.6, h * 0.62, fill=1, stroke=1)
        c.setFont(SYM if UNICODE else BODY, 5.8)
        c.setFillColor(INK)
        for semi, label in self.marks.items():
            if semi in whites:  # white-key names under the keyboard
                c.drawCentredString((whites.index(semi) + 0.5) * kw, 2.5, plain_acc(label))
            else:               # black-key names above it, so neighbours never collide
                o, r = divmod(semi, 12)
                pos = {1: 0, 3: 1, 6: 3, 8: 4, 10: 5}[r]
                c.drawCentredString((o * 7 + pos + 1) * kw, base + h + 3, plain_acc(label))


def keyboard_for(names: list[str], width=150, octaves=2) -> Keyboard:
    """Mark concrete notes (from C4) on a two-octave keyboard, labelled with their spelling."""
    marks = {}
    for n in names:
        nn = parse_note(n if any(ch.isdigit() for ch in n) else n + "4")
        rel = nn.midi - 60
        while rel < 0:
            rel += 12
        while rel > 24:
            rel -= 12
        marks.setdefault(rel, nn.pitch_class_name)
    root_rel = parse_note(names[0] if any(ch.isdigit() for ch in names[0]) else names[0] + "4").midi - 60
    return Keyboard(marks, root=root_rel, width=width, octaves=octaves)


class CircleOfFifths(Flowable):
    def __init__(self, size=260):
        super().__init__()
        self.size = size

    def wrap(self, aw, ah):
        return self.size, self.size

    def draw(self):
        c = self.canv
        r = self.size / 2
        cx = cy = r
        c.setStrokeColor(RULE)
        c.setFillColor(SOFT)
        c.circle(cx, cy, r - 2, fill=1, stroke=1)
        c.setFillColor(colors.white)
        c.circle(cx, cy, r * 0.58, fill=1, stroke=1)
        c.setFillColor(SOFT)
        c.circle(cx, cy, r * 0.30, fill=1, stroke=1)
        for i, e in enumerate(_CIRCLE):
            ang = math.pi / 2 - i * math.pi / 6
            c.setStrokeColor(RULE)
            a2 = ang + math.pi / 12
            c.line(cx + r * 0.30 * math.cos(a2), cy + r * 0.30 * math.sin(a2),
                   cx + (r - 2) * math.cos(a2), cy + (r - 2) * math.sin(a2))
            major = e["major"] + (f"/{e['enharmonic']}" if e["enharmonic"] else "")
            c.setFillColor(ACCENT)
            c.setFont(SYM if UNICODE else BOLD, 10.5)
            c.drawCentredString(cx + r * 0.80 * math.cos(ang), cy + r * 0.80 * math.sin(ang) - 4, plain_acc(major))
            c.setFillColor(INK)
            c.setFont(SYM if UNICODE else BODY, 8.2)
            c.drawCentredString(cx + r * 0.46 * math.cos(ang), cy + r * 0.46 * math.sin(ang) - 3,
                                plain_acc(e["relative_minor"] + "m"))
            f = e["fifths"]
            sig = "" if f == 0 else (f"{f}♯" if f > 0 else f"{-f}♭") if UNICODE else (f"{f}#" if f > 0 else f"{-f}b")
            c.setFont(SYM if UNICODE else BODY, 6.5)
            c.setFillColor(colors.HexColor("#6b6b6b"))
            c.drawCentredString(cx + r * 0.66 * math.cos(ang), cy + r * 0.66 * math.sin(ang) - 2.5, sig)
        c.setFillColor(INK)
        c.setFont(BODY, 7.5)
        c.drawCentredString(cx, cy + 3, "major outside")
        c.drawCentredString(cx, cy - 7, "minor inside")


# ------------------------------------------------------------ document

class Book(BaseDocTemplate):
    def __init__(self, path):
        super().__init__(path, pagesize=PAGE, leftMargin=MARGIN, rightMargin=MARGIN,
                         topMargin=MARGIN + 4, bottomMargin=MARGIN + 6,
                         title="The Rules of Harmony", author="midi-composer-mcp",
                         subject="Scales, chords, harmony, counterpoint and rhythm, explained for musicians")
        frame = Frame(MARGIN, MARGIN + 6, WIDTH, PAGE[1] - 2 * MARGIN - 10, id="f", leftPadding=0,
                      rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates([
            PageTemplate(id="plain", frames=[frame]),
            PageTemplate(id="main", frames=[frame], onPageEnd=self._decorate),
        ])
        self.chapter = ""
        self._chapter_no = None

    def _decorate(self, c, doc):
        c.saveState()
        c.setFont(ITALIC, 7.8)
        c.setFillColor(colors.HexColor("#7a7266"))
        c.drawString(MARGIN, PAGE[1] - MARGIN + 2, "The Rules of Harmony")
        c.drawRightString(PAGE[0] - MARGIN, PAGE[1] - MARGIN + 2, self.chapter)
        c.setStrokeColor(RULE)
        c.setLineWidth(0.4)
        c.line(MARGIN, PAGE[1] - MARGIN - 1, PAGE[0] - MARGIN, PAGE[1] - MARGIN - 1)
        c.setFont(BODY, 8.5)
        c.drawCentredString(PAGE[0] / 2, MARGIN - 6, str(doc.page))
        c.restoreState()

    def afterFlowable(self, f):
        if isinstance(f, Paragraph) and f.style.name in ("h1", "h2"):
            text = re.sub(r"<[^>]+>", "", f.getPlainText())
            level = 0 if f.style.name == "h1" else 1
            if text == "Contents":
                return
            if level == 0:
                self.chapter = text
                if self._chapter_no:
                    text = f"{self._chapter_no}  {text}"
                    self._chapter_no = None
            key = f"sec{id(f)}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=level, closed=level > 0)
            self.notify("TOCEntry", (level, text, self.page, key))


class _ChapterNo(Flowable):
    """Zero-size marker: numbers the next chapter heading in the table of contents."""

    def __init__(self, number):
        super().__init__()
        self.number = number

    def wrap(self, aw, ah):
        return 0, 0

    def draw(self):
        self.canv._doctemplate._chapter_no = self.number


def chapter(number: int, heading: str, lead: str) -> list:
    return [NextPageTemplate("main"), PageBreak(), _ChapterNo(number), P(f"CHAPTER {number}", "kicker"),
            P(heading, "h1"), Spacer(1, 2), P(lead, "lead")]


# ------------------------------------------------------------ content

SCALE_GROUPS = [
    ("The major scale and its modes", ["major", "dorian", "phrygian", "lydian", "mixolydian", "natural minor", "locrian"],
     "Seven scales that share one set of notes, each starting on a different degree. Brightest to darkest: "
     "lydian, major (ionian), mixolydian, dorian, natural minor (aeolian), phrygian, locrian."),
    ("Minor variants", ["harmonic minor", "melodic minor"],
     "Natural minor has no leading tone (the note a semitone below the tonic). Raising the 7th gives harmonic minor; "
     "raising the 6th as well smooths the resulting augmented second and gives melodic minor."),
    ("Modes of melodic minor", ["dorian b2", "lydian augmented", "lydian dominant", "mixolydian b6", "locrian #2", "altered"],
     "The modern jazz palette: each mode fits a particular chord colour, from the lydian-dominant 7♯11 to the altered dominant."),
    ("Harmonic major, harmonic minor relatives and folk colours",
     ["harmonic major", "phrygian dominant", "double harmonic", "hungarian minor", "hungarian major", "neapolitan minor",
      "neapolitan major", "persian", "ukrainian dorian", "enigmatic"],
     "Scales with augmented seconds and other distinctive steps — the sound of flamenco, klezmer, Eastern Europe and the Middle East."),
    ("Pentatonics and blues", ["major pentatonic", "minor pentatonic", "blues", "major blues", "egyptian"],
     "Five- and six-note scales without the most tense intervals; almost any combination of their notes sounds good."),
    ("Symmetric scales", ["whole tone", "augmented", "diminished whole-half", "diminished half-whole", "prometheus"],
     "Scales built from a repeating pattern of steps. Because they repeat, they have fewer distinct transpositions and blur the sense of a home note."),
    ("Bebop and eight-note scales", ["bebop dominant", "bebop major", "spanish 8-tone"],
     "Seven-note scales with one extra passing note, so that in running eighth notes the chord tones fall on the beats."),
    ("Japanese and world pentatonics", ["hirajoshi", "in sen", "iwato", "kumoi", "yo", "balinese pelog"],
     "Five-note scales from Japanese and Indonesian traditions, here in 12-tone equal temperament."),
    ("The complete chromatic scale", ["chromatic"], "All twelve notes: not a key, but the full palette."),
]

CHORD_GROUPS = [
    ("Triads and their relatives", ["major", "minor", "diminished", "augmented", "power chord", "suspended 2", "suspended 4"]),
    ("Sixth and seventh chords", ["major 6", "minor 6", "dominant 7", "major 7", "minor 7", "minor major 7", "diminished 7",
                                  "half-diminished", "augmented 7", "augmented major 7", "dominant 7 flat 5",
                                  "dominant 7 sus 4", "dominant 7 sus 2"]),
    ("Added-tone chords", ["add 9", "minor add 9", "add 4", "six nine", "minor six nine"]),
    ("Ninth chords", ["dominant 9", "major 9", "minor 9", "dominant 7 flat 9", "dominant 7 sharp 9"]),
    ("Sharp-eleven (lydian) chords", ["dominant 7 sharp 11", "major 7 sharp 11"]),
    ("Eleventh and thirteenth chords", ["dominant 11", "minor 11", "dominant 13", "major 13", "minor 13"]),
]


def _check_coverage():
    grouped = [n for _, names, _ in SCALE_GROUPS for n in names]
    assert sorted(grouped) == sorted(SCALES), set(SCALES) ^ set(grouped)
    grouped = [n for _, names in CHORD_GROUPS for n in names]
    assert sorted(grouped) == sorted(CHORDS), set(CHORDS) ^ set(grouped)


def front_matter(story, toc):
    story += [Spacer(1, 150), P("The Rules of Harmony", "title"), Spacer(1, 8),
              P("A musician's field guide to scales, chords, harmony,<br/>counterpoint and rhythm", "subtitle"),
              Spacer(1, 40), P(f"{len(SCALES)} scales · {len(CHORDS)} chord types · the rules that connect them", "center"),
              Spacer(1, 190),
              P("Generated from the rule engine of <i>midi-composer-mcp</i>. Every table, example and catalogue entry "
                "in this book is produced by the same deterministic code that the composing tools run.", "center"),
              PageBreak()]
    story += [P("Contents", "h1"), Spacer(1, 6), toc, PageBreak()]
    story += [P("How to read this book", "h1"), P(
        "This book explains the building blocks of Western tonal music — notes, intervals, scales, chords, keys — and "
        "the rules composers use to connect them: harmony, voice leading, reharmonization, melody craft, counterpoint, "
        "rhythm and form. It starts from zero, but it does not stay there: each chapter ends where a working musician "
        "needs it to.", "lead")]
    story += [P("Some conventions used throughout:", "body")]
    story += bullets([
        "<b>Note names</b> are letters with accidentals: C, F#, Bb. A name without an octave number is a "
        "<i>pitch class</i> — every C on the piano. With a number it is one concrete key: C4 is middle C, C5 the C above it.",
        "<b>Keyboard diagrams</b> show one or two octaves from C. The home note (root) is dark, the other members are orange, "
        "and each marked key is labelled with its correct spelling.",
        "<b>Degrees</b> are written 1, 2, b3, #4… — the scale step, altered relative to the major scale. "
        "<b>Roman numerals</b> name chords by the degree they stand on: I, ii, V7, bVII.",
        "<b>Pro notes</b> (tinted boxes) add detail for experienced readers. Beginners can skip them without losing the thread.",
    ])
    story += box("Why spelling matters",
                 "Bb and A# are the same key on a piano, but not the same note in music. F major contains Bb, never A#, "
                 "because a scale uses each letter once. Correct spelling keeps intervals readable (a third always "
                 "spans three letters) and is why this book, like the tools behind it, writes Cb, E#, or a double flat "
                 "when the theory calls for one.")


def ch_notes(story):
    story += chapter(1, "Notes, octaves and spelling",
                     "Twelve pitches repeat in every octave. Seven letters name them, and accidentals fill the gaps.")
    story += [P("The piano keyboard makes the system visible: seven white keys named C D E F G A B, and five black keys "
                "between them. Moving from any key to its nearest neighbour is a <b>semitone</b> (half step); two "
                "semitones make a <b>whole tone</b>. Between E–F and B–C there is no black key, so those white keys "
                "are only a semitone apart."),
              keyboard_for(["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"], width=WIDTH * 0.8),
              P("A <b>sharp</b> (#) raises a note by a semitone and a <b>flat</b> (b) lowers it; a double sharp or double "
                "flat moves it two. So every black key has two names — C# is also Db — and even white keys can be "
                "respelled: E# sounds as F, Cb as B. These are <b>enharmonic</b> spellings: same sound, different meaning."),
              P("An <b>octave</b> is the distance to the next note with the same letter; its frequency is exactly double. "
                "Octaves are numbered from C upward, so B3 is directly below C4. That is why Cb4 is the same key as B3, "
                "and B#3 the same key as C4 — the number belongs to the letter, not to the key."),
              ]
    rows = [["Note", "MIDI number", "Frequency (Hz)"]]
    for n in ["A0", "C2", "C3", "C4", "A4", "C5", "C6", "C8"]:
        m = parse_note(n).midi
        rows.append([n + (" (middle C)" if n == "C4" else " (tuning A)" if n == "A4" else ""), m,
                     f"{440 * 2 ** ((m - 69) / 12):.2f}"])
    story += [P("Concrete pitches, MIDI and frequency", "h3"), table(rows, [WIDTH * 0.4, WIDTH * 0.25, WIDTH * 0.35])]
    story += box("Pro note",
                 "MIDI numbers run from 0 (C-1) to 127 (G9); middle C is 60 and each semitone adds one. In equal "
                 "temperament every semitone multiplies the frequency by the twelfth root of two (about 1.0595). "
                 "Other temperaments — just intonation, the gamelan tunings behind pelog — divide the octave "
                 "differently; this book stays in twelve equal steps.")


def ch_intervals(story):
    story += chapter(2, "Intervals",
                     "An interval is the distance between two notes, named by how many letters it spans and how many semitones it holds.")
    story += [P("Count the letters from the lower note to the upper one, including both: C up to E spans C-D-E, three "
                "letters, so it is a <b>third</b>. The number alone is not enough — C–E and C–Eb are both thirds — so a "
                "<b>quality</b> is added: <b>major</b> or <b>minor</b> for 2nds, 3rds, 6ths and 7ths; <b>perfect</b> for "
                "unisons, 4ths, 5ths and octaves. Stretch any of them by a semitone and it becomes <b>augmented</b>; "
                "shrink it and it becomes <b>diminished</b>.")]
    rows = [["Interval", "Short", "Semitones", "From C", "Character"]]
    char = {
        "P1": "the same note", "m2": "sharp rub, tense", "M2": "a step; mild", "m3": "dark, sad colour",
        "M3": "bright, stable", "P4": "open; wants to fall to the 3rd", "A4": "the tritone: restless", "d5": "the tritone, spelled as a 5th",
        "P5": "hollow, strong", "m6": "sweet, melancholy", "M6": "warm, open", "m7": "bluesy tension",
        "M7": "yearning, bright", "P8": "the octave: the same note higher",
    }
    for b in ["C", "Db", "D", "Eb", "E", "F", "F#", "Gb", "G", "Ab", "A", "Bb", "B", "C5"]:
        r = interval_between("C4", b if b[-1].isdigit() else b + "4")
        rows.append([r["name"], r["short"], r["semitones"], f"C → {b.rstrip('5')}", char.get(r["short"], "")])
    story += [table(rows, [WIDTH * 0.25, WIDTH * 0.09, WIDTH * 0.15, WIDTH * 0.14, WIDTH * 0.37])]
    story += [P("Same sound, different interval", "h3"),
              P("Because intervals are counted by letters, C–F# is an augmented 4th but C–Gb a diminished 5th, although "
                "both are six semitones. C up to B# is an augmented 7th (twelve semitones!), and B#3 up to C4 a "
                "diminished second of zero semitones. The spelling tells you how the note behaves: an augmented 4th "
                "wants to open outwards, a diminished 5th to close inwards."),
              P("Consonance and dissonance", "h3"),
              P("In two-voice writing the <b>consonances</b> are the perfect unison, octave and fifth (the "
                "<i>perfect consonances</i>) and the major and minor thirds and sixths (the <i>imperfect consonances</i>). "
                "Everything else — seconds, sevenths, the tritone, and, when two voices sound alone, the perfect "
                "fourth — is a <b>dissonance</b> that needs careful handling. Chapter 9 turns this into rules."),
              P("Compound intervals and inversion", "h3"),
              P("Intervals wider than an octave are <b>compound</b>: a 9th is an octave plus a 2nd, a 10th an octave "
                f"plus a 3rd (C4 → E5 is a {interval_between('C4', 'E5')['name']}). Turning an interval upside down "
                "(moving the lower note up an octave) <b>inverts</b> it: the numbers add to nine and the qualities swap "
                "— major↔minor, augmented↔diminished, perfect stays perfect. So a major 3rd inverts to a minor 6th.")]


def scale_entry(scale) -> KeepTogether:
    c_notes = [n.name for n in scale_notes(scale, parse_note("C4"))][:-1]
    labels = degree_labels(scale)
    header = title(scale.name) + (f'  <font name="{ITALIC}" size="8.5">also {", ".join(scale.aliases)}</font>' if scale.aliases else "")
    left = [P(header, "entry"),
            P(f'<font color="#8a3b12">Degrees</font>  {"  ".join(labels)}<br/>'
              f'<font color="#8a3b12">On C</font>  {notes([n.rstrip("0123456789-") for n in c_notes])}', "small"),
            P(scale.description, "small")]
    kb = keyboard_for(c_notes, width=WIDTH * 0.36, octaves=1)
    t = Table([[left, kb]], colWidths=[WIDTH * 0.60, WIDTH * 0.40])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 4), ("TOPPADDING", (0, 0), (-1, -1), 3),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 5), ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE)]))
    return KeepTogether([t])


def ch_scales(story):
    story += chapter(3, "Scales",
                     f"A scale is a set of notes ordered from a home note. This chapter explains how scales are built, then catalogues all {len(SCALES)} of them.")
    story += [P("A <b>scale</b> picks some of the twelve pitch classes and orders them from a <b>tonic</b>. Its identity is "
                "its pattern of steps: the major scale is whole–whole–half–whole–whole–whole–half (W W H W W W H). "
                "Start that pattern on any note and you get that note's major scale; the letters each appear once, "
                "which is why the spelling comes out right (F major has Bb, D major has F# and C#)."),
              P("Scales are described by their <b>degrees</b> relative to the major scale: natural minor is "
                "1 2 b3 4 5 b6 b7, dorian 1 2 b3 4 5 6 b7. The degree list is the fastest way to compare scales — and to "
                "transpose them: to play any scale on any root, apply its degrees to that root's major scale."),
              P("<b>Modes</b> are scales that share their notes with a parent but start elsewhere. D dorian uses the "
                "white keys from D; so do E phrygian and G mixolydian. Same notes, different home — and a different "
                "mood, because the half steps fall in new places relative to the tonic.")]
    story += box("Try it",
                 "Play C major (all white keys from C), then A natural minor (all white keys from A). The notes are the "
                 "same; only the home note changed. That pair — a major key and the minor key three semitones below "
                 "— are <i>relative</i> keys.")
    for title, names, intro in SCALE_GROUPS:
        story += [CondPageBreak(120), P(title, "h2"), P(intro, "small"), Spacer(1, 2)]
        story += [scale_entry(SCALES[n]) for n in names]


def chord_entry(chord) -> KeepTogether:
    tones = [t.name for t in chord_notes(chord, parse_note("C4"))]
    symbol = f"C{chord.symbol}"
    header = f"{acc(title(chord.name))}  <font name='{BOLD}' color='#8a3b12'>{acc(symbol)}</font>"
    # symbol-like aliases are written after a root (Cmaj, C-7); word aliases ("power") are names
    words = [a for a in chord.aliases if " " in a or re.fullmatch(r"[a-z-]{5,}", a) and a not in ("minmaj7",)]
    symbols = [f"C{a}" for a in chord.aliases if a not in words]
    also = ", ".join(symbols) + (("; called " if symbols else "called ") + ", ".join(words) if words else "")
    left = [Paragraph(header, STYLES["entry"]),
            P(f'<font color="#8a3b12">Formula</font>  {"  ".join(l for l, _ in chord.degrees)}'
              f'<br/><font color="#8a3b12">On C</font>  {notes([t.rstrip("0123456789-") for t in tones])}'
              + (f'<br/><font color="#8a3b12">Also written</font>  {also}' if also else ""), "small"),
            P(chord.description, "small")]
    kb = keyboard_for(tones, width=WIDTH * 0.40)
    t = Table([[left, kb]], colWidths=[WIDTH * 0.57, WIDTH * 0.43])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 4), ("TOPPADDING", (0, 0), (-1, -1), 3),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 5), ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE)]))
    return KeepTogether([t])


def ch_chords(story):
    story += chapter(4, "Chords",
                     f"Three or more notes sounding together. Most chords are stacks of thirds; this chapter shows how they are built and named, then lists all {len(CHORDS)} types.")
    story += [P("Take every other note of a scale — 1, 3, 5 — and you have a <b>triad</b>. Its quality depends on the "
                "thirds it is made of: major third + minor third gives a <b>major</b> triad (C E G), minor + major a "
                "<b>minor</b> triad (C Eb G), two minor thirds a <b>diminished</b> triad, two major thirds an "
                "<b>augmented</b> one. Keep stacking thirds and you get sevenths (1 3 5 7), ninths, elevenths and thirteenths."),
              P("A <b>chord symbol</b> is the root plus a suffix: C (major), Cm, C7, Cmaj7, Cm7b5, C13. The suffix is "
                "case-sensitive in one place that matters: <b>m</b> is minor, <b>M</b> is major (CM7 = Cmaj7). A minus sign "
                "means minor in jazz charts (C-7 = Cm7), and in older charts a minus or plus before a number flattens or "
                "sharpens it (C7-9 = C7b9, C7+5 = C7#5)."),
              P("Inversions and slash chords", "h3"),
              P("A chord is in <b>root position</b> when its root is the lowest note. With the 3rd in the bass it is in "
                "<b>first inversion</b>, with the 5th <b>second inversion</b>, with the 7th <b>third inversion</b>. "
                "A <b>slash chord</b> names the bass explicitly: C/E is C major over E (first inversion); C/D puts a "
                "non-chord tone in the bass. The inversion is named by which chord member is in the bass — the 5th in "
                "the bass is always second inversion, whatever the chord.")]
    story += box("Pro note",
                 "Added-tone chords (add9, add4, 6) keep the triad and add a colour without the 7th; sus chords "
                 "<i>replace</i> the 3rd with a 2nd or 4th. Upper extensions are usually voiced selectively: an 11th "
                 "chord often drops its 3rd (which clashes with the 11th), a 13th often drops the 11th.")
    for title, names in CHORD_GROUPS:
        story += [CondPageBreak(120), P(title, "h2")]
        story += [chord_entry(CHORDS[n]) for n in names]


def diatonic_table(root, scale, sevenths):
    d = diatonic_chords(root, scale, sevenths)
    rows = [["Degree", "Numeral", "Chord", "Notes", "Function"]]
    for c in d["chords"]:
        rows.append([f'{c["degree"]} · {c.get("degree_name", "")}', c.get("roman", ""), c["symbol"],
                     notes(c["notes"]), c.get("harmonic_function", "")])
    return table(rows, [WIDTH * 0.26, WIDTH * 0.13, WIDTH * 0.14, WIDTH * 0.27, WIDTH * 0.2])


def ch_harmony(story):
    story += chapter(5, "Harmony in a key",
                     "Build a triad on every degree of a scale and you have the chords of that key — the raw material of every progression.")
    story += [P("Stacking thirds from each degree of C major, using only notes of the scale, gives seven chords. Their "
                "qualities follow from the scale's step pattern, and they are the same in every major key: "
                "I ii iii IV V vi vii°. Uppercase numerals are major chords, lowercase minor, ° diminished."),
              P("C major — triads", "h3"), diatonic_table("C", "major", False),
              P("C major — seventh chords", "h3"), diatonic_table("C", "major", True),
              P("Harmonic function", "h3"),
              P("Chords group into three families by what they do. <b>Tonic</b> chords (I, and its substitutes vi and iii) "
                "feel like home. <b>Subdominant</b> chords (IV, ii) move away. <b>Dominant</b> chords (V, vii°) create "
                "tension that wants to resolve home: their leading tone (B in C major) pulls up to the tonic and their "
                "7th (F) pulls down to the 3rd. Most progressions cycle tonic → subdominant → dominant → tonic."),
              P("Minor keys", "h3"),
              P("Natural minor lacks a leading tone, so its v chord is minor and weak. Composers borrow the raised 7th of "
                "harmonic minor to get a major V (E–G#–B in A minor) and the powerful V7. Numerals in this book are "
                "written relative to the major scale: in A natural minor the chord on the third degree is bIII."),
              P("A natural minor — triads", "h3"), diatonic_table("A", "natural minor", False),
              P("A harmonic minor — seventh chords", "h3"), diatonic_table("A", "harmonic minor", True)]
    e7 = analyze_progression(["Am", "E7", "Am"], "A", "natural minor")["chords"][1]
    story += [P("Is a chord in the key?", "h3"),
              P("A chord belongs to a key only if <b>every</b> one of its notes is in the scale — not just its root. "
                f"E7 in A natural minor stands on degree {e7['degree']}, but its G# is foreign, so it is a "
                f"<i>chromatic</i> (borrowed) chord: the analysis gives {acc(e7['roman'])} with non-scale note "
                f"{notes(e7['non_scale_notes'])}. In A harmonic minor the same E7 is fully diatonic.")]
    prog = analyze_progression(["C", "A7", "Dm", "G7", "C"], "C", "major")["chords"]
    rows = [["Chord", "Numeral", "In key?", "Why"]]
    for c in prog:
        rows.append([c["symbol"], c["roman"], "yes" if c["in_key"] else "no",
                     c.get("function", "") if c["in_key"] else "contains " + notes(c["non_scale_notes"])])
    story += [P("Analyzing C – A7 – Dm – G7 – C in C major:", "body"),
              table(rows, [WIDTH * 0.18, WIDTH * 0.18, WIDTH * 0.16, WIDTH * 0.48])]
    story += box("Pro note",
                 "Numerals follow spelling: in C, a Gb chord is bV and an F# chord #IV, even though they sound the same. "
                 "The A7 above is chromatic because of its C#; functionally it is the dominant of ii (V7/ii), which "
                 "chapter 7 builds on purpose.")
    story += [P("Common progressions", "h3")]
    rows = [["Name", "Numerals", "In C major"]]
    for name, degs in [("Pop axis", "I V vi IV"), ("Doo-wop", "I vi IV V"), ("Jazz ii–V–I", "ii V I"),
                       ("Circle descent", "vi ii V I"), ("Royal road", "IV V iii vi"), ("Plagal cadence", "IV I")]:
        rows.append([name, degs, " ".join(degrees_to_chords("C", "major", degs)["symbols"])])
    minor = degrees_to_chords("A", "natural minor", "i bVII bVI", False)["symbols"]
    rows.append(["Andalusian cadence", "i bVII bVI V", " ".join(minor) + " E (in A minor; V from harmonic minor)"])
    story += [table(rows, [WIDTH * 0.26, WIDTH * 0.26, WIDTH * 0.48])]


def ch_circle(story):
    story += chapter(6, "The circle of fifths",
                     "Arrange the twelve keys a fifth apart and they form a circle in which neighbours share all but one note.")
    story += [CircleOfFifths(size=WIDTH * 0.72), Spacer(1, 8),
              P("Moving clockwise raises the key by a perfect fifth and adds one sharp (or removes a flat); moving "
                "counter-clockwise lowers it by a fifth and adds a flat. The inner ring shows each major key's "
                "<b>relative minor</b>, which shares its key signature.")]
    rows = [["Major", "Signature", "Accidentals", "Relative minor"]]
    for e in _CIRCLE:
        f = e["fifths"]
        rows.append([e["major"] + (f" / {e['enharmonic']}" if e["enharmonic"] else ""),
                     "none" if f == 0 else f"{abs(f)} {'sharp' if f > 0 else 'flat'}{'s' if abs(f) > 1 else ''}",
                     " ".join(e["accidentals"]) or "—", e["relative_minor"] + " minor"])
    story += [table(rows, [WIDTH * 0.2, WIDTH * 0.18, WIDTH * 0.42, WIDTH * 0.2])]
    story += [P("Closely related keys", "h3"),
              P("The keys one step either side, plus the relative minors of all three, differ by at most one accidental. "
                "From C major they are A minor, G major, E minor, F major and D minor. They are the natural "
                "destinations for a modulation or a contrasting bridge: the change is audible but smooth."),
              P("Sharps are added in the order F C G D A E B, flats in the reverse order B E A D G C F. A key with more "
                "than seven accidentals is <i>theoretical</i> — D# major would need double sharps — and is written as "
                "its enharmonic instead (Eb major).")]


def ch_voice_leading(story):
    story += chapter(7, "Voice leading and reharmonization",
                     "How chords move matters as much as which chords you choose. Then: three classic ways to recolour a progression.")
    vl = voice_leading(["C", "Am", "F", "G", "C"])["voicings"]
    rows = [["Chord", "Voicing (low → high)", "Movement"]]
    prev = None
    for v in vl:
        total = None if prev is None else sum(abs(a - b) for a, b in zip(sorted(prev), sorted(v["midi"])))
        move = "—" if total is None else f"{total} semitone{'s' if total != 1 else ''}"
        rows.append([v["symbol"], notes(v["notes"]), move])
        prev = v["midi"]
    story += [P("Play every chord in root position and the whole hand jumps. Good <b>voice leading</b> treats each chord "
                "note as a singer: keep notes that two chords share (<b>common tones</b>) where they are, and move the "
                "others to the nearest available note. The best voicing is the one with the least total movement, "
                "each voice going to exactly one voice of the next chord."),
              table(rows, [WIDTH * 0.15, WIDTH * 0.5, WIDTH * 0.35]),
              P("A slash chord keeps its named bass at the bottom and voice-leads the rest above it.", "small")]
    story += [P("Secondary dominants", "h2"),
              P("Any major or minor chord of a key can be made to feel like a temporary tonic by preceding it with "
                "<i>its own</i> dominant — the dominant-seventh chord a perfect fifth above it. This is the "
                "<b>secondary dominant</b>, written V7/x."),
              ]
    rows = [["Target", "Its dominant", "Notes", "Label"]]
    for t, lab in [("Dm", "V7/ii"), ("Em", "V7/iii"), ("F", "V7/IV"), ("G", "V7/V"), ("Am", "V7/vi")]:
        r = secondary_dominant(t)
        rows.append([t, r["symbol"], notes(r["notes"]), lab])
    story += [table(rows, [WIDTH * 0.18, WIDTH * 0.22, WIDTH * 0.35, WIDTH * 0.25])]
    story += [P("Tritone substitution", "h2"),
              P("A dominant seventh's two guide tones — its 3rd and 7th — form a tritone. The dominant a tritone away "
                "contains the <i>same</i> tritone, spelled the other way round (G7 has B and F; Db7 has F and Cb). "
                "So Db7 can stand in for G7, and the bass slides down by semitone into C. Only dominants can be "
                "substituted this way; a plain major triad is read as the dominant seventh it implies.")]
    rows = [["Dominant", "Substitute", "Shared guide tones"]]
    for d in ["G7", "C7", "D7", "E7", "A7", "B7"]:
        r = tritone_substitute(d)
        root, ctype, _bass = parse_chord_symbol(d)
        orig = [t.name for t in chord_notes(ctype, root)]
        rows.append([d, r["symbol"], f"{note(orig[1])} and {note(orig[3])}"])
    story += [table(rows, [WIDTH * 0.25, WIDTH * 0.25, WIDTH * 0.5])]
    story += [P("Negative harmony", "h2"),
              P("Negative harmony (after Ernst Levy, popularized by Jacob Collier) mirrors every note around the axis "
                "halfway between the tonic and its fifth. In C the axis lies between E and Eb, so E↔Eb, G↔C, D↔F, "
                "and a major chord becomes a minor one: the tonic C major turns into C minor, the dominant G into F "
                "minor. Functions survive — the mirror of a dominant still pulls home — but the colour darkens. The "
                "rule is the same in every key: a note r semitones above the tonic maps to 7 − r semitones above it.")]
    mapping = negative_harmony(["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"], "C")["notes"]
    orig = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
    story += [table([["Note"] + orig, ["Mirror in C"] + mapping], [WIDTH * 0.16] + [WIDTH * 0.07] * 12, header=True, zebra=False)]
    rows = [["Chord in C", "Mirror", "Chord in D", "Mirror"]]
    for c, d in [("C E G", "D F# A"), ("F A C", "G B D"), ("G B D", "A C# E"), ("A C E", "B D F#")]:
        rows.append([notes(c.split()), notes(negative_harmony(c, "C")["notes"]),
                     notes(d.split()), notes(negative_harmony(d, "D")["notes"])])
    story += [Spacer(1, 6), table(rows, [WIDTH * 0.25] * 4)]


def ch_melody(story):
    story += chapter(8, "Melody craft",
                     "Melodies are built from scale steps, shaped by repetition and variation, and fitted to the harmony.")
    story += [P("Scale degrees and the number line", "h3"),
              P("Writing a melody as scale degrees — 1 2 3 5 8 — makes it portable: the same numbers give a melody in any "
                "key or mode. Degrees above 7 continue into the next octave (8 is the tonic an octave up, 9 the 2nd). "
                "Below the tonic the line continues without a zero: −1 is the step just below the tonic, −2 the step "
                "below that, −7 the tonic an octave down."),
              ]
    ex = notes_from_degrees("C4", "major", [-3, -1, 1, 2, 3, 5, 8])["notes"]
    ex_m = notes_from_degrees("A3", "natural minor", [-3, -1, 1, 2, 3, 5, 8])["notes"]
    story += [table([["Degrees", "−3", "−1", "1", "2", "3", "5", "8"], ["From C4, major"] + ex,
                     ["From A3, natural minor"] + ex_m],
                    [WIDTH * 0.3] + [WIDTH * 0.1] * 7)]
    g = motif_grammar("ABAC", {"A": [1, 2, 3, 5], "B": {"vary": "A", "transpose": -2},
                               "C": {"vary": "A", "invert": True}}, kind="degrees")
    story += [P("Motifs: repetition and variation", "h3"),
              P("Memorable melodies repeat. A <b>motif</b> — a short cell — is stated, repeated, and varied. The classic "
                "variations are <b>transposition</b> (shift every note by the same number of scale steps), "
                "<b>inversion</b> (mirror the contour: up becomes down), <b>retrograde</b> (play it backwards) and "
                "<b>rotation</b> (start it from a different note). Forms such as AABA or ABAC arrange the results."),
              table([["Label", "Rule", "Degrees", "In C major"]] + [
                  [lab, rule, " ".join(str(d) for d in g["motifs"][lab]),
                   notes(notes_from_degrees("C5", "major", g["motifs"][lab])["notes"])]
                  for lab, rule in [("A", "the motif"), ("B", "A transposed down 2 steps"), ("C", "A inverted")]],
                  [WIDTH * 0.1, WIDTH * 0.35, WIDTH * 0.2, WIDTH * 0.35])]
    story += [P("Sequences", "h3"),
              P("A <b>sequence</b> restates a motif at successive pitch levels, usually a step apart — the engine of "
                "Baroque episodes and countless pop bridges. A <i>diatonic</i> sequence stays in the key, so the "
                "interval qualities change slightly from copy to copy; a chromatic passing note keeps its distance "
                "from the scale note beneath it."),
              P("Harmonizing a melody", "h3"),
              P("To choose chords under a melody, list the chords that contain each melody note, then prefer the one "
                "that shares the most notes with the previous chord. That single rule — maximum common tones — "
                "produces smooth, singable harmony; ties go to the melody note sitting low in the chord (the root "
                "first), then to familiar chord qualities, then to simpler chords."),
              ]
    story += box("Pro note",
                 "Two refinements make the rule musical: <i>snap to scale</i> (move stray notes to the nearest scale "
                 "note, ties downward) keeps a line compatible with diatonic chords, and forbidding the same pitch "
                 "set twice in a row (C6 after Am7 is the same four notes) forces genuine harmonic motion.")
    m = ["A4", "B4", "C5", "D5", "C5", "B4", "A4"]
    t1 = tintinnabuli_voice(m, "Am", position="superior")["t_voice"]
    t2 = tintinnabuli_voice(m, "Am", position="inferior")["t_voice"]
    story += [P("Tintinnabuli (Arvo Pärt)", "h2"),
              P("Pärt's <i>tintinnabuli</i> technique pairs a stepwise melodic voice (the M-voice) with a voice that "
                "only ever plays notes of one triad (the T-voice) — like a bell ringing the tonic chord. For each "
                "melody note the T-voice takes the nearest triad note above it (<i>superior</i>), below it "
                "(<i>inferior</i>), or alternately; 'second position' skips to the next triad note. The T-voice can "
                "never clash with the harmony, which is why the texture sounds so luminous."),
              table([["M-voice"] + m, ["T-voice above"] + t1, ["T-voice below"] + t2],
                    [WIDTH * 0.23] + [WIDTH * 0.11] * 7)]


def ch_counterpoint(story):
    story += chapter(9, "Species counterpoint",
                     "Johann Joseph Fux's 1725 method for writing independent melodies that sound good together — still the best course in voice independence.")
    story += [P("Counterpoint adds a new melody to a given one, the <b>cantus firmus</b> (a plain line in whole notes, "
                "beginning and ending on the tonic). Fux taught it in five <b>species</b> of increasing rhythmic "
                "freedom. The rules below are the ones the counterpoint tool enforces; intervals are judged by their "
                "spelling, so a diminished seventh is a dissonance even though it spans as many semitones as a major sixth."),
              P("Rules for every species", "h3")]
    story += bullets([
        "<b>Strong beats are consonant</b>: unison/octave, fifth, thirds and sixths. The perfect fourth counts as a dissonance in two voices.",
        "<b>Begin and end on a perfect consonance</b>; the last note is the tonic, reached by step (usually the leading tone rising, or the 2nd falling). A lower voice begins on the unison or octave — a fifth below would suggest another key.",
        "<b>No parallel or direct fifths and octaves</b>: never move both voices in the same direction into a perfect consonance. Contrary motion is best, oblique next, similar motion into imperfect consonances acceptable.",
        "<b>Melodic lines sing</b>: no leaps of a tritone, a seventh, or any augmented or diminished interval, and none larger than an octave. Steps are the norm; leaps are rare and small.",
        "<b>Keep the voices apart but close</b>: the counterpoint stays on its side of the cantus, generally within a tenth.",
    ])
    story += [P("The five species", "h3")]
    story += bullets([
        "<b>First species</b> (1:1): note against note, all consonant; imperfect consonances preferred in the middle.",
        "<b>Second species</b> (2:1): two notes per cantus note. The first is consonant; the second may be a dissonant <i>passing tone</i> that fills a third by step in one direction. No note is struck twice in a row inside a bar.",
        "<b>Third species</b> (4:1): four notes per cantus note. Dissonances are passing or neighbour tones, approached and left by step, and never next to another dissonance.",
        "<b>Fourth species</b> (syncopation): each note is struck on the weak beat as a consonance and tied over the barline. If the tied note now clashes, it is a <i>suspension</i> and must resolve down by step to a consonance: 7–6, 4–3 or 9–8 above the cantus; 2–3, 4–5 or 9–10 below. A 7–8 resolution below is forbidden, and parallels are judged between the struck weak-beat notes.",
        "<b>Fifth species</b> (florid): a free mix of the others, with a prepared suspension at the cadence.",
    ])
    cf = ["C5", "D5", "E5", "F5", "E5", "D5", "C5"]
    story += [P("One cantus, five species", "h3"),
              P(f"The cantus {notes(cf)} in C major with a counterpoint above it. The table shows the "
                "counterpoint and the interval on each downbeat.", "body")]
    rows = [["Species", "Counterpoint", "Downbeat intervals"]]
    for sp in (1, 2, 3, 4, 5):
        r = species_counterpoint(cf, "C", "major", species=sp)
        rows.append([f"{sp} ({r['ratio']})", notes(r["counterpoint"]), "  ".join(r["downbeat_intervals"])])
    story += [table(rows, [WIDTH * 0.16, WIDTH * 0.52, WIDTH * 0.32])]
    story += box("Why these rules",
                 "Parallel fifths and octaves are forbidden because perfect consonances blend so completely that two "
                 "voices moving in parallel fuse into one — the independence the exercise is about disappears. "
                 "Dissonance handling (prepare, sound on a weak beat or as a suspension, resolve by step) is the "
                 "grammar of tension and release that tonal music still runs on.")


def ch_rhythm_form(story):
    story += chapter(10, "Rhythm and form",
                     "Patterns in time: onset grids, Euclidean rhythms, the grooves of world music, and the large-scale shape of a song.")
    story += [P("Rhythm as a grid", "h3"),
              P("A rhythm can be written as a row of steps: <b>O</b> a strong (accented) hit, <b>o</b> a weak hit, "
                "<b>.</b> a rest. Sixteen steps of a quarter beat each make one bar of 4/4 in sixteenth notes; eight "
                "steps of half a beat make a bar in eighths."),
              P("Euclidean rhythms", "h3"),
              P("Spread k hits as evenly as possible over n steps (Bjorklund's algorithm, as described by Godfried "
                "Toussaint) and a remarkable number of traditional rhythms appear. Three hits in eight steps is the "
                "Cuban <i>tresillo</i>; five in eight the <i>cinquillo</i>; five in sixteen the bossa-nova bass.")]
    rows = [["Hits / steps", "Pattern", "Known as"]]
    known = {(3, 8): "tresillo (Cuba)", (5, 8): "cinquillo (Cuba)", (2, 5): "khafif-e-ramal (13th-century Persia)",
             (3, 7): "ruchenitza (Bulgaria)", (4, 9): "aksak (Turkey)", (5, 12): "Venda clapping (South Africa)",
             (7, 12): "the standard West African bell pattern", (5, 16): "bossa-nova bass (Brazil)",
             (7, 16): "a samba necklace (Brazil)"}
    for (k, n), name in known.items():
        rows.append([f"{k} / {n}", euclidean_rhythm(k, n)["pattern"], name])
    story += [table(rows, [WIDTH * 0.2, WIDTH * 0.4, WIDTH * 0.4])]
    story += [P("A groove library", "h3")]
    rows = [["Groove", "Pattern", "Description"]]
    for name, (pattern, step, desc) in GROOVES.items():
        rows.append([name.replace("_", " "), pattern, desc])
    story += [table(rows, [WIDTH * 0.2, WIDTH * 0.3, WIDTH * 0.5])]
    story += [P("Song form", "h3"),
              P("A song is a sequence of sections. Letter forms name repeated material: AABA is the 32-bar standard "
                "(three statements of a strain around a contrasting bridge); verse–chorus forms alternate a "
                "storytelling verse with a recurring chorus, often with a pre-chorus to build into it and a bridge "
                "for contrast before the last chorus. Conventional lengths, in bars:")]
    roles = {"intro": "sets the mood, often a reduced groove", "verse": "tells the story; lower energy",
             "prechorus": "builds tension into the chorus", "chorus": "the hook: highest energy, repeated words",
             "bridge": "contrast — new chords, often a new key area", "outro": "winds down or fades",
             "drop": "the energy release in dance music", "break": "a pause: instruments drop out",
             "fill": "a one-bar transition, usually drums", "hook": "a short, repeated signature phrase"}
    rows = [["Section", "Bars", "Role"]] + [[k, v, roles.get(k, "")] for k, v in _DEFAULT_BARS.items() if k != "pre-chorus"]
    story += [table(rows, [WIDTH * 0.2, WIDTH * 0.1, WIDTH * 0.7])]
    story += box("Try it",
                 "Take one four-chord loop, a motif and a groove. Write the verse with the motif low and sparse, the "
                 "chorus with the motif transposed up and the full groove, and a bridge on the subdominant key from "
                 "the circle of fifths. That is a complete, coherent song built entirely from the rules in this book.")


def appendix(story):
    story += [NextPageTemplate("main"), PageBreak(), P("APPENDIX", "kicker"), P("Glossary", "h1")]
    terms = [
        ("Accidental", "A sign that raises (#) or lowers (b) a note by a semitone; doubled for two."),
        ("Cadence", "A closing progression, e.g. V–I (authentic), IV–I (plagal), V–vi (deceptive)."),
        ("Cantus firmus", "The given melody against which counterpoint is written."),
        ("Chromatic", "Using notes outside the key; a chromatic chord contains at least one."),
        ("Common tone", "A note shared by two consecutive chords."),
        ("Degree", "A note's position in a scale, 1 for the tonic, written relative to major (b3, #4)."),
        ("Diatonic", "Belonging to the key: made only of scale notes."),
        ("Dominant", "The 5th degree and the chord on it; the chord of tension that resolves to the tonic."),
        ("Enharmonic", "Two spellings of the same pitch, such as F# and Gb."),
        ("Guide tones", "The 3rd and 7th of a chord, which define its quality."),
        ("Interval", "The distance between two notes, named by letter count and quality."),
        ("Inversion", "A chord with a note other than the root in the bass; or an interval turned upside down."),
        ("Leading tone", "The note a semitone below the tonic, which pulls up to it."),
        ("Mode", "A scale that starts on another degree of a parent scale."),
        ("Parallel motion", "Two voices moving in the same direction by the same interval."),
        ("Pitch class", "All notes with one name regardless of octave: every C."),
        ("Relative key", "The major and minor keys sharing one key signature (C major and A minor)."),
        ("Suspension", "A consonant note held over into a chord where it clashes, then resolved down by step."),
        ("Tonic", "The home note of a key or scale."),
        ("Tritone", "The interval of three whole tones (six semitones): augmented 4th or diminished 5th."),
        ("Voice leading", "The way individual lines move from chord to chord."),
    ]
    story += [table([["Term", "Meaning"]] + [[f"<b>{t}</b>", m] for t, m in terms], [WIDTH * 0.25, WIDTH * 0.75])]
    story += [P("About this book", "h2"),
              P("This book is generated by <i>docs/book/build_book.py</i> in the midi-composer-mcp repository. The "
                "catalogue entries, tables and worked examples are computed by the same deterministic functions the "
                "composing tools expose to an AI assistant — the scale and chord databases, the diatonic harmony "
                "resolver, the analysis and voice-leading rules, the reharmonization tools, the tintinnabuli and "
                "species-counterpoint engines, and the rhythm generators. When a rule in the code changes, "
                "rebuilding the book updates every example with it.")]


def build(path: str) -> str:
    _check_coverage()
    doc = Book(path)
    toc = TableOfContents()
    toc.levelStyles = [STYLES["toc0"], STYLES["toc1"]]
    story: list = []
    front_matter(story, toc)
    for fn in (ch_notes, ch_intervals, ch_scales, ch_chords, ch_harmony, ch_circle, ch_voice_leading,
               ch_melody, ch_counterpoint, ch_rhythm_form, appendix):
        fn(story)
    doc.multiBuild(story)
    return path


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "the-rules-of-harmony.pdf")
    print(build(out))
