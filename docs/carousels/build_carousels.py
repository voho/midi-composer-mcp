"""Hudební teorie v kostce – Czech LinkedIn carousels (2:3, 4 x 6 in), generated from the rule tables.

    python docs/carousels/build_carousels.py [output_dir]      # needs: pip install -e ".[book]"

Eight decks: major scales in all keys, church modes, the chords of a key
(T-S-D), chord symbols, the circle of fifths, negative harmony, Fux
counterpoint and Euclidean rhythms. Every note, chord, interval and pattern is
computed by the server's deterministic tools; this file only lays them out and
names them the Czech way (H = English B, B = English B♭, -is/-es endings,
dur/moll, Czech interval and chord terminology).

Design: Avenir Next Condensed (macOS) for all text, Helvetica Neue for ♭/♯,
one visual per slide, one accent colour per deck.
"""

from __future__ import annotations

import math
import re
import os
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

from midi_composer_mcp.chords import CHORDS, chord_notes, match_chords, parse_chord_symbol
from midi_composer_mcp.circle import _CIRCLE, circle_of_fifths
from midi_composer_mcp.counterpoint import species_counterpoint
from midi_composer_mcp.diatonic import diatonic_chords
from midi_composer_mcp.generate import GROOVES, euclidean_rhythm
from midi_composer_mcp.harmony import negative_harmony
from midi_composer_mcp.notes import Note, parse_note
from midi_composer_mcp.scales import SCALES, degree_labels, scale_notes

W, H = 4 * 72, 6 * 72  # 2:3
M = 20

# ------------------------------------------------------------------ fonts

_SYS = "/System/Library/Fonts"
try:
    for _name, _idx in (("Heavy", 8), ("Bold", 0), ("Demi", 2), ("Medium", 5), ("Regular", 7), ("Italic", 4)):
        pdfmetrics.registerFont(TTFont(_name, f"{_SYS}/Avenir Next Condensed.ttc", subfontIndex=_idx))
    pdfmetrics.registerFont(TTFont("Sym", f"{_SYS}/HelveticaNeue.ttc", subfontIndex=10))
    pdfmetrics.registerFont(TTFont("Arrow", f"{_SYS}/Supplemental/Arial Unicode.ttf"))
    pdfmetrics.registerFontFamily("Medium", normal="Medium", bold="Demi", italic="Italic", boldItalic="Demi")
    HEAVY, DEMI, MED, SYM, ARROW = "Heavy", "Demi", "Medium", "Sym", "Arrow"
except Exception:  # pragma: no cover - other systems fall back to the built-in fonts
    HEAVY, DEMI, MED, SYM, ARROW = "Helvetica-Bold", "Helvetica-Bold", "Helvetica", "Helvetica", "Helvetica"

# ------------------------------------------------------------------ palette

INK = colors.HexColor("#141824")
SUB = colors.HexColor("#4a5160")
MUTED = colors.HexColor("#8a8f99")
PAPER = colors.HexColor("#f6f2ea")
CARD = colors.white
LINE = colors.HexColor("#e2dbcf")
GOOD = colors.HexColor("#2e8b57")
T_COL = colors.HexColor("#2f6fad")   # tonika
S_COL = colors.HexColor("#e2a022")   # subdominanta
D_COL = colors.HexColor("#d6452f")   # dominanta
FUNC = {"tonic": T_COL, "subdominant": S_COL, "dominant": D_COL}
FUNC_CS = {"tonic": "T", "subdominant": "S", "dominant": "D"}


def tint(c, k):
    """Mix a colour with white (k = 0 -> the colour, 1 -> white)."""
    return colors.Color(c.red + (1 - c.red) * k, c.green + (1 - c.green) * k, c.blue + (1 - c.blue) * k)


# ------------------------------------------------------------- Czech naming

def cz(note: Note | str, lower: bool = False) -> str:
    """Czech note name: H = B, B = B♭, sharps -is, flats -es (Es, As, B), doubles -isis/-eses."""
    n = parse_note(note.rstrip("0123456789-") if isinstance(note, str) else note.name)
    acc = n.accidental
    if n.letter == "B":
        name = {0: "H", 1: "His", 2: "Hisis", -1: "B", -2: "Heses"}[acc]
    elif acc > 0:
        name = n.letter + "is" * acc
    elif acc < 0:
        if n.letter == "A":
            name = "As" + "as" * (-acc - 1)          # As, Asas
        elif n.letter == "E":
            name = "Es" + "es" * (-acc - 1)          # Es, Eses
        else:
            name = n.letter + "es" * -acc            # Ces, Des, Fes, Ges, Deses...
    else:
        name = n.letter
    return name.lower() if lower else name


def key_cs(tonic: str, minor: bool = False) -> str:
    return f"{cz(tonic, lower=True)} moll" if minor else f"{cz(tonic)} dur"


CZ_SUFFIX = {
    "": "", "m": "mi", "dim": "°", "aug": "+", "5": "5", "sus2": "sus2", "sus4": "sus4", "6": "6", "m6": "mi6",
    "7": "7", "maj7": "maj7", "m7": "mi7", "mMaj7": "mi(maj7)", "dim7": "dim7", "m7b5": "mi7♭5", "7#5": "7♯5",
    "maj7#5": "maj7♯5", "7b5": "7♭5", "7sus4": "7sus4", "7sus2": "7sus2", "add9": "add9", "madd9": "mi(add9)",
    "add4": "add4", "6/9": "6/9", "m6/9": "mi6/9", "9": "9", "maj9": "maj9", "m9": "mi9", "7b9": "7♭9", "7#9": "7♯9",
    "7#11": "7♯11", "maj7#11": "maj7♯11", "11": "11", "m11": "mi11", "13": "13", "maj13": "maj13", "m13": "mi13",
}


def chord_cs(symbol: str) -> str:
    """'F#m7b5' -> 'Fismi7♭5', 'Bb' -> 'B', 'B' -> 'H', 'Am/C' -> 'Ami/C'."""
    root, ctype, bass = parse_chord_symbol(symbol)
    return cz(root) + CZ_SUFFIX[ctype.symbol] + (f"/{cz(bass)}" if bass is not None else "")


def plural_cs(n: int, one: str, few: str, many: str) -> str:
    return one if n == 1 else few if 2 <= n <= 4 else many


def signature_cs(fifths: int) -> str:
    if fifths == 0:
        return "bez předznamenání"
    n = abs(fifths)
    return f"{n} " + (plural_cs(n, "křížek", "křížky", "křížků") if fifths > 0 else plural_cs(n, "béčko", "béčka", "béček"))


def interval_cs(label: str) -> str:
    """'P1/P8' -> 'č1/8', 'M6' -> 'v6', 'm7' -> 'm7', 'A4' -> 'zv4', 'd5' -> 'zm5'."""
    if label == "P1/P8":
        return "č1/8"
    q = label.rstrip("0123456789")
    return {"P": "č", "M": "v", "m": "m", "A": "zv", "d": "zm"}.get(q, q) + label[len(q):]


def degree_sym(label: str) -> str:
    return label.replace("#", "♯").replace("b", "♭")


def roman_sym(numeral: str) -> str:
    """'bVII' -> '♭VII', '#iv°' -> '♯iv°' (in Czech, a bare 'b' reads as the note B♭)."""
    k = 0
    while k < len(numeral) and numeral[k] in "b#":
        k += 1
    return degree_sym(numeral[:k]) + numeral[k:]


def interval_between_cs(low: str, high: str) -> str:
    """The real (compound) interval as a Czech abbreviation: č8, v9, m10, zv4..."""
    from midi_composer_mcp.harmony import interval_between
    a, b = parse_note(low), parse_note(high)
    if a.midi > b.midi:
        a, b = b, a
    short = interval_between(a.name, b.name)["short"]
    q = short.rstrip("0123456789")
    return {"P": "č", "M": "v", "m": "m", "A": "zv", "d": "zm", "AA": "2×zv", "dd": "2×zm"}.get(q, q) + short[len(q):]


_LOF = {"F": -1, "C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5}


def practical_cs(root: Note, minor: bool) -> str:
    """' (= Fismi)' when a chord's key would need more than 7 accidentals (Gesmi, Fes...), else ''."""
    fifths = _LOF[root.letter] + 7 * root.accidental - (3 if minor else 0)
    if abs(fifths) < 7:
        return ""
    from midi_composer_mcp.notes import spell_pitch_class
    other = spell_pitch_class(root.pitch_class, prefer_flats=fifths > 0)
    return f" (= {cz(other)}{'mi' if minor else ''})"


# ------------------------------------------------------------------ slides

_NBSP = "\u00a0"


def czech_spacing(text: str) -> str:
    """Czech typography: no line break after a one-letter preposition/conjunction, before dur/moll, around ↔."""
    parts = re.split(r"(<[^>]+>)", text)
    for i, part in enumerate(parts):
        if part.startswith("<"):
            continue
        part = re.sub(r"(?<![\w\u00C0-\u017F])([ksvzouaiKSVZOUAI]) ", "\\1" + _NBSP, part)
        part = re.sub(r" (dur|moll)\b", _NBSP + "\\1", part)
        part = part.replace(" ↔ ", _NBSP + "↔" + _NBSP)
        parts[i] = part
    return "".join(parts)


def rich(text: str) -> str:
    """♭/♯ in Helvetica Neue and arrows in Arial Unicode (Avenir has neither); Czech spacing."""
    text = czech_spacing(text)
    out = []
    for ch in text:
        out.append(f'<font name="{SYM}">{ch}</font>' if ch in "♭♯" else
                   f'<font name="{ARROW}">{ch}</font>' if ch in "→↔↦" else ch)
    return "".join(out)


def pstyle(size, font=MED, color=INK, leading=None, align=TA_LEFT):
    return ParagraphStyle("p", fontName=font, fontSize=size, leading=leading or size * 1.22, textColor=color,
                          alignment=align)


class Slide:
    def __init__(self, deck, bg=PAPER):
        self.d, self.c = deck, deck.c
        self.c.setFillColor(bg)
        self.c.rect(0, 0, W, H, fill=1, stroke=0)

    def text(self, markup, x, y, width, st) -> float:
        """Draw a paragraph whose top edge is at y; return its bottom edge."""
        p = Paragraph(rich(markup), st)
        _, h = p.wrap(width, H)
        p.drawOn(self.c, x, y - h)
        return y - h

    def label(self, text, x, y, color=None, size=7.8):
        self.c.setFont(DEMI, size)
        self.c.setFillColor(color or self.d.accent)
        self.c.drawString(x, y, " ".join(text.upper()))

    def kicker(self, text):
        self.label(text, M, H - 30)

    def title(self, text, y, size=30, color=INK, width=W - 2 * M):
        return self.text(text, M, y, width, pstyle(size, HEAVY, color, size * 1.02))

    def body(self, text, y, size=12, color=SUB, width=W - 2 * M, x=M, align=TA_LEFT):
        return self.text(text, x, y, width, pstyle(size, MED, color, size * 1.3, align))

    def card(self, x, y, w, h, top="", big="", cap="", fill=CARD, edge=LINE, top_col=None, big_col=INK,
             big_size=None, radius=7, cap_col=None):
        c = self.c
        c.setFillColor(fill)
        c.setStrokeColor(edge or fill)
        c.setLineWidth(0.8)
        c.roundRect(x, y - h, w, h, radius, fill=1, stroke=1 if edge else 0)
        size = big_size or (19 if h >= 56 else 15 if h >= 42 else 12.5)
        plain = big.replace("<b>", "").replace("</b>", "")
        fit = (w - 6) / max(1, len(plain)) / 0.52  # condensed heavy: ~0.52 em per character
        size = min(size, fit)
        if top:
            self.text(top, x + 4, y - 6, w - 8, pstyle(7.5, DEMI, top_col or self.d.accent, 9, TA_CENTER))
        if big:
            yb = y - (max(h * 0.30, 16.5) if top else h * 0.18)
            self.text(big, x + 2, yb, w - 4, pstyle(size, HEAVY, big_col, size * 1.05, TA_CENTER))
        if cap:
            self.text(cap, x + 4, y - h + 13, w - 8, pstyle(7.3, MED, cap_col or MUTED, 8.5, TA_CENTER))

    def grid(self, items, y, cols, h, gap=7, **kw) -> float:
        cw = (W - 2 * M - gap * (cols - 1)) / cols
        for i, item in enumerate(items):
            r, k = divmod(i, cols)
            opts = dict(kw)
            if isinstance(item, dict):
                opts.update(item)
                item = (opts.pop("top", ""), opts.pop("big", ""), opts.pop("cap", ""))
            self.card(M + k * (cw + gap), y - r * (h + gap), cw, h, *item, **opts)
        return y - math.ceil(len(items) / cols) * (h + gap)

    def strip(self, items, y, h=34, gap=4, **kw) -> float:
        """One row of equal cards across the slide."""
        cw = (W - 2 * M - gap * (len(items) - 1)) / len(items)
        for i, item in enumerate(items):
            opts = dict(kw)
            if isinstance(item, dict):
                opts.update(item)
                item = (opts.pop("top", ""), opts.pop("big", ""), opts.pop("cap", ""))
            self.card(M + i * (cw + gap), y, cw, h, *item, **opts)
        return y - h - gap

    def keyboard(self, y, marks: dict[int, str], root: int | None = 0, special: set | None = None,
                 width=None, octaves=2, x=None) -> float:
        """Piano keys from C (top edge at y). Marked keys are filled; the root is darker; labels sit
        on the white keys, and above the keyboard for black keys. Returns the bottom edge."""
        c = self.c
        width = width or W - 2 * M
        kw = width / (7 * octaves + 1)
        kh = kw * 3.3
        x0 = M if x is None else x
        top = y - 12
        base = top - kh
        whites = [o * 12 + s for o in range(octaves) for s in (0, 2, 4, 5, 7, 9, 11)] + [12 * octaves]
        special = special or set()

        def fill(semi, default):
            if semi == root:
                return self.d.dark
            if semi in special:
                return self.d.accent2
            if semi in marks:
                return self.d.accent
            return default

        c.setLineWidth(0.6)
        for i, semi in enumerate(whites):
            c.setFillColor(fill(semi, colors.white))
            c.setStrokeColor(colors.HexColor("#b9b2a6"))
            c.roundRect(x0 + i * kw, base, kw, kh, 2.2, fill=1, stroke=1)
        blacks = [(x0 + (o * 7 + pos + 1) * kw - kw * 0.31, o * 12 + semi)
                  for o in range(octaves) for pos, semi in ((0, 1), (1, 3), (3, 6), (4, 8), (5, 10))]
        for bx, semi in blacks:
            c.setFillColor(fill(semi, INK))
            c.setStrokeColor(INK)
            c.roundRect(bx, base + kh * 0.36, kw * 0.62, kh * 0.64, 1.6, fill=1, stroke=1)
        size = min(7.6, kw * 0.52)
        for semi, lab in marks.items():
            c.setFont(DEMI, size if len(lab) <= 3 else size * 0.86)
            if semi in whites:
                c.setFillColor(colors.white)
                c.drawCentredString(x0 + (whites.index(semi) + 0.5) * kw, base + 5, lab)
            else:
                bx = next(bx for bx, s in blacks if s == semi)
                c.setFillColor(INK)
                c.drawCentredString(bx + kw * 0.31, top + 3, lab)
        return base - 6

    def footer(self, dark=False):
        c = self.c
        col = colors.white if dark else MUTED
        c.setFillColor(col)
        c.setFont(DEMI, 6.6)
        c.drawString(M, 14, " ".join(self.d.series.upper()))
        more = self.d.n < self.d.total
        c.setFont(MED, 7.5)
        c.drawRightString(W - M - (16 if more else 0), 14, f"{self.d.n} / {self.d.total}")
        if more:  # a drawn arrow: swipe on
            c.setStrokeColor(col)
            c.setLineWidth(1.1)
            ax = W - M - 3
            c.line(ax - 9, 16.5, ax, 16.5)
            c.line(ax - 3, 19.5, ax, 16.5)
            c.line(ax - 3, 13.5, ax, 16.5)


class Deck:
    series = "Hudební teorie v kostce"

    def __init__(self, path, name, accent, accent2, dark, total):
        self.c = canvas.Canvas(path, pagesize=(W, H))
        self.c.setTitle(name)
        self.c.setAuthor("midi-composer-mcp")
        self.c.setSubject(self.series)
        self.name, self.accent, self.accent2, self.dark = name, accent, accent2, dark
        self.total, self.n = total, 0

    def new(self, bg=PAPER) -> Slide:
        self.n += 1
        return Slide(self, bg)

    def done(self, s: Slide, dark=False):
        s.footer(dark)
        self.c.showPage()

    def _dark_frame(self, kicker):
        s = self.new(self.dark)
        self.c.setFillColor(self.accent)
        self.c.rect(0, H - 9, W, 9, fill=1, stroke=0)
        s.label(kicker, M, H - 40, tint(self.accent, 0.35), 8)
        return s

    def cover(self, kicker, title, sub, visual=None):
        s = self._dark_frame(kicker)
        y = s.title(title, H - 52, 38, colors.white)
        s.body(sub, y - 12, 13, tint(self.accent, 0.72))
        if visual:
            visual(s)
        self.done(s, dark=True)

    def closing(self, points: list[str]):
        s = self._dark_frame("Shrnutí")
        y = s.title("Ulož si to na příště", H - 52, 30, colors.white) - 14
        for p in points:
            self.c.setFillColor(self.accent)
            self.c.circle(M + 4, y - 7.5, 3.2, fill=1, stroke=0)
            y = s.body(p, y, 12.5, colors.white, W - 2 * M - 16, x=M + 16) - 10
        s.body("Všechny tóny, akordy a vzorce v této sérii vypočítal open-source nástroj <i>midi-composer-mcp</i> – "
               "hudební teorie, podle které umí skládat i AI.", 72, 9, tint(self.accent, 0.6))
        self.done(s, dark=True)

    def save(self):
        assert self.n == self.total, (self.name, self.n, self.total)
        self.c.save()


# --------------------------------------------------------------- visuals

KEYS = ["C", "G", "D", "A", "E", "B", "F#", "Db", "Ab", "Eb", "Bb", "F"]


def marks_for(names: list[str]) -> dict[int, str]:
    out = {}
    for n in names:
        nn = parse_note(n if n[-1].isdigit() else n + "4")
        rel = nn.midi - 60
        while rel < 0:
            rel += 12
        while rel > 24:
            rel -= 12
        out.setdefault(rel, cz(nn, lower=True))
    return out


def scale_on(tonic: str, scale="major") -> list[str]:
    return [n.name for n in scale_notes(SCALES[scale], parse_note(tonic + "4"))]


def names_cs(names) -> str:
    return " ".join(cz(n, lower=True) for n in names)


def mini_circle(s: Slide, cx, cy, r, highlight: str | None = None, labels=True):
    """The circle of fifths as 12 wedges; the highlighted key is filled, its neighbours tinted."""
    c = s.c
    pcs = [parse_note(e["major"]).pitch_class for e in _CIRCLE]
    hi = pcs.index(parse_note(highlight).pitch_class) if highlight else None
    for i in range(12):
        is_hi = hi is not None and i == hi
        near = hi is not None and (i - hi) % 12 in (1, 11)
        c.setFillColor(s.d.accent if is_hi else tint(s.d.accent, 0.6) if near else colors.white)
        c.setStrokeColor(LINE)
        c.setLineWidth(0.7)
        p = c.beginPath()
        p.moveTo(cx, cy)
        p.arcTo(cx - r, cy - r, cx + r, cy + r, 105 - i * 30, -30)
        p.close()
        c.drawPath(p, fill=1, stroke=1)
    c.setFillColor(PAPER)
    c.circle(cx, cy, r * 0.44, fill=1, stroke=0)
    if not labels:
        return
    for i, e in enumerate(_CIRCLE):
        ang = math.radians(90 - i * 30)
        is_hi = hi is not None and i == hi
        name = cz(e["major"]) + (f"/{cz(e['enharmonic'])}" if e["enharmonic"] else "")
        size = r * (0.15 if len(name) <= 3 else 0.105)
        c.setFont(HEAVY if is_hi else DEMI, size)
        c.setFillColor(colors.white if is_hi else INK)
        c.drawCentredString(cx + r * 0.73 * math.cos(ang), cy + r * 0.73 * math.sin(ang) - size * 0.35, name)
        c.setFont(MED, r * 0.085)
        c.setFillColor(SUB)
        c.drawCentredString(cx + r * 0.31 * math.cos(ang), cy + r * 0.31 * math.sin(ang) - r * 0.03,
                            cz(e["relative_minor"], lower=True))


def step_marks(s: Slide, y, x0, kw, steps):
    """Brackets over consecutive white keys labelled C (celý tón) or P (půltón)."""
    c = s.c
    for i, st in enumerate(steps):
        xa, xb = x0 + (i + 0.5) * kw, x0 + (i + 1.5) * kw
        half = st in ("P", "½")
        col = s.d.accent if half else SUB
        c.setStrokeColor(col)
        c.setLineWidth(1.5 if half else 0.9)
        c.line(xa + 3, y, xb - 3, y)
        c.line(xa + 3, y, xa + 3, y - 4)
        c.line(xb - 3, y, xb - 3, y - 4)
        c.setFont(HEAVY if half else DEMI, 11)
        c.setFillColor(col)
        c.drawCentredString((xa + xb) / 2, y + 4, st)


def chord_row(s: Slide, y, chords, h=36, size=10.5):
    items = [{"top": roman_sym(ch["roman"]), "big": chord_cs(ch["symbol"]), "top_col": FUNC.get(ch.get("harmonic_function"), s.d.accent)}
             for ch in chords]
    return s.strip(items, y, h=h, big_size=size, radius=5)


def check_mark(c, x, y, good: bool):
    c.setFillColor(GOOD if good else D_COL)
    c.circle(x, y, 9, fill=1, stroke=0)
    c.setStrokeColor(colors.white)
    c.setLineWidth(2)
    if good:
        c.line(x - 4, y, x - 1, y - 3.5)
        c.line(x - 1, y - 3.5, x + 4.5, y + 4)
    else:
        c.line(x - 3.5, y - 3.5, x + 3.5, y + 3.5)
        c.line(x - 3.5, y + 3.5, x + 3.5, y - 3.5)


# ============================================================== 1. STUPNICE

def deck_scales(path):
    d = Deck(path, "Durové stupnice", colors.HexColor("#e4572e"), colors.HexColor("#f3a712"),
             colors.HexColor("#1d1f33"), 4 + len(KEYS) + 1)
    d.cover("Stupnice", "Durové stupnice ve všech tóninách",
            "Jeden vzorec, dvanáct tónin – a proč má F dur tón b, a ne ais.",
            lambda s: s.keyboard(160, marks_for(scale_on("C")[:-1]), root=0, octaves=1))

    s = d.new()
    s.kicker("Vzorec")
    y = s.title("1 · 1 · ½ · 1 · 1 · 1 · ½", H - 40, 28)
    y = s.body("Od libovolného tónu postupuj vzhůru o <b>celé tóny (1)</b> a <b>půltóny (½)</b> v tomto pořadí – "
               "vznikne durová stupnice. Půltóny leží mezi 3. a 4. a mezi 7. a 8. stupněm.", y - 8)
    kb_top = y - 30
    s.keyboard(kb_top, marks_for(scale_on("C")), root=0, octaves=1)
    step_marks(s, kb_top - 2, M, (W - 2 * M) / 8, ["1", "1", "½", "1", "1", "1", "½"])
    y = kb_top - 12 - (W - 2 * M) / 8 * 3.3 - 16
    s.body("Půltón mezi 7. a 8. stupněm (<b>citlivý tón</b> h → c) dodává duru jasný, uzavřený závěr.", y, 11.5)
    d.done(s)

    s = d.new()
    s.kicker("Pravidlo")
    y = s.title("Každý kmenový tón jen jednou", H - 40, 30)
    y = s.body("Stupnice použije každý kmenový tón (c d e f g a h) právě jednou – přirozený, zvýšený (<b>-is</b>), "
               "nebo snížený (<b>-es</b>). Proto se tatáž klávesa jednou jmenuje s křížkem a jindy s béčkem.", y - 8)
    s.grid([("F dur", "b", "4. stupeň: snížené h, ne ais"), ("D dur", "fis · cis", "3. a 7. stupeň"),
            ("Fis dur", "eis", "7. stupeň leží na bílé klávese f"), ("Ges dur", "ces", "4. stupeň leží na bílé klávese h")],
           y - 14, cols=2, h=70, big_size=21)
    d.done(s)

    s = d.new()
    s.kicker("Česká specialita")
    y = s.title("Pozor na H a B", H - 40, 30)
    y = s.body("V češtině (stejně jako v němčině) se bílá klávesa pod c jmenuje <b>h</b> – angličtina jí říká B. "
               "Naše <b>b</b> je o půltón níž, anglicky B♭.", y - 8)
    y = s.keyboard(y - 18, {9: "a", 10: "b", 11: "h", 12: "c"}, root=None, octaves=1)
    s.grid([("česky", "h", "anglicky B"), ("česky", "b", "anglicky B♭"),
            ("křížek ♯", "-is", "c → cis, f → fis, h → his"), ("béčko ♭", "-es", "d → des, e → es, a → as, h → b")],
           y - 6, cols=2, h=58, big_size=20)
    d.done(s)

    for key in KEYS:
        names = scale_on(key)
        focus = circle_of_fifths(key)["focus"]
        s = d.new()
        s.kicker("Durová stupnice")
        s.title(key_cs(key), H - 40, 40)
        mini_circle(s, W - M - 34, H - 62, 32, highlight=key, labels=False)
        s.c.setFont(HEAVY, 16)
        s.c.setFillColor(d.accent)
        s.c.drawString(M, H - 104, "  ".join(cz(n, lower=True) for n in names))
        y = s.keyboard(H - 112, marks_for(names[:-1]), root=parse_note(names[0]).midi - 60, octaves=2)
        acc_list = ", ".join(cz(a, lower=True) for a in focus["accidentals"]) or "žádná znaménka"
        y = s.grid([("předznamenání", signature_cs(focus["fifths"]), acc_list),
                    ("paralelní moll", key_cs(focus["relative_minor"].split()[0], True), "stejné předznamenání"),
                    ("dominantová tónina", key_cs(focus["dominant"]), "o kvintu výš"),
                    ("subdominantová tónina", key_cs(focus["subdominant"]), "o kvintu níž")],
                   y - 2, cols=2, h=56, big_size=16)
        s.label("Akordy tóniny", M, y - 6)
        chord_row(s, y - 12, diatonic_chords(key, "major")["chords"], h=42, size=12)
        d.done(s)

    d.closing(["<b>Vzorec:</b> 1 1 ½ 1 1 1 ½ (1 = celý tón, ½ = půltón).",
               "<b>Pravopis:</b> každý kmenový tón jednou; ♯ = -is, ♭ = -es (výjimky: es, as, b).",
               "<b>H</b> = anglické B, <b>B</b> = anglické B♭.",
               "<b>O kvintu výš</b> přibude křížek (nebo ubude béčko), <b>o kvintu níž</b> naopak."])
    d.save()


# ============================================================== 2. MÓDY

MODES = [  # (scale, name, characteristic degree index, its label, what it is, mood)
    ("lydian", "lydický", 3, "♯4", "zvětšená kvarta", "Vzdušný a zasněný – oblíbený ve filmové hudbě."),
    ("major", "jónský", 6, "7", "velká septima, citlivý tón", "Jasný a stabilní – je to běžná durová stupnice."),
    ("mixolydian", "mixolydický", 6, "♭7", "malá septima", "Durový, ale bluesový – rock, funk, keltská hudba."),
    ("dorian", "dórský", 5, "6", "velká sexta", "Moll se světlou sextou – jazz, folk, soul."),
    ("natural minor", "aiolský", 5, "♭6", "malá sexta", "Klasická, smutná přirozená moll."),
    ("phrygian", "frygický", 1, "♭2", "malá sekunda", "Temný, se španělským nádechem."),
    ("locrian", "lokrický", 4, "♭5", "zmenšená kvinta", "Nestabilní – akord na tónice je zmenšený kvintakord."),
]


def mode_colour(i):
    k = i / 6
    return colors.Color(1 - 0.72 * k, 0.84 - 0.56 * k, 0.46 - 0.18 * k)


def deck_modes(path):
    d = Deck(path, "Církevní mody", colors.HexColor("#6c4ab6"), colors.HexColor("#f3a712"),
             colors.HexColor("#1f1834"), 3 + len(MODES) + 1)

    def cover_visual(s):
        for i, (_k, name, *_rest) in enumerate(MODES):
            s.c.setFillColor(mode_colour(i))
            s.c.roundRect(M, 196 - i * 21, W - 2 * M, 17, 4, fill=1, stroke=0)
            s.c.setFont(DEMI, 10)
            s.c.setFillColor(INK if i < 3 else colors.white)
            s.c.drawString(M + 8, 201 - i * 21, name)
    d.cover("Mody", "Sedm církevních modů", "Stejných sedm tónů, jiná tónika – a úplně jiná nálada.", cover_visual)

    s = d.new()
    s.kicker("Princip")
    y = s.title("Stejné tóny, jiná tónika", H - 40, 30)
    y = s.body("Hraj jen bílé klávesy. Od <b>c</b> je to jónský modus (dur), od <b>d</b> dórský, od <b>e</b> frygický, "
               "od <b>f</b> lydický, od <b>g</b> mixolydický, od <b>a</b> aiolský (přirozená moll) a od <b>h</b> lokrický.",
               y - 8)
    y = s.keyboard(y - 14, {0: "c", 2: "d", 4: "e", 5: "f", 7: "g", 9: "a", 11: "h"}, root=None, octaves=1)
    s.body("Se změnou tóniky se půltóny ocitnou na jiných stupních – a to mění charakter. Na dalších stranách "
           "stojí všechny mody na <b>c</b>, takže je snadno porovnáš.", y - 4, 11.5)
    d.done(s)

    s = d.new()
    s.kicker("Žebříček jasu")
    y = s.title("Od nejsvětlejšího k nejtemnějšímu", H - 40, 26) - 12
    for i, (_key, name, _deg, mark, what, _mood) in enumerate(MODES):
        s.c.setFillColor(mode_colour(i))
        s.c.roundRect(M, y - 34, 40, 30, 5, fill=1, stroke=0)
        s.text(mark, M, y - 10, 40, pstyle(15, HEAVY, INK if i < 3 else colors.white, 16, TA_CENTER))
        s.text(f"{name[0].upper()}{name[1:]}", M + 50, y - 5, W - 2 * M - 50, pstyle(13.5, HEAVY, INK, 15))
        s.text(what, M + 50, y - 21, W - 2 * M - 50, pstyle(9.5, MED, SUB, 11))
        y -= 39
    d.done(s)

    for i, (key, name, deg_idx, mark, what, mood) in enumerate(MODES):
        sc = SCALES[key]
        names = [n.name for n in scale_notes(sc, parse_note("C4"))][:-1]
        s = d.new()
        s.kicker("Modus od c")
        minor_type = 3 in SCALES[key].intervals  # minor third: written with a lowercase tonic, like a minor key
        y = s.title(f"{'c' if minor_type else 'C'} {name}", H - 40, 36)
        y = s.body(mood, y - 6, 12.5)
        y = s.keyboard(y - 12, marks_for(names), root=0, special={parse_note(names[deg_idx]).midi - 60}, octaves=1)
        items = [{"big": degree_sym(lab), "fill": d.accent2 if j == deg_idx else CARD,
                  "edge": None if j == deg_idx else LINE} for j, lab in enumerate(degree_labels(sc))]
        y = s.strip(items, y - 2, h=32, big_size=13, radius=5) - 2
        y = s.body(f"<b>Charakteristický stupeň {mark}</b> – {what}.", y, 12, INK)
        s.label("Akordy modu", M, y - 18)
        chord_row(s, y - 24, diatonic_chords("C", key)["chords"], h=46, size=12)
        d.done(s)

    d.closing(["<b>Jas:</b> lydický → jónský → mixolydický → dórský → aiolský → frygický → lokrický.",
               "Každý krok k temnějšímu modu sníží o půltón jeden stupeň, v pořadí 4 → 7 → 3 → 6 → 2 → 5.",
               "Modus od jiného tónu: vezmi jeho vzorec (např. dórský 1 2 ♭3 4 5 6 ♭7) a postav ho od nové tóniky."])
    d.save()


# ============================================================== 3. AKORDY V TÓNINĚ

def deck_diatonic(path):
    d = Deck(path, "Akordy v tónině", colors.HexColor("#1b998b"), colors.HexColor("#e2a022"),
             colors.HexColor("#132a2a"), 3 + len(KEYS) + 1)

    def cover_visual(s):
        items = [{"top": ch["roman"], "big": chord_cs(ch["symbol"]), "fill": FUNC[ch["harmonic_function"]],
                  "top_col": colors.white, "big_col": colors.white} for ch in diatonic_chords("C", "major")["chords"]]
        s.strip(items, 150, h=48, big_size=12, radius=5, edge=None)
    d.cover("Harmonie", "Sedm akordů každé durové tóniny", "I ii iii IV V vi vii° – vzorec se naučíš jednou a "
            "použiješ ho ve všech dvanácti durových tóninách.", cover_visual)

    s = d.new()
    s.kicker("Stavba")
    y = s.title("Tercie na tercii", H - 40, 30)
    y = s.body("Na každém stupni postav akord jen z tónů stupnice: základní tón, tercie a kvinta nad ním. "
               "V duru vyjde vždy totéž: <b>durové</b> akordy na I, IV a V, <b>mollové</b> na ii, iii a vi, "
               "<b>zmenšený</b> na vii°. Velká římská číslice = durový akord, malá = mollový, ° = zmenšený.", y - 8)
    y = s.keyboard(y - 14, {0: "c", 4: "e", 7: "g"}, root=0, octaves=1)
    s.body("Akord C: c + e (velká tercie) + g (malá tercie nad e) = durový kvintakord.", y - 2, 11.5)
    d.done(s)

    s = d.new()
    s.kicker("Funkce")
    y = s.title("T · S · D", H - 40, 34) - 6
    y = s.body("Akordy v tónině plní tři funkce. Barvy na dalších stranách ukazují, kam který patří.", y)
    for col, letter, name, desc in ((T_COL, "T", "tónika", "klid, domov – I, vi (iii je dvojznačný: T i D)"),
                                    (S_COL, "S", "subdominanta", "odchod z domova – IV, ii"),
                                    (D_COL, "D", "dominanta", "napětí, které chce domů – V, vii°")):
        s.card(M, y - 8, 56, 56, "", letter, "", fill=col, edge=None, big_col=colors.white, big_size=28)
        s.text(name, M + 68, y - 14, W - 2 * M - 68, pstyle(16, HEAVY, INK, 18))
        s.text(desc, M + 68, y - 34, W - 2 * M - 68, pstyle(11, MED, SUB, 13))
        y -= 66
    s.body("Většina harmonických průběhů krouží <b>T → S → D → T</b>.", y - 6, 12, INK)
    d.done(s)

    for key in KEYS:
        rows = diatonic_chords(key, "major")["chords"]
        s = d.new()
        s.kicker("Akordy v tónině")
        y = s.title(key_cs(key), H - 40, 38) - 10
        cw = (W - 2 * M - 7) / 2
        for i, ch in enumerate(rows):
            r, k = divmod(i, 2)
            col = FUNC[ch["harmonic_function"]]
            func = "T/D" if ch["degree"] == 3 else FUNC_CS[ch["harmonic_function"]]
            s.card(M + k * (cw + 7), y - r * 64, cw, 58, f'{roman_sym(ch["roman"])} · {func}',
                   chord_cs(ch["symbol"]), names_cs(ch["notes"]), fill=tint(col, 0.86), edge=None, top_col=col,
                   big_size=21, cap_col=SUB)
        y -= 4 * 64 + 2
        for lab, idx in (("Pop  I–V–vi–IV", (0, 4, 5, 3)), ("Jazz  ii–V–I", (1, 4, 0))):
            y = s.body(f"<b>{lab}:</b>  " + " – ".join(chord_cs(rows[i]["symbol"]) for i in idx), y, 11.5, INK) - 2
        d.done(s)

    d.closing(["<b>Durové:</b> I, IV, V. <b>Mollové:</b> ii, iii, vi. <b>Zmenšený:</b> vii°.",
               "<b>Funkce:</b> tónika (T), subdominanta (S), dominanta (D).",
               "<b>Pop:</b> I–V–vi–IV. <b>Jazz:</b> ii–V–I."])
    d.save()


# ============================================================== 4. AKORDOVÉ ZNAČKY

CHORD_CS_NAMES = {
    "major": "durový kvintakord", "minor": "mollový kvintakord", "diminished": "zmenšený kvintakord",
    "augmented": "zvětšený kvintakord", "suspended 4": "kvarta místo tercie", "suspended 2": "sekunda místo tercie",
    "major 6": "durový akord s přidanou sextou", "add 9": "durový akord s přidanou nónou",
    "six nine": "durový akord se sextou a nónou",
    "dominant 7": "dominantní (malý durový) septakord", "major 7": "velký durový septakord",
    "minor 7": "malý mollový septakord", "minor major 7": "velký mollový septakord",
    "half-diminished": "malý zmenšený septakord", "diminished 7": "zmenšený septakord",
    "augmented 7": "dominantní septakord se zvětšenou kvintou", "dominant 7 sus 4": "dominantní septakord s kvartou místo tercie",
    "dominant 9": "dominantní nónový akord", "major 9": "velký durový septakord s nónou",
    "minor 9": "malý mollový septakord s nónou", "dominant 13": "dominantní tercdecimový akord",
}
CHORD_PAGES = [
    ("Kvintakordy", ["major", "minor", "diminished", "augmented"]),
    ("Sus a přidaná sexta", ["suspended 4", "suspended 2", "major 6", "dominant 7 sus 4"]),
    ("Septakordy I", ["dominant 7", "major 7", "minor 7", "minor major 7"]),
    ("Septakordy II", ["half-diminished", "diminished 7", "augmented 7"]),
    ("Přidaná nóna", ["add 9", "six nine"]),
    ("Nónové akordy", ["dominant 9", "minor 9"]),
    ("Nóna a tercdecima", ["major 9", "dominant 13"]),
]


def deck_chords(path):
    d = Deck(path, "Akordové značky", colors.HexColor("#d1495b"), colors.HexColor("#f3a712"),
             colors.HexColor("#2a1720"), 3 + len(CHORD_PAGES) + 1)
    d.cover("Akordy", "Akordové značky bez tajemství",
            "Co přesně znamená Cmi7♭5, Cdim7 nebo C13 – vzorec, tóny a zvuk.",
            lambda s: s.grid([("", "Cmi7♭5", ""), ("", "Cdim7", ""), ("", "Cmaj7", ""), ("", "C13", "")], 176,
                             cols=2, h=48, fill=tint(d.dark, 0.12), edge=None, big_col=colors.white, big_size=20))

    s = d.new()
    s.kicker("Jak číst značku")
    y = s.title("Základní tón + přípona", H - 40, 28)
    y = s.body("Přípona je recept: říká, jaké intervaly akord obsahuje nad základním tónem.", y - 8)
    y = s.grid([("mi", "moll", "malá tercie"), ("7", "malá septima", "dominantní zvuk"),
                ("maj7", "velká septima", "měkký, jazzový"), ("°", "zmenšený", "zmenšená kvinta"),
                ("+", "zvětšený", "zvětšená kvinta"), ("sus4", "bez tercie", "kvarta místo tercie")],
               y - 10, cols=2, h=52, big_size=15)
    s.body("<b>Pozor:</b> mezinárodně se píše i <b>m</b> (Cm = Cmi), ale velké <b>M</b> znamená velkou septimu "
           "(CM7 = Cmaj7). Základní tón se čte česky: <b>B</b> = anglické B♭, <b>H</b> = anglické B, dále <b>Fis</b>, <b>Es</b>…",
           y - 2, 11)
    d.done(s)

    for heading, names in CHORD_PAGES:
        s = d.new()
        s.kicker("Vzorce od c")
        y = s.title(heading, H - 40, 28) - (8 if len(names) > 3 else 10)
        step = 80 if len(names) > 3 else 84
        for n in names:
            ct = CHORDS[n]
            tones = [t.name for t in chord_notes(ct, parse_note("C4"))]
            formula = "  ".join(degree_sym(lab) for lab, _ in ct.degrees)
            top = y
            sym = f"C{CZ_SUFFIX[ct.symbol]}"
            size = min(24, 100 / max(1, pdfmetrics.stringWidth(sym, HEAVY, 1)))
            if max(ct.intervals) <= 11:  # fits one octave: a big keyboard beside the text
                s.text(sym, M, top, 104, pstyle(size, HEAVY, d.accent, 25))
                s.text(CHORD_CS_NAMES[n], M, top - 26, 100, pstyle(9, DEMI, INK, 10.5))
                s.text(f"{formula}<br/>{names_cs(tones)}", M, top - 49, 100, pstyle(10, MED, SUB, 12))
                s.keyboard(top + 8, marks_for(tones), root=0, width=W - 2 * M - 108, octaves=1, x=M + 108)
                y = top - step
                rule = y + 5
            else:                         # extended chords: the name on top, two octaves below
                s.text(sym, M, top, 110, pstyle(size, HEAVY, d.accent, 25))
                s.text(CHORD_CS_NAMES[n], M + 112, top - 2, W - 2 * M - 112, pstyle(10, DEMI, INK, 11.5))
                s.text(f"{formula}   ·   {names_cs(tones)}", M + 112, top - 26, W - 2 * M - 112,
                       pstyle(10, MED, SUB, 12))
                y = s.keyboard(top - 36, marks_for(tones), root=0, octaves=2) - 8
                if n == "dominant 13":
                    y = s.body("Undecima (f) se v C13 obvykle vynechává – tře se s tercií e.", y + 2, 9.5, MUTED) - 4
                rule = y + 6
            s.c.setStrokeColor(LINE)
            s.c.setLineWidth(0.6)
            s.c.line(M, rule, W - M, rule)
        d.done(s)

    s = d.new()
    s.kicker("Obraty")
    y = s.title("Když v basu není základní tón", H - 40, 26)
    y = s.body("Značka s lomítkem říká, který tón zní v basu. Česká harmonie pro obraty používá tyto názvy:", y - 8)
    s.grid([("C/E", "sextakord", "1. obrat kvintakordu"), ("C/G", "kvartsextakord", "2. obrat kvintakordu"),
            ("C7/E", "kvintsextakord", "1. obrat septakordu"), ("C7/G", "terckvartakord", "2. obrat septakordu"),
            ("C7/B", "sekundakord", "3. obrat: septima (b) v basu")], y - 10, cols=2, h=56, big_size=15)
    d.done(s)

    d.closing(["<b>Vzorec</b> = intervaly nad základním tónem: 1 3 5, 1 ♭3 5, 1 3 5 ♭7…",
               "<b>mi</b> = moll, <b>7</b> = malá septima, <b>maj7</b> = velká septima.",
               "<b>Obraty:</b> sextakord a kvartsextakord; u septakordů kvintsextakord, terckvartakord, sekundakord."])
    d.save()


# ============================================================== 5. KVINTOVÝ KRUH

def deck_circle(path):
    d = Deck(path, "Kvintový kruh", colors.HexColor("#00798c"), colors.HexColor("#edae49"),
             colors.HexColor("#0f2a30"), 3 + len(KEYS) + 1)
    d.cover("Tóniny", "Kvintový kruh, tónina po tónině",
            "Předznamenání, paralelní moll a kam modulovat – pro všech dvanáct tónin.",
            lambda s: mini_circle(s, W / 2, 124, 88))

    s = d.new()
    s.kicker("Mapa")
    y = s.title("Dvanáct tónin po kvintách", H - 40, 28)
    mini_circle(s, W / 2, y - 120, 110)
    s.body("Po směru hodinových ručiček: o kvintu výš, tedy o křížek víc nebo o béčko méně. Proti směru: o kvintu "
           "níž, tedy o béčko víc nebo o křížek méně. Uvnitř jsou paralelní mollové tóniny.", y - 246, 11.5)
    d.done(s)

    s = d.new()
    s.kicker("Předznamenání")
    y = s.title("Pořadí křížků a béček", H - 40, 28)
    y = s.body("Křížky přibývají po kvintách nahoru, béčka po kvintách dolů:", y - 8)
    s.label("Křížky ♯", M, y - 14)
    y = s.strip([{"big": n} for n in ["fis", "cis", "gis", "dis", "ais", "eis", "his"]], y - 20, h=36, big_size=13)
    s.label("Béčka ♭", M, y - 8)
    y = s.strip([{"big": n} for n in ["b", "es", "as", "des", "ges", "ces", "fes"]], y - 14, h=36, big_size=13)
    s.body("Tónina s víc než sedmi znaménky je jen teoretická – místo Dis dur se píše Es dur. <b>Paralelní moll</b> "
           "má stejné předznamenání (C dur – a moll), <b>stejnojmenná moll</b> stejnou tóniku (C dur – c moll).",
           y - 6, 11)
    d.done(s)

    for key in KEYS:
        f = circle_of_fifths(key)["focus"]
        s = d.new()
        s.kicker("Kvintový kruh")
        s.title(key_cs(key), H - 40, 38)
        mini_circle(s, W / 2, H - 174, 90, highlight=key)
        acc_list = ", ".join(cz(a, lower=True) for a in f["accidentals"])
        y = s.grid([("předznamenání", signature_cs(f["fifths"]), acc_list or "žádná znaménka"),
                    ("paralelní moll", key_cs(f["relative_minor"].split()[0], True), "stejné předznamenání"),
                    ("dominantová tónina", key_cs(f["dominant"]), "o kvintu výš"),
                    ("subdominantová tónina", key_cs(f["subdominant"]), "o kvintu níž")],
                   H - 276, cols=2, h=48, big_size=14)
        rel = [key_cs(k["key"].split()[0], "minor" in k["key"]) for k in f["closely_related_keys"]]
        s.body("<b>Příbuzné tóniny:</b> " + ", ".join(rel), y - 2, 10.5, INK)
        d.done(s)

    d.closing(["<b>O kvintu výš</b> = o křížek víc (o béčko méně), <b>o kvintu níž</b> = o béčko víc (o křížek méně).",
               "<b>Paralelní moll</b> leží o malou tercii níž (C dur → a moll).",
               "<b>Modulace</b> do sousední tóniny na kruhu zní přirozeně a hladce."])
    d.save()


# ============================================================== 6. NEGATIVNÍ HARMONIE

def clock(s: Slide, cx, cy, r, tonic: str):
    """The 12 tones from the tonic in a circle, with the negative-harmony axis and the mirror pairs."""
    c = s.c
    t = parse_note(tonic)
    chromatic = {parse_note(n.name).pitch_class: n.name for n in scale_notes(SCALES["chromatic"], t)[:-1]}
    from midi_composer_mcp.notes import spell_pitch_class
    for pc in list(chromatic):  # chromatic tones: the simple spelling, never a double accidental
        chromatic[pc] = spell_pitch_class(pc, prefer_flats=True).name
    for n in scale_notes(SCALES["major"], t)[:-1]:  # the key's own tones keep the key's spelling
        chromatic[n.pitch_class] = n.name

    def pos(rel, rad):
        ang = math.radians(90 - (rel - 3.5) * 30)  # axis vertical, between the minor and major 3rd
        return cx + rad * math.cos(ang), cy + rad * math.sin(ang)

    c.setStrokeColor(tint(s.d.accent, 0.5))
    c.setLineWidth(1.0)
    for rel in range(12):
        m = (7 - rel) % 12
        if rel < m:
            x1, y1 = pos(rel, r * 0.8)
            x2, y2 = pos(m, r * 0.8)
            c.line(x1, y1, x2, y2)
    c.setStrokeColor(s.d.accent2)
    c.setLineWidth(2)
    c.setDash(4, 3)
    c.line(cx, cy + r + 8, cx, cy - r - 8)
    c.setDash()
    for rel in range(12):
        x, y = pos(rel, r)
        lab = cz(chromatic[(t.pitch_class + rel) % 12], lower=True)
        tonic_dot = rel == 0
        c.setFillColor(s.d.accent if tonic_dot else colors.white)
        c.setStrokeColor(LINE)
        c.circle(x, y, r * 0.18, fill=1, stroke=1)
        size = r * (0.14 if len(lab) <= 3 else 0.11)
        c.setFont(HEAVY if tonic_dot else DEMI, size)
        c.setFillColor(colors.white if tonic_dot else INK)
        c.drawCentredString(x, y - size * 0.35, lab)


def mirror_name(notes_, key):
    m = negative_harmony(notes_, key)["notes"]
    found = match_chords(m, include_partial=False, limit=1)["matches"]
    if not found:
        return names_cs(m), m
    root = parse_note(found[0]["root"])
    ctype = CHORDS[found[0]["chord_type"]]
    name = chord_cs(found[0]["root"] + ctype.symbol) + practical_cs(root, ctype.name == "minor")
    return name, m


def deck_negative(path):
    d = Deck(path, "Negativní harmonie", colors.HexColor("#3d348b"), colors.HexColor("#f18701"),
             colors.HexColor("#1a1733"), 3 + len(KEYS) + 1)
    d.cover("Reharmonizace", "Negativní harmonie",
            "Zrcadli akordy podle osy tóniny – dur se změní v moll a tah k tónice zůstane.",
            lambda s: clock(s, W / 2, 146, 80, "C"))

    s = d.new()
    s.kicker("Princip")
    y = s.title("Zrcadlo mezi velkou a malou tercií", H - 40, 26)
    y = s.body("Osa leží uprostřed mezi tónikou a její kvintou – v C dur mezi <b>e</b> a <b>es</b>. Každý tón se "
               "překlopí na druhou stranu osy: e ↔ es, g ↔ c, d ↔ f, a ↔ b, h ↔ as.", y - 8)
    clock(s, W / 2, y - 122, 98, "C")
    d.done(s)

    s = d.new()
    s.kicker("Pravidlo")
    y = s.title("Dur se mění v moll, tah k tónice zůstává", H - 40, 26)
    y = s.body("Durový akord se v zrcadle změní v mollový. Tónika C přejde v Cmi, dominanta G v Fmi a subdominanta F "
               "v Gmi – D a S si vymění role. Tah k tónice ale zůstane: Fmi vede k tónice plagálně.", y - 8)
    items = []
    for sym, rn in (("C", "I"), ("F", "IV"), ("G", "V"), ("Am", "vi")):
        root, ctype, _ = parse_chord_symbol(sym)
        tones = [t.name for t in chord_notes(ctype, root)]
        name, m = mirror_name(tones, "C")
        items.append({"top": f"{rn}  {chord_cs(sym)}  v zrcadle", "big": name,
                      "cap": f"{names_cs(tones)}  ↦  {names_cs(m)}"})
    s.grid(items, y - 12, cols=2, h=66, big_size=24)
    d.done(s)

    for key in KEYS:
        rows = diatonic_chords(key, "major")["chords"]
        s = d.new()
        s.kicker("Negativní harmonie")
        y = s.title(f"V {key_cs(key)}", H - 40, 32)
        clock(s, W / 2, y - 96, 76, key)
        items = []
        for i in (0, 3, 4, 5):
            ch = rows[i]
            name, m = mirror_name(ch["notes"], key)
            hint = ""
            if " (= " in name:
                name, hint = name.split(" (= ")
                hint = f"  (= {hint}"
            items.append({"top": f'{roman_sym(ch["roman"])}  {chord_cs(ch["symbol"])}  v zrcadle', "big": name,
                          "cap": names_cs(m) + hint})
        s.grid(items, y - 192, cols=2, h=54, big_size=19)
        d.done(s)

    d.closing(["<b>Osa</b> leží mezi velkou a malou tercií tóniky.",
               "<b>Vzorec:</b> tón o <i>r</i> půltónů nad tónikou se zrcadlí na tón o 7 − <i>r</i> půltónů nad ní.",
               "<b>Dur ↔ moll:</b> tónika zůstane tónikou, D a S si vymění role (G → Fmi, F → Gmi). "
               "Vyzkoušej to na posledním akordu fráze."])
    d.save()


# ============================================================== 7. KONTRAPUNKT

def piano_roll(s: Slide, x, y, w, h, r, bar_beats=4.0):
    """Cantus (grey) and counterpoint (accent) as bars on a pitch/time grid."""
    c = s.c
    cf = [parse_note(n).midi for n in r["cantus"]]
    cp = [parse_note(n).midi for n in r["counterpoint"]]
    lo, hi = min(cf + cp) - 1, max(cf + cp) + 1
    total = len(cf) * bar_beats
    grid = r["counterpoint_step_beats"]
    c.setFillColor(CARD)
    c.setStrokeColor(LINE)
    c.roundRect(x, y - h, w, h, 6, fill=1, stroke=1)

    def px(beat):
        return x + 7 + (w - 14) * beat / total

    def py(m):
        return y - h + 7 + (h - 14) * (m - lo) / (hi - lo)

    c.setStrokeColor(tint(LINE, 0.3))
    c.setLineWidth(0.6)
    for b in range(1, len(cf)):
        c.line(px(b * bar_beats), y - h + 5, px(b * bar_beats), y - 5)
    bh = max(3.4, (h - 14) / (hi - lo) * 0.85)
    for i, m in enumerate(cf):
        c.setFillColor(colors.HexColor("#c9c3b8"))
        c.roundRect(px(i * bar_beats) + 1, py(m) - bh / 2, px(bar_beats) - px(0) - 2, bh, 1.6, fill=1, stroke=0)
    onsets = [i for i, ch in enumerate(r["counterpoint_rhythm"]) if ch == "O"]
    for k, i in enumerate(onsets):
        end = onsets[k + 1] if k + 1 < len(onsets) else len(r["counterpoint_rhythm"])
        c.setFillColor(s.d.accent)
        c.roundRect(px(i * grid) + 1, py(cp[k]) - bh / 2, px((end - i) * grid) - px(0) - 2, bh, 1.6, fill=1, stroke=0)


def motion_demo(s: Slide, x, y, w, h, upper, lower, good: bool, caption: str):
    c = s.c
    c.setFillColor(CARD)
    c.setStrokeColor(LINE)
    c.roundRect(x, y - h, w, h, 6, fill=1, stroke=1)
    lo, hi = min(upper + lower) - 1, max(upper + lower) + 1

    def px(i):
        return x + 22 + (w - 44) * i / (len(upper) - 1)

    def py(m):
        return y - h + 26 + (h - 44) * (m - lo) / (hi - lo)

    for line, col in ((upper, s.d.accent), (lower, s.d.accent2)):
        c.setStrokeColor(col)
        c.setLineWidth(1.8)
        for i in range(len(line) - 1):
            c.line(px(i), py(line[i]), px(i + 1), py(line[i + 1]))
        c.setFillColor(col)
        for i, m in enumerate(line):
            c.circle(px(i), py(m), 4, fill=1, stroke=0)
    check_mark(c, x + w - 14, y - 14, good)
    c.setFont(DEMI, 9)
    c.setFillColor(INK)
    c.drawCentredString(x + w / 2, y - h + 9, caption)


def deck_counterpoint(path):
    cf = ["C5", "D5", "E5", "F5", "E5", "D5", "C5"]
    species = {1: ("1. druh", "Nota proti notě"), 2: ("2. druh", "Dvě noty proti jedné"),
               3: ("3. druh", "Čtyři noty proti jedné"), 4: ("4. druh", "Synkopy a průtahy"),
               5: ("5. druh", "Květnatý kontrapunkt")}
    d = Deck(path, "Kontrapunkt", colors.HexColor("#f26419"), colors.HexColor("#33658a"),
             colors.HexColor("#1e2a33"), 1 + 4 + 5 + 1)
    d.cover("Kontrapunkt", "Fuxova pravidla kontrapunktu",
            "Metoda z roku 1725 (Gradus ad Parnassum), podle které se dodnes učí samostatné vedení hlasů.",
            lambda s: piano_roll(s, M, 196, W - 2 * M, 118, species_counterpoint(cf, "C", "major", species=3)))

    s = d.new()
    s.kicker("Pravidlo 1")
    y = s.title("Na těžké době konsonance", H - 40, 28)
    y = s.body("Na těžkých dobách jen konsonance (jedinou výjimkou je připravený průtah ve 4. druhu). Ve dvojhlasu "
               "se i čistá kvarta počítá mezi disonance.", y - 8)
    s.label("Konsonance", M, y - 12, GOOD)
    y = s.grid([("dokonalé", "č1 · č5 · č8", "čistá prima, kvinta a oktáva"),
                ("nedokonalé", "m3 · v3 · m6 · v6", "malé a velké tercie a sexty")], y - 18, cols=1, h=48,
               big_size=18, top_col=GOOD)
    s.label("Disonance", M, y - 4, D_COL)
    s.grid([("", "m2 · v2 · č4", "sekundy a kvarta"), ("", "zv4 · zm5 · m7 · v7", "tritón a septimy")],
           y - 10, cols=1, h=44, big_size=16)
    d.done(s)

    s = d.new()
    s.kicker("Pravidlo 2")
    y = s.title("Žádné paralelní kvinty a oktávy", H - 40, 26)
    y = s.body("Když dva hlasy postoupí souběžně z čisté kvinty (oktávy) do další čisté kvinty (oktávy), splynou "
               "v jeden. Nejlepší je <b>protipohyb</b>, dobrý je <b>boční pohyb</b>.", y - 8)
    half = (W - 2 * M - 8) / 2
    motion_demo(s, M, y - 12, half, 126, [67, 69], [60, 62], False, "paralelní kvinty")
    motion_demo(s, M + half + 8, y - 12, half, 126, [67, 65], [60, 62], True, "protipohyb")
    s.body("Hlídej i <b>skryté kvinty a oktávy</b> (přímý pohyb do čisté kvinty či oktávy) a <b>akcentové kvinty "
           "a oktávy</b> (na dvou po sobě jdoucích těžkých dobách).", y - 150, 11.5)
    d.done(s)

    s = d.new()
    s.kicker("Pravidlo 3")
    y = s.title("Disonance musí mít důvod", H - 40, 28) - 10
    for head, txt in (("Průchodný tón", "na lehké době, přijde i odejde krokem stejným směrem"),
                      ("Střídavý tón", "odbočí o krok a vrátí se (až od 3. druhu)"),
                      ("Průtah", "příprava v konsonanci, ligatura přes těžkou dobu, rozvedení krokem dolů: "
                                 "7–6, 4–3 a 9–8 nad cantem, 2–3 pod ním")):
        s.card(M, y, W - 2 * M, 72)
        s.c.setFillColor(d.accent)
        s.c.roundRect(M, y - 72, 5, 72, 2, fill=1, stroke=0)
        s.text(head, M + 16, y - 12, W - 2 * M - 28, pstyle(15, HEAVY, INK, 17))
        s.text(txt, M + 16, y - 32, W - 2 * M - 28, pstyle(11, MED, SUB, 13))
        y -= 80
    d.done(s)

    s = d.new()
    s.kicker("Pravidlo 4")
    y = s.title("Zpěvná melodie", H - 40, 28)
    y = s.body("Hlas se má dát zazpívat: hlavně kroky, skoky malé a výjimečné. Zakázané melodické skoky:", y - 8)
    y = s.grid([("", "tritón", "zv4 · zm5"), ("", "septima", "m7 · v7"),
                ("", "zvětšené", "a zmenšené intervaly"), ("", "přes oktávu", "větší než č8")],
               y - 10, cols=2, h=54, big_size=18, big_col=D_COL)
    s.body("<b>Začátek:</b> dokonalá konsonance (je-li kontrapunkt pod cantem, jen č1 nebo č8). <b>Konec:</b> vždy "
           "č1 nebo č8 – poslední tón je tónika a přichází se k ní <b>krokem</b>.", y - 2, 11.5)
    d.done(s)

    for sp in (1, 2, 3, 4, 5):
        r = species_counterpoint(cf, "C", "major", species=sp)
        s = d.new()
        s.kicker("Jeden cantus, pět druhů")
        title = species[sp][1]
        y = s.title(title, H - 40, min(28, (W - 2 * M - 2) / pdfmetrics.stringWidth(title, HEAVY, 1)))
        s.label(species[sp][0], M, y - 12)
        piano_roll(s, M, y - 20, W - 2 * M, 176, r)
        y -= 208
        y = s.body("<b>Cantus firmus</b> (šedě): " + names_cs(r["cantus"]), y, 10.5) - 1
        line = "Kontrapunkt (barevně): " + names_cs(r["counterpoint"])
        fit = (W - 2 * M - 4) / pdfmetrics.stringWidth(line, MED, 1)
        y = s.body("<b>Kontrapunkt</b> (barevně): " + names_cs(r["counterpoint"]), y, min(10.5, fit)) - 1
        onsets = [i for i, ch in enumerate(r["counterpoint_rhythm"]) if ch == "O"]
        per_bar = []
        for bar, cantus_note in enumerate(r["cantus"]):
            beat = bar * r["cantus_step_beats"]
            k = max(j for j, i in enumerate(onsets) if i * r["counterpoint_step_beats"] <= beat + 1e-9)
            per_bar.append(interval_between_cs(cantus_note, r["counterpoint"][k]))
        s.body("<b>Intervaly na těžkých dobách:</b> " + "  ".join(per_bar), y, 10, MUTED)
        d.done(s)

    d.closing(["<b>Těžká doba</b> = konsonance (výjimka: připravený průtah); jinak disonance jen jako průchodný "
               "či střídavý tón na lehké době.",
               "<b>Žádné paralelní kvinty a oktávy</b>, přednost má protipohyb.",
               "<b>Zpěvná linka</b> a závěr na tónice krokem."])
    d.save()


# ============================================================== 8. EUKLIDOVSKÉ RYTMY

def necklace(s: Slide, cx, cy, r, pattern, label="", sub=""):
    c = s.c
    n = len(pattern)
    pts = [(cx + r * math.cos(math.radians(90 - i * 360 / n)), cy + r * math.sin(math.radians(90 - i * 360 / n)), ch)
           for i, ch in enumerate(pattern)]
    hits = [(x, y) for x, y, ch in pts if ch != "."]
    c.setStrokeColor(LINE)
    c.setLineWidth(0.8)
    c.circle(cx, cy, r, fill=0, stroke=1)
    if len(hits) == 2:
        c.setStrokeColor(s.d.accent)
        c.setLineWidth(1.4)
        c.line(*hits[0], *hits[1])
    elif len(hits) > 2:
        c.setFillColor(tint(s.d.accent, 0.78))
        c.setStrokeColor(s.d.accent)
        c.setLineWidth(1.4)
        p = c.beginPath()
        p.moveTo(*hits[0])
        for h in hits[1:]:
            p.lineTo(*h)
        p.close()
        c.drawPath(p, fill=1, stroke=1)
    dot = min(r * 0.17, 7)
    for x, y, ch in pts:
        c.setFillColor(s.d.dark if ch == "O" else s.d.accent if ch == "o" else colors.white)
        c.setStrokeColor(s.d.accent if ch != "." else LINE)
        c.setLineWidth(1)
        c.circle(x, y, dot, fill=1, stroke=1)
    if label:
        c.setFont(HEAVY, 14)
        c.setFillColor(INK)
        c.drawCentredString(cx, cy - r - 22, label)
    if sub:
        c.setFont(MED, 9)
        c.setFillColor(SUB)
        c.drawCentredString(cx, cy - r - 34, sub)


EUCLID = [((3, 8, 0), "tresillo · Kuba"), ((5, 8, 0), "cinquillo · Kuba"), ((2, 5, 0), "khafif-e-ramal · Persie"),
          ((3, 7, 0), "račenica · Bulharsko"), ((4, 9, 0), "aksak · Turecko"),
          ((5, 12, 0), "tleskání Vendů · Jižní Afrika"), ((7, 12, 0), "zvonkový vzorec · západní Afrika"),
          ((5, 16, 6), "bossa nova · Brazílie")]  # bossa nova = E(5,16) rotated to 3-3-4-3-3


def deck_euclid(path):
    d = Deck(path, "Euklidovské rytmy", colors.HexColor("#f26419"), colors.HexColor("#33658a"),
             colors.HexColor("#22223b"), 2 + 2 + 1 + 1)
    d.cover("Rytmus", "Euklidovské rytmy",
            "Rozlož <i>k</i> úderů co nejrovnoměrněji do <i>n</i> pulzů – a vyjdou rytmy z celého světa.",
            lambda s: necklace(s, W / 2, 146, 72, euclidean_rhythm(5, 8)["pattern"]))

    s = d.new()
    s.kicker("Princip")
    y = s.title("Rovnoměrně rozložené údery", H - 40, 28)
    y = s.body("Tři údery rozložené co nejrovnoměrněji do osmi pulzů (osmin jednoho taktu 4/4; Bjorklundův algoritmus) "
               "dají kubánské <b>tresillo</b>. Tmavý bod je přízvuk, barevný úder, prázdný pomlka.", y - 8)
    necklace(s, W / 2, y - 112, 84, euclidean_rhythm(3, 8)["pattern"], "E(3, 8)", "tresillo")
    d.done(s)

    for start in (0, 4):
        s = d.new()
        s.kicker("Katalog")
        s.title("Klasické euklidovské rytmy", H - 40, 26)
        for j, ((k, n, rot), name) in enumerate(EUCLID[start:start + 4]):
            rr, cc = divmod(j, 2)
            necklace(s, M + (W - 2 * M) * (0.25 + 0.5 * cc), H - 150 - rr * 150, 44,
                     euclidean_rhythm(k, n, rot)["pattern"], f"E({k}, {n})", name)
        d.done(s)

    s = d.new()
    s.kicker("Groove")
    y = s.title("Rytmické vzorce pro bicí", H - 40, 26) - 8
    groove_cs = {"four_on_floor": "four-on-the-floor", "backbeat": "backbeat (2 a 4)", "offbeat": "offbeat",
                 "eighths": "osminy", "sixteenths": "šestnáctiny", "tresillo": "tresillo", "cinquillo": "cinquillo",
                 "habanera": "habanera", "son_clave_32": "son clave 3-2"}
    for name, (pattern, _step, _desc) in list(GROOVES.items())[:9]:
        s.c.setFont(DEMI, 10)
        s.c.setFillColor(INK)
        s.c.drawString(M, y - 10, groove_cs[name])
        cell = (W - 2 * M - 84) / 16
        cell_len = len(pattern)
        pattern = pattern * (16 // cell_len)  # an 8-step cell repeats to fill the bar (second copy lighter)
        for i, ch in enumerate(pattern):
            repeat = i >= cell_len
            on = d.dark if ch == "O" else d.accent if ch == "o" else colors.white
            s.c.setFillColor(tint(on, 0.55) if repeat and ch != "." else on)
            s.c.setStrokeColor(LINE if ch == "." else (tint(d.accent, 0.5) if repeat else d.accent))
            s.c.roundRect(M + 84 + i * cell + 0.6, y - 14, cell - 1.2, 12, 2, fill=1, stroke=1)
        y -= 26
    s.body("Tmavé pole = přízvuk, barevné = úder, prázdné = pomlka. Pole = šestnáctina, řádek = takt 4/4 "
           "(osmipolové buňky se opakují, podruhé světleji). Four-on-the-floor = kopák na každou dobu, "
           "offbeat = údery mezi dobami.", y - 2, 10)
    d.done(s)

    d.closing(["<b>E(3, 8)</b> tresillo · <b>E(5, 8)</b> cinquillo · <b>E(5, 16)</b> bossa nova.",
               "Jeden vzorec na kopák, druhý na shaker – a groove je na světě.",
               "Posuň vzorec o pár pulzů (rotace) a vznikne nový."])
    d.save()


DECKS = [
    ("01-durove-stupnice.pdf", deck_scales), ("02-cirkevni-mody.pdf", deck_modes),
    ("03-akordy-v-tonine.pdf", deck_diatonic), ("04-akordove-znacky.pdf", deck_chords),
    ("05-kvintovy-kruh.pdf", deck_circle), ("06-negativni-harmonie.pdf", deck_negative),
    ("07-kontrapunkt.pdf", deck_counterpoint), ("08-euklidovske-rytmy.pdf", deck_euclid),
]


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out, exist_ok=True)
    for name, fn in DECKS:
        path = os.path.join(out, name)
        fn(path)
        print(path)


if __name__ == "__main__":
    main()
