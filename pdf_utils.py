"""PDF worksheet generation for WhatsPrep.

One public function: build_pdf(). It writes an A4 worksheet with the
questions up front and the answer key on its own final page, so a parent
can print everything except the last page for the child.

reportlab Paragraphs parse a small XML markup, so all platform text is
escaped before it goes in — a question containing "<" or "&" must never
crash the build.
"""

from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                PageBreak, HRFlowable)

_OPTION_LETTERS = "ABCDEFGH"


def _esc(value) -> str:
    return escape(str(value if value is not None else ""))


def build_pdf(path: str, child_name: str, level: str, topic_names: list,
              difficulty: str, questions: list) -> None:
    """Write the worksheet PDF to `path`.

    questions: list of dicts, each with:
      question (str)            required
      options  (list[str])      optional — MCQ choices, already worded
      answer   (str)            required for the key page
    """
    doc = SimpleDocTemplate(
        path, pagesize=A4,
        title=f"WhatsPrep Practice - {child_name}",
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm,
    )
    st = getSampleStyleSheet()
    h_title = ParagraphStyle("wp_title", parent=st["Title"], fontSize=20,
                             leading=24, spaceAfter=2)
    meta = ParagraphStyle("wp_meta", parent=st["Normal"], fontSize=10.5,
                          leading=14, textColor="#555555")
    note = ParagraphStyle("wp_note", parent=st["Normal"], fontSize=9,
                          leading=12, textColor="#777777")
    q_style = ParagraphStyle("wp_q", parent=st["Normal"], fontSize=11,
                             leading=16, spaceBefore=10)
    opt_style = ParagraphStyle("wp_opt", parent=st["Normal"], fontSize=11,
                               leading=16, leftIndent=16)
    key_style = ParagraphStyle("wp_key", parent=st["Normal"], fontSize=11,
                               leading=18)

    topics_line = ", ".join(_esc(t) for t in topic_names if t) or "Mixed topics"

    story = [
        Paragraph(f"WhatsPrep Practice — {_esc(child_name)}", h_title),
        Paragraph(f"{_esc(level)} &middot; {topics_line} &middot; "
                  f"{_esc(difficulty)} &middot; {len(questions)} questions", meta),
        Spacer(1, 4),
        Paragraph("Answer key is on the last page. This paper is for "
                  "self-practice: it is not marked by WhatsPrep and no "
                  "progress report is generated for it.", note),
        HRFlowable(width="100%", thickness=0.7, color="#DDDDDD",
                   spaceBefore=8, spaceAfter=4),
    ]

    for i, q in enumerate(questions, 1):
        story.append(Paragraph(f"<b>Q{i}.</b> {_esc(q.get('question'))}", q_style))
        options = q.get("options") or []
        if options:
            for j, opt in enumerate(options):
                letter = _OPTION_LETTERS[j] if j < len(_OPTION_LETTERS) else str(j + 1)
                text = _esc(opt)
                # Don't double-label options the platform already labels ("A) ...").
                if text[:2].upper().rstrip(".)") == letter:
                    story.append(Paragraph(text, opt_style))
                else:
                    story.append(Paragraph(f"({letter}) {text}", opt_style))
        else:
            # Open-ended: leave working space on the page.
            story.append(Spacer(1, 34))

    story.append(PageBreak())
    story.append(Paragraph("Answer Key (for parents)", st["Heading1"]))
    story.append(Paragraph("Cut here or keep this page aside before handing "
                           "the paper over.", note))
    story.append(Spacer(1, 8))
    for i, q in enumerate(questions, 1):
        story.append(Paragraph(f"<b>Q{i}:</b> {_esc(q.get('answer')) or '—'}",
                               key_style))

    doc.build(story)
    