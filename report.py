"""Create a printable PDF report from Stage 1 event records (no trained model implied)."""
from collections import Counter
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import simpleSplit
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
from reportlab.lib.pagesizes import A4, landscape

from .video_io import format_timestamp
from .presentation import describe_event, display_label


def create_event_report(path: str, video_filename: str, duration_sec: float, events, *, analysis_date=None):
    """Write a landscape A4 PDF. Values are detector output, not ground-truth labels."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    date = analysis_date or datetime.now().strftime('%d %b %Y')
    doc = SimpleDocTemplate(str(target), pagesize=landscape(A4), rightMargin=30, leftMargin=30,
                            topMargin=26, bottomMargin=30)
    title = ParagraphStyle('Title', fontName='Times-Bold', fontSize=23, leading=27, alignment=TA_CENTER)
    heading = ParagraphStyle('Heading', fontName='Times-Bold', fontSize=15, leading=19, spaceBefore=12, spaceAfter=8)
    cell = ParagraphStyle('Cell', fontName='Helvetica', fontSize=8, leading=11)
    header = ParagraphStyle('Header', parent=cell, fontName='Helvetica-Bold')
    small = ParagraphStyle('Small', fontName='Helvetica', fontSize=9, leading=14)
    story = [Paragraph('Event Detection Report', title), Spacer(1, 16)]
    metadata = [
        [Paragraph('<b>Object</b>', small), Paragraph('Cup', small), Paragraph('<b>Analysis Date</b>', small), Paragraph(str(date), small)],
        [Paragraph('<b>Video</b>', small), Paragraph(video_filename.replace('&','&amp;').replace('<','&lt;'), small), Paragraph('<b>Model Output</b>', small), Paragraph('Rule-based cup movement detection (Stage 1)', small)],
        [Paragraph('<b>Total Events</b>', small), Paragraph(str(len(events)), small), Paragraph('<b>Video Duration</b>', small), Paragraph(format_timestamp(duration_sec), small)],
    ]
    mt = Table(metadata, colWidths=[96, 275, 106, 300], hAlign='LEFT')
    mt.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
    story.extend([mt, Spacer(1, 10), Paragraph('1. Detected Events', heading)])
    columns = ['#', 'Start Time', 'End Time', 'Duration (s)', 'Event Type', 'Detection method', 'Description']
    data = [[Paragraph(c, header) for c in columns]]
    for i, event in enumerate(events, 1):
        description = describe_event(event.label).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
        data.append([Paragraph(str(i),cell), Paragraph(format_timestamp(event.start_time),cell),
                     Paragraph(format_timestamp(event.end_time),cell),
                     Paragraph(f'{event.duration:.2f}',cell), Paragraph(display_label(event.label).title(),cell),
                     Paragraph('Rule-based',cell), Paragraph(description,cell)])
    if not events:
        data.append([Paragraph('No events detected',cell)] + ['']*6)
    table = Table(data, colWidths=[28,91,91,70,100,75,322], repeatRows=1, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E8EAED')),
        ('GRID',(0,0),(-1,-1),0.5,colors.HexColor('#7C8289')),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
        ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
    ]))
    story.extend([table, Paragraph('2. Summary', heading)])
    counts = Counter(e.label for e in events)
    summaries = [f'A total of {len(events)} event(s) were detected in the video.']
    summaries += [f'{n} event(s): {display_label(label).lower()}.' for label, n in sorted(counts.items())]
    if not events:
        summaries.append('No motion events were recorded by the current detection rules.')
    summaries.append('Events are estimated from tracked position changes, not from a trained machine-learning classifier.')
    summaries.append('This prototype measures visible position changes; it cannot verify picking up, putting down, or contact with the table.')
    for line in summaries:
        story.append(Paragraph('&#8226; ' + line.replace('&','&amp;').replace('<','&lt;'),small))
        story.append(Spacer(1,4))
    def footer(canvas, document):
        canvas.setFont('Helvetica',8)
        canvas.setFillColor(colors.grey)
        canvas.drawRightString(landscape(A4)[0]-30,17,f'Page {document.page}')
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return target
