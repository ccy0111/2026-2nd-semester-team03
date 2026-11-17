"""Deterministic staff-notation SVG preview, not a full music engraving engine."""
import math

# Chromatic pitch -> natural letter index and sharp. C major spelling only.
SPELLING = ((0,False),(0,True),(1,False),(1,True),(2,False),(3,False),
            (3,True),(4,False),(4,True),(5,False),(5,True),(6,False))


def render_score(notes):
    """Quantize at 120 BPM to sixteenths; ties express longer durations.

    Instruments have separate treble staffs. Polyphonic rests, voice allocation,
    key/tempo estimation and percussion engraving are intentionally not inferred.
    """
    if len(notes) > 20000:
        raise ValueError("Score preview exceeds 20000 notes")
    groups = {}
    for program, pitch, start, stop in notes:
        if not all(math.isfinite(x) for x in (start, stop)) or stop <= start:
            continue
        onset = max(0, round(start * 8))
        end = max(onset + 1, round(stop * 8))
        # one unit = sixteenth, 16 units = a 4/4 measure
        pieces = []
        while onset < end:
            remaining = min(end - onset, 16 - onset % 16)
            size = max(x for x in (1,2,4,8,16) if x <= remaining)
            pieces.append((onset, size))
            onset += size
        for index, (onset, size) in enumerate(pieces):
            groups.setdefault((program, onset // 64), []).append(
                (pitch, onset % 64, size, index > 0, index < len(pieces)-1))
    # Reserve the complete pitch range (including stems/ledger lines) per system.
    layout = []
    cursor = 100
    for key, entries in sorted(groups.items()):
        offsets = [40 - (((p // 12 - 1) * 7 + SPELLING[p % 12][0]) - 30) * 5
                   for p, *_ in entries]
        above = max(0, 40 - min(offsets))
        below = max(40, max(offsets) + 24)
        top = cursor + above + 25
        layout.append((key, entries, top, cursor))
        cursor = top + below + 40
    height = max(180, cursor)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="{height}" viewBox="0 0 1100 {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<g fill="black" font-family="sans-serif" font-size="13">',
           '<text x="25" y="25">WebScore staff preview: 120 BPM, 4/4, sixteenth-note quantization</text>',
           '<text x="25" y="46">C-major spelling; no inferred keys, tempo changes, rests or independent voices. Percussion shown by MIDI pitch.</text>']
    if not groups:
        out.append('<text x="25" y="100">No notes detected.</text>')
    for (program, system), entries, top, label_y in layout:
        bottom = top + 40
        out.append(f'<text x="25" y="{label_y}">MIDI program {program}; measures {system*4+1}–{system*4+4}</text>')
        for line in range(5):
            y = top + line * 10
            out.append(f'<path d="M 55 {y} H 1075" stroke="black" fill="none"/>')
        out.append(f'<text x="60" y="{bottom-3}" font-size="42">𝄞</text><text x="92" y="{top+16}">4</text><text x="92" y="{top+34}">4</text>')
        for bar in range(5):
            x = 110 + bar * 240
            out.append(f'<path d="M {x} {top} V {bottom}" stroke="black"/>')
        for pitch, onset, size, tied_in, tied_out in entries:
            letter, sharp = SPELLING[pitch % 12]
            step = (pitch // 12 - 1) * 7 + letter
            y = bottom - (step - 30) * 5  # E4 is bottom treble line
            x = 120 + onset * 15
            fill = 'white' if size >= 8 else 'black'
            for ledger in range(bottom + 10, int(y) + 1, 10):
                out.append(f'<path d="M {x-10} {ledger} H {x+10}" stroke="black"/>')
            for ledger in range(top - 10, int(y) - 1, -10):
                out.append(f'<path d="M {x-10} {ledger} H {x+10}" stroke="black"/>')
            if sharp:
                out.append(f'<text x="{x-18}" y="{y+4}">♯</text>')
            out.append(f'<ellipse cx="{x}" cy="{y}" rx="6" ry="4" fill="{fill}" stroke="black" transform="rotate(-20 {x} {y})"/>')
            if size < 16:
                out.append(f'<path d="M {x+5} {y} V {y-30}" stroke="black"/>')
                for flag in range(2 if size == 1 else 1 if size == 2 else 0):
                    fy = y-30+flag*7
                    out.append(f'<path d="M {x+5} {fy} q 13 5 6 15" stroke="black" fill="none"/>')
            if tied_out:
                reach = min(size*15-5, 1070-x)
                out.append(f'<path d="M {x+2} {y+8} q {reach/2} 12 {reach} 0" stroke="black" fill="none"/>')
            if tied_in and onset == 0:
                out.append(f'<path d="M {x-12} {y+8} q 6 8 12 0" stroke="black" fill="none"/>')
    out.append('</g></svg>')
    return ''.join(out).encode('utf-8')
