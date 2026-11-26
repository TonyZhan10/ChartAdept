"""Render README figures from reported results and three saved inference cases."""
import json
from pathlib import Path

import cairosvg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'docs/assets'
SOURCE = json.loads((ASSETS / 'sources.json').read_text())
CASES = {key: json.loads((ROOT / 'examples/qualitative' / f'{key}.json').read_text()) for key in SOURCE}
INK = '#183442'
MUTED = '#526773'
TEAL = '#117E83'
BLUE = '#346D9E'
ORANGE = '#AD661E'
BG = '#F5F8FA'
LINE = '#D9E3E8'
FONT = font_manager.findfont('DejaVu Serif')
BOLD = font_manager.findfont(font_manager.FontProperties(family='DejaVu Serif', weight='bold'))
MONO = font_manager.findfont('DejaVu Sans Mono')


def font(size, bold=False, mono=False):
    return ImageFont.truetype(MONO if mono else BOLD if bold else FONT, size)


def wrap(draw, text, face, width):
    lines, current = [], ''
    for word in text.split():
        candidate = f'{current} {word}'.strip()
        if current and draw.textlength(candidate, font=face) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def paragraph(draw, text, xy, width, size=30, color=INK, bold=False, leading=1.32):
    x, y = xy
    face = font(size, bold=bold)
    for line in wrap(draw, text, face, width):
        draw.text((x, y), line, font=face, fill=color, anchor='lt')
        y += round(size * leading)
    return y


def new_case(title, case_id, calls, body_height):
    measure = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    question_lines = wrap(measure, CASES[case_id]['question'], font(30), 1472)
    height = 118 + len(question_lines) * round(30 * 1.32) + 54 + body_height
    canvas = Image.new('RGB', (1600, height), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((1, 1, 1598, height - 2), radius=26, fill='white', outline=LINE, width=3)
    draw.text((64, 44), title, font=font(46, bold=True), fill=INK, anchor='lt')
    label = f'{case_id}  ·  {calls}'
    width = draw.textlength(label, font=font(25))
    draw.text((1536 - width, 56), label, font=font(25), fill=MUTED, anchor='lt')
    bottom = paragraph(draw, CASES[case_id]['question'], (64, 118), 1472, size=30)
    draw.line((64, bottom + 24, 1536, bottom + 24), fill=LINE, width=2)
    return canvas, draw, bottom + 54


def chart(canvas, relative, box, crop_boxes=()):
    x, y, width, height = box
    with Image.open(ASSETS / relative) as original:
        original_size = original.size
        displayed = ImageOps.contain(original.convert('RGB'), (int(width), int(height)), Image.Resampling.LANCZOS)
    left = int(x + (width - displayed.width) / 2)
    top = int(y + (height - displayed.height) / 2)
    canvas.paste(displayed, (left, top))
    draw = ImageDraw.Draw(canvas)
    for index, rectangle in enumerate(crop_boxes):
        bounds = [left + rectangle[0] / original_size[0] * displayed.width,
                  top + rectangle[1] / original_size[1] * displayed.height,
                  left + rectangle[2] / original_size[0] * displayed.width,
                  top + rectangle[3] / original_size[1] * displayed.height]
        draw.rectangle(bounds, outline=(BLUE, ORANGE)[index], width=4)
    return left, top, displayed.width, displayed.height


def answer(draw, case_id, y):
    case = CASES[case_id]
    draw.rounded_rectangle((64, y, 1536, y + 76), radius=14, fill='#EAF5F3')
    draw.text((86, y + 17), 'Answer: ' + case['final_answer'], font=font(38, bold=True), fill=INK, anchor='lt')
    reference = 'Reference: ' + case['reference_answer']
    width = draw.textlength(reference, font=font(27))
    draw.text((1514 - width, y + 26), reference, font=font(27), fill=MUTED, anchor='lt')
    draw.text((64, y + 101), f'CharXiv-RQ · {case_id} · Recorded trajectory, with abridged step descriptions',
              font=font(22), fill=MUTED, anchor='lt')


def direct_case():
    case_id = 'RQ-1969'
    canvas, draw, top = new_case('Direct response', case_id, '0 tool calls', 918)
    draw.text((64, top), 'Original chart', font=font(26, bold=True), fill=MUTED, anchor='lt')
    chart(canvas, SOURCE[case_id]['input'], (64, top + 42, 920, 660))
    y = top + 70
    draw.text((1040, y), 'Observe', font=font(33, bold=True), fill=TEAL, anchor='lt')
    y = paragraph(draw, 'At the same π, the τ = 10 curve lies above the τ = 20 curve.',
                  (1040, y + 58), 450, size=30)
    y += 65
    draw.text((1040, y), 'Answer directly', font=font(33, bold=True), fill=TEAL, anchor='lt')
    y = paragraph(draw, 'A larger τ corresponds to a lower plotted value.', (1040, y + 58), 450, size=30)
    assert y < 940
    answer(draw, case_id, top + 740)
    canvas.save(ASSETS / 'case-direct.png', optimize=True)


def crop_case():
    case_id = 'RQ-2201'
    canvas, draw, top = new_case('Crop only', case_id, '2 Zoom calls', 1288)
    draw.text((64, top), 'Original chart and executed crop regions', font=font(26, bold=True), fill=MUTED, anchor='lt')
    chart(canvas, SOURCE[case_id]['input'], (170, top + 48, 1260, 405), SOURCE[case_id]['crop_boxes'])
    row = top + 510
    for index, (x, color, label) in enumerate([(64, BLUE, '1. Crop: Power'), (834, ORANGE, '2. Crop: FDR')]):
        draw.text((x, row), label, font=font(32, bold=True), fill=color, anchor='lt')
        chart(canvas, SOURCE[case_id]['crops'][index], (x, row + 54, 700, 450))
    paragraph(draw, 'Power ≈ 0.5 → Simes parameter ≈ 4', (64, row + 533), 700, size=29)
    paragraph(draw, 'At the same parameter, FDR ≈ 0.025', (834, row + 533), 700, size=29)
    answer(draw, case_id, top + 1110)
    canvas.save(ASSETS / 'case-crop.png', optimize=True)


def mixed_case():
    case_id = 'RQ-913'
    canvas, draw, top = new_case('Crop + Python', case_id, '2 Zoom calls + 1 Python call', 1288)
    draw.text((64, top), 'Original chart', font=font(26, bold=True), fill=MUTED, anchor='lt')
    chart(canvas, SOURCE[case_id]['input'], (64, top + 50, 900, 730), SOURCE[case_id]['crop_boxes'])
    for index, (x, color, label, low, high, span) in enumerate([
            (1030, BLUE, '1. Crop (a)', '-21.4', '-19.4', '2.0'),
            (1300, ORANGE, '2. Crop (c)', '-20.1', '-19.6', '0.5')]):
        draw.text((x, top), label, font=font(29, bold=True), fill=color, anchor='lt')
        chart(canvas, SOURCE[case_id]['crops'][index], (x, top + 52, 215, 500))
        draw.text((x, top + 583), f'{low} to {high}', font=font(25), fill=INK, anchor='lt')
        draw.text((x, top + 623), f'Range = {span}', font=font(28, bold=True), fill=color, anchor='lt')
    paragraph(draw, 'Read the two sets of endpoints, then compare their ranges.',
              (1030, top + 696), 490, size=28)
    y = top + 868
    draw.text((64, y), '3. Python: actual executed call', font=font(31, bold=True), fill=INK, anchor='lt')
    draw.rounded_rectangle((64, y + 52, 1536, y + 224), radius=14, fill='#F1F4F6')
    call = CASES[case_id]['tool_calls_recorded'][-1]
    for index, line in enumerate(call['arguments']['code'].splitlines()):
        draw.text((86, y + 71 + index * 35), line, font=font(25, mono=True), fill=INK, anchor='lt')
    draw.text((86, y + 181), 'stdout: ' + call['stdout'].strip(), font=font(26, mono=True), fill=TEAL, anchor='lt')
    answer(draw, case_id, top + 1128)
    canvas.save(ASSETS / 'case-crop-python.png', optimize=True)


def overview():
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="640" viewBox="0 0 1600 640">
<title>ChartAdept inference: answer directly or call tools and observe their results</title>
<defs><marker id="arrow" viewBox="0 0 12 12" refX="11" refY="6" markerWidth="14" markerHeight="14" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L12,6 L0,12 Z" fill="#466575"/></marker>
<clipPath id="tools"><rect x="1030" y="350" width="470" height="150" rx="24"/></clipPath></defs>
<rect width="1600" height="640" rx="28" fill="#F5F8FA"/>
<g font-family="DejaVu Serif, Georgia, serif" fill="#183442">
<text x="64" y="75" font-size="38" font-weight="bold">Answer directly or use a tool</text>
<text x="64" y="119" font-size="25" fill="#526773">Each real tool observation returns to the policy before its next decision.</text>
<rect x="64" y="200" width="290" height="250" rx="24" fill="white" stroke="#D9E3E8" stroke-width="2"/>
<text x="209" y="244" text-anchor="middle" font-size="28" font-weight="bold">Chart + question</text>
<path d="M110 280 V366 H305" fill="none" stroke="#637C89" stroke-width="3"/>
<rect x="134" y="327" width="30" height="39" fill="#BCD3DF"/><rect x="184" y="302" width="30" height="64" fill="#80B9C3"/><rect x="234" y="281" width="30" height="85" fill="#117E83"/>
<rect x="95" y="394" width="228" height="34" rx="9" fill="#EDF2F5"/>
<text x="209" y="419" text-anchor="middle" font-size="22">Question</text>
<path d="M354 325 H480" fill="none" stroke="#466575" stroke-width="3" marker-end="url(#arrow)"/>
<rect x="484" y="190" width="330" height="270" rx="28" fill="#183442"/>
<text x="649" y="265" text-anchor="middle" font-size="44" font-weight="bold" fill="white">ChartAdept</text>
<text x="649" y="312" text-anchor="middle" font-size="25" fill="#C7E0E5">Read · reason · choose</text>
<path d="M522 340 H776" stroke="#496674" stroke-width="2"/>
<text x="649" y="383" text-anchor="middle" font-size="25" fill="white">Select the next action</text>
<text x="649" y="424" text-anchor="middle" font-size="22" fill="#C7E0E5">Answer · Zoom · Python</text>
<path d="M814 245 H1048" fill="none" stroke="#466575" stroke-width="3" marker-end="url(#arrow)"/>
<text x="932" y="222" text-anchor="middle" font-size="24">Answer</text>
<rect x="1050" y="185" width="450" height="108" rx="24" fill="#E4F3EE" stroke="#ABD4C5" stroke-width="2"/>
<text x="1275" y="233" text-anchor="middle" font-size="31" font-weight="bold" fill="#176B62">Final answer</text>
<text x="1275" y="270" text-anchor="middle" font-size="22" fill="#176B62">Return the complete response</text>
<path d="M814 392 H1028" fill="none" stroke="#466575" stroke-width="3" marker-end="url(#arrow)"/>
<text x="922" y="368" text-anchor="middle" font-size="24">Act</text>
<g clip-path="url(#tools)"><rect x="1030" y="350" width="235" height="150" fill="#EAF2FB"/><rect x="1265" y="350" width="235" height="150" fill="#FFF1DF"/></g>
<rect x="1030" y="350" width="470" height="150" rx="24" fill="none" stroke="#D9E3E8" stroke-width="2"/>
<text x="1147" y="402" text-anchor="middle" font-size="26" font-weight="bold" fill="#346D9E">Zoom / Crop</text>
<text x="1147" y="446" text-anchor="middle" font-size="22" fill="#346D9E">Returned image</text>
<text x="1382" y="402" text-anchor="middle" font-size="26" font-weight="bold" fill="#AD661E">Python</text>
<text x="1382" y="446" text-anchor="middle" font-size="22" fill="#AD661E">stdout / image</text>
<path d="M1265 500 V580 H649 V462" fill="none" stroke="#466575" stroke-width="3" marker-end="url(#arrow)"/>
<text x="974" y="552" text-anchor="middle" font-size="25" fill="#526773">Observe the actual result</text>
</g></svg>'''
    (ASSETS / 'overview.svg').write_text(svg)
    cairosvg.svg2png(bytestring=svg.encode(), write_to=str(ASSETS / 'overview.png'))


def results():
    data = json.loads((ROOT / 'results/reported-main-table.json').read_text())
    ours = next(x for x in data['models'] if x['key'] == 'chartadept162')
    base = next(x for x in data['models'] if x['key'] == 'qwen3vl4b_agent')
    metrics = data['metric_order']
    labels = ['CharXiv · Reasoning', 'CharXiv · Descriptive', 'ChartQA', 'ChartQAPro',
              'ChartBench · Numerical', 'ChartBench · Binary', 'ChartX',
              'ReachQA · Reasoning', 'ReachQA · Recognition', 'ChartMuseum', 'Average (10 metrics)']
    values = np.array([float(ours['accuracy_percent'][m]) for m in metrics] + [float(ours['Avg10'])])
    reference = np.array([float(base['accuracy_percent'][m]) for m in metrics] + [float(base['Avg10'])])
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 14,
                         'svg.fonttype': 'none', 'axes.edgecolor': LINE,
                         'text.color': INK, 'axes.labelcolor': INK,
                         'xtick.color': MUTED, 'ytick.color': INK})
    fig, ax = plt.subplots(figsize=(12, 9.7), dpi=160)
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=.27, right=.83, top=.83, bottom=.15)
    y = np.arange(len(labels))
    ax.axhspan(9.5, 10.55, color='#EFF7F5', zorder=0)
    ax.barh(y - .19, reference, height=.32, color='#A4B7C7', label='Qwen3-VL-4B-Instruct (agent)', zorder=3)
    ax.barh(y + .19, values, height=.32, color=TEAL, label='ChartAdept (4B)', zorder=3)
    for i, (a, b) in enumerate(zip(values, reference)):
        ax.text(b + .7, i - .19, f'{b:.2f}', va='center', fontsize=12, color=MUTED)
        ax.text(a + .7, i + .19, f'{a:.2f}', va='center', fontsize=12, color=TEAL, fontweight='bold')
        ax.text(111, i, f'+{a-b:.2f}', va='center', ha='center', fontsize=14, color=TEAL, clip_on=False)
    ax.set_yticks(y, labels)
    ax.get_yticklabels()[-1].set_fontweight('bold')
    ax.set_ylim(10.65, -.65)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel('Accuracy (%)', labelpad=12, fontsize=15)
    ax.grid(axis='x', color=LINE, linewidth=.7, zorder=0)
    ax.tick_params(axis='y', length=0, pad=10)
    for name in ('top', 'right', 'left'):
        ax.spines[name].set_visible(False)
    fig.text(.045, .955, 'Seven benchmarks. Ten metrics.', fontsize=25, fontweight='bold')
    fig.text(.045, .915, 'Reported results with the same base model and agent interface', fontsize=16, color=MUTED)
    handles, names = ax.get_legend_handles_labels()
    fig.legend(handles, names, loc='upper left', bbox_to_anchor=(.26, .89), ncol=2, frameon=False, fontsize=13)
    fig.text(.892, .852, 'Gain (pp)', ha='center', fontsize=13, color=MUTED)
    fig.text(.045, .025, 'Source: main-paper Table 2 · 32,500 questions · benchmark-specific decoding settings',
             fontsize=11, color=MUTED)
    fig.savefig(ASSETS / 'results.png', dpi=160, bbox_inches='tight', pad_inches=.16)
    fig.savefig(ASSETS / 'results.svg', bbox_inches='tight', pad_inches=.16)
    svg_path = ASSETS / 'results.svg'
    svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text().splitlines()) + '\n')
    plt.close(fig)


if __name__ == '__main__':
    overview()
    results()
    direct_case()
    crop_case()
    mixed_case()
    print('Rendered overview, reported results, and all three recorded cases.')
