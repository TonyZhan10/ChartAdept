# README figures and attribution

The README's main image is **Figure 1 from the ChartAdept paper**, supplied by the authors. `paper-overview.pdf` is an unchanged copy of the paper's `ChartAdept_Main.pdf`; `paper-overview.png` is a high-resolution display rendering of that PDF. Its illustrative interactions and demonstration review retain the original content, layout, and fonts.

`overview.svg` / `overview.png` remain available as an optional schematic of the released inference interface. `results.svg` / `results.png` visualize the reported values in `results/reported-main-table.json`. These original figures and their rendering code use the repository's Apache 2.0 license.

## Recorded case illustrations

The three `case-*.png` illustrations, the seven chart/crop images in `source/`, and the associated records in `examples/qualitative/` use **CC BY-SA 4.0**, following the [CharXiv dataset license](https://huggingface.co/datasets/princeton-nlp/CharXiv). See the [CC BY-SA 4.0 terms](https://creativecommons.org/licenses/by-sa/4.0/).

Attribution: **CharXiv, Wang et al. (2024), distributed by princeton-nlp**, and the original figure authors linked below. Source-paper identifiers are taken from the official CharXiv validation metadata.

| Case | Original figure source | Illustration |
|---|---|---|
| RQ-1969 | [arXiv:2202.10721](https://arxiv.org/abs/2202.10721) | `case-direct.png` |
| RQ-2201 | [arXiv:2104.08279](https://arxiv.org/abs/2104.08279) | `case-crop.png` |
| RQ-913 | [arXiv:2010.15793](https://arxiv.org/abs/2010.15793) | `case-crop-python.png` |

Changes in the illustrations: chart images are scaled for display, recorded crop regions are outlined, and questions, abridged model observations, tool code, stdout, and answers are arranged into panels. The source chart files and saved tool-returned crop files are copied unchanged. Python code and stdout are from the recorded call. `sources.json` maps each case to its source assets and original paper.

These are three held-out evaluation examples. No paper training samples or data construction/verification pipeline are included.

## Regeneration

From the repository root:

Install **Times New Roman** locally before regeneration. Font binaries are not included in this repository; the renderer requires the named font and does not silently substitute another family.

```bash
pip install -r requirements-visuals.txt
python scripts/render_readme_visuals.py
```

The paper-figure preview requires Poppler's `pdftoppm` and preserves the supplied PDF's fonts. Generated titles, labels, annotations, and code snippets use **Times New Roman** throughout. Original chart pixels retain their original typography. Matplotlib, Pillow, and CairoSVG render the generated figures; SVG exports retain editable vector text and shapes and require the same installed font for identical appearance.
