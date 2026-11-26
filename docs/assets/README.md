# README figures and attribution

`overview.svg` / `overview.png` are an original schematic of the released inference interface. `results.svg` / `results.png` visualize the reported values in `results/reported-main-table.json`. These original figures and their rendering code use the repository's Apache 2.0 license.

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

```bash
pip install -r requirements-visuals.txt
python scripts/render_readme_visuals.py
```

The renderer uses Matplotlib, Pillow, and CairoSVG. It uses DejaVu Serif, a documented substitute for Times New Roman, which was unavailable in the rendering environment. Python code uses DejaVu Sans Mono. SVG exports retain editable vector text and shapes; the case plates contain the original raster chart assets.
