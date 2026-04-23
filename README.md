# Yoga Pose Angle Analysis (YPAA)

A computer vision pipeline that measures active vs. passive range of motion using yoga poses, and computes a novel metric — the **Musculoskeletal Wellness Index (MWI)** — linking functional mobility to clinical outcomes.

Built as the core system of my IE University thesis, **awarded Best Thesis (CS & AI, 2025).**

<!-- Add a screenshot or GIF here of the pipeline annotating a pose. This is the single highest-impact change you can make. -->
<!-- ![YPAA in action](docs/ypaa_demo.png) -->

## What it does

The pipeline takes standardized yoga pose images, extracts joint landmarks with MediaPipe, and computes the angular difference between a participant's *passive* (assisted) and *active* (unassisted) range of motion. The resulting MWI quantifies how much control someone has over their own functional movement — not just how flexible they are.

Applied to a pilot study of 9 participants, expanded via a literature-informed resampling strategy (designed with a statistics professor), the metric showed significant associations with:

- **Chronic pain** (t = 2.9, p = .02)
- **Recent injury** (t = 2.5, p = .04)
- **Regular strength training** (improved mobility, t = 3.1, p = .02)
- Moderate inverse correlations with quality of life and mental well-being

Pain status and strength training remained the most influential predictors in multivariable regression (adjusted R² = 0.68).

## Pipeline

1. **Pose classification** across 5 standardized poses
2. **Landmark detection** via MediaPipe Pose
3. **Angle computation** using vector trigonometry at the relevant joints
4. **AROM vs. PROM comparison** per participant per pose
5. **MWI derivation** and CSV export

## Running it

```bash
pip install -r requirements.txt

# Analysis mode
python main.py --mode analyze --image_folder path/to/images --output_csv results.csv

# Debug mode (interactive landmark inspection)
python main.py --mode debug --image_folder path/to/images --output_csv debug_results.csv
```

## Stack

`Python` · `MediaPipe` · `OpenCV` · `NumPy` · `pandas` · `matplotlib`

## Project structure
.
|-- yoga_pose_analyzer.py       # Core angle computation and MWI logic
|-- main.py                     # CLI entry point
|-- assemble_master_dataset.py  # Optional dataset assembly
|-- debug_tool.py               # Interactive landmark debugger
|-- requirements.txt
`-- data/                       # Excluded - participant privacy

## Thesis

*Quantifying Mobility: A Comparative Study of Passive and Active Range of Motion Using Yoga-Based Pose Analysis.* IE University, 2025. Awarded Best Thesis, CS & AI cohort.

## Credits

- **MWI concept:** Irene Alda
- **Development, pipeline, and study design:** Manuela Miranda
- **Statistical methodology (resampling strategy):** Dae-Jin Lee
- **Code assistance:** LLMs

## Contact

[LinkedIn](https://www.linkedin.com/in/manuela-miranda-5a6516204) · [manuelamirandacarv@gmail.com](mailto:manuelamirandacarv@gmail.com)

## License

[MIT](./LICENSE)
