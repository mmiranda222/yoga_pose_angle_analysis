# Yoga Pose Angle Analysis (YPAA)

The Yoga Pose Angle Analysis (YPAA) system is a computer vision-based pipeline designed to analyze active vs. passive range of motion using yoga poses. It computes a novel metric—the **Musculoskeletal Wellness Index (MWI)**—to quantify musculoskeletal health.

## 📌 Project Overview

This project uses MediaPipe and OpenCV to:
- Automatically classify yoga poses (between 5 options)
- Detect joint landmarks
- Measure angles using trigonometric calculations
- Compute AROM vs. PROM differences
- Derive the MWI per participant

The pipeline was used in a research study comparing musculoskeletal health between yoga practitioners and non-practitioners.

## 🧠 Features

- Pose recognition and classification (active/passive variants)
- Angle calculation for each pose
- MWI computation and export to CSV
- Debug image generation with annotated joint landmarks
- Support for both batch processing and debugging modes

## ⚙️ Dependencies

```bash
pip install -r requirements.txt
```

## 🛠️ Major Libraries Used

- `mediapipe`: For pose estimation and landmark tracking.
- `opencv-python`: For image loading, processing, and visualization.
- `numpy`: For numerical calculations, especially angle computation.
- `matplotlib`: For plotting and visualizing results.
- `pandas`: For handling tabular data and exporting analysis outputs.

## 🚀 Usage
Run analysis mode:
```bash
python main.py --mode analyze --image_folder path/to/images --output_csv results.csv
```
Run debug mode (opens debug software to play around with landmarks in the picture):
```bash
python main.py --mode debug --image_folder path/to/images --output_csv debug_results.csv
```

## 🗂️ Project Structure
.
├── yoga_pose_analyzer.py       # Core logic
├── main.py                     # CLI interface
├── assemble_master_dataset.py # Dataset assembly (optional)
├── debug_tool.py               # Image debugging utility
├── requirements.txt            # Python dependencies
└── data/                       # (Removed from GitHub for privacy reasons)

## 🔐 Data Privacy
The data/ folder is excluded from this repository to comply with participant privacy and ethical considerations.

## 📄 Thesis Reference
This project was developed as part of a thesis titled:  
*Quantifying Mobility: A Comparative Study of Passive and Active Range of Motion Using Yoga-Based Pose Analysis*

## 👥 Contributions
- **MWI concept**: Irene Alda  
- **Development, implementation, and study**: Manuela Miranda  
- **Synthetic data support**: Dae-Jin Lee  
- **Code assistance**: Large Language Models (LLMs)
  
## 📫 Contact
For any questions or collaboration opportunities, please open an issue on this repository or reach out via email if applicable.

## 🛠 License

This project is licensed under the [MIT License](LICENSE).

