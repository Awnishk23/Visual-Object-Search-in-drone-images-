# 🔍 VisDrone Object Detection Dashboard

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFCA?style=for-the-badge&logo=github&logoColor=black)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-GPU-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Dataset](https://img.shields.io/badge/VisDrone-2019-00ff88?style=for-the-badge)

**A YOLOv8-powered object detection system trained on drone imagery, with an interactive Streamlit dashboard for real-time inference.**

</div>

---

## 📸 Overview

This project trains a **YOLOv8m** model on the [VisDrone 2019](https://github.com/VisDrone/VisDrone-Dataset) dataset to detect 10 object classes in aerial/drone images. It includes a polished dark-mode Streamlit app for uploading images and running live inference with bounding box visualizations.

---

## ✨ Features

- 🚁 **Drone-optimized detection** — trained on VisDrone2019-DET with 6,471 training images
- 🎯 **10 object classes** — pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor
- 📊 **Interactive dashboard** — confidence threshold slider, IoU control, per-class breakdown
- 📥 **Downloadable results** — export annotated images directly from the UI
- ⚡ **Fast inference** — ~51ms per image on Tesla T4 GPU
- 🧩 **Supports YOLOv5 & YOLOv8** — switchable model type in settings

---

## 📁 Project Structure

```
├── app.py                          # Streamlit dashboard application
├── train_visdrone.ipynb            # Kaggle training notebook
├── requirements.txt                # Python dependencies
└── Drone_Project/
    └── training_run/
        ├── weights/
        │   ├── best.pt             # Best model weights (use this!)
        │   └── last.pt             # Last epoch weights
        ├── results.png             # Training curves
        └── labels.jpg              # Dataset label distribution
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/your-username/visdrone-detection.git
cd visdrone-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download model weights

Place `best.pt` (from `Drone_Project/training_run/weights/`) into your working directory, or point the dashboard to its folder path.

### 4. Launch the dashboard
```bash
streamlit run app.py
```

### 5. Use the app

1. Set the **model folder path** in the sidebar (where `best.pt` lives)
2. Select **YOLOv8 (ultralytics)** as the model type
3. Click **🔄 Load Model**
4. Upload a drone image (JPG/PNG)
5. Click **🚀 Detect Objects**

---

## 🏋️ Training Details

| Parameter | Value |
|---|---|
| **Model** | YOLOv8m (medium) |
| **Dataset** | VisDrone2019-DET |
| **Train images** | 6,471 |
| **Val images** | 548 |
| **Epochs** | 40 |
| **Image size** | 640×640 |
| **Batch size** | 8 |
| **Optimizer** | AdamW (lr=0.000714) |
| **GPU** | Tesla T4 (14GB) |
| **Training time** | ~2.8 hours |

---

## 📈 Model Performance (Final Epoch)

| Class | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| **All** | 0.557 | 0.426 | 0.427 | 0.253 |
| Car | 0.754 | 0.802 | 0.806 | 0.560 |
| Bus | 0.788 | 0.546 | 0.606 | 0.442 |
| Van | 0.541 | 0.476 | 0.474 | 0.335 |
| Truck | 0.572 | 0.411 | 0.415 | 0.279 |
| Motor | 0.573 | 0.485 | 0.479 | 0.210 |
| Pedestrian | 0.582 | 0.458 | 0.465 | 0.207 |
| People | 0.599 | 0.339 | 0.352 | 0.135 |
| Bicycle | 0.314 | 0.208 | 0.172 | 0.077 |
| Tricycle | 0.509 | 0.330 | 0.324 | 0.179 |
| Awning-tricycle | 0.341 | 0.201 | 0.177 | 0.109 |

> 📝 Cars dominate performance (mAP50: 0.806) due to their abundance in the dataset. Small objects like bicycles and awning-tricycles are harder to detect — common in drone imagery due to low resolution at altitude.

---

## 🖥️ Dashboard Settings

| Setting | Description | Default |
|---|---|---|
| Model folder path | Directory containing `best.pt` | `./` |
| Model type | YOLOv8 or YOLOv5 | YOLOv8 |
| Confidence threshold | Minimum detection confidence | 0.40 |
| IoU threshold (NMS) | Non-max suppression overlap cutoff | 0.45 |

---

## 📦 Requirements

```txt
streamlit
ultralytics
torch
torchvision
opencv-python
Pillow
numpy
```

Install all at once:
```bash
pip install streamlit ultralytics torch torchvision opencv-python Pillow numpy
```

---

## 🗂️ Dataset

This project uses the **VisDrone2019-DET** dataset — a large-scale benchmark collected by drone platforms across 14 cities in China, covering diverse weather, lighting, and density conditions.

- 📦 **6,471 training images**, **548 validation images**
- 🏷️ **10 object categories** focused on urban traffic and pedestrians
- 🌐 Available on [Kaggle](https://www.kaggle.com/datasets/banuprasadb/visdrone-dataset)

---

## 💡 Tips for Better Results

- Use a **confidence threshold of 0.25–0.40** for a good precision/recall balance
- For crowded scenes, lower the **IoU threshold** (e.g., 0.35) to reduce missed detections
- The model performs best on **overhead drone imagery** similar to the training distribution
- Images taken from **higher altitudes** may yield lower recall due to small object sizes

---

## 🔮 Future Improvements

- [ ] Add video/webcam stream support
- [ ] Export detections as CSV/JSON
- [ ] Try SAHI (Slicing Aided Hyper Inference) for better small object detection
- [ ] Fine-tune with more epochs or larger model (YOLOv8l/x)
- [ ] Add class-wise confidence filtering

---

## 📄 License

This project is released under the **MIT License**. The VisDrone dataset is subject to its own [terms of use](https://github.com/VisDrone/VisDrone-Dataset).

---

<div align="center">
  <sub>Built with ❤️ using YOLOv8 + Streamlit</sub>
</div>
