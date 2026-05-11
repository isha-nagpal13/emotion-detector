# 🎭 Emotion Detector using OpenCV & DeepFace

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?logo=opencv)
![DeepFace](https://img.shields.io/badge/DeepFace-0.0.79%2B-orange)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

A **Computer Vision** project that detects human emotions from a static image using **OpenCV** for face detection and **DeepFace** for emotion recognition.

---

## 📸 Demo

| Input Image | Output |
|-------------|--------|
| Person's face (test.jpg) | Face boxed in green + emotion label displayed |

**Detectable Emotions:** `Happy` · `Sad` · `Angry` · `Fear` · `Surprise` · `Disgust` · `Neutral`

---

## 🧠 How It Works

```
test.jpg  →  OpenCV loads image  →  Haar Cascade detects face
          →  DeepFace analyzes emotion  →  Draw box + label  →  output.jpg
```

1. **OpenCV** reads the image and converts it to grayscale
2. **Haar Cascade Classifier** (a pre-trained face detector built into OpenCV) locates faces
3. **DeepFace** analyzes the face region and predicts the dominant emotion
4. A **green rectangle** is drawn around each face, with the **emotion label** shown above it
5. The annotated image is displayed in a window and saved as `output.jpg`

---

## 🗂️ Project Structure

```
emotion-detector/
│
├── app.py              # Main script — all logic lives here
├── requirements.txt    # Python dependencies
├── test.jpg            # Your input image (add this yourself)
├── output.jpg          # Auto-generated result image
└── README.md           # This file
```

---

## ⚙️ Requirements

- Python **3.8 or higher**
- A front-facing photo named `test.jpg` in the project folder

---

## 🚀 Getting Started

### Step 1 — Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/emotion-detector.git
cd emotion-detector
```

### Step 2 — Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **Note:** DeepFace will automatically download the FER model (~200MB) on the **first run**. This is normal — it only happens once.

### Step 4 — Add Your Test Image

Place a clear, front-facing photo in the project folder and name it:

```
test.jpg
```

**Tips for best results:**
- Use a well-lit, front-facing photo
- Avoid heavy filters or obstructions
- A photo with one or more visible faces works best

### Step 5 — Run the App

```bash
python app.py

## 🛠️ Technologies Used

| Tool | Purpose |
|------|---------|
| [Python](https://python.org) | Core programming language |
| [OpenCV](https://opencv.org) | Image loading, face detection, drawing |
| [DeepFace](https://github.com/serengil/deepface) | Deep learning-based emotion recognition |
| [Haar Cascade](https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html) | Fast, lightweight face detection model |

---
<img width="1919" height="1031" alt="Screenshot 2026-04-15 194435" src="https://github.com/user-attachments/assets/72b9c435-b3d0-43f7-ba41-ddfc5aa9ac48" />

