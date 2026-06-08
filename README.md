
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

# Screenhots
<img width="590" height="516" alt="Screenshot 2026-06-09 002850" src="https://github.com/user-attachments/assets/37b241c7-8a40-418b-8e15-3041d68e688a" />

#Author
Isha Nagpal

