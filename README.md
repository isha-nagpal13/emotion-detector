# 🎭 Emotion Detector using OpenCV & DeepFace

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?logo=opencv)
![DeepFace](https://img.shields.io/badge/DeepFace-0.0.79%2B-orange)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

A beginner-friendly **Computer Vision** project that detects human emotions from a static image using **OpenCV** for face detection and **DeepFace** for emotion recognition.

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
```

---

## 💻 Running in VS Code — Step by Step

1. **Open VS Code** → `File` → `Open Folder` → select the `emotion-detector` folder
2. **Open Terminal** in VS Code: `` Ctrl + ` `` (backtick)
3. Create and activate a virtual environment (see Step 2 above)
4. Install packages: `pip install -r requirements.txt`
5. Add `test.jpg` to the folder
6. Open `app.py` and press **F5** to run, or type `python app.py` in the terminal
7. A window will appear showing your image with the detected emotion
8. **Press any key** to close the window

---

## 📦 Uploading to GitHub

```bash
# 1. Initialize git in the project folder
git init

# 2. Add all files
git add .

# 3. Commit your changes
git commit -m "Initial commit: Emotion Detector project"

# 4. Create a new repo on github.com, then connect it
git remote add origin https://github.com/YOUR_USERNAME/emotion-detector.git

# 5. Push to GitHub
git branch -M main
git push -u origin main
```

> 💡 **Tip:** Add a `test.jpg` sample image to your repo so others can test it immediately.

---

## 📖 Beginner Explanation — What Does Each Part Do?

| Concept | Plain English |
|---------|--------------|
| `cv2.imread()` | Opens the image file and loads it into memory as a grid of pixels |
| `cv2.cvtColor()` | Converts the colorful image to black & white (grayscale) — works better for face detection |
| `CascadeClassifier` | A pre-trained AI model that knows what a human face looks like — included free with OpenCV |
| `detectMultiScale()` | Scans the image at many zoom levels to find faces of any size |
| `DeepFace.analyze()` | Sends the face to a neural network trained on thousands of facial expressions to predict the emotion |
| `cv2.rectangle()` | Draws a colored box around the face |
| `cv2.putText()` | Writes text (the emotion name) onto the image |
| `cv2.imshow()` | Opens a window and displays the final annotated image |

---

## 🛠️ Technologies Used

| Tool | Purpose |
|------|---------|
| [Python](https://python.org) | Core programming language |
| [OpenCV](https://opencv.org) | Image loading, face detection, drawing |
| [DeepFace](https://github.com/serengil/deepface) | Deep learning-based emotion recognition |
| [Haar Cascade](https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html) | Fast, lightweight face detection model |

---

