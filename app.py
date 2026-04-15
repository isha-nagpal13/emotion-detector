# ============================================================
# 🎭 Emotion Detector using OpenCV + DeepFace
# Author: Your Name
# Description: Detects human emotions from a static image
# ============================================================

# Step 1: Import the required libraries
import cv2                          # OpenCV - for image processing and face detection
from deepface import DeepFace       # DeepFace - for emotion analysis
import sys                          # sys - for graceful error exits
import os                           # os - for checking if file exists

# ============================================================
# Step 2: Configuration — set your image filename here
# ============================================================
IMAGE_PATH = "test.jpg"             # Make sure this file is in the same folder as app.py

# ============================================================
# Step 3: Check if the image file exists
# ============================================================
if not os.path.exists(IMAGE_PATH):
    print(f"[ERROR] Image file '{IMAGE_PATH}' not found!")
    print("Please place a file named 'test.jpg' in the same folder as app.py.")
    sys.exit(1)

# ============================================================
# Step 4: Load the image using OpenCV
# ============================================================
print("[INFO] Loading image...")
image = cv2.imread(IMAGE_PATH)      # Reads the image from disk into a NumPy array (BGR format)

if image is None:
    print("[ERROR] Could not read the image. Make sure it's a valid .jpg or .png file.")
    sys.exit(1)

# ============================================================
# Step 5: Convert image to grayscale for face detection
# ============================================================
# Haar Cascade works faster and better on grayscale images
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ============================================================
# Step 6: Load the Haar Cascade face detector
# ============================================================
# This is a pre-trained XML model that comes built-in with OpenCV
# It detects faces based on patterns of light and dark regions (Haar features)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ============================================================
# Step 7: Detect faces in the image
# ============================================================
print("[INFO] Detecting faces...")

# detectMultiScale scans the image for faces at multiple sizes
# scaleFactor=1.1  → image is reduced by 10% each step (catches faces at different sizes)
# minNeighbors=5   → how many nearby detections to confirm a face (higher = less false positives)
# minSize=(30,30)  → ignore anything smaller than 30x30 pixels
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

# Check if any faces were found
if len(faces) == 0:
    print("[WARNING] No face detected in the image.")
    print("Try a clearer front-facing photo with good lighting.")
    sys.exit(0)

print(f"[INFO] {len(faces)} face(s) detected!")

# ============================================================
# Step 8: Analyze emotion using DeepFace + draw on image
# ============================================================
for i, (x, y, w, h) in enumerate(faces):
    # x, y → top-left corner of the face rectangle
    # w    → width of the face box
    # h    → height of the face box

    print(f"\n[INFO] Analyzing emotion for face #{i + 1}...")

    # Crop the face region from the original image for DeepFace
    face_region = image[y:y + h, x:x + w]

    try:
        # DeepFace analyzes the face for emotion
        # enforce_detection=False → prevents crash if DeepFace can't re-detect the face
        result = DeepFace.analyze(
            img_path=face_region,
            actions=["emotion"],        # We only need emotion (skip age, gender, race)
            enforce_detection=False
        )

        # DeepFace returns a list of results — get the first one
        dominant_emotion = result[0]["dominant_emotion"]
        print(f"[RESULT] Detected Emotion: {dominant_emotion.upper()}")

    except Exception as e:
        dominant_emotion = "Unknown"
        print(f"[WARNING] Could not analyze emotion: {e}")

    # --------------------------------------------------------
    # Step 9: Draw a green rectangle around the detected face
    # --------------------------------------------------------
    # cv2.rectangle(image, top-left, bottom-right, color BGR, thickness)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)   # Green box, 2px thick

    # --------------------------------------------------------
    # Step 10: Display the emotion text above the face box
    # --------------------------------------------------------
    label = dominant_emotion.capitalize()

    # Calculate text position — place it just above the rectangle
    text_x = x
    text_y = y - 10 if y - 10 > 10 else y + 20  # Prevent text from going off-screen

    # cv2.putText(image, text, position, font, scale, color BGR, thickness)
    cv2.putText(
        image,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,   # Clean, readable font
        0.9,                         # Font scale
        (0, 255, 0),                 # Green color
        2                            # Thickness
    )

# ============================================================
# Step 11: Show the final output image in a window
# ============================================================
print("\n[INFO] Displaying result... (Press any key to close the window)")
cv2.imshow("Emotion Detector - Press any key to exit", image)
cv2.waitKey(0)          # Wait until user presses any key
cv2.destroyAllWindows() # Close the OpenCV window cleanly

# ============================================================
# Step 12: Save the output image (optional)
# ============================================================
output_path = "output.jpg"
cv2.imwrite(output_path, image)
print(f"[INFO] Output saved as '{output_path}'")
print("\n✅ Done! Emotion detection complete.")
