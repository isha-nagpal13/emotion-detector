
import cv2                          # OpenCV - for image processing and face detection
from deepface import DeepFace       # DeepFace - for emotion analysis
import sys                          # sys - for graceful error exits
import os                           # os - for checking if file exists


IMAGE_PATH = "test.jpg"             # Make sure this file is in the same folder as app.py


if not os.path.exists(IMAGE_PATH):
    print(f"[ERROR] Image file '{IMAGE_PATH}' not found!")
    print("Please place a file named 'test.jpg' in the same folder as app.py.")
    sys.exit(1)


print("[INFO] Loading image...")
image = cv2.imread(IMAGE_PATH)      # Reads the image from disk into a NumPy array (BGR format)

if image is None:
    print("[ERROR] Could not read the image. Make sure it's a valid .jpg or .png file.")
    sys.exit(1)


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


print("[INFO] Detecting faces...")


faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)


if len(faces) == 0:
    print("[WARNING] No face detected in the image.")
    print("Try a clearer front-facing photo with good lighting.")
    sys.exit(0)

print(f"[INFO] {len(faces)} face(s) detected!")


for i, (x, y, w, h) in enumerate(faces):
   

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
