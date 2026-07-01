import cv2
import os

# ==========================================
# CONFIGURATION
# ==========================================

MODE = "WEBCAM"      # IMAGE or WEBCAM

IMAGE_PATH = "dataset/archive/ORL-Organised/7/3_7.jpg"

TRAINER_PATH = "biometric_lab/trainer.yml"

HAAR_CASCADE_PATH = (
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# Lower = stricter matching boundary
MATCH_THRESHOLD = 75.0

# ==========================================
# INITIALIZATION
# ==========================================

face_cascade = cv2.CascadeClassifier(
    HAAR_CASCADE_PATH
)

recognizer = cv2.face.LBPHFaceRecognizer_create()

if not os.path.exists(TRAINER_PATH):
    raise FileNotFoundError(
        f"Model not found: {TRAINER_PATH}"
    )

recognizer.read(TRAINER_PATH)

# ==========================================
# FACE PROCESSING (TASK 4 CORE LOGIC)
# ==========================================

def process_frame(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:

        cv2.putText(
            frame,
            "No Face Detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        return frame

    for (x, y, w, h) in faces:

        face_roi = gray[y:y+h, x:x+w]

        # Match training size dimensions
        face_roi = cv2.resize(
            face_roi,
            (100, 100)
        )

        # Predict Person ID and calculate spatial dissimilarity distance
        label, distance = recognizer.predict(
            face_roi
        )

        # Make the Match / No Match determination based on threshold rules
        if distance <= MATCH_THRESHOLD:
            status = "AUTHENTICATED"
            display_label = str(label)
            box_color = (0, 255, 0)
        else:
            status = "UNKNOWN / DENIED"
            display_label = "Unknown"
            box_color = (0, 0, 255)

        # Draw the visual bounding overlay boxes
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            box_color,
            2
        )

        text_id = f"Person ID: {display_label}"
        text_dist = f"Distance: {distance:.2f}"
        text_status = f"Status: {status}"

        cv2.putText(
            frame,
            text_id,
            (x, y-45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            box_color,
            2
        )

        cv2.putText(
            frame,
            text_dist,
            (x, y-25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            box_color,
            1
        )

        cv2.putText(
            frame,
            text_status,
            (x, y-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            box_color,
            2
        )

        # Output local match logging variables safely to console
        print("\n--- MATCH RESULT ---")
        print(f"Person ID : {display_label}")
        print(f"Distance  : {distance:.2f}")
        print(f"Status    : {status}")

    return frame


# ==========================================
# IMAGE MODE (1:1 Verification Testing)
# ==========================================

if MODE == "IMAGE":

    print(
        f"[INFO] Processing {IMAGE_PATH}"
    )

    img = cv2.imread(
        IMAGE_PATH
    )

    if img is None:
        print("Could not load image.")
    else:
        result = process_frame(img)
        cv2.imshow("Face Recognition", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# ==========================================
# WEBCAM MODE (1:N Identification Testing)
# ==========================================

elif MODE == "WEBCAM":

    print(
        "[INFO] Webcam started (Q = quit)"
    )

    cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        output = process_frame(frame)
        cv2.imshow("Live Recognition", output)

        if (
            cv2.waitKey(1)
            & 0xFF
            == ord("q")
        ):
            break

    cap.release()
    cv2.destroyAllWindows()