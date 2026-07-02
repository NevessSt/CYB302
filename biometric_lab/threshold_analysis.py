import os
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "dataset/archive/ORL-Organised"
TRAINER_PATH = "trainer.yml"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Thresholds to evaluate
THRESHOLDS = range(30, 91, 5)

# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(TRAINER_PATH):
    print("ERROR: trainer.yml not found.")
    exit()

if not os.path.exists(DATASET_PATH):
    print("ERROR: Dataset not found.")
    exit()

print("===========================================")
print("TASK 5 - MATCHING THRESHOLD ANALYSIS")
print("===========================================\n")

# ============================================================
# LOAD MODEL
# ============================================================

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(TRAINER_PATH)

face_detector = cv2.CascadeClassifier(CASCADE_PATH)

print("[OK] Trainer Loaded")
print("[OK] Haar Cascade Loaded\n")

# ============================================================
# STORAGE
# ============================================================

results = []

total_images = 0
correct_predictions = 0

print("Scanning dataset...\n")

# ============================================================
# PROCESS DATASET
# ============================================================

for person in sorted(os.listdir(DATASET_PATH)):

    folder = os.path.join(DATASET_PATH, person)

    if not os.path.isdir(folder):
        continue

    actual_id = int(person)

    for image_name in sorted(os.listdir(folder)):

        image_path = os.path.join(folder, image_name)

        image = cv2.imread(image_path)

        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) == 0:
            print(f"No face detected -> {image_name}")
            continue

        x, y, w, h = faces[0]

        roi = gray[y:y+h, x:x+w]

        try:
            predicted_id, distance = recognizer.predict(roi)

        except cv2.error:
            continue

        total_images += 1

        if predicted_id == actual_id:
            correct_predictions += 1

        results.append({
            "actual": actual_id,
            "predicted": predicted_id,
            "distance": distance,
            "image": image_name
        })

        print(
            f"{image_name:12s}"
            f" | Actual:{actual_id:2d}"
            f" | Pred:{predicted_id:2d}"
            f" | Dist:{distance:6.2f}"
        )

# ============================================================
# SUMMARY
# ============================================================

accuracy = (correct_predictions / total_images) * 100 if total_images else 0

print("\n===========================================")
print("DATASET SUMMARY")
print("===========================================")

print(f"Images Processed : {total_images}")
print(f"Correct Matches  : {correct_predictions}")
print(f"Wrong Matches    : {total_images-correct_predictions}")
print(f"Recognition Accuracy : {accuracy:.2f}%")

print("\nPrediction collection completed successfully.\n")

# ============================================================
# THRESHOLD EVALUATION
# ============================================================

print("===========================================")
print("THRESHOLD ANALYSIS")
print("===========================================\n")

analysis_results = []

print(
    f"{'THR':<6}"
    f"{'TA':<6}"
    f"{'FA':<6}"
    f"{'TR':<6}"
    f"{'FR':<6}"
    f"{'FAR':<10}"
    f"{'FRR':<10}"
    f"{'ACC':<10}"
)

for threshold in THRESHOLDS:

    TA = 0
    FA = 0
    TR = 0
    FR = 0

    for r in results:

        actual = r["actual"]
        predicted = r["predicted"]
        distance = r["distance"]

        accepted = distance <= threshold

        if actual == predicted:

            if accepted:
                TA += 1
            else:
                FR += 1

        else:

            if accepted:
                FA += 1
            else:
                TR += 1

    FAR = FA / (FA + TR) if (FA + TR) > 0 else 0
    FRR = FR / (FR + TA) if (FR + TA) > 0 else 0
    GAR = TA / (TA + FR) if (TA + FR) > 0 else 0
    Precision = TA / (TA + FA) if (TA + FA) > 0 else 0
    Accuracy = (TA + TR) / (TA + TR + FA + FR)

    analysis_results.append({
        "Threshold": threshold,
        "TA": TA,
        "FA": FA,
        "TR": TR,
        "FR": FR,
        "FAR": FAR,
        "FRR": FRR,
        "GAR": GAR,
        "Precision": Precision,
        "Accuracy": Accuracy
    })

    print(
        f"{threshold:<6}"
        f"{TA:<6}"
        f"{FA:<6}"
        f"{TR:<6}"
        f"{FR:<6}"
        f"{FAR:<10.4f}"
        f"{FRR:<10.4f}"
        f"{Accuracy:<10.4f}"
    )

# ============================================================
# EQUAL ERROR RATE (EER)
# ============================================================

eer_result = min(
    analysis_results,
    key=lambda x: abs(x["FAR"] - x["FRR"])
)

eer = (eer_result["FAR"] + eer_result["FRR"]) / 2

print("\n===========================================")
print("BEST OPERATING POINT")
print("===========================================")

print(f"Threshold : {eer_result['Threshold']}")
print(f"FAR       : {eer_result['FAR']:.4f}")
print(f"FRR       : {eer_result['FRR']:.4f}")
print(f"GAR       : {eer_result['GAR']:.4f}")
print(f"Accuracy  : {eer_result['Accuracy']:.4f}")
print(f"Precision : {eer_result['Precision']:.4f}")
print(f"EER       : {eer:.4f}")

# ============================================================
# SAVE RESULTS TO CSV
# ============================================================

with open("threshold_results.csv", "w", newline="") as csvfile:

    writer = csv.writer(csvfile)

    writer.writerow([
        "Threshold",
        "TrueAccept",
        "FalseAccept",
        "TrueReject",
        "FalseReject",
        "FAR",
        "FRR",
        "GAR",
        "Precision",
        "Accuracy"
    ])

    for r in analysis_results:

        writer.writerow([
            r["Threshold"],
            r["TA"],
            r["FA"],
            r["TR"],
            r["FR"],
            round(r["FAR"], 4),
            round(r["FRR"], 4),
            round(r["GAR"], 4),
            round(r["Precision"], 4),
            round(r["Accuracy"], 4)
        ])

print("\n✓ threshold_results.csv created successfully.")
# ============================================================
# PART 3 - VISUALIZATION & FINAL REPORT
# ============================================================

print("\nGenerating graphs...")

thresholds = [r["Threshold"] for r in analysis_results]
far = [r["FAR"] for r in analysis_results]
frr = [r["FRR"] for r in analysis_results]
gar = [r["GAR"] for r in analysis_results]
accuracy = [r["Accuracy"] for r in analysis_results]
precision = [r["Precision"] for r in analysis_results]

# ============================================================
# FAR vs FRR
# ============================================================

plt.figure(figsize=(8,5))
plt.plot(thresholds, far, 'r-o', label="FAR")
plt.plot(thresholds, frr, 'b-s', label="FRR")

plt.axvline(
    eer_result["Threshold"],
    color="green",
    linestyle="--",
    label=f"EER Threshold = {eer_result['Threshold']}"
)

plt.xlabel("Threshold")
plt.ylabel("Rate")
plt.title("False Acceptance Rate vs False Rejection Rate")
plt.grid(True)
plt.legend()

plt.savefig("far_frr_curve.png", dpi=300)
plt.close()

# ============================================================
# ROC CURVE
# ============================================================

plt.figure(figsize=(6,6))

plt.plot(far, gar, marker='o')
plt.plot([0,1],[0,1],'k--')

plt.xlabel("False Acceptance Rate")
plt.ylabel("Genuine Acceptance Rate")
plt.title("ROC Curve")
plt.grid(True)

plt.savefig("roc_curve.png", dpi=300)
plt.close()

# ============================================================
# EER CURVE
# ============================================================

difference = [abs(f-f2) for f, f2 in zip(far, frr)]

plt.figure(figsize=(8,5))

plt.plot(thresholds, difference, color="purple", marker='o')

plt.axvline(
    eer_result["Threshold"],
    linestyle="--",
    color="red",
    label="Estimated EER"
)

plt.xlabel("Threshold")
plt.ylabel("|FAR - FRR|")
plt.title("Equal Error Rate Estimation")
plt.grid(True)
plt.legend()

plt.savefig("eer_curve.png", dpi=300)
plt.close()

# ============================================================
# ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(8,5))

plt.plot(thresholds, accuracy, marker='o', color='green')

plt.xlabel("Threshold")
plt.ylabel("Accuracy")
plt.title("Recognition Accuracy vs Threshold")
plt.grid(True)

plt.savefig("accuracy_curve.png", dpi=300)
plt.close()

# ============================================================
# PRECISION GRAPH
# ============================================================

plt.figure(figsize=(8,5))

plt.plot(thresholds, precision, marker='s', color='orange')

plt.xlabel("Threshold")
plt.ylabel("Precision")
plt.title("Precision vs Threshold")
plt.grid(True)

plt.savefig("precision_curve.png", dpi=300)
plt.close()

# ============================================================
# CONFUSION MATRIX
# ============================================================

best = eer_result

matrix = np.array([
    [best["TA"], best["FR"]],
    [best["FA"], best["TR"]]
])

plt.figure(figsize=(6,6))

plt.imshow(matrix, cmap="Blues")

plt.title("Confusion Matrix")

plt.xticks([0,1],["Accept","Reject"])
plt.yticks([0,1],["Genuine","Impostor"])

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            matrix[i,j],
            ha="center",
            va="center",
            color="black",
            fontsize=12
        )

plt.colorbar()

plt.savefig("confusion_matrix.png", dpi=300)
plt.close()

# ============================================================
# RECOMMENDED THRESHOLD
# ============================================================

print("\n==============================================")
print("RECOMMENDED OPERATING THRESHOLD")
print("==============================================")

print(f"Threshold : {best['Threshold']}")
print(f"Accuracy  : {best['Accuracy']*100:.2f}%")
print(f"FAR       : {best['FAR']*100:.2f}%")
print(f"FRR       : {best['FRR']*100:.2f}%")
print(f"GAR       : {best['GAR']*100:.2f}%")
print(f"Precision : {best['Precision']*100:.2f}%")
print(f"EER       : {eer*100:.2f}%")

print("\nGenerated Files:")
print("  ✓ threshold_results.csv")
print("  ✓ far_frr_curve.png")
print("  ✓ roc_curve.png")
print("  ✓ eer_curve.png")
print("  ✓ confusion_matrix.png")
print("  ✓ accuracy_curve.png")
print("  ✓ precision_curve.png")

# ============================================================
# DISCUSSION
# ============================================================

print("\n==============================================")
print("TASK 5 DISCUSSION")
print("==============================================")

print("""
LOW THRESHOLD
-------------
• More users are accepted.
• Lower False Rejection Rate (FRR).
• Higher False Acceptance Rate (FAR).
• Better usability but weaker security.

HIGH THRESHOLD
--------------
• Fewer users are accepted.
• Lower FAR.
• Higher FRR.
• Stronger security but reduced usability.

The best threshold depends on the application:
- Banking or military systems usually prefer a lower FAR, even if FRR increases.
- Smartphones and attendance systems may tolerate a slightly higher FAR to improve user convenience.

Note:
This evaluation was performed using the enrolled ORL dataset. Because the recognizer was trained on all available images, the reported performance is likely optimistic. A separate training/testing split would provide a more realistic estimate of real-world performance.
""")

print("\n==============================================")
print("TASK 5 COMPLETED SUCCESSFULLY")
print("==============================================")

