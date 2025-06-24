import cv2
import mediapipe as mp
import csv
import os
import sys

label = sys.argv[1]  # e.g. "cover_drive" or "no_pose"
save_path = f'pose_data/{label}.csv'

# Create directory if it doesn't exist
os.makedirs('pose_data', exist_ok=True)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Recolor and process
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Show image
        cv2.putText(image, f'Collecting: {label}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Pose Collector', image)

        # Handle keypress
        key = cv2.waitKey(10)
        if key & 0xFF == ord('s'):
            if results.pose_landmarks:
                pose_row = []
                for lm in results.pose_landmarks.landmark:
                    pose_row.extend([lm.x, lm.y])
                pose_row.append(label)

                # Save to CSV
                file_exists = os.path.isfile(save_path)
                with open(save_path, mode='a', newline='') as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    if not file_exists:
                        headers = [f'x{i}' for i in range(33)] + [f'y{i}' for i in range(33)] + ['label']
                        csv_writer.writerow(headers)
                    csv_writer.writerow(pose_row)
                print(f"✅ Saved pose: {label}")

        elif key & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()