import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('pose_classifier.joblib')
scaler = joblib.load('scaler.pkl')

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Start webcam
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            pose_row = []
            for lm in landmarks:
                pose_row.extend([lm.x, lm.y])
            
            # Make prediction
            X = np.array(pose_row).reshape(1, -1)
            X_scaled = scaler.transform(X)
            prediction = model.predict(X_scaled)
            prob = model.predict_proba(X_scaled).max()

            # Show prediction on image
            cv2.putText(image, f'{prediction[0]} ({prob:.2f})', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display
        cv2.imshow('Pose Classification', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()