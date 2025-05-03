import cv2
import mediapipe as mp
import numpy as np
import time
from pose_utils import calculate_angle

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

DOWN_THRESHOLD = 140
UP_THRESHOLD = 110
REPS_TARGET = 8


def reset_globals():
    global current_reps, exercise_stage, feedback, feedback_flags
    global hold_counter, tempo_times, prev_angle, prev_time

    current_reps = 0
    exercise_stage = None

    feedback = {
        "low_range": 0,
        "short_hold": 0,
        "fast_movement": 0,
        "inconsistent_tempo": 0,
        "fast_down": 0,
        "body_tilt": 0,
        "low_lift": 0
    }

    feedback_flags = {k: False for k in feedback}
    hold_counter = 0
    tempo_times = []
    prev_angle = None
    prev_time = None


reset_globals()


def extract_leg_raise_landmarks(results):
    landmarks = results.pose_landmarks.landmark
    return {
        "shoulder": [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
        "hip": [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y],
        "knee": [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    }


def draw_leg_raise_left(image, results):
    landmarks = results.pose_landmarks.landmark
    points = [
        mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        mp_pose.PoseLandmark.LEFT_HIP.value,
        mp_pose.PoseLandmark.LEFT_KNEE.value
    ]

    for idx in points:
        x = int(landmarks[idx].x * image.shape[1])
        y = int(landmarks[idx].y * image.shape[0])
        cv2.circle(image, (x, y), 15, (0, 0, 255), -1)

    connections = [
        (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_HIP.value),
        (mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value)
    ]
    for connection in connections:
        x1 = int(landmarks[connection[0]].x * image.shape[1])
        y1 = int(landmarks[connection[0]].y * image.shape[0])
        x2 = int(landmarks[connection[1]].x * image.shape[1])
        y2 = int(landmarks[connection[1]].y * image.shape[0])
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 5)

    return image


def feedback_text(key, count):
    messages = {
        "low_range": f"Leg raise was too low ({count} times)",
        "short_hold": f"Held leg too short ({count} times)",
        "fast_movement": f"Movement was too fast ({count} times)",
        "inconsistent_tempo": f"Inconsistent tempo ({count} times)",
        "fast_down": f"Lowered leg too quickly ({count} times)",
        "body_tilt": f"Bent body to the side ({count} times)",
        "low_lift": f"Leg lift was too low ({count} times)"
    }
    return messages.get(key, "")


def reset_feedback_flags():
    for key in feedback_flags:
        feedback_flags[key] = False


def process_frame(frame):
    global exercise_stage, current_reps, feedback, prev_angle, prev_time

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    height, width, _ = image.shape

    if current_reps >= REPS_TARGET:
        stars = calculate_stars()
        sorted_feedback = sorted(feedback.items(), key=lambda x: x[1], reverse=True)
        tips = [k for k, v in sorted_feedback if v > 0][:2]
        exercise_name = "Leg raise"

        try:
            with open('feedback_data.txt', 'w') as f:
                f.write(f"{exercise_name}\n")
                f.write(f"{stars}\n")
                for tip in tips:
                    if feedback[tip] > 0:
                        f.write(f"{tip}:{feedback[tip]}\n")
            print("✅ feedback_data.txt saved.")
        except Exception as e:
            print("❌ Failed to write feedback file:", e)
        reset_globals()
        return None

    if results.pose_landmarks:
        image = draw_leg_raise_left(image, results)
        lm = extract_leg_raise_landmarks(results)
        shoulder, hip, knee = lm["shoulder"], lm["hip"], lm["knee"]

        angle = calculate_angle(shoulder, hip, knee)
        hip_px = np.multiply(hip, [width, height]).astype(int)

        # Draw ellipse representing the angle
        v1 = np.array(shoulder) - np.array(hip)
        v2 = np.array(knee) - np.array(hip)
        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)
        angle1 = np.degrees(np.arctan2(v1[1], v1[0]))
        angle2 = np.degrees(np.arctan2(v2[1], v2[0]))
        if angle1 < 0: angle1 += 360
        if angle2 < 0: angle2 += 360
        start_angle = min(angle1, angle2)
        end_angle = max(angle1, angle2)
        cv2.ellipse(image, tuple(hip_px), (60, 60), 0, start_angle, end_angle, (255, 0, 0), 5)

        # Draw angle text
        cv2.putText(image, f"{int(angle)} deg", (hip_px[0] - 20, hip_px[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 10)
        cv2.putText(image, f"{int(angle)} deg", (hip_px[0] - 20, hip_px[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 4)

        # Stage transition and feedback logic
        if angle > DOWN_THRESHOLD:
            if exercise_stage == "up":
                # Check for fast lowering (if previous angle dropped quickly)
                if prev_angle and (prev_angle - angle) > 20 and not feedback_flags["fast_down"]:
                    feedback["fast_down"] += 1
                    feedback_flags["fast_down"] = True
            exercise_stage = "down"

        elif angle < UP_THRESHOLD:
            if exercise_stage == "down":
                exercise_stage = "up"
                current_reps += 1

                # Rep evaluation feedback (one check per rep)
                # Check if the rep was held too short (using rep duration)
                rep_duration = time.time() - prev_time if prev_time else 0
                if rep_duration < 1.0 and not feedback_flags["short_hold"]:
                    feedback["short_hold"] += 1
                    feedback_flags["short_hold"] = True

                if angle > 105 and not feedback_flags["low_range"]:
                    feedback["low_range"] += 1
                    feedback_flags["low_range"] = True

                if abs(shoulder[0] - hip[0]) > 0.08 and not feedback_flags["body_tilt"]:
                    feedback["body_tilt"] += 1
                    feedback_flags["body_tilt"] = True

                if angle > 120 and not feedback_flags["low_lift"]:
                    feedback["low_lift"] += 1
                    feedback_flags["low_lift"] = True

                if prev_angle is not None and abs(angle - prev_angle) > 45 and not feedback_flags["fast_movement"]:
                    feedback["fast_movement"] += 1
                    feedback_flags["fast_movement"] = True

                reset_feedback_flags()  # Ready for the next rep

        # Update timing and angle history for next frame evaluation
        prev_angle = angle
        prev_time = time.time()

        # Draw rep counter and exercise name
        cv2.putText(image, "Leg raise", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 18)
        cv2.putText(image, "Leg raise", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 6)
        counter_text = f"Reps: {current_reps}/{REPS_TARGET}"
        cv2.putText(image, counter_text, (20, 1020), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 18)
        cv2.putText(image, counter_text, (20, 1020), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (242, 187, 5), 6)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def calculate_stars():
    non_zero_issues = [v for v in feedback.values() if v > 0]
    issue_count = len(non_zero_issues)
    total_issues = sum(non_zero_issues)

    if issue_count == 0:
        return 3
    elif issue_count <= 2 and total_issues <= 3:
        return 2
    else:
        return 1
