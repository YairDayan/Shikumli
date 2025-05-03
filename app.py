from flask import Flask, render_template, Response, request
import cv2
import os
from video_processor import process_frame

app = Flask(__name__)


@app.route('/')
def home():
    streak_days = 8  # Replace with dynamic logic if needed
    return render_template('home.html', streak_days=streak_days)


@app.route('/choose')
def choose():
    # Optional: clear file here too for safety
    try:
        os.remove("feedback_data.txt")
    except FileNotFoundError:
        pass
    return render_template("choose_exercise.html")


@app.route('/workout')
def workout():
    # Critical: ensure no previous result is left behind
    try:
        os.remove("feedback_data.txt")
    except FileNotFoundError:
        pass

    exercise = request.args.get("exercise", "leg_raise")
    return render_template("index.html", exercise=exercise)


@app.route('/video_feed')
def video_feed():
    def generate_frames():
        cap = cv2.VideoCapture(0)
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = process_frame(frame)
            if frame is None:
                break

            _, buffer = cv2.imencode('.jpg', frame)
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'

        cap.release()

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/complete')
def complete():
    exercise = "Workout"
    stars = 0
    tips = []
    try:
        with open("feedback_data.txt", "r") as f:
            lines = f.readlines()
            exercise = lines[0].strip()
            stars = int(lines[1].strip())
            for line in lines[2:]:
                key, count = line.strip().split(":")
                tips.append(f"{key.replace('_', ' ').capitalize()} ({count} times)")
    except Exception as e:
        print("Error reading feedback:", e)

    return render_template("workout_complete.html", stars=stars, tips=tips, exercise=exercise)


@app.route('/status')
def status():
    return {"finished": os.path.exists("feedback_data.txt")}


if __name__ == '__main__':
    app.run(debug=True)
