from datetime import datetime
import os
import cv2
import numpy as np
from flask import Flask, render_template, Response, request, redirect, url_for, send_from_directory
from ultralytics import YOLO
import sqlite3

app = Flask(__name__)
DB_NAME = 'VV.db'

def init_db():
    """Initialize the database and create the table if it doesn't exist."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Mask_Detection (
            Detection_ID INTEGER PRIMARY KEY AUTOINCREMENT,
            Camera_ID TEXT NOT NULL,
            Timestamp TEXT NOT NULL,
            No_of_Detections INTEGER NOT NULL,
            Image BLOB      
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def log_detection(camera_id, no_of_detections,image_data):
    """Log a detection event into the database."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute('''
        INSERT INTO Mask_Detection (Camera_ID, Timestamp, No_of_Detections, Image)
        VALUES (?, ?, ?, ?)
    ''', (camera_id, timestamp, no_of_detections, image_data))
    print("Called")
    conn.commit()
    conn.close()

# Load the YOLOv8 model
model = YOLO("best.pt")
names = model.model.names

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_webcam')
def start_webcam():
    return render_template('webcam.html')

def detect_objects_from_webcam():
    count = 0
    cap = cv2.VideoCapture(0)  # 0 for the default webcam
    camera_id = "Laptop Webcam"  # Replace with your camera ID
    no_mask_detections = 0

    # Cache to store track IDs of recently detected "No Mask" persons
    recent_detections = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count % 2 != 0:
            continue

        # Resize the frame to (1020, 600)
        frame = cv2.resize(frame, (1020, 600))

        # Run YOLOv8 tracking on the frame
        results = model.track(frame, persist=True)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, class_id, track_id in zip(boxes, class_ids, track_ids):
                label = names[class_id]
                x1, y1, x2, y2 = box

                # Check for "No Mask" class (Assume class 1 = "No Mask")
                if label.lower() == "mask_weared_incorrect" and track_id not in recent_detections:
                    recent_detections.add(track_id)  # Add to cache
                    no_mask_detections += 1

                    # Save the frame as an image
                    _, image_buffer = cv2.imencode('.jpg', frame)
                    image_data = image_buffer.tobytes()

                    # Log detection to the database
                    log_detection(camera_id, 1, image_data)

                    # Draw bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f'{track_id} - {label}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{track_id} - {label}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Optionally, clear recent detections after a certain period
        if len(recent_detections) > 100:  # Limit cache size
            recent_detections.clear()

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/webcam_feed')
def webcam_feed():
    return Response(detect_objects_from_webcam(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
# def upload_video():
#     if 'file' not in request.files:
#         return redirect(request.url)
    
#     file = request.files['file']
#     if file.filename == '':
#         return redirect(request.url)

#     # Save the uploaded file to the uploads folder
#     if not os.path.exists('uploads'):
#         os.makedirs('uploads')
    
#     file_path = os.path.join('uploads', file.filename)
#     file.save(file_path)

#     # Redirect to the video playback page after upload
#     return redirect(url_for('play_video', filename=file.filename))


@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    # Save the uploaded file to the uploads folder
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)
    print("Path: ",file_path)
    # Process the video for mask detections and save snapshots to DB
    process_video_for_detections(video_path=file_path)

    # Redirect to the video playback page after processing
    return redirect(url_for('play_video', filename=file.filename))

def process_video_for_detections(video_path):
    """Process a video for mask detections and save snapshots to the database."""
    cap = cv2.VideoCapture(video_path)
    count = 0
    camera_id = os.path.basename(video_path)  # Use the video filename as the camera ID

    # Cache to store track IDs of recently detected "No Queue" persons
    recent_detections = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count % 2 != 0:
            continue

        # Resize the frame to (1020, 600)
        frame = cv2.resize(frame, (1020, 600))

        # Run YOLOv8 tracking on the frame
        results = model.track(frame, persist=True)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, class_id, track_id in zip(boxes, class_ids, track_ids):
                label = names[class_id]
                x1, y1, x2, y2 = box

                # Draw bounding box and label on the frame
                color = (0, 255, 0) if label.lower() != "mask_weared_incorrect" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{track_id} - {label}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # Check for "No Queue" class (Assume class 1 = "No Queue")
                if label.lower() == "mask_weared_incorrect" and track_id not in recent_detections:
                    recent_detections.add(track_id)  # Add to cache

                    # Save the frame with bounding boxes as an image
                    _, image_buffer = cv2.imencode('.jpg', frame)
                    image_data = image_buffer.tobytes()

                    # Log detection to the database
                    log_detection(camera_id, 1, image_data)

    cap.release()

@app.route('/uploads/<filename>')
def play_video(filename):
    return render_template('play_video.html', filename=filename)

@app.route('/video/<path:filename>')
def send_video(filename):
    return send_from_directory('uploads', filename)

def detect_objects_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    count=0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count % 2 != 0:
           continue
        
        # Resize the frame to (1020, 600)
        frame = cv2.resize(frame, (1020, 600))

        # Run YOLOv8 tracking on the frame
        results = model.track(frame, persist=True)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, class_id, track_id in zip(boxes, class_ids, track_ids):
                c = names[class_id]
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{track_id} - {c}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed/<filename>')
def video_feed(filename):
    video_path = os.path.join('uploads', filename)
    return Response(detect_objects_from_video(video_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)