import os
import cv2
import sqlite3
import tensorflow as tf
from flask import Flask, request, jsonify, render_template, send_file
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Load the model
MODEL = tf.saved_model.load('model')

# Create directories for uploaded and processed videos
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Initialize SQLite database
DB_NAME = 'VV.db'

def initialize_database():
    """Initialize the SQLite database and create the CrowdControl table."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS CrowdControl (
            Detection_ID INTEGER PRIMARY KEY AUTOINCREMENT,
            Camera_ID TEXT NOT NULL,
            Timestamp TEXT NOT NULL,
            No_of_Detections INTEGER NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def log_detection_to_db(camera_id, no_of_detections):
    """Insert a detection log into the CrowdControl table."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute('''
        INSERT INTO CrowdControl (Camera_ID, Timestamp, No_of_Detections)
        VALUES (?, ?, ?)
    ''', (camera_id, timestamp, no_of_detections))
    conn.commit()
    conn.close()


# Call this once to initialize the database
initialize_database()

def process_video_alternative(video_path, model, output_path, threshold=0.25, frame_skip=10, detection_threshold=15):
    """Efficient frame-by-frame video processing, skipping frames periodically, with people detection."""
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Use the 'mp4v' codec for MP4 files
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    print(f"Processing video: {video_path}")
    print(f"Total frames: {total_frames}, Resolution: {frame_width}x{frame_height}, FPS: {fps}")

    frame_count = 0
    processed_frame_count = 0
    total_people_detected = 0
    camera_id = os.path.basename(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames based on frame_skip
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        frame_count += 1
        processed_frame_count += 1

        # Convert BGR frame (OpenCV) to RGB for TensorFlow
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = tf.convert_to_tensor(rgb_frame, dtype=tf.uint8)[tf.newaxis, ...]

        # Detect objects
        results = model(input_tensor)

        # Draw bounding boxes for detections
        boxes = results['detection_boxes'].numpy()[0]
        classes = results['detection_classes'].numpy()[0]
        scores = results['detection_scores'].numpy()[0]

        frame_people_detected = 0

        # Count the number of people detected in this frame
        for i in range(int(results['num_detections'][0])):
            if classes[i] == 1 and scores[i] > threshold:
                ymin, xmin, ymax, xmax = boxes[i]
                left, top, right, bottom = (int(xmin * frame_width), int(ymin * frame_height),
                                            int(xmax * frame_width), int(ymax * frame_height))
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                frame_people_detected += 1

        total_people_detected += frame_people_detected

        # If the detection threshold is exceeded, log to the database
        if frame_people_detected >= detection_threshold:
            log_detection_to_db(camera_id, frame_people_detected)

        # Write processed frame to output video
        out.write(frame)

    cap.release()
    out.release()
    print(f"Video processing completed. Output saved at: {output_path}")
    print(f"Total people detected: {total_people_detected}")
    return total_people_detected


def process_webcam_feed(model, threshold=0.25):
    """Process live webcam feed for people detection."""
    cap = cv2.VideoCapture(0)  # Open webcam
    total_people_detected = 0
    frame_count = 0
    frame_skip = 10

    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return -1

    print("Processing live webcam feed...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames to reduce computation
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        frame_count += 1

        # Convert BGR frame (OpenCV) to RGB for TensorFlow
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = tf.convert_to_tensor(rgb_frame, dtype=tf.uint8)[tf.newaxis, ...]

        # Detect objects
        results = model(input_tensor)

        # Extract detections
        boxes = results['detection_boxes'].numpy()[0]
        classes = results['detection_classes'].numpy()[0]
        scores = results['detection_scores'].numpy()[0]

        # Count the number of people detected in this frame
        for i in range(int(results['num_detections'][0])):
            if classes[i] == 1 and scores[i] > threshold:
                total_people_detected += 1

        # Draw bounding boxes for detections
        for i in range(int(results['num_detections'][0])):
            if classes[i] == 1 and scores[i] > threshold:
                ymin, xmin, ymax, xmax = boxes[i]
                left, top, right, bottom = (int(xmin * frame.shape[1]), int(ymin * frame.shape[0]),
                                            int(xmax * frame.shape[1]), int(ymax * frame.shape[0]))
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Show the processed frame
        cv2.imshow('Webcam Feed', frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Normalize total people detected by the number of processed frames
    normalized_count = total_people_detected / (frame_count / frame_skip)
    print(f"Total people detected from webcam: {normalized_count}")
    return normalized_count

@app.route('/webcam', methods=['GET'])
def webcam_feed():
    """Handle live webcam feed processing."""
    total_detections = process_webcam_feed(MODEL, threshold=0.25)
    if total_detections == -1:
        return jsonify({'error': 'Unable to access the webcam'}), 500

    return jsonify({
        'message': 'Webcam feed processed successfully',
        'total_detections': total_detections
    })

@app.route('/')
def index():
    """Render the homepage."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle video uploads and processing."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], 'processed_' + file.filename)

        # Save uploaded file
        file.save(input_path)

        # Process the video and get total detections
        total_detections = process_video_alternative(input_path, MODEL, output_path, threshold=0.25, frame_skip=10)

        return jsonify({
            'message': 'Video processed successfully',
            'total_detections': total_detections,
            'download_link': f'/download?filename={os.path.basename(output_path)}'
        })

@app.route('/download')
def download_file():
    """Download the processed video."""
    filename = request.args.get('filename')
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)