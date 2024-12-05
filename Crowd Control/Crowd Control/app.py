import cv2
import tensorflow as tf

# Load the model
loaded_model = tf.saved_model.load('model')


def process_video_alternative(video_path, model, output_path, threshold=0.5, frame_skip=10):
    """Efficient frame-by-frame video processing, skipping frames periodically, with people detection."""
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Use the 'XVID' codec for AVI files
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

    print(f"Processing video: {video_path}")
    print(f"Total frames: {total_frames}, Resolution: {frame_width}x{frame_height}, FPS: {fps}")

    frame_count = 0
    processed_frame_count = 0

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

        # Count the number of people detected in this frame
        people_detected = 0
        for i in range(int(results['num_detections'][0])):
            if classes[i] == 1 and scores[i] > threshold:
                ymin, xmin, ymax, xmax = boxes[i]
                left, top, right, bottom = (int(xmin * frame_width), int(ymin * frame_height),
                                            int(xmax * frame_width), int(ymax * frame_height))
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                people_detected += 1

        # Write processed frame to output video
        out.write(frame)

        # Print the number of people detected in this frame
        print(f"Frame {frame_count}/{total_frames}: {people_detected} people detected.")

        # Periodically report progress
        if processed_frame_count % 10 == 0:
            print(f"Processed {processed_frame_count} frames (skipping {frame_skip - 1} frames in between)...")

    cap.release()
    out.release()
    print(f"Video processing completed. Output saved at: {output_path}")


video_path = 'input/samplevid.mp4'
output_path = 'output/vid2.avi'  # AVI format

# Call the function
process_video_alternative(video_path, loaded_model, output_path, threshold=0.25, frame_skip=10)
