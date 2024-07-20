import tensorflow as tf
import numpy as np
import cv2

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path='3.tflite')
interpreter.allocate_tensors()

# Define the output file name and parameters
output_file = 'output_video.mp4'
fps = 30.0  # Adjust as needed
frame_size = (480, 480)  # Adjust as needed

# Initialize VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use appropriate codec based on file extension
video_writer = cv2.VideoWriter(output_file, fourcc, fps, frame_size)

def draw_keypoints(frame, keypoints, confidence_threshold):
    height, width, _ = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [height, width, 1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)

def draw_connections(frame, keypoints, edges, confidence_threshold):
    height, width, _ = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [height, width, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) and (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

EDGES = {
    (0, 1): 'm',
    (1, 3): 'm',
    (0, 2): 'c',
    (2, 4): 'c',
    (0, 5): 'm',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c',
    (0, 6): 'c',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
}

cap = cv2.VideoCapture('<input-file.mp4>')

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Resize the frame to the desired YouTube video size
    frame = cv2.resize(frame, frame_size)

    # Reshape image
    img = frame.copy()
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 256, 256)
    input_image = tf.cast(img, dtype=tf.float32)

    # Set-up input and output
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Make Predictions
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    key_points_with_scores = interpreter.get_tensor(output_details[0]['index'])

    # Draw keypoints and connections
    draw_keypoints(frame, key_points_with_scores[0, 0], confidence_threshold=0.3)
    draw_connections(frame, key_points_with_scores[0, 0], EDGES, confidence_threshold=0.3)
    
    cv2.imshow('MoveNet Lightning', frame)
    
    # Write the frame to the video file
    video_writer.write(frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    
# Release the VideoWriter object
video_writer.release()

cap.release()
cv2.destroyAllWindows()
