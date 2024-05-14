import tensorflow as tf
import numpy as np
import cv2
import math 

interpreter = tf.lite.Interpreter(model_path='3.tflite')
interpreter.allocate_tensors()

# Specify the path to the logo image
logo_path = '<your-logo.png>'

# Define the output file name and parameters
output_file = 'output_video.mp4'
fps = 30.0  # Adjust as needed
frame_size = (480, 480)  # Adjust as needed

# Initialize VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use appropriate codec based on file extension
video_writer = cv2.VideoWriter(output_file, fourcc, fps, frame_size)

def draw_keypoints(frame, keypoints, confidence_threshold):
    x, y, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)

def draw_connections(frame, keypoints, edges, confidence_threshold):
    x, y, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) and (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

def draw_chest_circle(frame, key_points_with_scores, confidence_threshold, radius=20, color=(0, 0, 255)):
    
    x, y, c = frame.shape
    shaped = np.squeeze(np.multiply(key_points_with_scores, [y, x, 1]))

    # Assuming the correct indices for left and right shoulder points are 5 and 6, respectively
    left_shoulder_point = shaped[5, :2] 
    right_shoulder_point = shaped[6, :2]  

    # Print the shoulder points
    print("Left Shoulder Point:", left_shoulder_point)
    print("Right Shoulder Point:", right_shoulder_point)
        
    # Calculate the midpoint between left and right shoulder points
    midpoint = ((left_shoulder_point[0] + right_shoulder_point[0]) / 2, 
                (left_shoulder_point[1] + right_shoulder_point[1]) / 2)

    # Define a small offset to move the circle below the midpoint
    offset = 90  # You can adjust this value based on your preference

    # Calculate the position for the circle below the midpoint
    circle_position = (int(midpoint[1]), int(midpoint[0]) + offset)

    # Draw a little red circle just below the midpoint
    #cv2.circle(frame, circle_position, 10, (0, 0, 255), -1)

    # Read the logo image with an alpha channel (transparency)
    logo_img = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
    
    # Define the fixed logo height and width
    fixed_logo_height, fixed_logo_width = 35, 60  # Adjust these values based on your preference
 
    # Resize the logo to the fixed size
    logo_img = cv2.resize(logo_img, (fixed_logo_width, fixed_logo_height))

     # Calculate the distance between left and right shoulder points
    shoulder_distance = np.linalg.norm(left_shoulder_point - right_shoulder_point)

    # Calculate the horizontal distance between left and right shoulder points
    horizontal_distance = abs(left_shoulder_point[1] - right_shoulder_point[1])

    # Calculate the scaling factor for width based on the horizontal distance
    width_scaling_factor = map_horizontal_distance_to_scaling_factor(horizontal_distance)

    # Resize the logo image based on the width scaling factor
    logo_img_resized = cv2.resize(logo_img, (int(fixed_logo_width * width_scaling_factor), fixed_logo_height))

    # Calculate the position for the resized logo
    logo_position = (
        int(midpoint[1] - logo_img_resized.shape[1] // 2),
        int(midpoint[0] + offset - logo_img_resized.shape[0])
    )

    # Create a mask for the logo using the alpha channel
    logo_alpha = logo_img_resized[:, :, 3] / 255.0
    logo_fg = logo_img_resized[:, :, :3] * np.expand_dims(logo_alpha, axis=2)

    # Ensure that the logo is placed within the frame boundaries
    start_y, end_y = logo_position[1], logo_position[1] + logo_img_resized.shape[0]
    start_x, end_x = logo_position[0], logo_position[0] + logo_img_resized.shape[1]

    if 0 <= start_y < frame.shape[0] and 0 <= end_y < frame.shape[0] and 0 <= start_x < frame.shape[1] and 0 <= end_x < frame.shape[1]:
        logo_bg = frame[start_y:end_y, start_x:end_x] * (1 - np.expand_dims(logo_alpha, axis=2))
        frame[start_y:end_y, start_x:end_x] = logo_fg + logo_bg

def map_horizontal_distance_to_scaling_factor(horizontal_distance):
    # Define your scaling logic here
    # For example, a simple linear mapping
    min_distance = 0.0
    max_distance = 100.0
    min_scaling_factor = 0.5
    max_scaling_factor = 1.5

    # Map horizontal distance to scaling factor using a linear function
    scaling_factor = ((horizontal_distance - min_distance) / (max_distance - min_distance)) * (max_scaling_factor - min_scaling_factor) + min_scaling_factor

    return scaling_factor
 

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

cap = cv2.VideoCapture('sahil-posing.mp4')

while cap.isOpened:
    ret, frame = cap.read()

    # Resize the frame to the desired YouTube video size
    frame = cv2.resize(frame, (480, 480))

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
    
    # Print the shape of key_points_with_scores for debugging
    print("key_points_with_scores shape:", key_points_with_scores.shape)
    
    # Extract key points from the tensor
    keypoints = key_points_with_scores[0, 0, :, :2]
    confidence_scores = key_points_with_scores[0, 0, :, 2]
    
    # Print the shape of keypoints for debugging
    print("keypoints shape:", keypoints.shape)

    # Draw a circle on the chest
    draw_chest_circle(frame, key_points_with_scores, 0.3)
    
    cv2.imshow('MoveNet Lightning', frame)
    
    # Write the frame to the video file
    video_writer.write(frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    
# Release the VideoWriter object
video_writer.release()

cap.release()
cv2.destroyAllWindows()
