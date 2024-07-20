# Motion Detection - TensorLite: Keypoint Annotator

## Overview
This Python script utilizes TensorFlow Lite and OpenCV to perform real-time pose estimation on video frames using the MoveNet model. It annotates key points on the human body and draws lines between them to visualize motion. This allows for real-time motion detection throughout the video.

## Installation
1. **Clone the Repository**: 
    ```bash
    git clone https://github.com/mahaboobsabGoa/MotionDetection-movenet.git
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Provide Input Video**:
    - Ensure you have an input video file in MP4 format and replace `'<input-file.mp4>'` in the code with the actual path to your input video.

## Usage
1. **Run the Script**:
    ```bash
    python movenet.py
    ```

2. **Output**:
    - The script will open a window displaying the real-time video feed annotated with key points and connections between them.
    - It will also generate an `output_video.mp4` file in the same directory, which contains the annotated video.

3. **Exit**:
    - Press 'q' on the keyboard to exit the script.

## Details
- **Model**: 
    - The script uses the MoveNet model in TensorFlow Lite format for single pose estimation.

- **Keypoint Annotation**:
    - Key points on the human body are annotated with circles, and connections between them are drawn to visualize the body parts such as arms, legs, and the torso.
    - This helps in real-time motion detection by clearly showing the movements of different body parts.

## Requirements
- Python 3.x
- TensorFlow Lite
- NumPy
- OpenCV (cv2)
