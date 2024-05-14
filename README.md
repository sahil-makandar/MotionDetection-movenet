# Motion Detection - TensorLite: Logo Annotator

## Overview
MoveNet Lightning is a Python script that utilizes TensorFlow Lite and OpenCV to perform real-time pose estimation on video frames using the MoveNet model. It annotates key points on the human body and dynamically adjusts the size and position of a logo image to overlay it on the chest area of the detected person. This ensures that the logo maintains a consistent appearance relative to the person's movements throughout the video.

## Installation
1. **Clone the Repository**: 
    ```bash
    git clone https://github.com/mahaboobsabGoa/MotionDetection-movenet.git
    ```

2. **Install Dependencies**:
   ```bash
    pip install -r requirements.txt
    ```

3. **Provide Logo Image**:
    - Replace `<your-logo.png>` with your desired logo image. Ensure it has an alpha channel for transparency.

## Usage
1. **Run the Script**:
    ```bash
    python movenet.py
    ```

2. **Output**:
    - The script will open a window displaying the real-time video feed annotated with key points and the dynamically adjusted logo on the chest area of the detected person.
    - It will also generate an `output_video.mp4` file in the same directory, which contains the annotated video.

3. **Exit**:
    - Press 'q' on the keyboard to exit the script.

## Details
- **Model**: 
    - The script uses the MoveNet model in TensorFlow Lite format for single pose estimation.

- **Logo Overlay**:
    - The script dynamically adjusts the size and position of the logo to overlay it on the chest area of the detected person.
    - The logo's size and position change according to the movements of the person, ensuring that it maintains a consistent appearance relative to the person's chest throughout the video.

- **Visualization**:
    - Key points are annotated with circles, and connections between them are drawn on the video frames.
    - The connections represent body parts such as arms, legs, and the torso.

## Requirements
- Python 3.x
- TensorFlow Lite
- NumPy
- OpenCV (cv2)
