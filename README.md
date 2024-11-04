# Face, Smile, and Emotion Detection in Video

This project implements a real-time face, smile, and emotion detection system using OpenCV and the FER (Facial Emotion Recognition) library. It processes video input to identify faces, detect smiles, and classify emotions, providing a visual display of the results.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [How It Works](#how-it-works)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Features

- **Face Detection**: Utilizes a deep learning model to detect faces in video frames.
- **Smile Detection**: Employs a Haar Cascade Classifier to identify smiles within detected faces.
- **Emotion Recognition**: Uses the FER library to analyze facial expressions and classify emotions.
- **Real-time Processing**: Processes video frames in real-time for immediate feedback.

## Requirements

Before running the project, ensure you have the following installed:

- **Python 3.x**: The programming language used for the implementation.
- **OpenCV**: Library for computer vision tasks.
- **NumPy**: Library for numerical operations in Python.
- **FER**: Library for facial emotion recognition.

You can install the required libraries using pip:
pip install opencv-python numpy fer


## Installation

1. **Clone the Repository**:
   Open your terminal and run the following command to clone the repository:

   git clone https://github.com/yourusername/face-smile-emotion-detection.git
   cd face-smile-emotion-detection


2. **Download the Pre-trained Models**:
   - **Caffe Model**: Download the `res10_300x300_ssd_iter_140000.caffemodel` file, which is used for face detection. You can find it in the OpenCV repository or from the links provided in their documentation.
   - **Prototxt File**: Download the `deploy.prototxt` file, which is the model configuration for the Caffe model.

3. **Update File Paths**:
   In the main script (`your_script_name.py`), update the paths for the Caffe model, prototxt file, and your video file. Make sure they point to the correct locations on your system.
   model_file = r"C:\path\to\res10_300x300_ssd_iter_140000.caffemodel"
   config_file = r"C:\path\to\deploy.prototxt"
   video_path = r"C:\path\to\your_video_file.mp4"


## Usage

1. **Run the Script**:
   In your terminal, navigate to the project directory and execute the script:

   python your_script_name.py


2. **View Results**:
   - A window will open displaying the video with detected faces outlined by rectangles.
   - Smiles will be indicated with green rectangles, and recognized emotions will be displayed as text.

3. **Exit the Video Playback**:
   Press q on your keyboard to close the video window and stop the script.

## File Structure
│
├── your_script_name.py                                # Main script for face, smile, and emotion detection
├── res10_300x300_ssd_iter_140000.caffemodel           # Caffe model file for face detection
├── deploy.prototxt                                    # Model configuration file
└── haarcascade_smile.xml                              # Haar Cascade for smile detection (included with OpenCV)


## How It Works

1. **Video Capture**: The script initializes a video capture object to read frames from the specified video file.

2. **Face Detection**: For each frame, the script processes it through a deep learning model to detect faces. The model outputs bounding boxes around detected faces.

3. **Smile Detection**: For each detected face, the script extracts the region of interest (ROI) and applies a Haar Cascade Classifier to check for smiles.

4. **Emotion Recognition**: The extracted face ROI is analyzed using the FER library, which provides a list of emotions with their corresponding confidence scores.

5. **Display Results**: Detected faces, smiles, and emotions are visualized on the video frames, which are then displayed in real-time.

## Troubleshooting

- **Model Not Found**: Ensure that the paths to the Caffe model and prototxt file are correct. If the files are missing, download them again.
- **Performance Issues**: If the video playback is slow, consider lowering the video resolution or processing every other frame by modifying the frame count logic.
- **No Faces Detected**: Ensure that the video is of good quality and that faces are clearly visible. Adjust the confidence threshold if necessary.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
