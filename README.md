
# GestureTrackController

## Overview
GestureTrackController is a Python application that allows users to control audio playback features such as play, pause, track navigation, and volume adjustments through hand gestures. This application utilizes OpenCV, MediaPipe, and Pygame to detect hand landmarks and translate gestures into control commands.

## Prerequisites
Before running this application, ensure you have the following installed:
- Python 3.8 or higher
- OpenCV
- MediaPipe
- numpy
- pygame

## Installation

### Python Installation
Download and install Python from [python.org](https://www.python.org/downloads/).

### Library Installation
Use pip to install the required Python libraries:
```bash
pip install opencv-python mediapipe numpy pygame
```

### Clone the Repository
```bash
git clone https://github.com/Aman-Sunesh/esture-Track-Controller-using-MediaPipe-OpenCV.git
cd Gesture-Track-Controller-using-MediaPipe-OpenCV
```

## Usage
To run the application, navigate to the application directory and run:
```bash
python GestureTrackController.py
```

## Functionality

### Audio Control Gestures
- **Play/Pause**:
  - **Gesture**: Show 0 fingers to pause and 1 finger to play or unpause the audio.
- **Track Navigation**:
  - **Next Track**: Show 2 fingers to move to the next track.
  - **Previous Track**: Show 3 fingers to return to the previous track.
- **Volume Control**:
  - **Volume Up**: Show 4 fingers to increase the volume.
  - **Volume Down**: Show 5 fingers to decrease the volume.

### Visual Indicators
- **Volume Level and Current Track**: Displayed on the screen.
- **Number of Detected Fingers**: Shows how many fingers are detected, which corresponds to different controls.

## Important Notes
Ensure your webcam is enabled and properly configured before running the application. Lighting conditions and background can significantly affect the accuracy of hand gesture detection.

## Troubleshooting

### Webcam not Detected
Ensure that your webcam is connected and correctly installed. You may need to update your webcam drivers or try a different USB port.

### Library Errors
If you encounter errors related to missing libraries, ensure all the required Python packages are installed via pip.
