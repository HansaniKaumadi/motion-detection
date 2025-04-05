# Motion Detection with Human Identification

This project uses a Raspberry Pi and a webcam to detect human motion in real-time. It utilizes the MobileNet SSD (Single Shot Multibox Detector) pre-trained deep learning model to identify human presence in video frames. When significant movement is detected, the system saves a screenshot of the scene with a timestamped filename.

## Requirements

- Raspberry Pi (any model with a camera interface)
- USB Webcam or Raspberry Pi Camera Module
- Python 3.x
- OpenCV (cv2)
- NumPy

## Installation

### 1. Install Dependencies
Ensure that your Raspberry Pi is set up with the latest version of Python and necessary libraries. Install the required libraries via `pip`:

```bash
sudo apt update
sudo apt install python3-opencv
sudo apt install python3-numpy
