# Vision Based Audio Tracking System

Gesture Media Tracking System which is a Python application that uses a webcam to track hand movements and control media playback on a computer. Instead of using a mouse or keyboard, the user can interact which the sytem simply by using hand gestures. The user can change the systems volume, play or pause music, and allows to skip to the next song using hand gestures in front of the camera.

---
## Demo
<video src="Demo.mp4" controls></video>

## Features

- 🤏 **Pinch Gesture** – Control system volume  
- ✊ **Fist Gesture** – Play / Pause media  
- ✌️ **Two-Finger Gesture** – Skip to next track  
- 🎥 Real-time hand tracking using MediaPipe  
- 🎚 Smooth volume transitions (no jitter)  
- 🪟 Works with Spotify, YouTube, VLC, and other media players  

---

## Technologies Used

- **Python**
- **OpenCV** – Webcam access and UI rendering
- **MediaPipe (Hands – Tasks API)** – Hand landmark detection
- **pycaw** – System volume control (Windows)
- **keyboard** – Media key simulation

---

## Requirements

- Windows OS  
- Python 3.11 or 3.12  
- Webcam  

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/ppatel2301/Vision-Based-Audio-Tracking-System.git
   cd Vision-Based-Audio-Tracking-System
