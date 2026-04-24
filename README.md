# Vision-Based PCB Alignment Detection System

A real-time PCB orientation detection system built using **Raspberry Pi 4B**, a **USB webcam**, **Python**, **OpenCV**, and a **Tkinter GUI**.

The system compares a live PCB image against a captured reference region using **ORB feature matching** and classifies the PCB as correctly or incorrectly aligned. Based on the result, it can trigger GPIO outputs for conveyor continuation or buzzer alerts.

---

## 🔎 Overview

Accurate PCB placement is important in wave soldering lines, where incorrect board orientation can cause soldering defects, rework, or production delays.

This project automates PCB orientation checking using computer vision. A reference PCB region is selected through the GUI, and incoming PCB frames are compared against it in real time.

The system includes:

- Live USB camera feed
- User-selectable ROI
- ORB feature matching
- Motion stability detection
- Rotation-angle estimation
- Tkinter-based GUI
- Raspberry Pi GPIO output support

---

## ✨ Features

- Real-time PCB orientation detection
- ORB keypoint detection and descriptor matching
- Homography-based rotation angle estimation
- Motion detection to avoid false checks while the PCB is moving
- Tkinter GUI for reference capture and ROI adjustment
- GPIO output for buzzer and conveyor/motor signal
- Cross-platform test version for Windows development

---

## 🖼️ Preview

### GUI Screenshot

![GUI Screenshot](images/gui_screenshot.png)

### Sample Detection Images

| Reference PCB | Wrong Orientation |
|---|---|
| ![Reference PCB](samples/reference_pcb.jpg) | ![Wrong Orientation PCB](samples/wrong_orientation_pcb.jpg) |

---

## ⚙️ System Architecture

| Module | Description |
|---|---|
| Processing Unit | Raspberry Pi 4B |
| Camera | USB webcam |
| Software | Python, OpenCV, Tkinter |
| Vision Algorithm | ORB feature matching + homography |
| GUI | Tkinter full-screen interface |
| Output Control | Raspberry Pi GPIO |
| Alerts | Buzzer and conveyor/motor signal |

---

## 🛠️ Workflow

1. The USB camera captures the live PCB image.
2. The user draws a reference ROI using the GUI.
3. The selected ROI is captured as the reference PCB.
4. The system monitors the detection region.
5. Motion detection checks whether the PCB is stationary.
6. ORB keypoints and descriptors are extracted.
7. Feature matches are used to estimate rotation angle.
8. If the PCB is aligned, the system shows **OK** and activates the conveyor/motor signal.
9. If the PCB is misaligned, the system shows **Wrong** and triggers the buzzer alert.

---

## 🧠 Detection Logic

The detection pipeline uses:

- ORB feature extraction
- Hamming-distance descriptor matching
- Homography estimation with RANSAC
- Rotation angle calculation from the homography matrix
- Frame differencing for motion stability
- ROI brightness and contour checking for PCB presence

Default parameters:

| Parameter | Value |
|---|---|
| Angle threshold | 10° |
| Motion threshold | 20 |
| Stability time | 0.3 s |
| ORB features | 1000 |
| Minimum matches | 10 |
| Frame size | 800 × 450 |
| Buzzer GPIO | BCM 18 |
| Motor/Conveyor GPIO | BCM 23 |

---

## 📁 Repository Structure

```text
pcb-alignment-detection-opencv/
├── src/
│   ├── pcb_orientation_detector_pi.py
│   └── pcb_orientation_detector_crossplatform.py
├── images/
│   └── gui_screenshot.png
├── samples/
│   ├── reference_pcb.jpg
│   └── wrong_orientation_pcb.jpg
├── requirements.txt
├── .gitignore
└── README.md
