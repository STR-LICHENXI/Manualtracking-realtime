# Manualtracking-realtime

This is a real-time interactive visual effects project built with Python, MediaPipe, and OpenCV. It uses live hand landmark tracking to generate dynamic image surfaces that respond directly to hand movement, recreating the visual style commonly seen in the viral **#manualtracking** trend without frame-by-frame manual editing.

## Project Background

Basically I came across the recent **#manualtracking** trend on Instagram, Douyin/TikTok, WeChat Channels, Bilibili, and other short-video platforms.

At first, I thought these effects were all created through frame-by-frame editing as most of them actually are. That made me curious about whether similar interactions could be reproduced directly in real time using computer vision.

I therefore attempted to recreate the effect with Python, MediaPipe, and OpenCV, using hand landmarks as control points for image deformation and interactive geometry.

After the prototype was already working, I looked further into the original source of the visual style and found that Instagram creator **[@wxll.hx](https://www.instagram.com/wxll.hx/)** had also been using MediaPipe together with TouchDesigner for real-time interactive visuals.

This discovery changed how I viewed the project. Instead of presenting it as the first real-time version of the effect, I continued developing it as an independent Python and OpenCV implementation inspired by the original interactive work and the wider **#manualtracking** trend.

The current prototype explores how similar real-time interactions can be implemented without TouchDesigner, using MediaPipe hand landmarks, lightweight geometry construction, and OpenCV-based texture mapping.

This repository does not use or redistribute the original creator's TouchDesigner project files.

## System Architecture

The application runs from a standard webcam and uses MediaPipe Hand Landmarker to detect up to two hands in real time.

Each detected hand provides 21 landmark points. These landmarks are used as dynamic control points for constructing deformable image surfaces, while OpenCV performs affine transformations, perspective mapping, alpha blending, and real-time rendering.

A lightweight state machine controls the interaction sequence. An effect is activated when the hands touch, but it is only displayed after the hands separate and move apart. Additional touch-and-hold gestures switch between the three visual modes.

## Interactive Effects

The current prototype includes three modes.

### 1. 2D Foldable Surface

Uses the thumb and index finger landmarks from both hands to create a semi-transparent deformable image surface.

The texture follows hand movement in real time and can rotate, scale, twist, overlap, and visually fold as the relative positions of the fingers change.

### 2. Multi-Surface Effect

Connects the five fingertips of both hands to construct multiple semi-transparent image surfaces.

The fingertips form two closed five-point structures, while corresponding fingertips are connected with a sci-fi-inspired line frame. The generated geometry changes continuously as the hands move, rotate, and change orientation.

### 3. Fan Effect

Uses the index fingertip of one hand as the focal point and connects it to the five fingertips of the opposite hand.

The five resulting triangular surfaces form a closed fan-like structure that expands and deforms in real time.

## Demo

A full demonstration video is available here:

[View Sample Video](./Video%20sample/sample%20video.mp4)

## Quick Start
```bash
git clone https://github.com/STR-LICHENXI/Manualtracking-realtime.git
cd Manualtracking-realtime
pip install -r requirements.txt
python app.py
```

## Tech Stack

**Language:** Python  
**Computer Vision:** MediaPipe Hand Landmarker  
**Rendering:** OpenCV  
**Core Libraries:** NumPy, MediaPipe, OpenCV  
**Algorithms:** Hand Landmark Tracking, State Machine, Affine Transformation, Perspective Transformation, Alpha Blending, Dynamic Geometry Mapping

No custom neural network training is required. The project uses MediaPipe's pretrained hand landmark model as the real-time visual input layer.

## Inspiration

Visual inspiration: **[@wxll.hx](https://www.instagram.com/wxll.hx/)**

The original visual concept and creative direction belong to the original creator.

This repository contains an independent Python implementation developed using MediaPipe and OpenCV. It does not include or redistribute the creator's TouchDesigner project files.
