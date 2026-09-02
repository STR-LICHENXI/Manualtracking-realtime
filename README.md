# manualtracking-realtime

A real-time interactive visual effects project built with Python, MediaPipe, and OpenCV. It uses live hand landmark tracking to generate dynamic image surfaces that respond directly to hand movement, recreating the visual style commonly seen in the viral **#manualtracking** trend without frame-by-frame manual editing.

## Project Background

This project was inspired by the recent popularity of **#manualtracking** visual effects across Instagram, TikTok, Douyin, and other short-video platforms.

The project was inspired by the video of Instagram creator **[@wxll.hx](https://www.instagram.com/wxll.hx/)**, whose work combines real-time hand tracking with generative visual effects.

After seeing a large number of manual-tracking and editing-based recreations, this project explores a different implementation approach: rebuilding similar interactions from scratch using Python, MediaPipe, and OpenCV.

The current prototype tracks both hands in real time and uses hand landmarks as dynamic control points for image deformation, folding, and multi-surface visual effects.

This repository is an independent implementation and does not use or redistribute the original creator's TouchDesigner project files.

## System Architecture

The application runs from a standard webcam and uses MediaPipe Hand Landmarker to detect up to two hands in real time.

Each detected hand provides 21 landmark points. These landmarks are used to construct dynamic geometric surfaces, while OpenCV performs affine and perspective transformations to map image textures onto the generated geometry.


The current prototype includes three modes.

### 1. 2D Foldable Surface

Uses the thumb and index finger landmarks from both hands to create a dynamic image surface.

The texture follows hand movement in real time and can rotate, scale, and visually fold as the relative positions of the fingers change.

### 2. Multi-Surface Effect

Connects the five fingertips of both hands to construct multiple semi-transparent image surfaces.

The generated geometry changes continuously as the hands move, rotate, and change orientation, producing a deformable spatial visual effect.

### 3. Fan Effect

Uses one hand as a central anchor and connects it to the five fingertips of the opposite hand.

The five resulting triangular surfaces form a dynamic fan-like structure that expands and deforms with hand movement.

## Demo

Will be added soon. 

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

This repository contains an independent Python implementation developed using MediaPipe and OpenCV and does not include or redistribute the creator's TouchDesigner project files.
