# Football Analysis Project

## Introduction

This project aims to analyze football (soccer) videos using advanced computer vision and machine learning techniques. It detects and tracks players, referees, and the ball, assigns players to teams based on jersey colors, measures ball possession, estimates camera movement, and calculates player movements and speeds.

## Features

- Object detection and tracking using YOLO and ByteTrack
- Team assignment based on jersey color using K-means clustering
- Ball possession analysis
- Camera movement estimation using optical flow
- Player movement analysis
- Video annotation with detected objects and calculated metrics

## Requirements

- Python 3.x
- OpenCV
- NumPy
- Pandas
- scikit-learn
- ultralytics (YOLO)
- supervision

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/football-analysis-project.git
   cd football-analysis-project
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Download the YOLO model weights and place them in the `models` directory.

## Usage

1. Place your input video in the `Input_video` directory.

2. Run the main script:
   ```
   python main.py
   ```

3. The annotated output video will be saved in the `Output_video` directory.

## Project Structure

- `main.py`: The main script that orchestrates the entire analysis process.
- `trackers.py`: Contains the `Tracker` class for object detection and tracking.
- `team_assigner.py`: Implements the `TeamAssigner` class for assigning players to teams based on jersey colors.
- `camera_movement_estimator.py`: Contains the `CameraMovementEstimator` class for estimating camera movement between frames.
- `player_ball_assigner.py`: Implements the `PlayerBallAssigner` class for determining ball possession.
- `utils/`: Directory containing utility functions for bounding box operations, video processing, etc.

## Customization

You can adjust various parameters in the code to fine-tune the analysis:

- In `trackers.py`: Modify YOLO confidence thresholds and other detection parameters.
- In `team_assigner.py`: Adjust the K-means clustering parameters for team assignment.
- In `camera_movement_estimator.py`: Fine-tune the optical flow parameters for camera movement estimation.
- In `player_ball_assigner.py`: Modify the `max_player_ball_distance` to adjust ball possession detection sensitivity.

## Contributing

Contributions to this project are welcome! Please fork the repository and submit a pull request with your changes.
