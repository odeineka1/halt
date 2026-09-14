# Halt!
 
A real-time two-player computer vision game inspired by "Red Light, Green Light" (from *Squid Game*). Players move on a "green light" and must freeze on "red light". If the system detects motion during a red light, that player loses.

## How it works
 
1. **Person detection** — a YOLOv8 (`yolov8n`) model detects people in each webcam frame and draws a bounding box around them.
2. **Player assignment** — each detected person is assigned to "Player 1" or "Player 2" based on which half of the frame their bounding box mostly occupies (`detect_person()`).
3. **Motion detection** — OpenCV's `BackgroundSubtractorMOG2` isolates moving pixels within each player's bounding box. If the amount of motion exceeds a threshold, the player is flagged as moving.
4. **Light cycle** — a traffic-light-style indicator (grey → green → yellow → red) cycles automatically with a randomized-feeling delay. During red light, any detected motion eliminates that player ("You Lost!").
5. **Win/lose conditions** — the round ends when both players are eliminated, the timer runs out, or a player presses spacebar to signal a win.

## Tech stack
 
- **Python 3**
- **OpenCV** — video capture, background subtraction, rendering
- **Ultralytics YOLOv8** — real-time person detection
- **Pygame** — sound effects (game-over sound cue)
- **NumPy** — motion-intensity calculation

## Setup
 
```bash
pip install -r requirements.txt
```
 
Download the YOLOv8n weights (`yolov8n.pt`) into a `yolo-Weights/` folder in the project root — see [Ultralytics YOLOv8 docs](https://docs.ultralytics.com/) for the download link.
 
## Run
 
```bash
python main.py
```
 
- Stand in front of your webcam. The screen is split into two zones (Player 1 / Player 2).
- Move freely during green/yellow light. Freeze during red light or you'll be eliminated.
- Press **spacebar** to declare a win manually; press **q** to quit.
## Known limitations
 
- Currently supports exactly two players via a fixed left/right screen split rather than dynamic player tracking — a player standing near the center can be misassigned.
- The motion threshold (`35000`) was tuned empirically for a specific camera distance/lighting setup and may need adjustment in other environments.
- No player elimination visuals/persistence between rounds; game state resets only on restart.
## Possible improvements
 
- Replace the fixed left/right zone split with per-player object tracking (e.g. persistent IDs via a tracker like ByteTrack) to support more than two players.
- Make the motion threshold adaptive to lighting/camera conditions instead of a fixed constant.
- Add a start/lobby screen and visual countdown instead of console-only timing.

## Testing

A small test suite (`test_main.py`) covers `detect_person()`, the function
that assigns a detected bounding box to Player 1 or Player 2. Run it with:

\```bash
python3 -m pip install pytest
python3 -m pytest test_main.py
\```