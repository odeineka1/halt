import cv2 
import time 
import pygame 
from ultralytics import YOLO 
import numpy as np 

model = YOLO("yolo-Weights/yolov8n.pt") 
classNames = ["person"] 

# For separating the background from moving objects
# history: the number of frames to be taken into account
# varThreshold: the lower the value, the more it detects even minor movements
# detectShadows: whether to detect shadows
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True) 


# Constants: game rules and timing
countdown_time = 60  # round length in seconds before "time's up" ends the game for everyone

# Threshold tuned empirically for a person standing ~2m from the camera in a
# normally lit room; raise it if small movements (breathing, sway) trigger
# false eliminations, lower it if real movement isn't being caught
motion_threshold = 35000

# Constants: traffic light appearance and timing
circle_radius = 35
circle_thickness = 2
# Positions of the three traffic-light circles (top-center of frame, stacked vertically).
# circle_center2 = top light, circle_center3 = middle light, circle_center1 = bottom light
circle_center1 = (640, 225)
circle_center3 = (640, 150)
circle_center2 = (640, 75)

rectangle_color = (25, 25, 25)  # Black
rectangle_thickness = 2
rectangle_sp1 = (600, 25)
rectangle_sp2 = (680, 275)

# Constants: on-screen text positions for each player's status/result
PLAYER1_STATUS_POS = (20, 400)
PLAYER1_LOST_POS = (175, 400)
PLAYER1_WIN_POS = (150, 360)
PLAYER2_STATUS_POS = (1050, 400)
PLAYER2_LOST_POS = (850, 400)
PLAYER2_WIN_POS = (900, 360)
CENTER_WIN_POS = (520, 360)
GAME_OVER_POS = (480, 360)

# Game state (mutated as the game runs)
circle_color1 = (128, 128, 128)  # Grey color
circle_color2 = (0, 255, 0)  # Green color
circle_color3 = (128, 128, 128)
last_color_change_time = time.time()

start_time = cv2.getTickCount()
elapsed_time = 0
player1_game_over = False  # If there are more than two players, an instantiation/spawning algorithm must be implemented for when a player enters the camera viewport
player2_game_over = False  # 
is_player1_dead = False
is_player2_dead = False

# 0 = still alive (no red-light motion caught yet), 1 = eliminated
motiond1 = 0
motiond2 = 0
end = 0
time_is_up = False


def detect_person(x1, y1, x2, y2, frame_width):
    """Assign a detected bounding box to Player 1 (left half) or Player 2 (right half).

    Args:
        x1, y1, x2, y2: pixel coordinates of the detection's bounding box
            (top-left and bottom-right corners).
        frame_width: width of the video frame in pixels, used to find the
            left/right dividing line down the middle of the screen.

    Returns:
        int: 1 if the box belongs to the left-side player, 2 if it belongs
        to the right-side player. If the box straddles the midline, the
        player whose side contains the larger portion of the box is chosen.
    """
    mid_x = frame_width // 2
    if x2 <= mid_x:
        return 1
    elif x1 >= mid_x:
        return 2
    else:
        left_area = (mid_x - x1) * (y2 - y1)
        right_area = (x2 - mid_x) * (y2 - y1)
        return 1 if left_area >= right_area else 2


def play_sound(sound_file):
    """Play an audio file once, from the start, using pygame's music mixer.

    Args:
        sound_file: path to an audio file (e.g. "ak-47.mp3") to play.
            Re-initializes the mixer each call, so this is only meant for
            occasional one-off sound effects, not rapid/overlapping playback.
    """
    pygame.mixer.init()
    pygame.mixer.music.load(sound_file)
    pygame.mixer.music.play()


def detect_players(frame, fg_mask, width_f):
    """Run YOLO person detection for one frame and update elimination state.

    For every detected person, draws a bounding box (red if they're moving
    enough during a red light to be eliminated, magenta otherwise) and a
    "<N> person" label. Updates the module-level motiond1/motiond2 flags
    when a player is caught moving during a red light.
    """
    global motiond1, motiond2

    results = model(frame, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            cls = int(box.cls[0])
            if cls < len(classNames) and classNames[cls] == 'person':
                roi_fg = fg_mask[y1:y2, x1:x2]
                motion_value = np.sum(roi_fg) / 255

                if motion_value > motion_threshold:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    person = detect_person(x1, y1, x2, y2, width_f)
                    if person == 1:
                        if circle_color1 == (0, 0, 255):
                            motiond1 = 1
                    elif person == 2:
                        if circle_color1 == (0, 0, 255):  # therefore, each player should have their own 'motiond' variable
                            motiond2 = 1
                    cv2.putText(frame, str(person) + " person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)


def update_traffic_light():
    """Advance the traffic light's green -> yellow -> red -> green cycle.

    Seconds between light-cycle changes: 4s is the default "hold" time for
    green/red; the yellow (transition) light is deliberately much shorter
    (1s) to mimic a real traffic light's quick amber phase.
    """
    global circle_color1, circle_color2, circle_color3, last_color_change_time

    current_time = time.time()
    delay = 4
    if circle_color3 == (255, 255, 0):
        delay = 1

    if current_time - last_color_change_time > delay:
        last_color_change_time = current_time
        if circle_color2 == (0, 255, 0):
            circle_color3 = (0, 255, 255)
            circle_color2 = (128, 128, 128)
        elif circle_color3 == (0, 255, 255):
            circle_color1 = (0, 0, 255)
            circle_color3 = (128, 128, 128)
        else:
            circle_color2 = (0, 255, 0)
            circle_color1 = (128, 128, 128)


def render_hud(frame, remaining_time, players_remained):
    """Draw all the fixed UI elements: divider line, traffic light, status
    text for each player, the countdown timer, and the players-remaining count.
    """
    cv2.line(frame, (640, 0), (640, 720), (211, 211, 211), 2)
    cv2.rectangle(frame, rectangle_sp1, rectangle_sp2, rectangle_color, -1)
    cv2.circle(frame, circle_center1, circle_radius, circle_color1, -1)
    cv2.circle(frame, circle_center2, circle_radius, circle_color2, -1)
    cv2.circle(frame, circle_center3, circle_radius, circle_color3, -1)

    # Draw each player's status text exactly once per frame, at its own
    # fixed position, so the two players' text never overlaps
    if motiond1 == 1:
        cv2.putText(frame, "You Lost!", PLAYER1_LOST_POS, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
    else:
        cv2.putText(frame, "Not Detected", PLAYER1_STATUS_POS, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if motiond2 == 1:
        cv2.putText(frame, "You Lost!", PLAYER2_LOST_POS, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
    else:
        cv2.putText(frame, "Not Detected", PLAYER2_STATUS_POS, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, "Time: {}".format(remaining_time), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, "Players remained: {}".format(players_remained), (20, 675), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


def update_game_state():
    """Update elimination/game-over flags from the current motion state."""
    global player1_game_over, player2_game_over, time_is_up

    elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
    remaining_time = countdown_time - int(elapsed_time)

    if remaining_time <= 0:
        time_is_up = True

    if motiond1 == 1:  # Implement motion detection for each player present in the camera view,
        player1_game_over = True  # and assign playerN_game_over = True if motion is detected
    if motiond2 == 1:
        player2_game_over = True

    return remaining_time


def handle_spacebar(frame):
    """Show "You Won!" inside the surviving player's own zone if spacebar
    was pressed. Returns True if the game should end this frame.
    """
    global end

    if cv2.waitKey(1) & 0xFF == ord(' '):
        if motiond1 == 0 and motiond2 == 1:
            cv2.putText(frame, "You Won!", PLAYER1_WIN_POS, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
        elif motiond2 == 0 and motiond1 == 1:
            cv2.putText(frame, "You Won!", PLAYER2_WIN_POS, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
        else:
            cv2.putText(frame, "You Won!", CENTER_WIN_POS, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

        win_start_time = time.time()
        while time.time() - win_start_time < 2:
            cv2.imshow('Frame', frame)
            cv2.waitKey(1)
        end = 1
        return True
    return False


def handle_game_over(frame):
    """Play the elimination gunshot cue and the end-of-round "Game Over"
    screen when the round has actually ended. Returns True if the game
    loop should stop this frame.
    """
    global is_player1_dead, is_player2_dead, end

    if (player1_game_over and not is_player1_dead) or (player2_game_over and not is_player2_dead):
        if player1_game_over:
            is_player1_dead = True
        if player2_game_over:
            is_player2_dead = True
        play_sound("gunshot.mp3")

    if ((player1_game_over and player2_game_over) or time_is_up) and end == 0:
        cv2.putText(frame, "Game Over", GAME_OVER_POS, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        if time_is_up:
            play_sound("ak-47.mp3")
        game_over_start_time = time.time()
        while time.time() - game_over_start_time < 2:
            cv2.imshow('Frame', frame)
            cv2.waitKey(1)
        end = 1
        return True
    return False


def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1280, 720))
        height_f, width_f = frame.shape[:2]

        fg_mask = bg_subtractor.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)

        detect_players(frame, fg_mask, width_f)
        update_traffic_light()

        # Implement detection of the number of players on screen
        players_remained = 2
        remaining_time = update_game_state()

        render_hud(frame, remaining_time, players_remained)

        if handle_spacebar(frame):
            break

        if handle_game_over(frame):
            break

        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()