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

cap = cv2.VideoCapture(0) 
start_time = cv2.getTickCount() 
countdown_time = 70   
elapsed_time = 0 
player1_game_over = False # If there are more than two players, an instantiation/spawning algorithm 
player2_game_over = False # must be implemented for when a player enters the camera viewport. 
is_player1_dead = False
is_player2_dead = False
motiond1 = 0 
motiond2 = 0
end = 0 
time_is_up  = False

circle_color1 = (128, 128, 128)  # Grey color
circle_color2 = (0, 255, 0)  # Green color
circle_color3 = (128, 128, 128) 
circle_radius = 35
circle_thickness = 2
circle_center1 = (640, 225)
circle_center3 = (640, 150)
circle_center2 = (640, 75)

rectangle_color = (25, 25, 25) #Black
rectangle_thickness = 2
rectangle_sp1 = (600, 25)
rectangle_sp2 = (680, 275)
last_color_change_time = time.time()

def detect_person(x1, y1, x2, y2, frame_width):
    mid_x = frame_width // 2
    if x2 <= mid_x:
        return 1 
    elif x1 >= mid_x:
        return 2
    else:
        left_area = (mid_x - x1) * (y2 - y1)
        right_area = (x2 - mid_x) * (y2 - y1)
        return 1 if left_area >= right_area else 2

# def rounded_rectangle(src, top_left, bottom_right, radius=1, color=255, thickness=1, line_type=cv2.LINE_AA):
#     p1 = top_left
#     p2 = (bottom_right[1], top_left[1])
#     p3 = (bottom_right[1], bottom_right[0])
#     p4 = (top_left[0], bottom_right[0])

#     height = abs(bottom_right[0] - top_left[1])

#     if radius > 1:
#         radius = 1

#     corner_radius = int(radius * (height/2))
#     if thickness < 0:

#         #big rect
#         top_left_main_rect = (int(p1[0] + corner_radius), int(p1[1]))
#         bottom_right_main_rect = (int(p3[0] - corner_radius), int(p3[1]))

#         top_left_rect_left = (p1[0], p1[1] + corner_radius)
#         bottom_right_rect_left = (p4[0] + corner_radius, p4[1] - corner_radius)

#         top_left_rect_right = (p2[0] - corner_radius, p2[1] + corner_radius)
#         bottom_right_rect_right = (p3[0], p3[1] - corner_radius)

#         all_rects = [
#         [top_left_main_rect, bottom_right_main_rect], 
#         [top_left_rect_left, bottom_right_rect_left], 
#         [top_left_rect_right, bottom_right_rect_right]]

#         [cv2.rectangle(src, rect[0], rect[1], color, thickness) for rect in all_rects]

#     # draw straight lines
#     cv2.line(src, (p1[0] + corner_radius, p1[1]), (p2[0] - corner_radius, p2[1]), color, abs(thickness), line_type)
#     cv2.line(src, (p2[0], p2[1] + corner_radius), (p3[0], p3[1] - corner_radius), color, abs(thickness), line_type)
#     cv2.line(src, (p3[0] - corner_radius, p4[1]), (p4[0] + corner_radius, p3[1]), color, abs(thickness), line_type)
#     cv2.line(src, (p4[0], p4[1] - corner_radius), (p1[0], p1[1] + corner_radius), color, abs(thickness), line_type)

#     # draw arcs
#     cv2.ellipse(src, (p1[0] + corner_radius, p1[1] + corner_radius), (corner_radius, corner_radius), 180.0, 0, 90, color ,thickness, line_type)
#     cv2.ellipse(src, (p2[0] - corner_radius, p2[1] + corner_radius), (corner_radius, corner_radius), 270.0, 0, 90, color , thickness, line_type)
#     cv2.ellipse(src, (p3[0] - corner_radius, p3[1] - corner_radius), (corner_radius, corner_radius), 0.0, 0, 90,   color , thickness, line_type)
#     cv2.ellipse(src, (p4[0] + corner_radius, p4[1] - corner_radius), (corner_radius, corner_radius), 90.0, 0, 90,  color , thickness, line_type)

#     return src

def play_sound(sound_file): 
    pygame.mixer.init() 
    pygame.mixer.music.load(sound_file) 
    pygame.mixer.music.play() 

while True:
    ret, frame = cap.read() 
    if not ret: 
        break 

    frame = cv2.resize(frame, (1280, 720)) 
    height_f, width_f = frame.shape[:2]


    # Detect movements
    results = model(frame, stream=True) 
    fg_mask = bg_subtractor.apply(frame) 
    _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY) 

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
                if motion_value > 35000: 
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3) 
                    # motiond = 1 
                    person = detect_person(x1, y1, x2, y2, width_f)
                    if person == 1:
                        if circle_color1 == (0, 0, 255): 
                            motiond1 = 1

                        if motiond1 == 1:
                            pl1_text = "You Lost!"
                            coord = (175, 400)
                            arg1 = 2
                            arg2 = 4
                        else:
                            pl1_text = "Motion Detected"
                            coord = (20, 400)
                            arg1 = 1
                            arg2 = 2
                        cv2.putText(frame, pl1_text, coord, cv2.FONT_HERSHEY_SIMPLEX, arg1, (0, 0, 255), arg2)
                    elif person == 2:
                        if circle_color1 == (0, 0, 255): # therefore, each player should have their own 'motiond' variable
                            motiond2 = 1

                        if motiond2 == 1:
                            pl2_text = "You Lost!"
                            coord = (850, 400)
                            arg1 = 2
                            arg2 = 4
                        else:
                            pl2_text = "Motion Detected"
                            coord = (1000, 400)
                            arg1 = 1
                            arg2 = 2
                        cv2.putText(frame, pl2_text, coord, cv2.FONT_HERSHEY_SIMPLEX, arg1, (0, 0, 255), arg2)
                    cv2.putText(frame, str(person) + " person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2) 
                else: 
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3) 
                    # pl1_text = "Not Detected"
                    if motiond1 == False:
                        cv2.putText(frame, "Not Detected", (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else: 
                        cv2.putText(frame, "You Lost!", coord, cv2.FONT_HERSHEY_SIMPLEX, arg1, (0, 0, 255), arg2)

                    if motiond2 == False:
                        cv2.putText(frame, "Not Detected", (1050, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "You Lost!", coord, cv2.FONT_HERSHEY_SIMPLEX, arg1, (0, 0, 255), arg2)
                    
                #cv2.putText(frame, classNames[cls], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2) 
    
    current_time = time.time() 
    # delay = random.randint(4, 10) 
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



    cv2.line(frame, (640, 0), (640, 720), (211, 211, 211), 2)  
    cv2.rectangle(frame, rectangle_sp1, rectangle_sp2, rectangle_color, -1)
    # frame = rounded_rectangle(frame, rectangle_sp1, rectangle_sp2, color = rectangle_color, radius = 0.5, thickness=-1)
    cv2.circle(frame, circle_center1, circle_radius, circle_color1, -1)
    cv2.circle(frame, circle_center2, circle_radius, circle_color2, -1)  
    cv2.circle(frame, circle_center3, circle_radius, circle_color3, -1)  
    
    elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
    remaining_time = countdown_time - int(elapsed_time)

    # Implement detection of the number of players on screen
    players_remained = 2 

    # End the game for everyone if the game time expires
    if remaining_time <= 0:
        time_is_up = True

    if motiond1 == 1: # Implement motion detection for each player present in the camera view,
        player1_game_over = True # and assign playerN_game_over = True if motion is detected
    if motiond2 == 1: # Implement motion detection for each player present in the camera view,
        player2_game_over = True # and assign playerN_game_over = True if motion is detected

    cv2.putText(frame, "Time: {}".format(remaining_time), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    # Screen text indicating how many players are left
    cv2.putText(frame, "Players remained: {}".format(players_remained), (20, 675), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Change to Spacebar
    if cv2.waitKey(1) & 0xFF == ord(' '):
        
        cv2.putText(frame, "You Won!", (520, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, ), 4)

        win_start_time = time.time()
        while time.time() - win_start_time < 2:
            cv2.imshow('Frame', frame)
            cv2.waitKey(1)
        end = 1
        break
    
    # if (player1_game_over == True and is_player1_dead == False) or (player2_game_over == True and is_player2_dead == False):
    #     if player1_game_over == True:
    #         is_player1_dead == True
    #     if player2_game_over == True:
    #         is_player2_dead == True
    #     play_sound("gunshot.mp3")
    #     sound_delay = time.time() 

    if player1_game_over == True:
        pl1_text = "You Lost!"

    if ((player1_game_over == True and player2_game_over == True) or time_is_up == True) and end == 0: 
        cv2.putText(frame, "Game Over", (480, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4) 
        if time_is_up == True:
            play_sound("ak-47.mp3")
        game_over_start_time = time.time() 
        while time.time() - game_over_start_time < 2: 
            cv2.imshow('Frame', frame)
            cv2.waitKey(1) 
        end = 1     
        break
     
    cv2.imshow('Frame', frame) 
     
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break 


cap.release() 
cv2.destroyAllWindows()