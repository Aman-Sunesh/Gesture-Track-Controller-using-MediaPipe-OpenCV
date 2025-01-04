import cv2
import mediapipe as mp
import numpy as np
import time
import sys
import pygame
import HandTrackingModule as htm

pTime = 0
camera_index = 0

audioList = ['1.mp3', '2.mp3', '3.mp3', '4.mp3', '5.mp3']
current_track = 0

# Constants
TRIGGER_THRESHOLD = 10  # Increase the threshold for more stable gesture detection
COOLDOWN_PERIOD = 1.0  # Cooldown period in seconds after a gesture triggers an action

last_triggered_time = 0  # Time when the last gesture was triggered

pygame.mixer.init()
pygame.mixer.music.load(audioList[current_track])


cap = None
while camera_index <= 10:
    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        print(f"Camera opened with index {camera_index}")
        break
    print(f"Failed to open camera with index {camera_index}. Trying next index...")
    camera_index += 1

if not cap or not cap.isOpened():
    print("Error: Could not open any webcam.")
    sys.exit()


detector = htm.handDetector(
    static_image_mode=False,
    max_num_hands=1,         
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


def loadNextTrack():
    global current_track
    current_track = (current_track + 1) % len(audioList)
    pygame.mixer.music.load(audioList[current_track])
    pygame.mixer.music.play()
    print(f"Playing Next Track: {audioList[current_track]}")

def loadPreviousTrack():
    global current_track
    current_track = (current_track - 1) % len(audioList)
    pygame.mixer.music.load(audioList[current_track])
    pygame.mixer.music.play()
    print(f"Playing Previous Track: {audioList[current_track]}")

def volumeUp():
    volume = pygame.mixer.music.get_volume() + 0.1
    pygame.mixer.music.set_volume(min(volume, 1.0))
    print(f"Volume Up: {pygame.mixer.music.get_volume():.2f}")

def volumeDown():
    volume = pygame.mixer.music.get_volume() - 0.1
    pygame.mixer.music.set_volume(max(volume, 0.0))
    print(f"Volume Down: {pygame.mixer.music.get_volume():.2f}")


stable_gesture = None
stable_count = 0
TRIGGER_THRESHOLD = 5  # how many consecutive frames before we trigger an action
last_triggered_gesture = None  # so we don't trigger the same gesture again and again in one hold

def trigger_gesture_action(gesture_id):
    global last_triggered_gesture, last_triggered_time

    current_time = time.time()
    if current_time - last_triggered_time < COOLDOWN_PERIOD:
        return  # Skip if we're in cooldown

    # Avoid re-triggering the same gesture if user holds it
    if gesture_id == last_triggered_gesture:
        return

    # Define actions based on gesture_id
    gesture_actions = {
        0: lambda: (pygame.mixer.music.pause(), print("Paused audio")),
        1: lambda: (pygame.mixer.music.unpause() if pygame.mixer.music.get_busy() else pygame.mixer.music.play(), print("Unpaused or Started audio")),
        2: loadNextTrack,
        3: loadPreviousTrack,
        4: volumeUp,
        5: volumeDown
    }

    # Execute the action if it exists
    action = gesture_actions.get(gesture_id)
    if action:
        action()
        last_triggered_gesture = gesture_id
        last_triggered_time = current_time  # Update the last triggered time


while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    frame = cv2.flip(frame, 1)
    frame, results = detector.findHands(frame, draw=True)

    if results and hasattr(results, 'multi_hand_landmarks') and results.multi_hand_landmarks:
        num_fingers = detector.count_fingers(frame)
    else:
        num_fingers = -1  # No hand detected

    # Debounce logic
    if num_fingers == -1:
        stable_gesture = None
        stable_count = 0
    else:
        if num_fingers == stable_gesture:
            stable_count += 1
        else:
            stable_gesture = num_fingers
            stable_count = 1

        if stable_count >= TRIGGER_THRESHOLD:
            trigger_gesture_action(stable_gesture)

    # Calculate & display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime
    cv2.putText(frame, f'FPS: {int(fps)}',
                (10, 45), cv2.FONT_HERSHEY_PLAIN,
                2, (0, 255, 0), 2)

    if num_fingers >= 0:
        cv2.putText(frame, f'Fingers: {num_fingers}',
                    (10, 85), cv2.FONT_HERSHEY_PLAIN,
                    2, (255, 0, 0), 2)

    cv2.imshow('Hand Audio Control', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
