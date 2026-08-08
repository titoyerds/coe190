"""
Detect hand gestures on streams.

Usage:
    $ python gesture.py --mode single
"""

import cv2
import time
import os

from utils.utils import enhance_image
from circuit_detection import detect_circuit

CAM_W = 1280
CAM_H = 720
TEXT_COLOR = (243,236,27)

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, CAM_W)
    cap.set(4, CAM_H)

    # Create capture folder and cooldown
    os.makedirs("captures", exist_ok=True)
    capture_cooldown = 3.0  # seconds between captures
    last_capture_time = time.time()

    text_duration = 2.0  # How long to show the text (in seconds)
    show_capture_text_until = 0.0 # This will store the time when the text should disappear

    # CAPTURE THRESHOLD
    # ** YOU MUST TUNE THIS VALUE **
    # This is the minimum pixel area of the paper to trigger a capture.
    # Hold the paper up to the camera and check the 'Area: X' text to find a good value.
    CAPTURE_AREA_THRESHOLD = 100000

    while True:
        _, img = cap.read()
        img = cv2.flip(img, 1)

        display_img = img.copy()

        # call your detect_circuit(img) function
        paper_box = detect_circuit(img)
        if paper_box is not None:
            # cv2.polylines(img, [paper_box], True, (0,255,0), 3)

            # Calculate the area
            current_area = cv2.contourArea(paper_box)

            # Optional: Display the current area to help you tune the threshold
            cv2.putText(display_img, f'Area: {int(current_area)}', (50, 50), 0, 0.8, TEXT_COLOR, 2, lineType=cv2.LINE_AA)

            # Check if area is above threshold AND cooldown has passed
            if current_area > CAPTURE_AREA_THRESHOLD:
                current_time = time.time()
                if current_time - last_capture_time > capture_cooldown:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join("captures", f"capture_{timestamp}.jpg")
                    cv2.imwrite(filename, img)
                    print(f"[INFO] Frame captured: {filename}")

                    enhanced = enhance_image(img, scale_factor=1.25, sharpen_strength=1.2)
                    cv2.imshow("Enhanced Circuit", enhanced)
                    cv2.imwrite("enhanced_circuit.png", enhanced)
                    show_capture_text_until = current_time + text_duration

                    last_capture_time = current_time

        if time.time() < show_capture_text_until:
            cv2.putText(display_img, "Frame Captured!", (50, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3, lineType=cv2.LINE_AA)
        
        cv2.imshow('Frame Capture', display_img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            cv2.destroyAllWindows()
            break

main()
