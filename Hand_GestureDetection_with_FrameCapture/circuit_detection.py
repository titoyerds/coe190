import cv2
import numpy as np

def detect_circuit_box(image):
    """
    Finds the largest circuit-like contour in an image.

    """
    
    # --- Calculate max area ---
    img_height, img_width, _ = image.shape
    # Set a max area threshold (e.g., 60% of total screen)
    # This stops the code from detecting the entire wall.
    # You can tune this 0.6 value.
    max_area = (img_height * img_width) * 0.4
    
    # --- Step 1: Preprocessing ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    
    # --- Step 2: Adaptive Thresholding ---
    thresh = cv2.adaptiveThreshold(blurred, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 21, 5)
    
    # Remove Noise
    open_kernel = np.ones((5, 5), np.uint8)
    thresh_opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, open_kernel)
    
    # --- Step 3: Morphological Closing ---
    kernel = np.ones((15, 15), np.uint8)
    closed = cv2.morphologyEx(thresh_opened, cv2.MORPH_CLOSE, kernel)

    # --- Step 4: Find Contours ---
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None  # No contours found

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_contour = contours[0]

    # --- Step 5: Check and Return Contour ---
    
    # Get the area of the largest contour
    contour_area = cv2.contourArea(largest_contour)

    # Add max_area check ---
    # We now check if the contour is too small (noise)
    # OR if it's too big (the background wall).
    
    # Tune this min value (5000) if needed
    min_area = 30000 
    
    if contour_area < min_area or contour_area > max_area:
        return None

    # If it passes both checks, it's probably the paper
    peri = cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)
    
    return approx