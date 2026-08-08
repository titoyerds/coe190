import cv2
import numpy as np

def order_points(pts):
    """
    Orders the 4 corner points in top-left, top-right,
    bottom-right, bottom-left order.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def crop_and_warp_paper(image):
    """
    Finds the paper using adaptive thresholding and morphological
    operations to handle uneven lighting and shadows.
    """
    orig = image.copy()
    
    # --- Step 1: Preprocessing ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply a slightly larger blur to smooth out noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # --- Step 2: Adaptive Thresholding (The Key Step) ---
    # This handles uneven lighting and shadows by calculating
    # a threshold for local regions of the image.
    # THRESH_BINARY_INV: Makes light areas (paper) white (255)
    # and dark areas (background/lines) black (0).
    thresh = cv2.adaptiveThreshold(blurred, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 21, 5)
    
    # --- DEBUG STEP 1 ---
    print("Showing 'Adaptive Threshold'. Paper should be white (maybe noisy). Press key.")
    cv2.imshow("Adaptive Threshold", thresh)
    cv2.waitKey(0)

    # --- Step 3: Morphological Closing ---
    # The circuit lines create "gaps" in the white paper.
    # We use a large kernel to "close" these gaps and merge
    # the paper into one solid white blob.
    kernel = np.ones((15, 15), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # --- DEBUG STEP 2 ---
    print("Showing 'Closed Image'. Paper should be a solid blob. Press key.")
    cv2.imshow("Closed Image (Solid Blob)", closed)
    cv2.waitKey(0)

    # --- Step 4: Find Contours (on the 'closed' image) ---
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    paper_contour = None
    
    if len(contours) == 0:
        print("Error: No contours found after closing.")
        return None
    
    # Find the largest contour and approximate its 4 corners
    c = contours[0]
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    if len(approx) == 4:
        paper_contour = approx
    else:
        # Fallback if approximation isn't 4 points (e.g., thumb in the way)
        # This is less accurate but better than failing.
        print("Warning: Did not find 4 corners. Using bounding box of largest contour.")
        x, y, w, h = cv2.boundingRect(c)
        paper_contour = np.array([
            [[x, y]],
            [[x + w, y]],
            [[x + w, y + h]],
            [[x, y + h]]
        ], dtype=int)

    # --- DEBUG STEP 3 ---
    debug_image = orig.copy()
    cv2.drawContours(debug_image, [paper_contour], -1, (0, 255, 0), 3) # Draw green box
    print("Showing 'Contour Found'. Green box MUST be on the paper. Press key.")
    cv2.imshow("Contour Found (Press any key)", debug_image)
    cv2.waitKey(0)

    # --- Step 5: Apply Perspective Transform ---
    points = np.squeeze(paper_contour, axis=1).astype("float32")
    rect = order_points(points)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    if maxWidth <= 0 or maxHeight <= 0:
        print("Error: The contour found has no area. Cannot warp.")
        return None

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Get and apply the transformation
    M = cv2.getPerspectiveTransform(rect, dst)
    # Warp the *original color image*
    warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))
    
    # Convert the final *warped color image* to grayscale
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    return warped_gray

def upscale_and_sharpen_circuit(image, scale_factor=2, sharpen_strength=1.0):
    """
    Upscales the given grayscale image and applies a sharpening filter.

    Args:
        image (numpy.ndarray): The input grayscale image (your cropped_circuit).
        scale_factor (int): How much to scale up the image (e.g., 2 for 2x, 3 for 3x).
        sharpen_strength (float): Controls how strong the sharpening effect is.
                                  1.0 is standard. Higher values (e.g., 1.5, 2.0)
                                  will make it sharper but can introduce artifacts.

    Returns:
        numpy.ndarray: The upscaled and sharpened image.
    """
    if image is None:
        print("Error: Input image for upscaling and sharpening is None.")
        return None

    # --- 1. Upscaling ---
    # Using INTER_CUBIC for better quality interpolation during upscaling
    upscaled = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    # --- 2. Sharpening ---
    # Define a sharpening kernel
    # This is a common 3x3 sharpening kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], dtype=np.float32)

    # Apply the sharpening filter
    # cv2.filter2D takes (src, ddepth, kernel)
    # ddepth = -1 means the output image will have the same depth as the input
    sharpened = cv2.filter2D(upscaled, -1, kernel * sharpen_strength)

    # Blend the sharpened image with the original upscaled image to control strength
    # A simple blend: result = original * (1 - alpha) + sharpened * alpha
    # Where alpha is sharpen_strength (adjusting how much the sharpened version contributes)
    # For more control, we already have sharpen_strength in the kernel.
    # Let's just return the sharpened result directly for now,
    # as the kernel itself is designed to enhance edges by subtracting blurred content.

    return sharpened


# --- Main execution (Modified to include upscaling and sharpening) ---
if __name__ == "__main__":
    image = cv2.imread("captures/capture_20251104_154408.jpg")
    
    if image is None:
        print("Error: Image 'capture_20251104_144658.jpg' not found.")
    else:
        print("Showing 'Original Image'. Press any key to start.")
        cv2.imshow("Original Image (Press any key)", image)
        cv2.waitKey(0)
        
        cropped_circuit = crop_and_warp_paper(image) # This is your successful cropping part
        
        if cropped_circuit is not None:
            print("Cropping complete! Showing raw cropped circuit.")
            cv2.imshow("Raw Cropped Circuit", cropped_circuit)
            cv2.imwrite("final_warped_circuit.png", cropped_circuit)
            print("Successfully saved 'final_warped_circuit.png'")
            cv2.waitKey(0)

            # --- New: Upscale and Sharpen ---
            print("\nNow performing upscaling and sharpening...")
            # Experiment with scale_factor (e.g., 2, 3, 4) and sharpen_strength (e.g., 0.8, 1.2, 1.5)
            upscaled_sharpened_circuit = upscale_and_sharpen_circuit(cropped_circuit, 
                                                                       scale_factor=2, 
                                                                       sharpen_strength=1.2)
            
            if upscaled_sharpened_circuit is not None:
                print("Upscaling and sharpening complete! Showing final result.")
                cv2.imshow("Upscaled & Sharpened Circuit", upscaled_sharpened_circuit)
                cv2.imwrite("upscaled_sharpened_circuit.png", upscaled_sharpened_circuit)
                print("Successfully saved 'upscaled_sharpened_circuit.png'")
                cv2.waitKey(0)
            else:
                print("Failed to upscale and sharpen the circuit.")
        else:
            print("Failed to crop the image. Cannot proceed with upscaling/sharpening.")
        
        cv2.destroyAllWindows()