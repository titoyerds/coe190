import cv2
import numpy as np

def get_latest_capture(folder="captures", ext=".jpg"):
    import os
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(ext)]
    return max(files, key=os.path.getmtime) if files else None

def enhance_image(image, scale_factor=1.5, sharpen_strength=1.0):
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