import cv2
import numpy as np
import os
from PIL import Image  # Importing PIL for consistency

def resize_image(image, target_size=(512, 512)):
   
    height, width = image.shape[:2]

    # aspect ratio
    aspect_ratio = width / height

    # new dimensions
    if width > height:
        new_width = target_size[0]
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = target_size[1]
        new_width = int(new_height * aspect_ratio)

    # Ensuring dimensions don't exceed target size
    new_width = min(new_width, target_size[0])
    new_height = min(new_height, target_size[1])

    # Resize image
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # black canvas of the target size
    canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    # Calculating offsets to center the image in the canvas
    x_offset = (target_size[0] - new_width) // 2
    y_offset = (target_size[1] - new_height) // 2

    # Placing the resized image on the canvas
    canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized

    return canvas

def remove_background(image):
   
    mask = np.zeros(image.shape[:2], np.uint8)

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    height, width = image.shape[:2]
    border = 10  
    rect = (border, border, width - 2*border, height - 2*border)

    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    result = image * mask2[:, :, np.newaxis]

    # Replacing black background with white (for better processing)
    white_background = np.ones_like(image, np.uint8) * 255
    result = np.where(result == 0, white_background, result)

    return result

def preprocess_image(image_array, skip_background_removal=False):
  
    print(f"preprocess_image: image_array.shape = {image_array.shape}")  
    image = resize_image(image_array)
    print("resize_image() successful") 

    if not skip_background_removal:
        try:
            image = remove_background(image)
            print("remove_background() successful")  
        except Exception as e:
            print(f"Warning: Background removal failed: {e}")
            print("Proceeding with original image.")

    print("Preprocessing completed successfully")  
    return image

