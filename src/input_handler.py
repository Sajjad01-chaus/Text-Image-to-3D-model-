import os
import numpy as np
from src.img_process import preprocess_image
from PIL import Image
from PIL import UnidentifiedImageError
import time  

def is_image_file(input_path):
   
    print(f"is_image_file: input_path = '{input_path}'")  
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    _, ext = os.path.splitext(input_path.lower())
    print(f"is_image_file: ext = '{ext}'")  
    result = ext in image_extensions
    print(f"is_image_file: result = {result}") 
    return result

def process_input(input_value, skip_background_removal=False):
   
    print(f"process_input: input_value = '{input_value}'")  

    if not os.path.exists(input_value):
        print(f"process_input: File does not exist: {input_value}")  
        return 'text', input_value

    if os.path.isfile(input_value) and is_image_file(input_value):
        retries = 3
        for attempt in range(retries):
            try:
                img = Image.open(input_value)
                print("process_input: Image.open() successful")  
                processed_image = preprocess_image(np.array(img), skip_background_removal)
                print("process_input: preprocess_image() successful")  #
                return 'image', processed_image
            except FileNotFoundError:
                print(f"process_input: FileNotFoundError (attempt {attempt + 1}/{retries})") 
                if attempt < retries - 1:
                    time.sleep(0.1)
                else:
                    print("process_input: FileNotFoundError after retries")  
                    return 'text', input_value
            except UnidentifiedImageError:
                print("process_input: UnidentifiedImageError")  
                return 'text', input_value
            except Exception as e:
                print(f"process_input: Unexpected error: {e}") 
                return 'text', input_value
    else:
        print("process_input: Treating input as text")  
        processed_text = input_value.strip()
        return 'text', processed_text