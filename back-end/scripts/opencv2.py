import cv2
import numpy as np
import pytesseract
from pytesseract import Output

def remove_fixed_position_text(image_path, output_path):
    # Load the original image
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image. Check the file path.")
        return

    # Define the fixed position of the text (coordinates need to be manually adjusted)
    # Example: Assuming "HELLO WORLD" text in the provided image
    text_region = (100, 140, 200, 120)  # (x, y, width, height)

    # Create a mask for the text region
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    x, y, w, h = text_region
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    # Inpaint the original image using the text mask
    inpainted_image = cv2.inpaint(image, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)

    # Save the result
    cv2.imwrite(output_path, inpainted_image)
    print(f"Text removed and saved to {output_path}")

# Example usage
remove_fixed_position_text("../media/input/hello.png", "../media/output/out.png")