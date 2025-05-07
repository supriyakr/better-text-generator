import cv2
import numpy as np
import pytesseract
from pytesseract import Output

def remove_text(image_path, output_path):
    # Load the original image
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image. Check the file path.")
        return
  # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast further if needed
    gray = cv2.convertScaleAbs(gray, alpha=2.5, beta=0)

    # Apply adaptive thresholding to highlight text regions
    gray = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Check if text is detected
    detected_text = pytesseract.image_to_string(gray)
    if not detected_text.strip():
        print("No text detected. Check the image or preprocessing.")
        cv2.imwrite(output_path, image)  # Save the original image
        return

    # Use Tesseract to detect text regions
    d = pytesseract.image_to_data(gray, output_type=Output.DICT, config="--psm 6")
    
    # Create an empty mask
    mask = np.zeros_like(gray, dtype=np.uint8)

    # Iterate through detected text regions
    for i in range(len(d['text'])):
        if int(d['conf'][i]) > 40:  # Confidence threshold
            # Get the bounding box coordinates
            x, y, w, h = d['left'][i], d['top'][i], d['width'][i], d['height'][i]
            # Draw rectangles on the mask
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    # Debug: Save the mask for inspection
    cv2.imwrite("debug_mask.jpg", mask)

    # Inpaint the original image using the text mask
    inpainted_image = cv2.inpaint(image, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    
    # Save the result
    cv2.imwrite(output_path, inpainted_image)
    print(f"Text removed and saved to {output_path}")

# Example usage
remove_text("../media/output/out.png", "../media/output/out.png")