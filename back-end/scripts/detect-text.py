def get_text_coordinates(detector: TextDetector, image_filename: str) -> Sequence[TextBox]:
    """
    Detects text in the given image and returns a list of TextBox objects containing
    the coordinates and text for each detected text box.
    
    Args:
        detector: An instance of TextDetector (either AzureTextDetector or TesseractTextDetector).
        image_filename: Path to the image file.

    Returns:
        List of TextBox objects with the coordinates and detected text.
    """
    return detector.detect_text(image_filename)

# Example usage with TesseractTextDetector
tesseract_detector = TesseractTextDetector(tesseract_path="/usr/bin/tesseract")
text_boxes = get_text_coordinates(tesseract_detector, "path/to/image.jpg")

# Print the coordinates and text
for box in text_boxes:
    print(f"Text: {box.text}, Coordinates: ({box.x}, {box.y}), Width: {box.w}, Height: {box.h}")