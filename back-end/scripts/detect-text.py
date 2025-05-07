from dataclasses import dataclass
from typing import Sequence
from PIL import Image
import pytesseract

@dataclass
class TextBox:
    # (x, y) is the top left corner of a rectangle; the origin of the coordinate system is the top-left of the image.
    # x denotes the vertical axis, y denotes the horizontal axis.
    x: int
    y: int
    h: int
    w: int
    text: str = None

class TextDetector:
    def detect_text(self, image_filename: str) -> Sequence[TextBox]:
        pass

class TesseractTextDetector(TextDetector):
    """Uses the `tesseract` OCR library from Google to do text detection."""

    def __init__(self, tesseract_path: str):
        """
        Args:
          tesseract_path: The path where the `tesseract` library is installed, e.g., "/usr/local/bin/tesseract".
        """
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

    def detect_text(self, image_filename: str) -> Sequence[TextBox]:
        image = Image.open(image_filename)
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        boxes = [
            TextBox(left, top, width, height, text)
            for left, top, width, height, text in zip(
                data["left"], data["top"], data["width"], data["height"], data["text"]
            )
            if text.strip()  # Only keep non-empty text boxes
        ]
        return boxes

# Example usage
def get_text_coordinates(detector: TextDetector, image_filename: str) -> Sequence[TextBox]:
    """
    Detects text in the given image and returns a list of TextBox objects containing
    the coordinates and text for each detected text box.
    """
    return detector.detect_text(image_filename)

# Specify the Tesseract path based on where it's installed on your system.
tesseract_detector = TesseractTextDetector(tesseract_path="/opt/homebrew/bin/tesseract")

# Use an image file of your choice here
text_boxes = get_text_coordinates(tesseract_detector, "../media/output/output_image.png")

# Print the coordinates and text
for box in text_boxes:
    print(f"Text: {box.text}, Coordinates: ({box.x}, {box.y}), Width: {box.w}, Height: {box.h}")