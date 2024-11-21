import cv2
import numpy as np
import os
import easyocr
from google.cloud import vision
from model import Model  # SC-FEGAN model

# Set the path to your Google Cloud Service Account JSON key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./back-end/dev-acolyte-442123-f9-724f681a960e.json"

# SC-FEGAN Config
class Config:
    INPUT_SIZE = 256  # SC-FEGAN input size
    BATCH_SIZE = 1  # Batch size for processing
    CKPT_DIR = './checkpoints/'  # Path to SC-FEGAN checkpoint files

# Initialize SC-FEGAN Model
sc_fegan_model = Model(Config)
sc_fegan_model.load_demo_graph(Config)

def preprocess_image_for_ocr(image):
    """
    Preprocess the image for OCR without affecting the original image's color.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 30, 7, 21)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    return sharpened

def detect_text_with_easyocr(image_path):
    """
    Detect text and bounding boxes using EasyOCR.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Failed to load image. Check the file path.")
    preprocessed_image = preprocess_image_for_ocr(image)
    reader = easyocr.Reader(['en'])
    results = reader.readtext(preprocessed_image)
    boxes = []
    for (bbox, text, prob) in results:
        if prob > 0.5:
            (top_left, top_right, bottom_right, bottom_left) = bbox
            x_min = int(min(top_left[0], bottom_left[0]))
            y_min = int(min(top_left[1], top_right[1]))
            x_max = int(max(bottom_right[0], top_right[0]))
            y_max = int(max(bottom_right[1], bottom_left[1]))
            w = x_max - x_min
            h = y_max - y_min
            boxes.append({"x": x_min, "y": y_min, "w": w, "h": h, "text": text.strip()})
    return boxes

def detect_text_with_google_ocr(image_path):
    """
    Detect text and bounding boxes using Google Vision OCR.
    """
    client = vision.ImageAnnotatorClient()
    with open(image_path, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    boxes = []
    for text in response.text_annotations[1:]:
        vertices = text.bounding_poly.vertices
        x_min = int(min([vertex.x for vertex in vertices]))
        y_min = int(min([vertex.y for vertex in vertices]))
        x_max = int(max([vertex.x for vertex in vertices]))
        y_max = int(max([vertex.y for vertex in vertices]))
        w = x_max - x_min
        h = y_max - y_min
        boxes.append({"x": x_min, "y": y_min, "w": w, "h": h, "text": text.description.strip()})
    return boxes

def merge_boxes(boxes):
    """
    Merge overlapping bounding boxes using Intersection over Union (IoU).
    """
    def iou(box1, box2):
        x1, y1, w1, h1 = box1["x"], box1["y"], box1["w"], box1["h"]
        x2, y2, w2, h2 = box2["x"], box2["y"], box2["w"], box2["h"]
        xi1, yi1 = max(x1, x2), max(y1, y2)
        xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area, box2_area = w1 * h1, w2 * h2
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0
    merged = []
    while boxes:
        box = boxes.pop(0)
        overlapping = [b for b in boxes if iou(box, b) > 0.5]
        for b in overlapping:
            boxes.remove(b)
        merged.append(box)
    return merged

def create_text_mask(image, boxes):
    """
    Create a binary mask for the detected text areas.
    """
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for box in boxes:
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        padding = int(0.02 * max(w, h))  # Add slight padding
        x1, y1 = max(0, x - padding), max(0, y - padding)
        x2, y2 = min(image.shape[1], x + w + padding), min(image.shape[0], y + h + padding)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    return mask

def inpaint_with_sc_fegan(image, mask):
    """
    Inpaint the image using SC-FEGAN.
    """
    batch = np.zeros([1, Config.INPUT_SIZE, Config.INPUT_SIZE, 9], dtype=np.float32)
    resized_image = cv2.resize(image, (Config.INPUT_SIZE, Config.INPUT_SIZE)) / 127.5 - 1  # Normalize
    resized_mask = cv2.resize(mask, (Config.INPUT_SIZE, Config.INPUT_SIZE)) / 255.0
    batch[0, :, :, :3] = resized_image
    batch[0, :, :, 7:8] = resized_mask[:, :, np.newaxis]
    inpainted_output = sc_fegan_model.demo(Config, batch)
    return (inpainted_output[0] * 127.5 + 127.5).astype(np.uint8)

def process_image_with_layered_ocr(image_path, output_path):
    """
    Perform layered OCR, mask creation, and inpainting with SC-FEGAN.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Failed to load image. Check the file path.")

    # Detect text using EasyOCR
    easyocr_boxes = detect_text_with_easyocr(image_path)

    # Detect text using Google Vision OCR
    google_ocr_boxes = detect_text_with_google_ocr(image_path)

    # Merge boxes
    all_boxes = merge_boxes(easyocr_boxes + google_ocr_boxes)

    # Create mask
    mask = create_text_mask(image, all_boxes)

    # Inpaint with SC-FEGAN
    inpainted_image = inpaint_with_sc_fegan(image, mask)

    # Save output
    cv2.imwrite(output_path, inpainted_image)
    print(f"Processed image saved to {output_path}")

# Example usage
process_image_with_layered_ocr(
    image_path="./back-end/media/input/testing2.jpg",
    output_path="./back-end/media/output/text_removed_output.png"
)
