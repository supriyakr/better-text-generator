import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import easyocr
from google.cloud import vision
import subprocess
import os
from Segmind import inpaint_image_with_stable_diffusion

# Set the path to your service account JSON key
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./back-end/dev-acolyte-442123-f9-724f681a960e.json"

def run_lama_inpainting(image_path, mask_path, output_path, lama_path="./back-end/lama"):
    """
    Run LaMa inpainting on an image with a mask.
    :param image_path: Path to the input image.
    :param mask_path: Path to the binary mask.
    :param output_path: Path to save the inpainted image.
    :param lama_path: Path to the LaMa repository.
    """
    # Prepare LaMa input directory
    input_dir = os.path.join(lama_path, "data")
    os.makedirs(input_dir, exist_ok=True)
    output_dir = os.path.join(lama_path, "output")
    os.makedirs(output_dir, exist_ok=True)
  

    # Copy input image and mask to LaMa's input directory
    os.system(f"cp {image_path} {input_dir}/image.png")
    os.system(f"cp {mask_path} {input_dir}/mask.png")

    # Run LaMa inpainting using the specified model checkpoint
    lama_command = (
        f"python {lama_path}/bin/predict.py "
        f"--config {lama_path}/configs/prediction/default.yaml"
        f"model.path={lama_path}/big-lama/best.ckpt"
        f"indir={input_dir} "
        f"outdir={output_dir}/{output_path} "
    )
    # Run the command
    result = subprocess.run(lama_command, shell=True, text=True, capture_output=True)

    if result.returncode != 0:
        raise RuntimeError(f"LaMa inpainting failed: {result.stderr}")

    # Copy the output image to the desired path
    inpainted_image_path = os.path.join(output_dir, "image.png")
    if not os.path.exists(inpainted_image_path):
        raise FileNotFoundError(f"Inpainted image not found at {inpainted_image_path}")

    print(f"Inpainted image saved to {output_path}")

def preprocess_image_for_ocr(image):
    """
    Preprocess the image for OCR without affecting the original image's color.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Denoise and sharpen
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 30, 7, 21)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # Sharpen kernel
    sharpened = cv2.filter2D(denoised, -1, kernel)

    return sharpened

def detect_text_with_easyocr(image_path):
    """
    Detect text and bounding boxes using EasyOCR.
    """
    # Load the original image (retains color for output)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Failed to load image. Check the file path.")

    # Preprocess a copy of the image for OCR
    preprocessed_image = preprocess_image_for_ocr(image)

    # Initialize EasyOCR
    reader = easyocr.Reader(['en'])
    results = reader.readtext(preprocessed_image)

    # Parse results into bounding boxes
    boxes = []
    for (bbox, text, prob) in results:
        if prob > 0.5:  # Confidence threshold
            (top_left, top_right, bottom_right, bottom_left) = bbox
            x_min = int(min(top_left[0], bottom_left[0]))
            y_min = int(min(top_left[1], top_right[1]))
            x_max = int(max(bottom_right[0], top_right[0]))
            y_max = int(max(bottom_right[1], bottom_left[1]))
            w = x_max - x_min
            h = y_max - y_min
            boxes.append({"x": x_min, "y": y_min, "w": w, "h": h, "text": text.strip()})

    return image, boxes

def detect_text_with_google_ocr(image_path):
    """
    Detect text and bounding boxes using Google Vision OCR.
    """
    client = vision.ImageAnnotatorClient()

    # Load the image into Google Vision
    with open(image_path, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)

    boxes = []
    # Skip the first annotation (full text) and process the rest
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

def create_mask(image, boxes):
    """
    Remove text from the given image using inpainting.
    """
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for box in boxes:
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]

        # Expand the box slightly to capture any residual pixels
        padding = int(0.02 * max(w, h))  # Adjust padding as needed
        x1 = max(x - padding, 0)
        y1 = max(y - padding, 0)
        x2 = min(x + w + padding, image.shape[1] - 1)
        y2 = min(y + h + padding, image.shape[0] - 1)

        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    return mask

def insert_new_text(image, inpainted_image, boxes, new_text, font_path, font_size=30):
    """
    Insert new text into the image at the position of the first text box only.
    Remove other texts.
    """
    # Convert the inpainted image to PIL format
    pil_image = Image.fromarray(cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    # Insert new text at the position of the first text box
    if boxes:
        first_box = boxes[0]
        x, y, w, h = first_box["x"], first_box["y"], first_box["w"], first_box["h"]

        current_font_size = font_size
        font = ImageFont.truetype(font_path, current_font_size)

        bbox = font.getbbox(new_text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Adjust font size to fit within the box
        while (text_width > w or text_height > h) and current_font_size > 1:
            current_font_size -= 1
            font = ImageFont.truetype(font_path, current_font_size)
            bbox = font.getbbox(new_text)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

        # Center the text within the box
        text_x = x + (w - text_width) / 2 - bbox[0]
        text_y = y + (h - text_height) / 2 - bbox[1]

        text_x = max(text_x, 0)
        text_y = max(text_y, 0)

        draw.text((text_x, text_y), new_text, font=font, fill="Black")

    return pil_image

def process_image_with_layered_ocr(image_path, output_path, new_text, font_path):
    """
    Perform layered OCR using EasyOCR and Google Vision.
    """
    # Step 1: EasyOCR for initial detection
    image, easyocr_boxes = detect_text_with_easyocr(image_path)

    # Step 2: Google OCR for refined detection
    google_ocr_boxes = detect_text_with_google_ocr(image_path)

    # Combine the boxes from both OCR methods
    all_boxes = easyocr_boxes + google_ocr_boxes

    # Remove text using the combined boxes
    mask = create_mask(image, all_boxes)

    # Save the input and mask images
    cv2.imwrite("input_image.png", image)
    cv2.imwrite("mask_image.png", mask) 

    # Run LaMa inpainting
    # Run LaMa
   # run_lama_inpainting("input_image.png", "mask_image.png", "inpainted_output.png")
    response = inpaint_image_with_stable_diffusion(image, mask);

    # Load the result for further processing
    inpainted_image = cv2.imread("inpainted_output.png")
    


    # Add new text at the position of the first detected text
    final_image = insert_new_text(image, inpainted_image, all_boxes, new_text, font_path)

    # Save the final image
    final_image.save(output_path)
    print(f"Processed image saved to {output_path}")

# Example Usage
process_image_with_layered_ocr(
    image_path="./back-end/media/input/poster11.png",
    output_path="./back-end/media/output/poster11.png",
    new_text="Everyone hopes for the Best",  # Provide the new text you want to insert
    font_path="./back-end/fonts/Arima.ttf"  # Path to your font file
)
