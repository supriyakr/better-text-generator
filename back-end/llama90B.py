import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests
import json
import os
import base64

# Set your API key and endpoint
OPENROUTER_API_KEY = "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"
YOUR_SITE_URL = "your_site_url"
YOUR_APP_NAME = "your_app_name"

def preprocess_image_for_ocr(image):
    """
    Preprocess the image for OCR without affecting the original image's color.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 30, 7, 21)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # Sharpen kernel
    sharpened = cv2.filter2D(denoised, -1, kernel)
    return sharpened

def detect_text_with_easyocr(image_path):
    """
    Detect text and bounding boxes using EasyOCR.
    """
    import easyocr
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Failed to load image. Check the file path.")
    preprocessed_image = preprocess_image_for_ocr(image)
    reader = easyocr.Reader(['en'])
    results = reader.readtext(preprocessed_image)
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

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
    # Prepend the required format header
    return f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('utf-8')}"

def remove_text_with_llama_vision(image_path, new_text):
    """
    Replace text in an image using the Llama Vision API.
    """
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": YOUR_SITE_URL,
        "X-Title": YOUR_APP_NAME,
    }
    encoded_image = encode_image_to_base64(image_path)
    
    data = {
        "model": "meta-llama/llama-3.2-90b-vision-instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                   {"type": "text", "text": f"Remove the text in this image"},
                    {"type": "image_url", "image_url": {"url": encoded_image}}
                ]
            }
        ]
    }
    
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        data=json.dumps(data)
    )
    
    if response.status_code == 200:
        output_data = response.json()
        choices = output_data.get("choices", [])
        if choices and "message" in choices[0]:
            message_content = choices[0]["message"]["content"]
            return message_content  # URL of the processed image
    else:
        raise Exception(f"API call failed with status code {response.status_code}: {response.text}")

def process_image_with_layered_ocr(image_path, output_path, new_text, font_path):
    """
    Process the image using Llama Vision for inpainting.
    """
    # Step 1: EasyOCR for bounding boxes
    image, easyocr_boxes = detect_text_with_easyocr(image_path)

    # Step 2: Replace text using Llama Vision API
    processed_image_url = remove_text_with_llama_vision(image_path, new_text)

    # Download the processed image
    response = requests.get(processed_image_url)
    if response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"Processed image saved to {output_path}")
    else:
        raise Exception(f"Failed to download processed image: {response.status_code}")

# Example Usage
process_image_with_layered_ocr(
    image_path="./back-end/media/input/poster11.png",
    output_path="./back-end/media/output/poster11.png",
    new_text="Everyone hopes for the Best",
    font_path="./back-end/fonts/Arima.ttf"
)