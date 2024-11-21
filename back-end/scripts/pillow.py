from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import argparse

# Set the font path (update to your font file location)
  # Update this to the correct path for your font

def generate_image(text: str, font_size: int = 50):
   
    # Open an existing image
    input_image_path =  "../media/input/sample.jpg"
    image = Image.open(input_image_path)
    
    # Create an ImageDraw object
    draw = ImageDraw.Draw(image)
    
  
    output_path="../media/output/output_image.png"
    image.save(output_path)
    FONT_PATH = "/fonts/Arima.ttf"

    # Load the font and define the size
    font = ImageFont.truetype(FONT_PATH, font_size)
    

    # Get the bounding box of the text
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

    # Position the text in the center of the image
    position = ((image.width - text_width) // 2, (image.height - text_height) // 2)
    
    # Add the text to the image
    draw.text(position, text, fill="white", font=font)

    # Save the image to the specified output path
 
    image.save(output_path)
    print(f"Image saved at {output_path}")

generate_image(text= "Baking Classes!!")

