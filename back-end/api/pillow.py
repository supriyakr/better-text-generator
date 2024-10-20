from fastapi import FastAPI
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from fastapi.responses import StreamingResponse

app = FastAPI()

# Set the font path (if you're using custom fonts, place them in a `fonts/` folder)
FONT_PATH = "fonts/Arima.ttf"  # Update this to the correct path for your font

@app.post("/generate-image/")
async def generate_image(text: str, font_size: int = 20):
    # Create a blank image with white background
    image = Image.new("RGB", (400, 200), "white")
    draw = ImageDraw.Draw(image)

    # Load the font and define the size
    font = ImageFont.truetype(FONT_PATH, font_size)

    # Position the text in the center of the image
    text_width, text_height = draw.textsize(text, font=font)
    position = ((image.width - text_width) // 2, (image.height - text_height) // 2)
    
    # Add the text to the image
    draw.text(position, text, fill="black", font=font)

    # Save the image to a bytes buffer
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    # Return the image as a response
    return StreamingResponse(buffer, media_type="image/png")

