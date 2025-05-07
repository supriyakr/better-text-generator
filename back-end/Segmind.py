import requests

def inpaint_image_with_stable_diffusion(init_image, mask_image):
    """
    Calls the Stable Diffusion API to perform inpainting on the provided image with a mask.

    Parameters:
    init_image (str): URL of the input image to be inpainted.
    mask_image (str): URL of the mask image defining the areas for inpainting.

    Returns:
    str: URL of the generated image if successful, or an error message if the request fails.
    """
    # API Endpoint and Key
    API_URL = "https://stablediffusionapi.com/api/v3/inpaint"
    API_KEY = "SG_ae37def2d2e19962"  # Replace with your actual API key

    # Prepare the payload
    payload = {
        "key": API_KEY,
        "prompt": "Replace the text and inpaint the masked area.",
        "negative_prompt": None,
        "init_image": init_image,
        "mask_image": mask_image,
        "width": 512,  # Image width
        "height": 512,  # Image height
        "samples": 1,
        "num_inference_steps": 30,
        "guidance_scale": 7.5,
        "strength": 0.7,
        "base64": "no",  # Set to "yes" if you want the response as base64-encoded image
        "seed": None
    }

    # Make the API request
    response = requests.post(API_URL, json=payload)

    # Process the response
    if response.status_code == 200:
        result = response.json()
        return result.get("output", "No output field in response.")  # Adjust based on actual JSON structure
    else:
        return f"Error: {response.status_code} - {response.text}"

# Example usage
init_image_url = "https://example.com/your_image.jpg"  # Replace with your input image URL
mask_image_url = "https://example.com/your_mask.jpg"  # Replace with your mask image URL

output = inpaint_image_with_stable_diffusion(init_image_url, mask_image_url)
print("Generated Image URL or Error:", output)