import streamlit as st
from PIL import Image
import time

# Title and description
st.title("AI-Powered Image Text Replacement Demo")
st.write("""
This app simulates how AI can identify garbled text in an image, clear it, and allow the user to replace it with desired text. 
Follow the steps below to see how it works!
""")

# Step 1: Upload image
st.header("Step 1: Upload Your Image")
uploaded_file = st.file_uploader("Upload an image with garbled text (JPEG/PNG format)", type=["jpeg", "png", "jpg"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image with Garbled Text", use_column_width=True)
    st.success("Image uploaded successfully!")
    
    # Adding delay to simulate processing
    with st.spinner("Processing the uploaded image to detect garbled text..."):
        time.sleep(3)  # Simulate detection delay
    
    # Step 2: Display cleared image
    st.header("Step 2: AI Clears the Garbled Text")
    st.write("The AI has identified and cleared the garbled text. Here is the updated image:")
    
    cleared_image_path = "./cleared-image.png"
    cleared_image = Image.open(cleared_image_path)
    st.image(cleared_image, caption="Image with Text Cleared", use_column_width=True)
    
    # Adding delay to simulate further processing
    with st.spinner("Preparing the cleared image..."):
        time.sleep(2)  # Simulate clearing delay
    
    # Step 3: User inputs desired text
    st.header("Step 3: Replace the Text")
    user_input = st.text_input("Enter the text you want to replace in the cleared area:")
    
    if user_input:
        st.success("Text entered successfully! Generating the updated image...")
        
        # Adding delay to simulate image generation
        with st.spinner("Generating the updated image with your text..."):
            time.sleep(3)  # Simulate text replacement delay

        # Step 4: Display image with new text
        st.header("Step 4: Updated Image with Your Text")
        new_text_image_path = "./new-text-image.png"
        new_text_image = Image.open(new_text_image_path)
        st.image(new_text_image, caption="Updated Image with Your Text", use_column_width=True)
else:
    st.info("Please upload an image to begin the process.")