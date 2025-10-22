from PIL import Image
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from gtts import gTTS
import gradio as gr
import os

# Image captioning (BLIP)
print("Loading BLIP model...")
captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

# Story generation (FLAN-T5)
print("Loading FLAN-T5 model...")
story_model_name = "google/flan-t5-small"
story_tokenizer = AutoTokenizer.from_pretrained(story_model_name)
story_model = AutoModelForSeq2SeqLM.from_pretrained(story_model_name)

print(" All models loaded successfully!")

def image_to_caption(pil_image):
    """Generate a caption from an image using BLIP."""
    caption_result = captioner(pil_image)
    caption_text = caption_result[0]['generated_text']
    return caption_text

def generate_story(caption_text, style="warm and playful", length="short"):
    """Generate a short, child-friendly story from a caption using FLAN-T5."""
    prompt = f"Write a {length}, child-friendly story in a {style} tone based on: {caption_text}"
    input_ids = story_tokenizer(prompt, return_tensors="pt").input_ids
    outputs = story_model.generate(input_ids, max_length=200, num_beams=4, early_stopping=True)
    story = story_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return story

def text_to_speech(text, outpath="story.mp3", lang="en"):
    """Convert text to speech using gTTS."""
    tts = gTTS(text=text, lang=lang)
    tts.save(outpath)
    return outpath

def run_pipeline(uploaded_image, voice_lang="en"):
    """Main pipeline: image → caption → story → audio"""
    if uploaded_image is None:
        return "No image uploaded", "", None
    
    # If uploaded_image is already a PIL image
    if isinstance(uploaded_image, Image.Image):
        img = uploaded_image
    else:
        img = Image.open(uploaded_image).convert("RGB")
    
    caption = image_to_caption(img)
    story = generate_story(caption)
    
    # Create temp directory if it doesn't exist
    os.makedirs("temp", exist_ok=True)
    mp3_path = f"temp/story_{abs(hash(story))%10_000}.mp3"
    text_to_speech(story, mp3_path, lang=voice_lang)
    
    return caption, story, mp3_path

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🖼️ Image → Short Audio Story")
    gr.Markdown("Upload an image and get a creative story with audio narration!")
    
    with gr.Row():
        img_in = gr.Image(label="Upload an image", type="pil")
        
        with gr.Column():
            lang = gr.Dropdown(
                ["en", "hi", "es"], 
                value="en", 
                label="TTS Language",
                info="English, Hindi, or Spanish"
            )
            btn = gr.Button("Generate Story & Audio", variant="primary")
            caption_out = gr.Textbox(label="Image Caption", interactive=False)
            story_out = gr.Textbox(label="Generated Story", interactive=False, lines=5)
            audio_out = gr.Audio(label="Audio Story")
    
    btn.click(
        run_pipeline, 
        inputs=[img_in, lang], 
        outputs=[caption_out, story_out, audio_out]
    )
    
    gr.Markdown("### How it works:")
    gr.Markdown("1. Upload an image\n2. Select your preferred language\n3. Click 'Generate Story & Audio'\n4. Get a caption, story, and downloadable audio!")

if __name__ == "__main__":
    demo.launch()
