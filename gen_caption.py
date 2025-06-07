from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch

# Load expressive captioning model
caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
caption_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning", use_fast=True)
caption_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning", use_fast=True)

def generate_caption(pil_image, seed):
    """
    Generate a single caption for the given image based on the specified seed.
    """
    # Preprocess the image
    pixel_values = caption_processor(images=pil_image, return_tensors="pt").pixel_values

    with torch.no_grad():
        if seed == 1:
            # Sampling with temperature
            output_ids = caption_model.generate(pixel_values, max_length=100, do_sample=True, temperature=1.0)
        elif seed == 2:
            # Default greedy decoding
            output_ids = caption_model.generate(pixel_values, max_length=100)
        elif seed == 3:
            # Sampling with higher temperature
            output_ids = caption_model.generate(pixel_values, max_length=100, do_sample=True, temperature=1.5)
        elif seed == 4:
            # Top-k sampling
            output_ids = caption_model.generate(pixel_values, max_length=100, do_sample=True, top_k=50)
        elif seed == 5:
            # Top-p (nucleus) sampling
            output_ids = caption_model.generate(pixel_values, max_length=100, do_sample=True, top_p=0.9)
        elif seed == 6:
            # Sampling with both top-k and top-p
            output_ids = caption_model.generate(pixel_values, max_length=100, do_sample=True, top_k=50, top_p=0.9)
        elif seed == 7:
            # Sampling with low temperature
            output_ids = caption_model.generate(pixel_values, max_length=100, do_sample=True, temperature=0.7)
        elif seed == 8:
            # Sampling with high temperature
            output_ids = caption_model.generate(pixel_values, max_length=100, do_sample=True, temperature=2.0)
        elif seed == 9:
            # Top-k sampling with low k
            output_ids = caption_model.generate(pixel_values, max_length=100, do_sample=True, top_k=10)
        elif seed == 10:
            # Top-p sampling with low p
            output_ids = caption_model.generate(pixel_values, max_length=100, do_sample=True, top_p=0.8)
        else:
            raise ValueError("Invalid seed value. Must be between 1 and 10.")

        return caption_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

def generate_captions(pil_image):
    """
    Generate 10 different captions for the given image using various decoding strategies.
    """
    captions = []
    for seed in range(1, 11):
        captions.append(generate_caption(pil_image, seed))
    return captions