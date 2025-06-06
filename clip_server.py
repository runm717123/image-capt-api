from fastapi import FastAPI, UploadFile, File
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration, VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import torch
import io

app = FastAPI()

# Load CLIP model + processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

@app.get("/test")
async def tokage():
    return {"message": "Hello World"}

# Load CLIP for classification
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load BLIP for captioning
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Custom labels
CUSTOM_LABELS = ["floorplan", "map", "photograph", "house", "kitchen", "bedroom", "dining room", "bathroom"]


# gpt_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
# feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
# tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

@app.post("/gen-image-data")
async def analyze_image(image: UploadFile = File(...)):
    contents = await image.read()
    pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

    # --- Classification (CLIP)
    clip_inputs = clip_processor(text=CUSTOM_LABELS, images=pil_image, return_tensors="pt", padding=True)
    with torch.no_grad():
        clip_outputs = clip_model(**clip_inputs)
        logits_per_image = clip_outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)[0]
    label_probs = {label: round(prob.item(), 4) for label, prob in zip(CUSTOM_LABELS, probs)}
    best_label = max(label_probs, key=label_probs.get)


    # --- Captioning (BLIP)
    blip_inputs = blip_processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        caption_ids = blip_model.generate(**blip_inputs, max_length=100)
        caption = blip_processor.decode(caption_ids[0], skip_special_tokens=True)

    return {
        "label": best_label,
        "label_probs": label_probs,
        "caption": caption
    }
