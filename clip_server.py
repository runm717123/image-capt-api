from fastapi import FastAPI, UploadFile, File
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import io
from gen_caption import generate_captions


app = FastAPI()

# Load CLIP for classification
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Custom labels
CUSTOM_LABELS = ["floorplan", "map", "photograph", "house", "kitchen", "bedroom", "dining room", "bathroom"]


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


    captions = generate_captions(pil_image)

    return {
        "label": best_label,
        "label_probs": label_probs,
        "captions": captions
    }

# Custom prompt to guide caption
PROMPT = "an attractive real estate photo of"

@app.post("/gen-captions")
async def caption_image(image: UploadFile = File(...)):
    contents = await image.read()
    pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    captions = generate_captions(pil_image)
    return {"captions": captions}

@app.get("/test")
async def tokage():
    return {"message": "Hello World"}