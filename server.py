from fastapi import FastAPI, Form, UploadFile, File
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import io
from gen_caption import generate_caption, generate_caption_blip, generate_captions, generate_captions_blip
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allows all domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Custom labels
CUSTOM_LABELS = [
    "floorplan",
    "map",
    "photograph",
    "house",
    "kitchen",
    "bedroom",
    "dining room",
    "living room",
    "bathroom",
    "certificate",
    "backyard",
    "unknown"
]


@app.post("/gen-image-data")
async def analyze_image(image: UploadFile = File(...)):
    contents = await image.read()
    pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

    # --- Classification (CLIP)
    clip_inputs = clip_processor(
        text=CUSTOM_LABELS, images=pil_image, return_tensors="pt", padding=True
    )
    with torch.no_grad():
        clip_outputs = clip_model(**clip_inputs)
        logits_per_image = clip_outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)[0]
    label_probs = {
        label: round(prob.item(), 4) for label, prob in zip(CUSTOM_LABELS, probs)
    }
    best_label = max(label_probs, key=label_probs.get)

    if best_label == "certificate":
        caption = "EPC Certificate of the house or property"
        best_label = "EPC"
        label_probs["EPC"] = label_probs.pop("certificate")
    elif best_label == "floorplan":
        caption = "Floorplan of the house or property"
    else:
        caption = generate_caption(pil_image, 1)

    return {"label": best_label, "label_probs": label_probs, "caption": caption}


@app.post("/gen-caption")
async def caption_image(seed: int = Form(...), image: UploadFile = File(...)):
    contents = await image.read()
    pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    caption = generate_caption(pil_image, seed)
    return {"caption": caption}


@app.post("/gen-captions")
async def caption_images(image: UploadFile = File(...)):
    contents = await image.read()
    pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    captions = generate_captions(pil_image)
    return {"captions": captions}


@app.get("/test")
async def tokage():
    return {"message": "Hello World"}

@app.post("/gen-caption2")
async def caption_image_blip(image: UploadFile = File(...)):
    contents = await image.read()
    pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    caption = generate_caption_blip(pil_image, 1)
    return {"caption": caption}


@app.post("/gen-captions2")
async def caption_images_blip(image: UploadFile = File(...)):
    contents = await image.read()
    pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    captions = generate_captions_blip(pil_image)
    return {"captions": captions}