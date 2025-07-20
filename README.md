# image-capt-api

![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-FFD21E?style=for-the-badge)

This is a FastAPI-based image analysis service that provides intelligent image classification and caption generation using state-of-the-art machine learning models. The API can identify image types (floorplan, photograph, certificate, etc.) and generate descriptive captions for uploaded images.

## Features

- **Image Classification**: Automatically categorizes images into predefined types (floorplan, map, photograph, house interiors, etc.)
- **Caption Generation**: Generates natural language descriptions of image content using BLIP models
- **RESTful API**: Simple HTTP endpoints for easy integration

## Demo

This API powers the image analysis functionality in the [CNTG Hugging Face Space](https://huggingface.co/spaces/runm717123/cntg) demo.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/image-capt-api.git
   cd image-capt-api
   ```
2. Install the required dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Optional: Install `pyTorch` with `CUDA`
   with `CUDA` enabled, the modal can run ~3x faster!. For more details for installation please refer to https://pytorch.org/get-started/locally

## Running the API

To start the API server, run the following command:

```bash
python -m uvicorn server:app --port 9001 --reload
```
