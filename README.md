# image-capt-api

This is a simple API for generating image type and captions using a pre-trained machine learning model.

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

The server will be accessible at http://127.0.0.1:9001.
---
title: Cntg
emoji: 💻
colorFrom: purple
colorTo: green
sdk: docker
app_port: 7860
suggested_hardware: cpu-basic
pinned: false
license: mit
short_description: Image caption and labelling
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
