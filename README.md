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

## Running the API
To start the API server, run the following command:
```bash
python -m uvicorn server:app --port 9001 --reload
```

The server will be accessible at http://127.0.0.1:9001.