# Gesture & Expression Recognition

This project streams webcam frames from the browser to a FastAPI backend that performs MediaPipe-based gesture and facial expression analysis. The client renders only the camera feed and the detection results returned by the server.

## Getting started

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Start the FastAPI server:

   ```bash
   uvicorn server:app --host 0.0.0.0 --port 4564 --reload
   ```

3. Open the interface at [http://localhost:4564](http://localhost:4564) and click **Start Camera** to begin streaming frames to the server for processing.

## Development notes

- All detection heuristics run on the server. The browser only captures frames and displays the emoji/label data returned by the API.
- Adjust the sensitivity slider in the UI to raise or lower the confidence threshold used by the backend classifiers.
- The backend retains a short per-session history and aggregates gesture/expression counts plus average confidence for quick feedback.
