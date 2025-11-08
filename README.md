# smart-bin

Real-time object detection demo built around the MobileNet SSD Caffe model. The repository ships with pretrained weights and a simple entry-point script that opens the default webcam, detects common objects, and overlays bounding boxes and class labels in real time.

## Overview

- **Entry point**: `real_time_object_detection.py`
- **Model assets**: `MobileNetSSD_deploy.prototxt.txt`, `MobileNetSSD_deploy.caffemodel`
- **Primary dependency stack**: `opencv-python`, `imutils`, `numpy`
- **Target use case**: Rapid prototyping and experimentation with MobileNet SSD on live camera feeds

## Installation

1. (Optional) Create and activate a virtual environment.
2. Install the required Python packages:
   ```bash
   pip install opencv-python imutils numpy
   ```
3. Ensure the provided Caffe model files remain in the project root or supply their paths via CLI arguments.

## Command-Line Interface (CLI)

Run the detector from the repository root:

```bash
python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel
```

### CLI arguments

| Flag | Required | Default | Description |
| ---- | -------- | ------- | ----------- |
| `--prototxt`, `-p` | Yes | – | Path to the MobileNet SSD Caffe deploy prototxt file. |
| `--model`, `-m` | Yes | – | Path to the pretrained MobileNet SSD Caffe model weights. |
| `--confidence`, `-c` | No | `0.3` | Minimum detection confidence (0–1) used to filter weak predictions. |

All arguments accept absolute or relative paths. Adjust `--confidence` upward (e.g., `0.6`) to limit detections to high-confidence predictions or downward (e.g., `0.2`) to increase recall.

## Pipeline Components

- **Model loader**: Uses `cv2.dnn.readNetFromCaffe` to load the MobileNet SSD network defined by `--prototxt` and `--model`.
- **Video stream**: `imutils.video.VideoStream` initializes the default camera (`src=0`) and handles threaded frame retrieval.
- **Class catalog**: `CLASSES` is a list of the 21 labels supported by the pretrained model. Customize this list if you swap in a different model trained on alternate classes.
- **Inference loop**:
  1. Grab the latest frame and resize to a width of 1000 pixels.
  2. Convert the frame into a 300×300 blob and feed it through the network.
  3. Iterate through detections, filter using `--confidence`, and compute pixel bounding boxes.
  4. Draw bounding boxes and class labels with a randomly initialized color palette.
  5. Display frames in an OpenCV window, exiting when the `q` key is pressed.
- **Performance monitor**: `imutils.video.FPS` tracks frames per second and prints summary statistics on exit.

## Usage Examples

- **High-confidence detections only**
  ```bash
  python real_time_object_detection.py \
      --prototxt MobileNetSSD_deploy.prototxt.txt \
      --model MobileNetSSD_deploy.caffemodel \
      --confidence 0.6
  ```

- **Using alternative model paths**
  ```bash
  python real_time_object_detection.py \
      --prototxt /path/to/custom.prototxt.txt \
      --model /path/to/custom.caffemodel
  ```

- **Embedding in another script**

  ```python
  import argparse
  from pathlib import Path

  import cv2
  import imutils
  import numpy as np
  from imutils.video import FPS, VideoStream

  def run_detection(prototxt: str, model: str, confidence: float = 0.3):
      net = cv2.dnn.readNetFromCaffe(prototxt, model)
      vs = VideoStream(src=0).start()
      fps = FPS().start()
      classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle",
                 "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
                 "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
                 "train", "tvmonitor"]
      colors = np.random.uniform(0, 255, size=(len(classes), 3))
      try:
          while True:
              frame = imutils.resize(vs.read(), width=1000)
              (h, w) = frame.shape[:2]
              blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                            0.007843, (300, 300), 127.5)
              net.setInput(blob)
              detections = net.forward()
              for i in np.arange(0, detections.shape[2]):
                  if detections[0, 0, i, 2] > confidence:
                      idx = int(detections[0, 0, i, 1])
                      box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                      (startX, startY, endX, endY) = box.astype("int")
                      label = f"{classes[idx]}: {detections[0, 0, i, 2] * 100:.0f}%"
                      cv2.rectangle(frame, (startX, startY), (endX, endY),
                                    colors[idx], 2)
                      y = startY - 15 if startY - 15 > 15 else startY + 15
                      cv2.putText(frame, label, (startX, y),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                  colors[idx], 2)
              cv2.imshow("Frame", frame)
              if (cv2.waitKey(1) & 0xFF) == ord("q"):
                  break
              fps.update()
      finally:
          fps.stop()
          cv2.destroyAllWindows()
          vs.stop()

  if __name__ == "__main__":
      parser = argparse.ArgumentParser()
      parser.add_argument("--prototxt", required=True)
      parser.add_argument("--model", required=True)
      parser.add_argument("--confidence", type=float, default=0.3)
      args = parser.parse_args()
      run_detection(args.prototxt, args.model, args.confidence)
  ```

  This example mirrors the shipped CLI but exposes `run_detection` for reuse in other Python applications.

## Output

- Annotated OpenCV window titled `Frame` with bounding boxes and class labels.
- Console logs:
  - `[INFO] loading model...`
  - `[INFO] starting video stream...`
  - FPS statistics when the session terminates.

## Customization Tips

- Increase `VideoStream(src=<index>)` to target additional or external cameras.
- Modify the frame resize width (default `1000`) to improve performance on low-powered devices.
- Swap in new Caffe models by replacing the prototxt/weights files and updating `CLASSES` accordingly.
- Persist detections by writing frames to disk with `cv2.VideoWriter` or emitting detection metadata to downstream systems.

## Troubleshooting

- **`ImportError: No module named cv2`**: Ensure `opencv-python` is installed in the active environment.
- **No camera detected**: Check that your webcam is accessible and not locked by another application. On Linux, verify device permissions for `/dev/video*`.
- **Slow FPS**: Lower the frame resize width or run on hardware with a dedicated GPU/OpenVINO backend.

---

For questions or enhancements, open an issue or submit a pull request.
