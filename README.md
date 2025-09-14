# YOLO-E Real-time Gaming Detection System

## 🤖 Why YOLO-E?

[YOLO-E (Real-Time Seeing Anything)](https://arxiv.org/html/2503.07465v1) represents a breakthrough in open-vocabulary object detection, delivering the flexibility to detect arbitrary objects through simple text prompts like "enemy soldier with rifle" without retraining. Unlike traditional YOLO models limited to fixed categories, YOLO-E achieves [+3.5 AP improvement over YOLO-Worldv2 with 1.4× faster inference](https://github.com/THU-MIG/yoloe) and [305+ FPS on NVIDIA T4 GPUs](https://docs.ultralytics.com/models/yoloe/), making it the first open-vocabulary model truly suitable for real-time gaming applications. Perfect for dynamic scenarios where enemy types, character skins, or game content changes frequently—simply update your text prompts instead of retraining entire models.

This project implements a client-server architecture using YOLO-E to identify enemies and other entities in games with minimal performance impact on the gaming PC, while leveraging the model's revolutionary "seeing anything" capabilities for maximum flexibility.

## 🎯 Features

- **Dual Monitor Setup**: Capture from primary monitor, display overlay on secondary
- **WebSocket Streaming**: Low-latency communication between client and GPU server
- **Open-Vocabulary Detection**: Use text prompts for custom enemy types without retraining
- **Performance Controls**: Frame skipping, quality adjustment, and scaling options
- **Window/Region Capture**: Support for specific game windows or screen regions

## 📋 Current Performance

- **FPS**: 4-5 FPS (optimization needed)
- **Latency**: High due to JPEG encoding/decoding and network overhead
- **Detection Quality**: Poor confidence scores, needs game-specific tuning

## 🚀 Quick Start

### Server Setup (GPU Machine)

```bash
# Install dependencies
pip install ultralytics fastapi uvicorn[standard] opencv-python-headless

# Download a YOLO-E model (adjust path in remote_server_yoloe_ws.py)
# Example: yoloe-11l.pt (detection only) or yoloe-11l-seg.pt (segmentation)

# Start server
uvicorn remote_server_yoloe_ws:app --host 0.0.0.0 --port 8765 --workers 1
```

### Client Setup (Gaming PC)

```bash
# Install dependencies
pip install websockets opencv-python dxcam screeninfo pywin32

# Configure server URL in local_client_yoloe_ws.py
SERVER_WS_URL = "ws://YOUR.SERVER.IP:8765/infer"

# Run client
python local_client_yoloe_ws.py
```

### Controls

- **F**: Toggle fullscreen overlay
- **Q**: Quit application

## ⚙️ Configuration

### Client (`local_client_yoloe_ws.py`)
```python
PROMPTS = ["[enemy] humanoid stone golem", "[friend] human"]
CONF_TH = 0.10          # Detection confidence threshold
IMGSZ = 1280            # Input image size for model
SEND_SCALE = 1.0        # Downscale before sending (0.5-0.75 for speed)
FRAME_SKIP = 1          # Send every Nth frame
JPEG_QUALITY = 70       # Compression quality
```

### Server (`remote_server_yoloe_ws.py`)
```python
MODEL_PATH = "/path/to/yoloe-model.pt"
DEFAULT_PROMPTS = ["enemy soldier with rifle", "robot soldier"]
```

## 📁 Architecture

```
Gaming PC (Client)           GPU Server
┌─────────────────┐         ┌──────────────────┐
│ Screen Capture  │ JPEG    │ YOLO-E Inference │
│ (dxcam)         │────────►│ (ultralytics)    │
│                 │         │                  │
│ Overlay Display │ JPEG    │ Annotation       │
│ (OpenCV)        │◄────────│ (results.plot)   │
└─────────────────┘         └──────────────────┘
```

## 🔧 Known Issues & Optimizations Needed

See the [Issues](../../issues) section for detailed optimization tasks:

- [ ] Switch to detection-only model (remove segmentation overhead)
- [ ] Reduce inference image size for better FPS
- [ ] Optimize network transmission
- [ ] Improve detection confidence with game-specific prompts
- [ ] Consider local inference option
- [ ] Add performance benchmarking tools

## 🛡️ Security Note

For testing over internet, consider using SSH tunneling:
```bash
ssh -L 8765:127.0.0.1:8765 user@your.server
# Then connect to ws://127.0.0.1:8765/infer
```

## 📦 Dependencies

### Client
- `websockets` - WebSocket communication
- `opencv-python` - Image processing and display
- `dxcam` - Fast screen capture on Windows
- `screeninfo` - Monitor detection
- `pywin32` - Window title capture (optional)

### Server
- `ultralytics` - YOLO-E model inference
- `fastapi` - WebSocket server framework
- `uvicorn` - ASGI server
- `opencv-python-headless` - Image processing
- `torch` - GPU acceleration (CUDA-enabled)

## 🤝 Contributing

Issues and pull requests are welcome! Please see the issues section for current optimization priorities.

## 📄 License

MIT License - feel free to modify and distribute.
