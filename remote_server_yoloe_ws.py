# remote_server_yoloe_ws.py
# -------------------------------------------------------------
# GPU server-side WebSocket that receives JPEG frames, runs YOLOE,
# and streams back annotated JPEG overlays.
#
# Start (on the GPU server):
#   pip install ultralytics fastapi uvicorn[standard] opencv-python-headless
#   # Also install a CUDA-enabled torch for your GPU (per pytorch.org)
#   uvicorn remote_server_yoloe_ws:app --host 0.0.0.0 --port 8765 --workers 1
#
# Optional security for quick tests: SSH tunnel from your PC
#   ssh -L 8765:127.0.0.1:8765 user@your.server
# Then connect your local client to ws://127.0.0.1:8765/infer
# -------------------------------------------------------------

import json
import traceback
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# Ultralytics
from ultralytics import YOLOE

# ----------------------- CONFIG -----------------------------
MODEL_PATH = r"/models/yoloe-11l-seg.pt"   # non -pf weights (adjust path on your server)
DEFAULT_PROMPTS: List[str] = [
    "enemy soldier with rifle",
    "armored enemy trooper",
    "robot soldier",
    "monster with armor",
]
CONF_TH = 0.10
IOU_TH = 0.65
IMGSZ  = 1280
RETINA_MASKS = False
HALF = True        # Use fp16 on GPU if available
JPEG_QUALITY = 70  # 60-85 is a good range
MAX_SIZE = 1280 * 1280 * 3  # Max raw image bytes (safety)
# --------------------- END CONFIG ---------------------------

app = FastAPI()
model: Optional[YOLOE] = None
active_prompts: List[str] = DEFAULT_PROMPTS.copy()


def ensure_model():
    global model
    if model is None:
        try:
            model = YOLOE(MODEL_PATH)
            if active_prompts:
                model.set_classes(active_prompts, model.get_text_pe(active_prompts))
        except Exception as e:
            print(f"Error loading model: {e}")
            traceback.print_exc()
            raise


@app.get("/")
async def root():
    return {"status": "ok", "endpoint": "/infer (WebSocket)"}


@app.websocket("/infer")
async def infer(ws: WebSocket):
    await ws.accept()
    try:
        # First message should be a small JSON config from the client
        cfg_text = await ws.receive_text()
        cfg = json.loads(cfg_text)

        # Allow client to update prompts and thresholds
        prompts = cfg.get("prompts")
        conf = float(cfg.get("conf", CONF_TH))
        iou = float(cfg.get("iou", IOU_TH))
        imgsz = int(cfg.get("imgsz", IMGSZ))
        retina_masks = bool(cfg.get("retina_masks", RETINA_MASKS))
        half = bool(cfg.get("half", HALF))
        jpeg_quality = int(cfg.get("jpeg_quality", JPEG_QUALITY))

        global active_prompts
        if prompts and isinstance(prompts, list):
            active_prompts = [str(p) for p in prompts]

        ensure_model()
        if active_prompts:
            # Reset classes for this session (safe & quick)
            model.set_classes(active_prompts, model.get_text_pe(active_prompts))

        # Ack
        await ws.send_text(json.dumps({"ready": True, "prompts": active_prompts}))

        while True:
            try:
                msg = await ws.receive()
                if "bytes" not in msg:
                    # Client might send a control text message
                    if msg.get("text") == "close":
                        break
                    continue

                data: bytes = msg["bytes"]
                if not data:
                    continue
                if len(data) > 20_000_000:  # 20 MB safety
                    continue

                # Decode JPEG -> BGR image
                arr = np.frombuffer(data, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                # YOLOE inference
                results = model.predict(
                    frame,
                    imgsz=imgsz,
                    conf=conf,
                    iou=iou,
                    retina_masks=retina_masks,
                    half=half,
                    device=0,        # assume GPU device 0; change if needed
                    verbose=False,
                    max_det=300,
                    persist=True
                )
                # Annotated overlay
                overlay = results[0].plot()

                # Encode to JPEG
                ok, buf = cv2.imencode(".jpg", overlay, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
                if not ok:
                    continue

                await ws.send_bytes(buf.tobytes())

            except Exception as e:
                print(f"Error in inference loop: {e}")
                traceback.print_exc()
                # Try to send error info to client
                try:
                    await ws.send_text(json.dumps({"error": str(e)}))
                except Exception:
                    pass
                # Continue processing instead of breaking
                continue

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        traceback.print_exc()
        # Try to send error info before closing
        try:
            await ws.send_text(json.dumps({"error": str(e)}))
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass
