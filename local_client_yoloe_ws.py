# local_client_yoloe_ws.py
# -------------------------------------------------------------
# Windows client that:
# 1) Captures your game screen (monitor or window),
# 2) Streams JPEG frames to the remote GPU server via WebSocket,
# 3) Displays the annotated overlay on your second monitor.
#
# Install:
#   pip install websockets opencv-python dxcam screeninfo
#
# Run:
#   python local_client_yoloe_ws.py
#
# Hotkeys:
#   f = toggle fullscreen, q = quit
# -------------------------------------------------------------

import asyncio
import json
import time
from typing import Optional, Tuple, List

import cv2
import numpy as np

import websockets
from screeninfo import get_monitors
import dxcam

# Optional window-title capture (requires pywin32)
try:
    import win32gui  # type: ignore
except Exception:
    win32gui = None


# ----------------------- CONFIG -----------------------------
SERVER_WS_URL = "ws://213.173.108.95:11961/infer"  # or ws://your.server.ip:8765/infer

PROMPTS: List[str] = [
    "[enemy] humanoid stone golem",
    "[friend] human",
]

CONF_TH = 0.10
IOU_TH = 0.65
IMGSZ  = 1280
RETINA_MASKS = False
HALF = True

# JPEG controls
SEND_SCALE = 1.0         # downscale before sending (e.g., 0.75 for faster FPS)
JPEG_QUALITY = 70        # 60-85 recommended

SRC_MONITOR = 0          # which monitor to capture
DST_MONITOR = 1          # where to display overlay
FRAME_SKIP = 1           # send every Nth frame
GAME_WINDOW_TITLE: Optional[str] = None  # e.g., "MyGame"
REGION: Optional[Tuple[int, int, int, int]] = None  # (left, top, right, bottom)
# --------------------- END CONFIG ---------------------------


def list_monitors():
    mons = get_monitors()
    print("Detected monitors:")
    for i, m in enumerate(mons):
        print(f"  [{i}] x={m.x} y={m.y} w={m.width} h={m.height}")
    return mons


def rect_from_window_title(title: str) -> Optional[Tuple[int, int, int, int]]:
    if win32gui is None:
        return None
    def enum_handler(hwnd, result):
        if win32gui.IsWindowVisible(hwnd):
            t = win32gui.GetWindowText(hwnd)
            if t and title.lower() in t.lower():
                rect = win32gui.GetWindowRect(hwnd)  # (l, t, r, b)
                result.append(rect)
    found = []
    win32gui.EnumWindows(enum_handler, found)
    if not found:
        return None
    l, t, r, b = found[0]
    return (l, t, r, b)


async def run_client():
    mons = list_monitors()
    if not mons:
        print("No monitors found.")
        return

    if REGION is not None:
        cap_region = REGION
    elif GAME_WINDOW_TITLE:
        cap_region = rect_from_window_title(GAME_WINDOW_TITLE)
        if cap_region is None:
            src = mons[SRC_MONITOR]
            cap_region = (src.x, src.y, src.x + src.width, src.y + src.height)
    else:
        src = mons[SRC_MONITOR]
        cap_region = (src.x, src.y, src.x + src.width, src.y + src.height)

    dst = mons[DST_MONITOR]
    print(f"Capture region: {cap_region}")
    print(f"Overlay window on monitor [{DST_MONITOR}] at ({dst.x},{dst.y}) size {dst.width}x{dst.height}")

    # Init capture
    cam = dxcam.create(output_color="BGR")
    _ = cam.grab(region=cap_region)

    # Prepare window
    win_name = "Remote Enemy Overlay"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(win_name, dst.x, dst.y)
    cv2.resizeWindow(win_name, dst.width, dst.height)
    is_fullscreen = False

    # Connect WebSocket
    async with websockets.connect(SERVER_WS_URL, max_size=2**27) as ws:
        # Send config
        cfg = {
            "prompts": PROMPTS,
            "conf": CONF_TH,
            "iou": IOU_TH,
            "imgsz": IMGSZ,
            "retina_masks": RETINA_MASKS,
            "half": HALF,
            "jpeg_quality": JPEG_QUALITY,
            # "persist" argument removed
        }
        await ws.send(json.dumps(cfg))
        ack = await ws.recv()
        print("Server ack:", ack)

        frame_id = 0
        t0 = time.time()
        fps = 0.0

        while True:
            frame = cam.grab(region=cap_region)
            if frame is None:
                continue

            # Optional downscale to speed up encode + inference
            if SEND_SCALE != 1.0:
                h, w = frame.shape[:2]
                nw, nh = int(w * SEND_SCALE), int(h * SEND_SCALE)
                frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)

            # Throttle send rate
            if frame_id % FRAME_SKIP == 0:
                # Encode JPEG
                ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                if not ok:
                    continue
                await ws.send(buf.tobytes())

                # Receive annotated overlay (JPEG)
                data = await ws.recv()
                if isinstance(data, str):
                    # maybe a text error
                    print("Server says:", data[:200])
                    continue
                arr = np.frombuffer(data, dtype=np.uint8)
                overlay = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if overlay is None:
                    continue
            else:
                # If skipping frames, just show the last frame to keep UI responsive
                overlay = cv2.resize(frame, (dst.width, dst.height), interpolation=cv2.INTER_LINEAR)

            # Fit to destination monitor
            if overlay.shape[1] != dst.width or overlay.shape[0] != dst.height:
                overlay = cv2.resize(overlay, (dst.width, dst.height), interpolation=cv2.INTER_LINEAR)

            # FPS text
            t1 = time.time()
            fps = 1.0 / max(t1 - t0, 1e-6)
            t0 = t1
            cv2.putText(overlay, f"{fps:.1f} FPS (remote)", (16, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(overlay, f"{fps:.1f} FPS (remote)", (16, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(win_name, overlay)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                is_fullscreen = not is_fullscreen
                cv2.setWindowProperty(
                    win_name, cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
                )
                if not is_fullscreen:
                    cv2.moveWindow(win_name, dst.x, dst.y)
                    cv2.resizeWindow(win_name, dst.width, dst.height)

            frame_id += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        asyncio.run(run_client())
    except KeyboardInterrupt:
        pass