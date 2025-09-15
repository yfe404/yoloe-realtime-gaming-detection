# local_client_yoloe_ws.py
# -------------------------------------------------------------
# Generic YOLO-E WebSocket Client with Automatic Resolution Handling
#
# Features:
# - Automatic resolution detection and reporting to server
# - Flexible monitor setup and window capture
# - Performance optimization with configurable scaling
# - Comprehensive error handling and recovery
# - Real-time performance monitoring
#
# Install: pip install websockets opencv-python dxcam screeninfo
# Usage: python local_client_yoloe_ws.py
# -------------------------------------------------------------

import asyncio
import json
import time
import traceback
from typing import Optional, Tuple, List, Dict, Any

import cv2
import numpy as np
import websockets
from screeninfo import get_monitors
import dxcam

# Optional window detection (Windows only)
try:
    import win32gui
except ImportError:
    win32gui = None

# ======================= CONFIGURATION =======================

# Server connection
SERVER_WS_URL = "ws://213.173.108.95:11961/infer"

# Detection prompts - what to look for
DETECTION_PROMPTS: List[str] = [
    "[enemy] humanoid stone golem",
    "[friend] human",
    "[vehicle] tank",
    "[item] weapon",
]

# Model parameters
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.65
INFERENCE_SIZE = 640
RETINA_MASKS = False
HALF_PRECISION = True
MIN_CONFIDENCE_DISPLAY = 0.3

# Performance settings
SEND_SCALE = 0.5           # Downscale frames for network (0.1-1.0)
JPEG_QUALITY = 50          # JPEG compression quality (10-95)
FRAME_SKIP = 2             # Send every Nth frame (1=all frames)

# Display configuration
SOURCE_MONITOR = 0         # Monitor to capture from
DISPLAY_MONITOR = 1        # Monitor to show results on
TARGET_WINDOW_TITLE: Optional[str] = None  # Capture specific window
CAPTURE_REGION: Optional[Tuple[int, int, int, int]] = None  # (x1,y1,x2,y2)

# UI settings
SHOW_CONFIDENCE = True
SHOW_CLASS_NAMES = True
SHOW_DEBUG_INFO = True
BOX_THICKNESS = 2
FONT_SCALE = 0.6
PERFORMANCE_REPORT_INTERVAL = 5.0  # seconds

# ============================================================

# Color palette for different detection classes
DETECTION_COLORS = [
    (0, 255, 0),      # Green
    (255, 0, 0),      # Blue  
    (0, 165, 255),    # Orange
    (255, 255, 0),    # Cyan
    (0, 0, 255),      # Red
    (255, 0, 255),    # Magenta
    (128, 0, 128),    # Purple
    (255, 255, 255),  # White
]

def discover_monitors() -> List:
    """Discover and display available monitors"""
    monitors = get_monitors()
    print("🖥️  Available monitors:")
    for i, monitor in enumerate(monitors):
        print(f"   [{i}] {monitor.width}x{monitor.height} at ({monitor.x}, {monitor.y})")
    return monitors

def find_window_region(title: str) -> Optional[Tuple[int, int, int, int]]:
    """Find window by title and return its region (Windows only)"""
    if not win32gui:
        return None
        
    def enum_windows_callback(hwnd, windows):
        if win32gui.IsWindowVisible(hwnd):
            window_title = win32gui.GetWindowText(hwnd)
            if window_title and title.lower() in window_title.lower():
                rect = win32gui.GetWindowRect(hwnd)
                windows.append(rect)
                
    windows = []
    win32gui.EnumWindows(enum_windows_callback, windows)
    return windows[0] if windows else None

def setup_capture_region(monitors: List) -> Tuple[int, int, int, int]:
    """Determine the capture region based on configuration"""
    
    # Priority 1: Explicit region
    if CAPTURE_REGION:
        print(f"📐 Using explicit capture region: {CAPTURE_REGION}")
        return CAPTURE_REGION
    
    # Priority 2: Window title
    if TARGET_WINDOW_TITLE:
        window_region = find_window_region(TARGET_WINDOW_TITLE)
        if window_region:
            print(f"🪟 Found window '{TARGET_WINDOW_TITLE}': {window_region}")
            return window_region
        else:
            print(f"⚠️  Window '{TARGET_WINDOW_TITLE}' not found, using monitor")
    
    # Priority 3: Source monitor
    if SOURCE_MONITOR < len(monitors):
        monitor = monitors[SOURCE_MONITOR]
        region = (monitor.x, monitor.y, monitor.x + monitor.width, monitor.y + monitor.height)
        print(f"🖥️  Using monitor {SOURCE_MONITOR}: {region}")
        return region
    
    # Fallback: Primary monitor
    monitor = monitors[0]
    region = (monitor.x, monitor.y, monitor.x + monitor.width, monitor.y + monitor.height)
    print(f"⚠️  Fallback to primary monitor: {region}")
    return region

def draw_detection(image: np.ndarray, detection: Dict[str, Any], 
                  show_confidence: bool = True, show_class: bool = True) -> None:
    """Draw a single detection on the image"""
    try:
        bbox = detection["bbox"]
        class_name = detection["class_name"]
        confidence = detection["confidence"]
        class_id = detection.get("class_id", 0)
        
        # Convert coordinates to integers
        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure coordinates are within image bounds
        height, width = image.shape[:2]
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))
        
        # Select color
        color = DETECTION_COLORS[class_id % len(DETECTION_COLORS)]
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, BOX_THICKNESS)
        
        # Draw center point
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(image, (center_x, center_y), 3, color, -1)
        
        # Prepare label
        label_parts = []
        if show_class:
            label_parts.append(class_name)
        if show_confidence:
            label_parts.append(f"{confidence:.2f}")
            
        if label_parts:
            label = " | ".join(label_parts)
            
            # Calculate label size
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 1
            )
            
            # Position label
            label_x = min(x1, width - label_width - 10)
            label_y = max(label_height + 5, y1)
            
            # Draw label background
            cv2.rectangle(
                image,
                (label_x, label_y - label_height - 5),
                (label_x + label_width + 5, label_y + 5),
                color,
                cv2.FILLED
            )
            
            # Draw label text
            cv2.putText(
                image, label, (label_x + 2, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 255), 1, cv2.LINE_AA
            )
        
        # Draw segmentation mask if available
        if "mask_points" in detection:
            mask_points = detection["mask_points"]
            if mask_points and len(mask_points) >= 3:
                points = np.array(mask_points, dtype=np.int32)
                
                # Clamp points to image bounds
                points[:, 0] = np.clip(points[:, 0], 0, width - 1)
                points[:, 1] = np.clip(points[:, 1], 0, height - 1)
                
                # Draw mask outline
                cv2.polylines(image, [points], True, color, 1)
                
                # Draw semi-transparent fill
                mask_overlay = image.copy()
                cv2.fillPoly(mask_overlay, [points], color)
                cv2.addWeighted(image, 0.8, mask_overlay, 0.2, 0, image)
                
    except Exception as e:
        print(f"⚠️  Error drawing detection: {e}")

def render_detections(image: np.ndarray, predictions: Dict[str, Any], 
                     show_confidence: bool, show_class: bool, show_debug: bool) -> np.ndarray:
    """Render all detections and overlay information"""
    overlay = image.copy()
    
    try:
        detections = predictions.get("detections", [])
        
        # Draw each detection
        for detection in detections:
            draw_detection(overlay, detection, show_confidence, show_class)
        
        # Draw information overlay
        info_lines = [
            f"Detections: {len(detections)}",
            f"Frame: {predictions.get('frame_id', 'N/A')}",
        ]
        
        # Add resolution information if available
        resolution_info = predictions.get("resolution", {})
        if resolution_info and show_debug:
            target = resolution_info.get("target", [0, 0])
            inference = resolution_info.get("inference", [0, 0])
            sent = resolution_info.get("sent_frame", [0, 0])
            scaling = resolution_info.get("scaling", [1.0, 1.0])
            
            info_lines.extend([
                f"Target: {target[0]}x{target[1]}",
                f"Sent: {sent[0]}x{sent[1]}",
                f"Inference: {inference[0]}x{inference[1]}",
                f"Scale: {scaling[0]:.2f}x{scaling[1]:.2f}"
            ])
        
        # Draw info panel
        for i, line in enumerate(info_lines):
            y_position = 30 + i * 25
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            x_position = max(10, overlay.shape[1] - text_size[0] - 15)
            
            # Background
            cv2.rectangle(
                overlay,
                (x_position - 8, y_position - 18),
                (x_position + text_size[0] + 8, y_position + 7),
                (0, 0, 0, 180),  # Semi-transparent black
                -1
            )
            
            # Text
            cv2.putText(
                overlay, line, (x_position, y_position),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA
            )
            
    except Exception as e:
        print(f"⚠️  Error rendering detections: {e}")
        traceback.print_exc()
    
    return overlay

async def run_detection_client():
    """Main client execution function"""
    print("🚀 Starting YOLO-E Gaming Detection Client")
    print("=" * 50)
    
    # Setup display
    monitors = discover_monitors()
    if not monitors:
        print("❌ No monitors detected")
        return
    
    # Determine capture and display regions
    capture_region = setup_capture_region(monitors)
    capture_width = capture_region[2] - capture_region[0]
    capture_height = capture_region[3] - capture_region[1]
    
    if DISPLAY_MONITOR < len(monitors):
        display_monitor = monitors[DISPLAY_MONITOR]
    else:
        display_monitor = monitors[0]
        print(f"⚠️  Display monitor {DISPLAY_MONITOR} not found, using monitor 0")
    
    print(f"📹 Capture: {capture_width}x{capture_height} from {capture_region}")
    print(f"🖥️  Display: Monitor {DISPLAY_MONITOR} ({display_monitor.width}x{display_monitor.height})")
    print(f"📡 Server: {SERVER_WS_URL}")
    print(f"⚙️  Send scale: {SEND_SCALE}, JPEG quality: {JPEG_QUALITY}, Frame skip: {FRAME_SKIP}")
    
    # Initialize screen capture
    camera = dxcam.create(output_color="BGR")
    test_frame = camera.grab(region=capture_region)
    if test_frame is None:
        print("❌ Failed to initialize screen capture")
        return
    
    # Setup display window
    window_name = "YOLO-E Gaming Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, display_monitor.x, display_monitor.y)
    cv2.resizeWindow(window_name, display_monitor.width, display_monitor.height)
    
    # UI state
    is_fullscreen = False
    show_confidence = SHOW_CONFIDENCE
    show_class_names = SHOW_CLASS_NAMES
    show_debug_info = SHOW_DEBUG_INFO
    
    # Performance tracking
    frame_times = []
    network_times = []
    last_predictions = None
    last_performance_report = time.time()
    
    # Frame counters
    total_frames = 0
    frames_sent = 0
    
    print("🔗 Connecting to server...")
    
    try:
        async with websockets.connect(SERVER_WS_URL, max_size=2**20) as websocket:
            
            # Send client configuration
            client_config = {
                "prompts": DETECTION_PROMPTS,
                "conf": CONFIDENCE_THRESHOLD,
                "iou": IOU_THRESHOLD,
                "imgsz": INFERENCE_SIZE,
                "retina_masks": RETINA_MASKS,
                "half": HALF_PRECISION,
                "min_confidence": MIN_CONFIDENCE_DISPLAY,
                # Resolution information for server
                "original_resolution": [capture_width, capture_height],
                "send_scale": SEND_SCALE,
            }
            
            await websocket.send(json.dumps(client_config))
            
            # Receive server acknowledgment
            server_response = await websocket.recv()
            server_info = json.loads(server_response)
            
            print(f"✅ Server ready: {server_info.get('status', 'unknown')}")
            print(f"🎯 Prompts loaded: {len(server_info.get('prompts_loaded', []))}")
            
            print("\n🎮 Detection active!")
            print("Hotkeys: F=fullscreen, Q=quit, C=confidence, N=class names, D=debug info")
            print("=" * 50)
            
            # Main processing loop
            while True:
                try:
                    loop_start = time.time()
                    
                    # Capture frame
                    frame = camera.grab(region=capture_region)
                    if frame is None:
                        continue
                    
                    original_frame = frame.copy()
                    
                    # Send frame for inference (with frame skipping)
                    if total_frames % FRAME_SKIP == 0:
                        # Scale frame for network transmission
                        if SEND_SCALE != 1.0:
                            new_height = int(frame.shape[0] * SEND_SCALE)
                            new_width = int(frame.shape[1] * SEND_SCALE)
                            send_frame = cv2.resize(frame, (new_width, new_height), 
                                                   interpolation=cv2.INTER_AREA)
                        else:
                            send_frame = frame
                        
                        # Encode and send
                        encode_success, frame_buffer = cv2.imencode(
                            ".jpg", send_frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                        )
                        
                        if not encode_success:
                            print("⚠️  Frame encoding failed")
                            continue
                        
                        network_start = time.time()
                        await websocket.send(frame_buffer.tobytes())
                        frames_sent += 1
                        
                        # Receive predictions
                        prediction_data = await websocket.recv()
                        network_time = time.time() - network_start
                        network_times.append(network_time)
                        
                        # Parse predictions
                        if isinstance(prediction_data, str):
                            try:
                                predictions = json.loads(prediction_data)
                                
                                if "error" in predictions:
                                    print(f"❌ Server error: {predictions['error']}")
                                    if predictions.get("fatal"):
                                        break
                                    continue
                                    
                                last_predictions = predictions
                                
                            except json.JSONDecodeError as e:
                                print(f"⚠️  JSON decode error: {e}")
                                continue
                        else:
                            print(f"⚠️  Unexpected response type: {type(prediction_data)}")
                            continue
                    
                    # Render predictions on original frame
                    if last_predictions:
                        display_frame = render_detections(
                            original_frame, last_predictions, 
                            show_confidence, show_class_names, show_debug_info
                        )
                    else:
                        display_frame = original_frame.copy()
                    
                    # Scale to display resolution if needed
                    if (display_frame.shape[1] != display_monitor.width or 
                        display_frame.shape[0] != display_monitor.height):
                        display_frame = cv2.resize(
                            display_frame, 
                            (display_monitor.width, display_monitor.height),
                            interpolation=cv2.INTER_LINEAR
                        )
                    
                    # Performance overlay
                    frame_time = time.time() - loop_start
                    frame_times.append(frame_time)
                    
                    # Limit history size
                    if len(frame_times) > 60:
                        frame_times = frame_times[-60:]
                        network_times = network_times[-30:]
                    
                    # Calculate metrics
                    avg_frame_time = sum(frame_times) / len(frame_times)
                    avg_network_time = sum(network_times) / len(network_times) if network_times else 0
                    fps = 1.0 / max(avg_frame_time, 1e-6)
                    
                    # Performance info overlay
                    perf_info = [
                        f"FPS: {fps:.1f}",
                        f"Network: {avg_network_time*1000:.0f}ms",
                        f"Frame: {total_frames} (sent: {frames_sent})",
                    ]
                    
                    if last_predictions:
                        detection_count = len(last_predictions.get("detections", []))
                        perf_info.append(f"Detections: {detection_count}")
                    
                    # Draw performance overlay
                    for i, info in enumerate(perf_info):
                        y_pos = 30 + i * 25
                        cv2.rectangle(display_frame, (10, y_pos - 20), (280, y_pos + 5), (0, 0, 0), -1)
                        cv2.putText(display_frame, info, (15, y_pos),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
                    
                    # Show frame
                    cv2.imshow(window_name, display_frame)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('q'):
                        print("👋 Quit requested")
                        break
                    elif key == ord('f'):
                        is_fullscreen = not is_fullscreen
                        if is_fullscreen:
                            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        else:
                            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                            cv2.moveWindow(window_name, display_monitor.x, display_monitor.y)
                            cv2.resizeWindow(window_name, display_monitor.width, display_monitor.height)
                    elif key == ord('c'):
                        show_confidence = not show_confidence
                        print(f"📊 Confidence display: {'ON' if show_confidence else 'OFF'}")
                    elif key == ord('n'):
                        show_class_names = not show_class_names
                        print(f"🏷️  Class names: {'ON' if show_class_names else 'OFF'}")
                    elif key == ord('d'):
                        show_debug_info = not show_debug_info
                        print(f"🐛 Debug info: {'ON' if show_debug_info else 'OFF'}")
                    
                    total_frames += 1
                    
                    # Periodic performance report
                    current_time = time.time()
                    if current_time - last_performance_report >= PERFORMANCE_REPORT_INTERVAL:
                        detection_count = len(last_predictions.get("detections", [])) if last_predictions else 0
                        send_ratio = (frames_sent / total_frames * 100) if total_frames > 0 else 0
                        
                        print(f"📊 Performance: {fps:.1f} FPS, "
                              f"network: {avg_network_time*1000:.0f}ms, "
                              f"detections: {detection_count}, "
                              f"send ratio: {send_ratio:.0f}%")
                        
                        last_performance_report = current_time
                
                except Exception as e:
                    print(f"⚠️  Error in main loop: {e}")
                    traceback.print_exc()
                    continue

    except Exception as e:
        print(f"💥 Connection error: {e}")
        traceback.print_exc()
        
    finally:
        cv2.destroyAllWindows()
        print(f"🏁 Client finished. Processed {total_frames} frames, sent {frames_sent}.")

def main():
    """Entry point"""
    try:
        asyncio.run(run_detection_client())
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()