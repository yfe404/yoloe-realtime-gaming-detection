# remote_server_yoloe_ws.py
# -------------------------------------------------------------
# Generic YOLO-E WebSocket Server with Automatic Resolution Handling
#
# Features:
# - Automatic coordinate scaling detection and handling
# - Support for any client resolution and scaling factor
# - Graceful degradation and error recovery
# - Performance optimization and monitoring
# - Clean signal handling
#
# Start: uvicorn remote_server_yoloe_ws:app --host 0.0.0.0 --port 8765 --workers 1
# -------------------------------------------------------------

import json
import signal
import sys
import time
import traceback
from typing import List, Optional, Dict, Any, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# Ultralytics
from ultralytics import YOLOE

# ======================= CONFIGURATION =======================
MODEL_PATH = r"/models/yoloe-11l-seg.pt"

# Default detection prompts - can be overridden by client
DEFAULT_PROMPTS: List[str] = [
    "enemy soldier with rifle",
    "armored enemy trooper", 
    "robot soldier",
    "monster with armor",
]

# Model parameters - can be overridden by client
DEFAULT_CONF_TH = 0.25
DEFAULT_IOU_TH = 0.65
DEFAULT_IMGSZ = 640
DEFAULT_RETINA_MASKS = False
DEFAULT_HALF = True
DEFAULT_MAX_DETECTIONS = 50
DEFAULT_MIN_CONFIDENCE = 0.3

# Performance limits
MAX_FRAME_SIZE = 20_000_000  # 20MB max frame size
PERFORMANCE_REPORT_INTERVAL = 30  # Report every N frames
# ============================================================

app = FastAPI()

# Global state
model: Optional[YOLOE] = None
active_prompts: List[str] = DEFAULT_PROMPTS.copy()
frame_counter = 0
shutdown_flag = False

def signal_handler(signum, frame):
    """Clean shutdown on SIGINT/SIGTERM"""
    global shutdown_flag
    print(f"\n⚠️  Received signal {signum}, shutting down gracefully...")
    shutdown_flag = True
    sys.exit(0)

# Register signal handlers for clean shutdown
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def ensure_model_loaded() -> None:
    """Load YOLO-E model if not already loaded"""
    global model
    if model is None:
        try:
            print(f"🔄 Loading YOLO-E model: {MODEL_PATH}")
            model = YOLOE(MODEL_PATH)
            if active_prompts:
                model.set_classes(active_prompts, model.get_text_pe(active_prompts))
            print(f"✅ Model loaded successfully with {len(active_prompts)} prompts")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            traceback.print_exc()
            raise

def detect_resolution_scaling(client_config: Dict[str, Any], sent_frame_shape: Tuple[int, int]) -> Tuple[int, int, float, float]:
    """
    Detect the actual target resolution and calculate scaling factors.
    
    Priority:
    1. Explicit original_resolution from client
    2. Infer from sent frame and known scaling factors
    3. Use sent frame size as fallback
    
    Returns: (target_width, target_height, width_scale, height_scale)
    """
    sent_width, sent_height = sent_frame_shape
    inference_size = client_config.get("imgsz", DEFAULT_IMGSZ)
    
    # Method 1: Explicit resolution from client (most reliable)
    if "original_resolution" in client_config:
        target_width, target_height = client_config["original_resolution"]
        width_scale = target_width / inference_size
        height_scale = target_height / inference_size
        
        print(f"🎯 Using explicit resolution: {target_width}x{target_height}")
        print(f"📏 Scaling factors: {width_scale:.3f}x{height_scale:.3f}")
        return target_width, target_height, width_scale, height_scale
    
    # Method 2: Try to infer from client metadata
    if "send_scale" in client_config and client_config["send_scale"] != 1.0:
        send_scale = float(client_config["send_scale"])
        # Reverse the client scaling to get original resolution
        target_width = int(sent_width / send_scale)
        target_height = int(sent_height / send_scale)
        width_scale = target_width / inference_size
        height_scale = target_height / inference_size
        
        print(f"🔍 Inferred resolution from send_scale {send_scale}: {target_width}x{target_height}")
        print(f"📏 Scaling factors: {width_scale:.3f}x{height_scale:.3f}")
        return target_width, target_height, width_scale, height_scale
    
    # Method 3: Fallback to sent frame size
    width_scale = sent_width / inference_size
    height_scale = sent_height / inference_size
    
    print(f"⚠️  Using sent frame size as target: {sent_width}x{sent_height}")
    print(f"📏 Scaling factors: {width_scale:.3f}x{height_scale:.3f}")
    return sent_width, sent_height, width_scale, height_scale

def extract_detections(results, frame_id: int, target_resolution: Tuple[int, int], 
                      inference_resolution: Tuple[int, int], sent_resolution: Tuple[int, int],
                      scaling_factors: Tuple[float, float], min_confidence: float) -> Dict[str, Any]:
    """
    Extract and scale detection results to target resolution.
    
    Args:
        results: YOLO-E prediction results
        frame_id: Current frame number
        target_resolution: (width, height) - Final target resolution for coordinates
        inference_resolution: (width, height) - Resolution used for inference
        sent_resolution: (width, height) - Resolution of frame sent by client  
        scaling_factors: (width_scale, height_scale) - Scaling from inference to target
        min_confidence: Minimum confidence threshold for detections
        
    Returns:
        Dictionary with detections and metadata
    """
    detections = []
    target_width, target_height = target_resolution
    inference_width, inference_height = inference_resolution
    sent_width, sent_height = sent_resolution
    width_scale, height_scale = scaling_factors
    
    if len(results[0].boxes) > 0:
        for i, box in enumerate(results[0].boxes):
            confidence = float(box.conf[0])
            
            # Filter by confidence
            if confidence < min_confidence:
                continue
                
            class_id = int(box.cls[0])
            
            # Get bounding box coordinates from inference resolution
            bbox = box.xyxy[0].cpu().numpy().tolist()
            x1, y1, x2, y2 = bbox
            
            # Scale coordinates to target resolution
            scaled_bbox = [
                x1 * width_scale,
                y1 * height_scale, 
                x2 * width_scale,
                y2 * height_scale
            ]
            
            # Clamp to target resolution bounds
            scaled_bbox = [
                max(0, min(scaled_bbox[0], target_width - 1)),
                max(0, min(scaled_bbox[1], target_height - 1)),
                max(0, min(scaled_bbox[2], target_width - 1)),
                max(0, min(scaled_bbox[3], target_height - 1))
            ]
            
            detection = {
                "id": i,
                "class_id": class_id,
                "class_name": results[0].names.get(class_id, f"class_{class_id}"),
                "confidence": confidence,
                "bbox": [float(coord) for coord in scaled_bbox],
            }
            
            # Add segmentation mask if available
            if (hasattr(results[0], 'masks') and 
                results[0].masks is not None and 
                i < len(results[0].masks.xy)):
                
                mask = results[0].masks.xy[i]
                if mask is not None and len(mask) > 0:
                    # Scale and simplify mask points
                    scaled_mask = []
                    step = max(1, len(mask) // 20)  # Limit to ~20 points for performance
                    
                    for point in mask[::step]:
                        scaled_point = [
                            max(0, min(float(point[0] * width_scale), target_width - 1)),
                            max(0, min(float(point[1] * height_scale), target_height - 1))
                        ]
                        scaled_mask.append(scaled_point)
                    
                    if scaled_mask:
                        detection["mask_points"] = scaled_mask
            
            detections.append(detection)
    
    # Return comprehensive result
    return {
        "frame_id": frame_id,
        "timestamp": time.time(),
        "detections": detections,
        "total_detections": len(detections),
        "resolution": {
            "target": list(target_resolution),          # Final coordinate space
            "inference": list(inference_resolution),    # YOLO inference resolution  
            "sent_frame": list(sent_resolution),        # Frame size sent by client
            "scaling": list(scaling_factors)            # Scaling factors applied
        },
        "model_info": {
            "confidence_threshold": min_confidence,
            "model_prompts": active_prompts,
            "total_classes": len(active_prompts)
        }
    }

@app.get("/")
async def root():
    """Health check and server info endpoint"""
    return {
        "status": "healthy", 
        "service": "YOLO-E WebSocket Server",
        "endpoint": "/infer (WebSocket)",
        "version": "2.0.0",
        "features": [
            "automatic_resolution_detection",
            "coordinate_scaling", 
            "performance_optimization",
            "graceful_shutdown",
            "comprehensive_debugging"
        ],
        "model_loaded": model is not None,
        "active_prompts": len(active_prompts)
    }

@app.websocket("/infer")
async def inference_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for real-time inference"""
    global frame_counter, shutdown_flag, active_prompts
    
    await websocket.accept()
    session_start = time.time()
    session_frames = 0
    
    # Performance tracking
    inference_times = []
    processing_times = []
    last_report_time = time.time()
    
    try:
        # === CLIENT CONFIGURATION PHASE ===
        print("📱 New client connecting...")
        
        config_data = await websocket.receive_text()
        client_config = json.loads(config_data)
        
        print(f"🔧 Client config: {client_config}")
        
        # Parse client configuration with defaults
        prompts = client_config.get("prompts", DEFAULT_PROMPTS)
        conf = float(client_config.get("conf", DEFAULT_CONF_TH))
        iou = float(client_config.get("iou", DEFAULT_IOU_TH))
        imgsz = int(client_config.get("imgsz", DEFAULT_IMGSZ))
        retina_masks = bool(client_config.get("retina_masks", DEFAULT_RETINA_MASKS))
        half = bool(client_config.get("half", DEFAULT_HALF))
        min_confidence = float(client_config.get("min_confidence", DEFAULT_MIN_CONFIDENCE))

        # Update active prompts if provided
        if prompts and isinstance(prompts, list):
            active_prompts = [str(p).strip() for p in prompts if str(p).strip()]
            print(f"🎯 Updated prompts ({len(active_prompts)}): {active_prompts}")

        # Ensure model is loaded with current prompts
        ensure_model_loaded()
        if active_prompts:
            model.set_classes(active_prompts, model.get_text_pe(active_prompts))

        # Send acknowledgment to client
        server_response = {
            "status": "ready",
            "prompts_loaded": active_prompts,
            "model_config": {
                "confidence_threshold": conf,
                "min_confidence_filter": min_confidence,
                "inference_size": imgsz,
                "max_detections": DEFAULT_MAX_DETECTIONS,
                "retina_masks": retina_masks,
                "half_precision": half
            },
            "server_info": {
                "version": "2.0.0",
                "features": ["auto_resolution", "coordinate_scaling"]
            }
        }
        
        await websocket.send_text(json.dumps(server_response))
        print(f"✅ Client configured. Server ready for inference.")

        # === INFERENCE LOOP ===
        resolution_detected = False
        target_resolution = None
        scaling_factors = None
        
        while not shutdown_flag:
            try:
                # Receive frame data
                message = await websocket.receive()
                
                if "bytes" not in message:
                    # Handle text messages (e.g., "close" command)
                    if message.get("text") == "close":
                        print("👋 Client requested disconnect")
                        break
                    continue

                frame_data = message["bytes"]
                
                # Validate frame data
                if not frame_data or len(frame_data) > MAX_FRAME_SIZE:
                    print(f"⚠️  Invalid frame: size={len(frame_data) if frame_data else 0}")
                    continue

                processing_start = time.time()
                
                # Decode image
                frame_array = np.frombuffer(frame_data, dtype=np.uint8)
                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                
                if frame is None:
                    print("⚠️  Failed to decode frame")
                    continue

                sent_height, sent_width = frame.shape[:2]
                
                # Detect resolution and scaling on first frame
                if not resolution_detected:
                    target_width, target_height, width_scale, height_scale = detect_resolution_scaling(
                        client_config, (sent_width, sent_height)
                    )
                    target_resolution = (target_width, target_height)
                    scaling_factors = (width_scale, height_scale)
                    resolution_detected = True
                
                # Prepare frame for inference
                if sent_width != imgsz or sent_height != imgsz:
                    inference_frame = cv2.resize(frame, (imgsz, imgsz))
                    inference_resolution = (imgsz, imgsz)
                else:
                    inference_frame = frame
                    inference_resolution = (sent_width, sent_height)

                # Run YOLO-E inference
                inference_start = time.time()
                
                results = model.predict(
                    inference_frame,
                    imgsz=imgsz,
                    conf=conf,
                    iou=iou,
                    retina_masks=retina_masks,
                    half=half,
                    device=0,
                    verbose=False,
                    max_det=DEFAULT_MAX_DETECTIONS,
                    save=False,
                    show=False
                )
                
                inference_time = time.time() - inference_start
                inference_times.append(inference_time)
                
                # Extract and scale detections
                detection_results = extract_detections(
                    results, frame_counter, target_resolution, inference_resolution, 
                    (sent_width, sent_height), scaling_factors, min_confidence
                )
                
                # Send results to client
                result_json = json.dumps(detection_results)
                await websocket.send_text(result_json)
                
                # Update counters
                frame_counter += 1
                session_frames += 1
                
                processing_time = time.time() - processing_start
                processing_times.append(processing_time)
                
                # Performance reporting
                if frame_counter % PERFORMANCE_REPORT_INTERVAL == 0:
                    current_time = time.time()
                    
                    # Calculate averages
                    recent_inference = inference_times[-PERFORMANCE_REPORT_INTERVAL:]
                    recent_processing = processing_times[-PERFORMANCE_REPORT_INTERVAL:]
                    
                    avg_inference = sum(recent_inference) / len(recent_inference)
                    avg_processing = sum(recent_processing) / len(recent_processing)
                    
                    # Calculate FPS
                    time_window = current_time - last_report_time
                    fps = PERFORMANCE_REPORT_INTERVAL / time_window if time_window > 0 else 0
                    
                    # Detection stats
                    detection_count = len(detection_results['detections'])
                    json_size = len(result_json)
                    
                    print(f"📊 Frame {frame_counter}: {detection_count} detections, "
                          f"inference: {avg_inference*1000:.1f}ms, "
                          f"total: {avg_processing*1000:.1f}ms, "
                          f"FPS: {fps:.1f}, "
                          f"JSON: {json_size}B")
                    
                    print(f"📐 Resolution: sent={sent_width}x{sent_height}, "
                          f"inference={inference_resolution[0]}x{inference_resolution[1]}, "
                          f"target={target_resolution[0]}x{target_resolution[1]}")
                    
                    last_report_time = current_time

            except WebSocketDisconnect:
                print("🔌 Client disconnected during processing")
                break
                
            except Exception as e:
                print(f"❌ Error in inference loop: {e}")
                traceback.print_exc()
                
                # Send error to client
                try:
                    error_response = {
                        "error": str(e),
                        "frame_id": frame_counter,
                        "timestamp": time.time(),
                        "recoverable": True
                    }
                    await websocket.send_text(json.dumps(error_response))
                except Exception:
                    print("Failed to send error response to client")
                    break

    except WebSocketDisconnect:
        print("🔌 Client disconnected")
        
    except Exception as e:
        print(f"💥 WebSocket error: {e}")
        traceback.print_exc()
        
        try:
            await websocket.send_text(json.dumps({"error": str(e), "fatal": True}))
        except Exception:
            pass
            
    finally:
        # Session cleanup
        session_duration = time.time() - session_start
        
        print(f"🏁 Session ended: {session_frames} frames in {session_duration:.1f}s "
              f"(avg: {session_frames/session_duration:.1f} FPS)")
        
        try:
            await websocket.close()
        except Exception:
            pass

if __name__ == "__main__":
    print("🚀 YOLO-E Gaming Detection Server v2.0")
    print("📡 WebSocket endpoint: ws://0.0.0.0:8765/infer")
    print("🌐 HTTP health check: http://0.0.0.0:8765/")
    print("⚠️  Press Ctrl+C for graceful shutdown")
    print("=" * 60)