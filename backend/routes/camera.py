import cv2
import logging
import threading
import time

from flask import Blueprint
from flask import Response
from flask import jsonify
from flask import request

logger = logging.getLogger(__name__)

camera_bp = Blueprint(
    "camera",
    __name__
)

camera = None
frame = None
lock = threading.Lock()
is_camera_running = False
last_camera_access = time.time()

CAMERA_INDEX = 0


# ===========================
# Camera Thread
# ===========================

def camera_reader():
    global frame, is_camera_running

    while is_camera_running:
        if camera is None:
            time.sleep(0.05)
            continue

        if time.time() - last_camera_access > 60:
            logger.info("Camera auto-stopped due to 60s inactivity.")
            stop_camera()
            break

        success, img = camera.read()

        if success:
            with lock:
                frame = img
            time.sleep(0.04)
        else:
            time.sleep(0.05)


# ===========================
# Start Camera
# ===========================

def start_camera(index=0):
    global camera, is_camera_running, last_camera_access

    stop_camera()
    last_camera_access = time.time()

    camera = cv2.VideoCapture(index)
    if not camera.isOpened():
        camera = None
        return False

    is_camera_running = True
    threading.Thread(
        target=camera_reader,
        daemon=True
    ).start()
    return True


# ===========================
# Stop Camera
# ===========================

def stop_camera():
    global camera, is_camera_running, frame

    is_camera_running = False

    if camera is not None:
        camera.release()
        camera = None

    with lock:
        frame = None


# ===========================
# Video Feed
# ===========================

@camera_bp.route("/video_feed")
def video_feed():
    global last_camera_access
    last_camera_access = time.time()

    def generate():
        global last_camera_access
        while is_camera_running:
            last_camera_access = time.time()
            current_frame = None
            with lock:
                if frame is not None:
                    current_frame = frame.copy()

            if current_frame is None:
                time.sleep(0.05)
                continue

            success, buffer = cv2.imencode(".jpg", current_frame)
            if not success:
                time.sleep(0.05)
                continue

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n'
                + buffer.tobytes()
                + b'\r\n'
            )
            time.sleep(0.04)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ===========================
# Capture Image
# ===========================

@camera_bp.route("/capture")
def capture():
    global last_camera_access
    last_camera_access = time.time()

    with lock:
        if frame is None:
            return jsonify({
                "success": False,
                "message": "No frame captured."
            }), 503
        current_frame = frame.copy()

    success, buffer = cv2.imencode(".jpg", current_frame)
    if not success:
        return jsonify({
            "success": False,
            "message": "Failed to encode frame."
        }), 500

    return Response(
        buffer.tobytes(),
        mimetype="image/jpeg"
    )


# ===========================
# Camera Status
# ===========================

@camera_bp.route("/camera_status")
def camera_status():

    return jsonify({
        "camera_running": is_camera_running and camera is not None
    })


# ===========================
# Process Captured Frame
# ===========================

@camera_bp.route("/process", methods=["POST"])
def process():
    global last_camera_access
    last_camera_access = time.time()

    with lock:
        if frame is None:
            return jsonify({
                "success": False,
                "message": "No frame available to process."
            }), 503
        current_frame = frame.copy()

    try:
        from services.processor import process_image
        mode = (request.args.get('mode') or
                (request.get_json(silent=True) or {}).get('mode', 'ai'))
        use_llm = (str(mode).strip().lower() != 'fast')
        result = process_image(current_frame, use_llm=use_llm)
        return jsonify(result)
    except Exception:
        logger.exception("Error during /process (webcam)")
        return jsonify({
            "success": False,
            "error": "Processing failed. Check server logs for details."
        }), 500


# ===========================
# Initialize On Demand
# ===========================

@camera_bp.route("/start", methods=["POST"])
def api_start_camera():
    started = start_camera(CAMERA_INDEX)
    if started:
        return jsonify({"success": True, "message": "Camera started."})
    else:
        return jsonify({"success": False, "message": "Could not open camera device."}), 500

@camera_bp.route("/stop", methods=["POST"])
def api_stop_camera():
    stop_camera()
    return jsonify({"success": True, "message": "Camera stopped."})