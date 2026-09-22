from flask import Blueprint
from flask import request
from flask import jsonify
from flask import Response

import cv2
import logging
import threading
import time

logger = logging.getLogger(__name__)

ipcamera_bp = Blueprint(
    "ipcamera",
    __name__
)

phone_camera = None
PHONE_URL = ""
ip_frame = None
ip_lock = threading.Lock()
is_phone_running = False

# ============================================
# Camera Thread
# ============================================

def ipcamera_reader():
    global ip_frame, is_phone_running

    while is_phone_running:
        if phone_camera is None:
            time.sleep(0.05)
            continue
        success, img = phone_camera.read()
        if success:
            with ip_lock:
                ip_frame = img.copy()
            time.sleep(0.01)
        else:
            time.sleep(0.05)


# ============================================
# Connect Phone Camera
# ============================================

@ipcamera_bp.route("/connect_phone", methods=["POST"])
def connect_phone():

    global phone_camera
    global PHONE_URL
    global is_phone_running
    global ip_frame

    data = request.get_json(silent=True) or {}
    raw_ip = str(data.get("ip", "")).strip()

    if not raw_ip:
        return jsonify({
            "success": False,
            "message": "IP Address Missing"
        }), 400

    # Sanitize user input: handle cases like http://192.168.1.5:8080/video or 192.168.1.5:8080
    clean_ip = raw_ip
    for prefix in ["http://", "https://"]:
        if clean_ip.startswith(prefix):
            clean_ip = clean_ip[len(prefix):]
    clean_ip = clean_ip.rstrip("/")
    if clean_ip.endswith("/video"):
        clean_ip = clean_ip[:-len("/video")].rstrip("/")

    PHONE_URL = f"http://{clean_ip}/video"

    # Stop any previous phone camera session cleanly
    is_phone_running = False
    if phone_camera is not None:
        phone_camera.release()
        phone_camera = None

    with ip_lock:
        ip_frame = None

    phone_camera = cv2.VideoCapture(PHONE_URL)

    if not phone_camera.isOpened():
        phone_camera = None
        return jsonify({
            "success": False,
            "message": f"Unable to connect to stream at {PHONE_URL}"
        })

    is_phone_running = True
    threading.Thread(
        target=ipcamera_reader,
        daemon=True
    ).start()

    return jsonify({
        "success": True,
        "message": "Phone Connected",
        "url": PHONE_URL
    })


# ============================================
# Disconnect
# ============================================

@ipcamera_bp.route("/disconnect_phone")
def disconnect_phone():

    global phone_camera
    global ip_frame
    global is_phone_running

    is_phone_running = False

    if phone_camera is not None:
        phone_camera.release()
        phone_camera = None

    with ip_lock:
        ip_frame = None

    return jsonify({
        "success": True
    })


# ============================================
# Status
# ============================================

@ipcamera_bp.route("/phone_status")
def phone_status():

    connected = False
    if is_phone_running and phone_camera is not None:
        connected = phone_camera.isOpened()

    return jsonify({
        "connected": connected,
        "url": PHONE_URL
    })


@ipcamera_bp.route("/video_feed")
def video_feed():
    def generate():
        while True:
            current_frame = None
            with ip_lock:
                if ip_frame is not None:
                    current_frame = ip_frame.copy()
            
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
            time.sleep(0.03)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

# ============================================
# Process Current Frame
# ============================================

@ipcamera_bp.route("/process", methods=["POST"])
def process():
    
    with ip_lock:
        if phone_camera is None or not phone_camera.isOpened() or ip_frame is None:
            return jsonify({
                "success": False,
                "message": "Phone camera is not connected or no frame available."
            })
        current_frame = ip_frame.copy()

    try:
        from services.processor import process_image
        mode = (request.args.get('mode') or
                (request.get_json(silent=True) or {}).get('mode', 'ai'))
        use_llm = (str(mode).strip().lower() != 'fast')
        result = process_image(current_frame, use_llm=use_llm)
        return jsonify(result)
    except Exception:
        logger.exception("Error during /ipcamera/process")
        return jsonify({
            "success": False,
            "error": "Processing failed. Check server logs for details."
        }), 500
