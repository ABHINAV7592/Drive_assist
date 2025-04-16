import json
import numpy as np
import cv2
import serial
import time
import threading
import pygame
from io import BytesIO
from PIL import Image
from channels.generic.websocket import AsyncWebsocketConsumer
from ultralytics import YOLO
from inference_sdk import InferenceHTTPClient

# Load YOLO models
traffic_model = YOLO(rf"D:\code\drivegaurd\Drive_guard_new\Drive_guard\drive_guard\traffic.pt")
pothole_model = YOLO(rf"D:\code\drivegaurd\Drive_guard_new\Drive_guard\drive_guard\pothole.pt")

# Initialize Roboflow client for drowsiness detection
DROWSINESS_CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="2yWFyy2GmaDnaD1vFXZw"
)

# Serial port configuration
SERIAL_PORT = "COM3"  # Change as needed
BAUD_RATE = 9600

# Initialize pygame for alarm
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound('alarm.wav')  # Provide your alarm sound file
alarm_playing = False
alarm_lock = threading.Lock()

def play_alarm():
    """Play alarm sound in a loop"""
    global alarm_playing
    with alarm_lock:
        if not alarm_playing:
            alarm_playing = True
            alarm_sound.play(0)  # -1 makes it loop indefinitely

def stop_alarm():
    """Stop the alarm sound"""
    global alarm_playing
    with alarm_lock:
        if alarm_playing:
            alarm_playing = False
            alarm_sound.stop()

def read_from_arduino():
    """Read data from the Arduino serial port."""
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        if ser.in_waiting > 0:
            return ser.readline().decode('utf-8').strip()
    except Exception as e:
        print(f"[ERROR] Serial error: {e}")
    return None


class TrafficDetectionConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        print("WebSocket Connected (Traffic Mode)")

    async def disconnect(self, close_code):
        print("WebSocket Disconnected (Traffic Mode)")

    async def receive(self, text_data=None, bytes_data=None):
        if bytes_data:
            try:
                frame_np = self.process_frame(bytes_data)
                # Increase confidence threshold to 0.5
                traffic_results = traffic_model(frame_np, conf=0.5)
                detections = self.parse_detections(traffic_results, frame_np, "traffic")
                sensor_data = read_from_arduino()

                cv2.imshow("Traffic Detection", frame_np)
                cv2.waitKey(1)

                # Only send detections if there are any
                if detections:
                    await self.send(json.dumps({
                        "message": "Frame processed (Traffic Mode)",
                        "detections": detections,
                        "sensor_data": sensor_data,
                    }))
                else:
                    await self.send(json.dumps({
                        "message": "No detections",
                        "detections": [],
                        "sensor_data": sensor_data,
                    }))
            except Exception as e:
                await self.send(json.dumps({"error": str(e)}))

    def process_frame(self, bytes_data):
        """Convert bytes data to a numpy array."""
        image = Image.open(BytesIO(bytes_data)).convert("RGB")
        frame_np = np.array(image, dtype=np.uint8)
        return cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

    def parse_detections(self, results, frame, label):
        """Parse YOLO detection results."""
        detections = []
        for box in results[0].boxes:
            confidence = float(box.conf[0].cpu().numpy())
            # Only consider detections with confidence >= 0.5
            if confidence >= 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_name = results[0].names[int(box.cls[0].cpu().numpy())]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                detections.append({"label": class_name, "confidence": confidence, "bbox": [x1, y1, x2, y2], "type": label})
        return detections


class DrowsinessCameraConsumer(AsyncWebsocketConsumer):
    def __init__(self):
        super().__init__()
        self.is_detection_active = False
        self.alarm_on = False
        self.temp_file = 'temp_frame.jpg'
        self.frame_counter = 0
        self.last_processed_frame = None
        
    async def connect(self):
        await self.accept()
        print("WebSocket Connected (Drowsiness Mode)")

    async def disconnect(self, close_code):
        print("WebSocket Disconnected (Drowsiness Mode)")
        self.is_detection_active = False
        threading.Thread(target=stop_alarm).start()

    async def receive(self, text_data=None, bytes_data=None):
        if text_data:
            try:
                data = json.loads(text_data)
                if data.get("action") == "start":
                    self.is_detection_active = True
                    self.frame_counter = 0  # Reset counter when starting
                    await self.send(json.dumps({"message": "Drowsiness detection started"}))
                elif data.get("action") == "stop":
                    self.is_detection_active = False
                    threading.Thread(target=stop_alarm).start()
                    await self.send(json.dumps({"message": "Drowsiness detection stopped"}))
            except Exception as e:
                await self.send(json.dumps({"error": str(e)}))
        elif bytes_data and self.is_detection_active:
            try:
                # Increment frame counter
                self.frame_counter += 1
                
                # Only process every 3rd frame (0, 3, 6, etc.)
                if self.frame_counter % 3 != 0:
                    # If we have a previous processed result, send that instead
                    if self.last_processed_frame is not None:
                        await self.send(json.dumps(self.last_processed_frame))
                    return
                
                # Decode the frame
                frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    print("Error: Invalid frame received")
                    await self.send(json.dumps({"error": "Invalid frame"}))
                    return

                # Save the frame to a temporary file
                cv2.imwrite(self.temp_file, frame)
                
                try:
                    # Send the frame to Roboflow for drowsiness detection
                    result = DROWSINESS_CLIENT.infer(self.temp_file, model_id="yolov8_dataset-hrf1x/4")
                    
                    # Process detection results
                    drowsy_result = self.process_drowsy_detection(frame, result)
                    self.last_processed_frame = drowsy_result  # Store the last result
                    
                    # Display the processed frame
                    cv2.imshow("Drowsiness Detection", frame)
                    cv2.waitKey(1)
                    
                    # Send the result back to the Flutter app
                    await self.send(json.dumps(drowsy_result))
                    
                except Exception as e:
                    print(f"Inference error: {e}")
                    error_result = {
                        "error": f"Inference error: {str(e)}",
                        "drowsy": False,
                        "message": "Error in drowsiness detection",
                        "alarm_on": False
                    }
                    self.last_processed_frame = error_result
                    await self.send(json.dumps(error_result))
                    
            except Exception as e:
                print(f"Error processing frame: {e}")
                await self.send(json.dumps({"error": str(e)}))

    def process_drowsy_detection(self, frame, predictions):
        """Process drowsiness detection results and draw on frame"""
        drowsy_detected = False
        yawning_detected = False
        message = "Eyes Open"
        
        if not predictions or 'predictions' not in predictions:
            return {
                "drowsy": False,
                "message": "No face detected",
                "alarm_on": False
            }
        
        for pred in predictions['predictions']:
            try:
                # Convert all coordinates to integers
                x = int(pred['x'])
                y = int(pred['y'])
                width = int(pred['width'])
                height = int(pred['height'])
                
                # Calculate box coordinates
                x1 = max(0, x - width // 2)
                y1 = max(0, y - height // 2)
                x2 = min(frame.shape[1], x + width // 2)
                y2 = min(frame.shape[0], y + height // 2)
                
                # Determine color based on class
                class_name = pred.get('class', '').lower()
                if 'drowsy' in class_name:
                    color = (0, 0, 255)  # Red for drowsy
                    drowsy_detected = True
                    message = "DROWSINESS ALERT!"
                elif 'yawn' in class_name:
                    color = (0, 165, 255)  # Orange for yawning
                    yawning_detected = True
                    if not drowsy_detected:  # Don't override drowsy message
                        message = "YAWNING DETECTED"
                else:
                    color = (0, 255, 0)  # Green for others
                
                # Draw rectangle
                thickness = 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                
                # Prepare text
                confidence = pred.get('confidence', 0)
                label = f"{class_name} {confidence:.2f}"
                
                # Calculate text position
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                # Draw text background
                cv2.rectangle(frame, 
                             (x1, y1 - text_height - 10), 
                             (x1 + text_width, y1), 
                             color, -1)
                
                # Put text
                cv2.putText(frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
            except Exception as e:
                print(f"Error drawing prediction: {e}")
                continue
        
        # Control alarm based on detections
        if drowsy_detected or yawning_detected:
            threading.Thread(target=play_alarm).start()
            self.alarm_on = True
            
            # Display warning text at the top of the frame
            warning_text = "ALERT! " + message
            text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            cv2.putText(frame, warning_text, (text_x, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            threading.Thread(target=stop_alarm).start()
            self.alarm_on = False
        
        # Prepare result to send back to the app
        result = {
            "drowsy": drowsy_detected or yawning_detected,
            "message": message,
            "alarm_on": self.alarm_on
        }
        
        return result


class PotholeDetectionConsumer(TrafficDetectionConsumer):
    async def receive(self, text_data=None, bytes_data=None):
        if bytes_data:
            try:
                frame_np = self.process_frame(bytes_data)
                # Increase confidence threshold to 0.5
                pothole_results = pothole_model(frame_np, conf=0.5)
                detections = self.parse_detections(pothole_results, frame_np, "pothole")
                sensor_data = read_from_arduino()

                cv2.imshow("Pothole Detection", frame_np)
                cv2.waitKey(1)

                # Only send detections if there are any
                if detections:
                    await self.send(json.dumps({
                        "message": "Frame processed (Pothole Mode)",
                        "detections": detections,
                        "sensor_data": sensor_data,
                    }))
                else:
                    await self.send(json.dumps({
                        "message": "No detections",
                        "detections": [],
                        "sensor_data": sensor_data,
                    }))
            except Exception as e:
                await self.send(json.dumps({"error": str(e)}))