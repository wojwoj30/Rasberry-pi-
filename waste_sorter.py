import os
import json
import time
import base64
import cv2
from dotenv import load_dotenv
from openai import OpenAI
from gpiozero import Servo
from gpiozero.pins.lgpio import LGPIOFactory

# Load Liquid AI credentials
load_dotenv()
client = OpenAI(
    base_url=os.getenv("LIQUID_BASE_URL", "https://api.liquid.ai/v1"),
    api_key=os.getenv("LIQUID_API_KEY")
)

# Servo diverter setup on GPIO 18
servo = Servo(18, pin_factory=LGPIOFactory(), min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)

SERVO_ANGLES = {
    "recycling": -1.0,
    "landfill":   0.0,
    "compost":    1.0
}

def actuate_bin(bin_target: str):
    angle = SERVO_ANGLES.get(bin_target.lower(), 0.0)
    print(f"[ACTUATOR] Directing diverter to {bin_target.upper()} (pos: {angle})")
    servo.value = angle
    time.sleep(1.5)
    servo.value = SERVO_ANGLES["landfill"]
    time.sleep(0.5)

def encode_frame(frame):
    resized = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_AREA)
    _, buffer = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return base64.b64encode(buffer).decode("utf-8")

def classify_waste(frame):
    b64_image = encode_frame(frame)
    prompt = (
        "Examine the waste item in the frame. Determine material composition and contamination. "
        "Respond ONLY with valid JSON containing keys: "
        "'material' (string), 'recyclable' (boolean), "
        "'bin' ('recycling'|'landfill'|'compost'), and 'confidence' (float 0.0-1.0)."
    )
    response = client.chat.completions.create(
        model="lfm3-vl-2b",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                ]
            }
        ],
        max_tokens=60,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

def run_sorter_cycle(image_path=None):
    if image_path and os.path.exists(image_path):
        frame = cv2.imread(image_path)
    else:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("[ERROR] Camera capture failed.")
            return

    print("[PIPELINE] Querying lfm3-vl-2b...")
    start_t = time.time()
    try:
        verdict = classify_waste(frame)
        latency = (time.time() - start_t) * 1000
        print(f"[DECISION] ({latency:.1f}ms): {verdict}")
        actuate_bin(verdict.get("bin", "landfill"))
    except Exception as e:
        print(f"[ERROR] Inference or parsing failed: {e}")
        actuate_bin("landfill")

if __name__ == "__main__":
    test_img = "test.jpg"
    if not os.path.exists(test_img):
        dummy = cv2.randn(cv2.Mat.zeros(512, 512, cv2.CV_8UC3), 128, 20)
        cv2.imwrite(test_img, dummy)
    run_sorter_cycle(test_img)
