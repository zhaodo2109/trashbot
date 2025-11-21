import requests, time, subprocess
from Adafruit_SSD1306 import SSD1306_128_64
from PIL import Image, ImageDraw, ImageFont

SERVER = "http://10.0.0.117:8080/predict"

# --- OLED Setup ---
oled = SSD1306_128_64(rst=None)
oled.begin()
oled.clear()
oled.display()

width = oled.width
height = oled.height
image = Image.new('1', (width, height))
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()

def oled_show(text):
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    draw.text((0, 0), text, font=font, fill=255)
    oled.image(image)
    oled.display()

def run_servo(cmd_list):
    try:
        subprocess.run(["python3", "servo.py"] + cmd_list, check=True)
    except Exception as e:
        oled_show("Servo Error")
        print("Servo Error:", e)

while True:
    try:
        r = requests.get(SERVER, timeout=1)
        data = r.json()

        category = data.get("category", "")

        oled_show(f"Detected:\n{category}")

        if category == "Recycle":
            oled_show("Recycle\nExecuting...")
            run_servo(["reset", "mid"])
            run_servo(["pick", "slow"])
            run_servo(["dropr", "slow"])

        elif category == "Organic":
            oled_show("Organic\nExecuting...")
            run_servo(["reset", "mid"])
            run_servo(["pick", "slow"])
            run_servo(["dropo", "slow"])

        elif category == "Trash":
            oled_show("Trash\nExecuting...")
            run_servo(["reset", "mid"])
            run_servo(["pick", "slow"])
            run_servo(["dropt", "slow"])

        else:
            oled_show(f"Unknown:\n{category}")

    except Exception as e:
        oled_show("Error\nNo server?")
        print("Error:", e)

    time.sleep(0.5)
