import requests, time, subprocess
import board
import digitalio
from PIL import Image, ImageDraw, ImageFont
import adafruit_rgb_display.st7735 as st7735

SERVER = "http://10.0.0.117:8080/predict"


def run_servo(cmd_list):
    try:
        subprocess.run(["python3", "servo.py"] + cmd_list, check=True)
    except Exception as e:
        print("Servo Error:", e)



BAUDRATE = 24000000
FONTSIZE = 22

spi = board.SPI()
cs_pin = digitalio.DigitalInOut(board.CE0)     # GPIO8
dc_pin = digitalio.DigitalInOut(board.D5)      # GPIO5
reset_pin = digitalio.DigitalInOut(board.D25)  # GPIO25

disp = st7735.ST7735R(
    spi,
    rotation=0,
    width=128,
    height=160,
    cs=cs_pin,
    dc=dc_pin,
    rst=reset_pin,
    baudrate=BAUDRATE,
)

width = disp.width
height = disp.height
image = Image.new("RGB", (width, height), (0, 0, 0))  # Black background
draw = ImageDraw.Draw(image)

font = ImageFont.truetype(
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", FONTSIZE
)

last_tft_text = ""



while True:
    try:
        r = requests.get(SERVER, timeout=1)
        data = r.json()
        print(f"Detected: {data}")

        category = data.get("category", "None")
        display_text = f"Detected: {category}"


        if display_text != last_tft_text:
            last_tft_text = display_text

            # Clear screen
            draw.rectangle((0, 0, width, height), fill=(0, 0, 0))

            # Calculate text size
            bbox = font.getbbox(display_text)
            fw = bbox[2] - bbox[0]
            fh = bbox[3] - bbox[1]

            # Draw centered text
            draw.text(
                ((width - fw) // 2, (height - fh) // 2),
                display_text,
                font=font,
                fill=(255, 255, 255)
            )

            disp.image(image)

        if category == "Recycle":
            print("Recycle detected ? executing recycle sequence")
            run_servo(["reset", "mid"])
            run_servo(["pick", "slow"])
            run_servo(["dropr", "slow"])

        elif category == "Organic":
            print("Organic detected ? executing organic sequence")
            run_servo(["reset", "mid"])
            run_servo(["pick", "slow"])
            run_servo(["dropo", "slow"])

        elif category == "Trash":
            print("Trash detected ? executing trash sequence")
            run_servo(["reset", "mid"])
            run_servo(["pick", "slow"])
            run_servo(["dropt", "slow"])

        else:
            print("Unknown category:", category)

    except Exception as e:
        print("Error:", e)

    time.sleep(0.5)

