import requests, time
import board
import digitalio
from PIL import Image, ImageDraw, ImageFont
import adafruit_rgb_display.st7735 as st7735

#SERVER = "http://172.20.96.167:8080/predict"
SERVER = "http://10.0.0.82:8080/predict"


BAUDRATE = 24000000
FONTSIZE = 12

spi = board.SPI()

# Your wiring:
cs_pin = digitalio.DigitalInOut(board.CE0)    # GPIO8 (CS)
dc_pin = digitalio.DigitalInOut(board.D5)     # GPIO5 (A0 / DC)
reset_pin = digitalio.DigitalInOut(board.D25) # GPIO25 (RESET)

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

# Prepare image buffer
width = disp.width
height = disp.height
image = Image.new("RGB", (width, height), (0, 0, 0))  # always black
draw = ImageDraw.Draw(image)

font = ImageFont.truetype(
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    FONTSIZE
)

last_text = ""


while True:
    try:
        r = requests.get(SERVER, timeout=1)
        data = r.json()

        category = data.get("category", "None")
        text = f"Detected: {category}"
        print(text)

        if text != last_text:
            last_text = text

            # Clear to black
            draw.rectangle((0, 0, width, height), fill=(0, 0, 0))

            # Center text
            bbox = font.getbbox(text)
            fw = bbox[2] - bbox[0]
            fh = bbox[3] - bbox[1]

            draw.text(
                ((width - fw) // 2, (height - fh) // 2),
                text,
                font=font,
                fill=(255, 255, 255)
            )

            disp.image(image)

    except Exception as e:
        print("Error:", e)

    time.sleep(0.5)
