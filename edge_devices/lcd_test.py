import time
import board
import busio
import digitalio
from PIL import Image, ImageDraw, ImageFont
import adafruit_rgb_display.st7735 as st7735

# SPI1
# SCK  -> GPIO 21 (Pin 40)
# MOSI -> GPIO 20 (Pin 38)
spi = busio.SPI(clock=board.D21, MOSI=board.D20, MISO=None)


# CONTROL PINS

cs_pin = digitalio.DigitalInOut(board.D16)     # CS  -> GPIO 16 (Pin 36)
dc_pin = digitalio.DigitalInOut(board.D5)      # A0  -> GPIO 5  (Pin 29)
reset_pin = digitalio.DigitalInOut(board.D26)  # RST -> GPIO 26 (Pin 37)

disp = st7735.ST7735R(
    spi,
    rotation=0,
    width=128,
    height=160,
    cs=cs_pin,
    dc=dc_pin,
    rst=reset_pin,
    baudrate=24000000,
)


width = disp.width
height = disp.height

image = Image.new("RGB", (width, height))
draw = ImageDraw.Draw(image)

font = ImageFont.truetype(
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    12
)

text = "Detected: Recycle"
bbox = font.getbbox(text)
tw = bbox[2] - bbox[0]
th = bbox[3] - bbox[1]

# Center text
draw.rectangle((0, 0, width, height), fill=0)
draw.text(
    ((width - tw) // 2, (height - th) // 2),
    text,
    font=font,
    fill=(255, 255, 255)
)

# Push to display
disp.image(image)

print("Displayed on screen!")
