from flask import Flask, Response
import cv2

app = Flask(__name__)

# Open default webcam
cap = cv2.VideoCapture(0)

def generate():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Encode frame as JPEG
        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route('/video')
def video():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '<h1>Camera Stream</h1><img src="/video" width="640" />'


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
