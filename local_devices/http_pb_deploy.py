from flask import Flask, Response
import tensorflow as tf
import numpy as np, cv2, time, requests, json, os, gc

# ---- Configuration ----

#CAMERA_URL = "http://10.15.186.103:8080/video" #school
#CAMERA_URL = "http://10.0.0.251:8080/video" #local

CAMERA_URL = "http://10.0.0.117:8080/video"        # local pi5


MODEL_PATH = "/home/cershy/tf-gpu/TACO/ssd_mobilenet_v2_taco_2018_03_29.pb"
LABELS_PATH = "/home/cershy/tf-gpu/TACO/data/annotations.json"

# ---- Load label map from TACO ----
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)
taco_labels = {cat["id"]: cat["name"] for cat in dataset["categories"]}

organic = {
    "Food waste",
    "Tissues",
    "Toilet tube",
    "Rope & strings",
}

recycle = {
    "Aluminium blister pack",
    "Aluminium foil",
    "Aerosol",
    "Battery",
    "Broken glass",
    "Carded blister pack",
    "Clear plastic bottle",
    "Corrugated carton",
    "Drink can",
    "Drink carton",
    "Egg carton",
    "Food Can",
    "Foam cup",
    "Glass bottle",
    "Glass cup",
    "Glass jar",
    "Magazine paper",
    "Meal carton",
    "Metal bottle cap",
    "Metal lid",
    "Normal paper",
    "Other carton",
    "Paper bag",
    "Paper cup",
    "Paper straw",
    "Pizza box",
    "Plastic bottle cap",
    "Plastic film",
    "Plastic lid",
    "Plastic straw",
    "Polypropylene bag",
    "Scrap metal",
    "Six pack rings",
    "Squeezable tube",
    "Tupperware",
    "Wrapping paper",
}

trash = {
    "Cigarette",
    "Crisp packet",
    "Disposable food container",
    "Disposable plastic cup",
    "Foam food container",
    "Garbage bag",
    "Other plastic",
    "Other plastic bottle",
    "Other plastic container",
    "Other plastic cup",
    "Other plastic wrapper",
    "Plastic glooves",
    "Plastic utensils",
    "Plastified paper bag",
    "Shoe",
    "Single-use carrier bag",
    "Styrofoam piece",
    "Unlabeled litter",
}
app = Flask(__name__)

# ---- Load TensorFlow model ----
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.disable_eager_execution()

detection_graph = tf.Graph()
with detection_graph.as_default():
    graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(MODEL_PATH, "rb") as f:
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="")

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(graph=detection_graph, config=config)
print("TensorFlow .pb model loaded")

# ---- Tensor names ----
image_tensor = detection_graph.get_tensor_by_name("image_tensor:0")
detection_boxes = detection_graph.get_tensor_by_name("detection_boxes:0")
detection_scores = detection_graph.get_tensor_by_name("detection_scores:0")
detection_classes = detection_graph.get_tensor_by_name("detection_classes:0")
num_detections = detection_graph.get_tensor_by_name("num_detections:0")

# ---- Read MJPEG stream ----
def read_mjpeg_stream(url):
    stream = requests.get(url, stream=True, timeout=10)
    bytes_data = b''
    for chunk in stream.iter_content(chunk_size=4096):
        bytes_data += chunk
        a = bytes_data.find(b'\xff\xd8')
        b = bytes_data.find(b'\xff\xd9')
        if a != -1 and b != -1:
            jpg = bytes_data[a:b+2]
            bytes_data = bytes_data[b+2:]
            frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                yield frame

# ---- Inference + Annotate ----
def gen_frames(threshold=0.5):
    print("Waiting for frames...")
    for frame in read_mjpeg_stream(CAMERA_URL):
        if frame is None:
            continue

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (300, 300))
        x = np.expand_dims(img_resized, axis=0)

        # TensorFlow inference
        boxes, scores, classes, num = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: x}
        )

        h, w, _ = frame.shape
        for i in range(int(num[0])):
            if scores[0][i] < threshold:
                continue
            y1, x1, y2, x2 = boxes[0][i]
            (x1, y1, x2, y2) = (int(x1*w), int(y1*h), int(x2*w), int(y2*h))
            class_id = int(classes[0][i]) - 1
            class_name = taco_labels.get(class_id, f"Class {class_id}")

            # classify category color
            if class_name in recycle:
                color, cat = (255, 0, 0), "Recycle"
            elif class_name in organic:
                color, cat = (0, 255, 0), "Organic"
            elif class_name in trash:
                color, cat = (0, 0, 255), "Trash"

            label = f"{cat}: {class_name} ({scores[0][i]*100:.0f}%)"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')
        gc.collect()
        time.sleep(0.03)

# ---- Flask Routes ----
@app.route('/')
def index():
    return '<h1>PB Model Server</h1><img src="/video" width="640" />'

@app.route('/video')
def video():
    print("/video endpoint accessed — starting stream")
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print("Model server running at http://0.0.0.0:8080/video")
    app.run(host="0.0.0.0", port=8080, threaded=True)
