from flask import Flask, Response
import tensorflow as tf
import numpy as np, cv2, time, requests, json, os, gc
from collections import deque

# -------------------- CONFIG --------------------
CAMERA_URL = "http://10.0.0.117:8080/video"
MODEL_PATH = "/home/cershy/tf-gpu/TACO/ssd_mobilenet_v2_taco_2018_03_29.pb"
LABELS_PATH = "/home/cershy/tf-gpu/TACO/data/annotations.json"

# -------------------- LOAD LABELS --------------------
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)
taco_labels = {cat["id"]: cat["name"] for cat in dataset["categories"]}

organic = {"Food waste","Tissues","Toilet tube","Rope & strings"}
recycle = {"Aluminium blister pack","Aluminium foil","Aerosol","Battery","Broken glass",
           "Carded blister pack","Clear plastic bottle","Corrugated carton","Drink can",
           "Drink carton","Egg carton","Food Can","Foam cup","Glass bottle","Glass cup",
           "Glass jar","Magazine paper","Meal carton","Metal bottle cap","Metal lid",
           "Normal paper","Other carton","Paper bag","Paper cup","Paper straw","Pizza box",
           "Plastic bottle cap","Plastic film","Plastic lid","Plastic straw",
           "Polypropylene bag","Scrap metal","Six pack rings","Squeezable tube",
           "Tupperware","Wrapping paper","Other plastic","Other plastic bottle",
           "Other plastic container","Other plastic cup"
}
trash = {"Cigarette","Crisp packet","Disposable food container","Disposable plastic cup",
         "Foam food container","Garbage bag","Other plastic wrapper","Plastic glooves",
         "Plastic utensils","Plastified paper bag","Shoe","Single-use carrier bag",
         "Styrofoam piece","Unlabeled litter"}

app = Flask(__name__)

# -------------------- LOAD TF MODEL --------------------
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

print("\nTensorFlow .pb model loaded successfully!\n")

# Tensor names
image_tensor = detection_graph.get_tensor_by_name("image_tensor:0")
detection_boxes = detection_graph.get_tensor_by_name("detection_boxes:0")
detection_scores = detection_graph.get_tensor_by_name("detection_scores:0")
detection_classes = detection_graph.get_tensor_by_name("detection_classes:0")
num_detections = detection_graph.get_tensor_by_name("num_detections:0")

# -------------------- STREAM READER --------------------
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

# -------------------- GHOST SUPPRESSION STATE --------------------
ghost_gate = deque(maxlen=3)     # need 3 consecutive valid detections
missed_frames = 0                # count absence of detection
ghost_free_detection = None      # output for /predict

STRONG_T = 0.70
WEAK_T = 0.45

# -------------------- FRAME GENERATOR WITH BOXES + GHOST FILTER --------------------
def gen_frames():
    global ghost_gate, missed_frames, ghost_free_detection

    print("Waiting for frames...")

    for frame in read_mjpeg_stream(CAMERA_URL):

        # Preprocess
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (300, 300))
        x = np.expand_dims(img_resized, axis=0)

        # Inference
        boxes, scores, classes, num = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: x}
        )

        h, w, _ = frame.shape
        best_class, best_cat = None, None
        best_score = 0
        best_box = None

        # -------------------- FIND REAL DETECTION --------------------
        for i in range(int(num[0])):
            score = float(scores[0][i])
            if score < WEAK_T:
                continue

            class_id = int(classes[0][i]) - 1
            class_name = taco_labels.get(class_id, f"Class {class_id}")

            # box dimensions
            y1, x1, y2, x2 = boxes[0][i]
            box_h = (y2 - y1) * h
            box_w = (x2 - x1) * w

            # remove tiny boxes (ghost noise)
            if box_h < 25 or box_w < 25:
                continue

            # strong detection gets priority
            if score > STRONG_T or score > best_score:
                best_score = score
                best_class = class_name
                best_box = (x1, y1, x2, y2)

        # category mapping
        if best_class:
            if best_class in recycle: best_cat = "Recycle"
            elif best_class in organic: best_cat = "Organic"
            elif best_class in trash: best_cat = "Trash"

        # -------------------- GHOST SUPPRESSION --------------------
        if best_class:
            ghost_gate.append(1)
            missed_frames = 0
        else:
            ghost_gate.append(0)
            missed_frames += 1

        if missed_frames >= 5:
            ghost_free_detection = None

        if sum(ghost_gate) == 3 and best_class:
            ghost_free_detection = {
                "category": best_cat,
                "class": best_class,
                "score": round(best_score, 3)
            }
            print("[DETECTED]", ghost_free_detection)

        # -------------------- DRAW BOUNDING BOXES --------------------
        if best_box and best_class:
            x1n = int(best_box[0] * w)
            y1n = int(best_box[1] * h)
            x2n = int(best_box[2] * w)
            y2n = int(best_box[3] * h)

            # color by category
            if best_cat == "Recycle": color = (255, 0, 0)
            elif best_cat == "Organic": color = (0, 255, 0)
            else: color = (0, 0, 255)

            label = f"{best_cat}: {best_class} ({best_score*100:.0f}%)"
            cv2.rectangle(frame, (x1n, y1n), (x2n, y2n), color, 2)
            cv2.putText(frame, label, (x1n, max(20, y1n - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # -------------------- STREAM FRAME --------------------
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')

        time.sleep(0.03)
        gc.collect()

# -------------------- FLASK ROUTES --------------------
@app.route("/")
def index():
    return '<h1>Ghost-Free PB Model Server</h1><img src="/video" width="640">'

@app.route("/video")
def video():
    print("/video requested → streaming")
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/predict")
def predict():
    if ghost_free_detection is None:
        return {"category": None, "class": None, "score": 0}
    return ghost_free_detection

# -------------------- RUN APP --------------------
if __name__ == "__main__":
    print("Server running at http://0.0.0.0:8080")
    app.run(host="0.0.0.0", port=8080, threaded=True)
