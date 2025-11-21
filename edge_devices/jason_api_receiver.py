import requests, time

#SERVER = "http://127.0.0.1:8080/predict"
#SERVER = "http://172.20.96.167:8080/predict"
SERVER = "http://10.15.158.43:8080/predict"

while True:
    try:
        r = requests.get(SERVER, timeout=1)
        data = r.json()
        print(f"Detected: {data}")
        # Example robot logic:
        if data["category"] == "Recycle":
            print("Move arm to recycle bin")
        elif data["category"] == "Organic":
            print("Move arm to compost bin")
        elif data["category"] == "Trash":
            print("Move arm to trash bin")
    except Exception as e:
        print("Error:", e)
    time.sleep(0.5)
