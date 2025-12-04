import requests, time, subprocess

SERVER = "http://10.15.139.47:8080/predict"

def run_servo(cmd_list):
    """Run servo.py with arguments safely."""
    try:
        subprocess.run(["python3", "servo_v3.py"] + cmd_list, check=True)
    except Exception as e:
        print("Servo Error:", e)

while True:
    try:
        r = requests.get(SERVER, timeout=1)
        data = r.json()
        print(f"Detected: {data}")

        category = data.get("category", "")

        if category == "Recycle":
            print("Recycle detected ? executing recycle sequence")
            #run_servo(["reset", "mid"])
            run_servo(["pick", "mid"])
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
