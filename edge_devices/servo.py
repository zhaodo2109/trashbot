#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Servo.py
Features:
 - Servo control using PCA9685 (channels 8-15)
 - Speed modes: slow / mid / fast
 - Persistent servo angles in .servo_state.json
 - Load servo presets dynamically from servo_presets.json
 - Supports nested action modes: sequence / parallel / single
 - Command line:
     python3 servo.py <channel> <angle> [speed]
     python3 servo.py reset [speed]
     python3 servo.py <preset> [speed]
"""

import sys
import time
import json
import os
import threading
from pca9685 import PCA9685


# ---------- Load preset file ----------
PRESET_FILE = "servo_presets.json"

def load_presets():
    """Load servo presets from external JSON file, or use defaults."""
    if os.path.exists(PRESET_FILE):
        try:
            with open(PRESET_FILE, "r") as f:
                presets = json.load(f)
                print(f"[INFO] Loaded presets from {PRESET_FILE}")
                return presets
        except Exception as e:
            print(f"[WARN] Failed to load {PRESET_FILE}: {e}")
            return {}
    else:
        print(f"[WARN] {PRESET_FILE} not found. Using default presets.")
        return {
            "set1": {
                "mode": "parallel",
                "actions": [
                    ["0", 0],
                    ["1", 130],
                    ["2", 70]
                ]
            },
            "set2": {
                "mode": "sequence",
                "actions": [
                    {"mode": "single", "actions": [["0", 180]]},
                    {"mode": "parallel", "actions": [["1", 70], ["2", 180]]}
                ]
            }
        }

PRESETS = load_presets()


class Servo:
    STATE_FILE = ".servo_state.json"

    def __init__(self, address=0x40, busnum=1):
        # Initialize PCA9685 driver
        self.pwm = PCA9685(address)
        self.pwm.set_pwm_freq(50)

        # Logical servo IDs 0–7 → physical PCA9685 channels 8–15
        self.servo_map = {str(i): 8 + i for i in range(8)}

        # Load saved servo angles from file
        self.current_angles = self.load_state()

        # Speed profiles: (angle step, delay seconds)
        self.speed_modes = {
            'slow': (1, 0.02),
            'mid':  (1, 0.01),
            'fast': (3, 0.01),
        }

    # ---------- File persistence ----------
    def load_state(self):
        """Load last known servo angles from file."""
        if os.path.exists(self.STATE_FILE):
            try:
                with open(self.STATE_FILE, "r") as f:
                    data = json.load(f)
                return {str(k): float(v) for k, v in data.items()}
            except Exception:
                return {}
        else:
            return {}

    def save_state(self):
        """Save current servo angles to file."""
        try:
            with open(self.STATE_FILE, "w") as f:
                json.dump(self.current_angles, f)
        except Exception as e:
            print(f"[WARN] Could not save servo state: {e}")

    # ---------- Utility functions ----------
    def angle_to_pulse(self, angle):
        """Convert angle (0–180°) to pulse width (µs)."""
        angle = max(0, min(180, angle))
        pulse = 500 + int(angle * 2000 / 180)
        return pulse

    def set_servo_pwm(self, logic_channel, angle):
        """Send PWM pulse to physical servo channel."""
        pulse = self.angle_to_pulse(angle)
        physical_channel = self.servo_map[str(logic_channel)]
        self.pwm.set_servo_pulse(physical_channel, pulse)

    # ---------- Main movement control ----------
    def move_to(self, channel, target_angle, speed='slow'):
        """
        Move servo to target_angle with given speed.
        speed: 'slow' | 'mid' | 'fast'
        """
        ch = str(channel)
        current = self.current_angles.get(ch)

        # First-time control → direct move
        if current is None:
            self.set_servo_pwm(ch, target_angle)
            self.current_angles[ch] = target_angle
            self.save_state()
            return

        # Check speed
        if speed not in self.speed_modes:
            raise ValueError(f"Invalid speed '{speed}', must be one of {list(self.speed_modes.keys())}")

        step, delay = self.speed_modes[speed]

        # Movement logic
        if speed == 'fast':
            self.set_servo_pwm(ch, target_angle)
        else:
            if target_angle > current:
                angles = range(int(current), int(target_angle) + 1, step)
            else:
                angles = range(int(current), int(target_angle) - 1, -step)

            for angle in angles:
                self.set_servo_pwm(ch, angle)
                time.sleep(delay)

        # Update and save angle
        self.current_angles[ch] = target_angle
        self.save_state()

    # ---------- Recursive Action Executor ----------
    def run_actions(self, actions, mode='sequence', speed='slow'):
        """Recursively execute servo actions (supports nested sequence / parallel)."""
        # Case 1: simple list of channel-angle pairs
        if isinstance(actions, list) and all(isinstance(a, list) for a in actions):
            if mode == 'parallel':
                threads = []
                for ch, angle in actions:
                    t = threading.Thread(target=self.move_to, args=(ch, angle, speed))
                    t.start()
                    threads.append(t)
                for t in threads:
                    t.join()
            else:  # sequence
                for ch, angle in actions:
                    self.move_to(ch, angle, speed)
            return

        # Case 2: list of grouped actions (nested)
        if isinstance(actions, list):
            for item in actions:
                m = item.get("mode", "sequence")
                sub_actions = item.get("actions", [])
                self.run_actions(sub_actions, mode=m, speed=speed)
            return

    # ---------- Run Preset ----------
    def run_preset(self, preset_name, speed='slow', mode=None):
        """Run a predefined servo position set (supports nested modes)."""
        preset = PRESETS.get(preset_name)
        if not preset:
            print(f"[ERROR] Preset '{preset_name}' not found.")
            return

        exec_mode = mode or preset.get("mode", "sequence")
        print(f"[PRESET] Executing '{preset_name}' ({exec_mode}, speed={speed}) ...")

        actions = preset.get("actions", [])
        self.run_actions(actions, mode=exec_mode, speed=speed)

        print(f"[PRESET] '{preset_name}' completed.")
        self.save_state()

    # ---------- Reset all servos ----------
    def reset_all(self, target_angle=90, speed='mid'):
        """Move all servos (0–7) to the same angle (default 90°)."""
        print(f"[RESET] Moving all servos to {target_angle}° (speed={speed}) ...")

        if speed not in self.speed_modes:
            raise ValueError(f"Invalid speed '{speed}', must be one of {list(self.speed_modes.keys())}")

        threads = []
        for ch in self.servo_map.keys():
            t = threading.Thread(target=self.move_to, args=(ch, target_angle, speed))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

        self.save_state()
        print("[RESET] All servos centered at 90°.")

    # ---------- Cleanup ----------
    def cleanup(self):
        """Close I2C bus."""
        self.pwm.close()


# ========== Command-line entry ==========
def main():
    """
    Command line usage:
      python3 servo.py <channel> <angle> [speed]
      python3 servo.py reset [speed]
      python3 servo.py set1 [speed]
      python3 servo.py set2 [speed]
    """
    # --- Handle reset command ---
    if len(sys.argv) >= 2 and sys.argv[1].lower() == "reset":
        speed = sys.argv[2] if len(sys.argv) == 3 else "fast"
        servo = Servo()
        servo.reset_all(speed=speed)
        servo.cleanup()
        return

    # --- Handle preset commands (set1, set2, etc.) ---
    if len(sys.argv) >= 2 and sys.argv[1].lower() in PRESETS:
        preset = sys.argv[1].lower()
        speed = sys.argv[2] if len(sys.argv) == 3 else "slow"
        servo = Servo()
        servo.run_preset(preset, speed)
        servo.cleanup()
        return

    # --- Normal servo movement command ---
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python3 servo.py <channel> <angle> [speed]")
        sys.exit(1)

    channel = sys.argv[1]
    angle = int(sys.argv[2])
    speed = sys.argv[3] if len(sys.argv) == 4 else "slow"

    servo = Servo()
    servo.move_to(channel, angle, speed)
    servo.cleanup()


if __name__ == "__main__":
    main()
