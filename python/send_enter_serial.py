#!/usr/bin/env python3
"""
send_enter_serial.py
Simple helper: open serial (auto-detect Arduino) and send an ASCII message each time
the user presses Enter in the console.

Usage:
  python send_enter_serial.py [--port COM3] [--baud 115200] [--msg "ENTER\n"]

If no --port given, the script tries to autodetect a likely Arduino port.
If no serial is found it prompts for a port or runs in dry-run mode (prints what it would send).
"""

import argparse
import sys
import time
from typing import Optional

import serial
import serial.tools.list_ports as list_ports


def find_arduino_port(max_index_search: int = 10) -> Optional[str]:
    """Try to find a serial port that looks like an Arduino/USB-serial device.
    Returns the port name like 'COM3' or None.
    """
    for p in list_ports.comports():
        d = (p.description or "").upper()
        h = (p.hwid or "").upper()
        if any(k in d for k in ["ARDUINO", "CH340", "USB-SERIAL", "USB SERIAL"]) or any(
            v in h for v in ["VID:2341", "VID:2A03", "VID:1A86"]
        ):
            return p.device
    # fallback: return first openable port if nothing matched
    ports = [p.device for p in list_ports.comports()]
    return ports[0] if ports else None


def open_serial_port(port: str, baud: int):
    try:
        ser = serial.Serial(port, baud, timeout=1)
        # some boards reset on serial open; wait a bit
        time.sleep(2.0)
        return ser
    except Exception as e:
        print(f"❌ Failed to open serial {port} @ {baud}: {e}")
        return None


def main():
    p = argparse.ArgumentParser(description="Send ASCII message to Arduino when Enter is pressed.")
    p.add_argument("--port", help="Serial port (e.g. COM3). If omitted, try autodetect.")
    p.add_argument("--baud", type=int, default=115200, help="Serial baudrate (default: 115200)")
    p.add_argument("--msg", default="ENTER\n", help=r"ASCII message to send (default: 'ENTER\\n')")
    args = p.parse_args()

    port = args.port
    if not port:
        print("No port specified. Trying autodetection...")
        port = find_arduino_port()
        if port:
            print(f"Autodetected port: {port}")
        else:
            print("No serial ports found on this machine.")

    ser = None
    if port:
        ser = open_serial_port(port, args.baud)
        if ser:
            print(f"🔌 Opened serial {port} @ {args.baud}")
        else:
            print("⚠️  Serial open failed; running in dry-run mode.")
    else:
        # ask user to enter a port manually or continue in dry-run
        choice = input("Enter a COM port (e.g. COM3) or press Enter to continue in dry-run: ")
        if choice.strip():
            ser = open_serial_port(choice.strip(), args.baud)
            if ser:
                print(f"🔌 Opened serial {choice.strip()} @ {args.baud}")

    print("Ready. Press Enter to send, type 'q' + Enter to quit.")
    print(f"Message to send (raw): {repr(args.msg)}")

    try:
        while True:
            # wait for a line (user presses Enter)
            line = sys.stdin.readline()
            if not line:
                # EOF
                break
            if line.strip().lower() in ("q", "quit", "exit"):
                print("Quitting.")
                break
            # send the configured ASCII message
            try:
                if ser is not None:
                    ser.write(args.msg.encode('ascii'))
                    ser.flush()
                    print(f"➡️ Sent: {args.msg.strip()}")
                else:
                    print(f"(dry-run) Would send: {args.msg.strip()}")
            except Exception as e:
                print("❌ Failed to send over serial:", e)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass
        print("Done.")


if __name__ == '__main__':
    main()
