# -*- coding: utf-8 -*-
"""
main.py — smoke test + mini console série
- Détecte une caméra disponible
- Liste les ports série (Arduino)
- Essaie un PING vers l'Arduino si détecté
- Permet d'envoyer manuellement: PING (p), GRAB (g), Quit (q)
"""

import time
import cv2
import serial
import serial.tools.list_ports as list_ports

# ---------- Caméra ----------
def find_camera(max_index=4):
    """Renvoie l'index de la première caméra dispo, ou None."""
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.release()
            return i
        cap.release()
    return None

# ---------- Série ----------
def list_serial_ports():
    return list(list_ports.comports())

def find_arduino_port():
    """Essaie de reconnaître un Arduino/clone (Arduino, CH340, USB-SERIAL, VID/PID connus)."""
    for p in list_ports.comports():
        d = (p.description or "").upper()
        h = (p.hwid or "").upper()
        if any(k in d for k in ["ARDUINO", "CH340", "USB-SERIAL", "USB SERIAL"]) \
           or any(v in h for v in ["VID:2341", "VID:2A03", "VID:1A86"]):
            return p.device
    return None

if __name__ == "__main__":
    print("✅ Environnement Python OK !")

    # --- Caméra ---
    cam_index = find_camera()
    if cam_index is None:
        print("ℹ️ Aucune caméra détectée (c’est OK si rien n’est branché).")
    else:
        print(f"📷 Caméra détectée sur l’index {cam_index}.")
        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        ok, _ = cap.read()
        print("   → Capture test:", "OK" if ok else "échec")
        cap.release()

    # --- Ports série ---
    ports = list_serial_ports()
    if not ports:
        print("ℹ️ Aucun port série détecté (Arduino non branché).")
    else:
        print("🔌 Ports série détectés :")
        for p in ports:
            print(f"   - {p.device} | {p.description}")

    # --- PING automatique si Arduino trouvé ---
    port = find_arduino_port()
    if port is None:
        print("⚠️ Aucun Arduino/clone reconnu. Branche la carte puis relance le script.")
        raise SystemExit(0)

    print(f"➡️ Connexion à l’Arduino sur {port} @115200 …")
    try:
        with serial.Serial(port, 115200, timeout=1) as ser:
            time.sleep(2)  # reset auto
            ser.write(b"PING\n")
            resp = ser.readline().decode(errors="ignore").strip()
            print("   Réponse Arduino:", resp or "(aucune)")

            # Mini console
            print("\nConsole: 'p' = PING, 'g' = GRAB (démo), 'q' = Quit")
            while True:
                cmd = input("> ").strip().lower()
                if cmd == "q": break
                elif cmd == "p": ser.write(b"PING\n")
                elif cmd == "g": ser.write(b"GRAB,SMALL,22\n")  # démo
                else: ser.write((cmd + "\n").encode("ascii"))
                time.sleep(0.05)
                while ser.in_waiting:
                    print("Arduino:", ser.readline().decode(errors="ignore").strip())

    except Exception as e:
        print("❌ Échec de connexion série :", e)
