# -- coding: utf-8 --
"""
vision_to_arduino.py — Vision + Arduino (<ripe,size> sur ENTER, O pour trappe)
------------------------------------------------------------------------------

- Ouvre la caméra
- Détecte la framboise rouge la plus grande
- Vérifie si elle est mûre
- Estime sa taille (petite / grande)
- Affiche "camera" + "mask_red"
- Quand on appuie sur ENTER :
    -> envoie sur le port série : "<ripe_bit,size_bit>\n"
       ripe_bit = 1 si mûre, 0 sinon
       size_bit = 0 si petite, 1 si grande
- Quand on appuie sur O :
    -> envoie "O\n" à l'Arduino (séquence trappe)
- Quitter : ESC ou 'q'

Dépendances :
    pip install opencv-python numpy pyserial
"""

import time
from dataclasses import dataclass, asdict
from typing import Optional
import json
import os
import subprocess
import re

import cv2
import numpy as np
import serial
import serial.tools.list_ports as list_ports


# =========================
# Configuration
# =========================
@dataclass
class Config:
    # --- Caméra ---
    cam_index: int = 1
    cam_width: int = 1280
    cam_height: int = 720
    show_window: bool = True

    # --- Détection "rouge" ---
    R_MIN: int = 120
    K_DOM: float = 1.05
    use_chromaticity: bool = True
    chroma_r_min: float = 0.34
    chroma_r_min_relaxed: float = 0.30
    chroma_s_min: int = 40

    # --- Nettoyage masque ---
    morph_open_ks: int = 5
    morph_close_ks: int = 7
    min_area_px: int = 120

    # --- Maturité ---
    red_area_frac_min: float = 0.22

    # --- Calibration (pixels → mm) ---
    REF_MM: float = 30.0
    REF_PX: float = 330.0
    override_mm_per_px: float = 0.0

    # --- Décision petite/grande ---
    small_large_threshold_mm: float = 25.0
    size_large_mm: float = 30.0
    size_small_mm: float = 10.0
    # Si >0, le seuil en pixels est prioritaire.
    small_large_threshold_px: int = 575
    # --- Bundle detection (3 raspberries) ---
    bundle_aspect_ratio_threshold: float = 1.5  # width/height ratio to detect bundles
    bundle_small_threshold_px: int = 1150  # 2× single threshold for bundle classification
    print_ascii: bool = True

    # --- Focus control ---
    use_focus_control: bool = True
    focus_autofocus: bool = False
    focus_value: int = 50
    focus_min: int = 0
    focus_max: int = 255
    focus_step: int = 15
    focus_auto_calibrate: bool = False
    focus_measure_frames: int = 2

    # --- Série vers Arduino ---
    serial_baud: int = 115200
    # Sur ta machine : Arduino Uno (COM3)
    forced_serial_port: str = "COM3"
    # Optional: 'DSHOW', 'MSMF' ou 'DEFAULT'
    forced_backend: Optional[str] = None


CFG = Config()
CONFIG_FILENAME = "vision_config.json"


def config_file_path() -> str:
    return os.path.join(os.path.dirname(_file_), CONFIG_FILENAME)


def save_config(cfg: Config) -> None:
    try:
        p = config_file_path()
        with open(p, "w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, indent=2)
        print(f"💾 Config saved -> {p}")
    except Exception as e:
        print("❌ Échec sauvegarde config:", e)


def load_config(cfg: Config) -> bool:
    p = config_file_path()
    if not os.path.isfile(p):
        return False
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in data.items():
            if k == "cam_index":
                continue
            if k == "forced_serial_port" and getattr(cfg, "_cli_serial_port_override", False):
                continue
            if k == "forced_backend" and getattr(cfg, "_cli_forced_backend_override", False):
                continue
            if hasattr(cfg, k):
                try:
                    setattr(cfg, k, v)
                except Exception:
                    pass
        print(f"🔁 Config loaded <- {p} (cam_index conservé: {cfg.cam_index})")
        return True
    except Exception as e:
        print("❌ Échec lecture config:", e)
        return False


# =========================
# Caméra
# =========================
def open_camera(cfg: Config):
    def get_dshow_device_names():
        names = []
        try:
            p = subprocess.run(
                ["ffmpeg", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
                capture_output=True,
                text=True,
                timeout=4,
            )
            out = p.stderr or p.stdout or ""
            ff_names = re.findall(r'"([^"]+)"', out)
            for n in ff_names:
                s = n.strip()
                if s:
                    names.append(s)
        except Exception:
            pass

        try:
            p2 = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "Get-PnpDevice -Class Camera | Select-Object -ExpandProperty FriendlyName",
                ],
                capture_output=True,
                text=True,
                timeout=3,
            )
            out2 = p2.stdout or ""
            for line in out2.splitlines():
                s = line.strip()
                if s:
                    names.append(s)
        except Exception:
            pass

        seen = set()
        res = []
        for n in names:
            if n and n not in seen:
                seen.add(n)
                res.append(n)
        return res

    def find_device_by_keywords(keywords):
        names = get_dshow_device_names()
        for n in names:
            up = n.upper()
            if any(k.upper() in up for k in keywords):
                return n
        return None

    try:
        logi_name = find_device_by_keywords(["LOGI", "LOGITECH", "C920", "C922", "C270", "WEBCAM"])
        if logi_name:
            try:
                cap_name = cv2.VideoCapture(f"video={logi_name}", cv2.CAP_DSHOW)
                cap_name.set(cv2.CAP_PROP_FRAME_WIDTH, CFG.cam_width)
                cap_name.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG.cam_height)
                ok, f = cap_name.read()
                if ok and f is not None:
                    print(f"📷 Caméra OK (name='{logi_name}', opened via DSHOW)")
                    return cap_name
                try:
                    cap_name.release()
                except Exception:
                    pass
            except Exception:
                pass
    except Exception:
        pass

    def try_open(index: int, backend, retries: int = 6, delay: float = 0.15):
        try:
            if backend is None:
                cap_ = cv2.VideoCapture(index)
            else:
                cap_ = cv2.VideoCapture(index, backend)
        except Exception:
            return None
        try:
            cap_.set(cv2.CAP_PROP_FRAME_WIDTH, CFG.cam_width)
            cap_.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG.cam_height)
        except Exception:
            pass
        if not cap_.isOpened():
            try:
                cap_.release()
            except Exception:
                pass
            return None

        for _ in range(retries):
            try:
                ok, frame = cap_.read()
            except Exception:
                ok, frame = False, None
            if ok and frame is not None:
                return cap_
            try:
                cap_.grab()
            except Exception:
                pass
            time.sleep(delay)

        try:
            cap_.release()
        except Exception:
            pass
        return None

    backend_map = {"DSHOW": cv2.CAP_DSHOW, "MSMF": cv2.CAP_MSMF, "DEFAULT": None}
    backends = []
    if getattr(cfg, "forced_backend", None):
        name = (cfg.forced_backend or "").upper()
        if name in backend_map:
            backends.append(backend_map[name])
    for b in (cv2.CAP_DSHOW, cv2.CAP_MSMF, None):
        if b not in backends:
            backends.append(b)

    def backend_name(b):
        if b is None:
            return "DEFAULT"
        if b == cv2.CAP_DSHOW:
            return "DSHOW"
        if b == cv2.CAP_MSMF:
            return "MSMF"
        return str(b)

    for b in backends:
        cap = try_open(cfg.cam_index, b)
        if cap is not None:
            print(f"📷 Caméra OK (index={cfg.cam_index}, backend={backend_name(b)})")
            return cap

    MAX_IDX = 8
    for b in backends:
        for i in range(MAX_IDX):
            cap = try_open(i, b)
            if cap is not None:
                print(
                    f"📷 Caméra détectée (index={i}, backend={backend_name(b)}) "
                    "— cam_index de la config n'est PAS modifié."
                )
                return cap

    print("⚠ Aucune caméra disponible.")
    return None


# =========================
# Série Arduino
# =========================
def find_arduino_port() -> Optional[str]:
    for p in list_ports.comports():
        d = (p.description or "").upper()
        h = (p.hwid or "").upper()
        if any(k in d for k in ["ARDUINO", "CH340", "USB-SERIAL", "USB SERIAL"]) or any(
            v in h for v in ["VID:2341", "VID:2A03", "VID:1A86"]
        ):
            return p.device
    return None


def open_serial(cfg: Config):
    if getattr(cfg, "forced_serial_port", ""):
        port = cfg.forced_serial_port
        print(f"🛈 Port série forcé: {port}")
    else:
        port = find_arduino_port()
        print(f"🛈 Port série autodétecté: {port}")
    if not port:
        print("ℹ Aucun Arduino détecté.")
        return None
    try:
        ser = serial.Serial(port, cfg.serial_baud, timeout=1)
        time.sleep(2)
        print(f"🔌 Arduino connecté sur {port} @ {cfg.serial_baud}")
        return ser
    except Exception as e:
        print("❌ Échec connexion série :", e)
        return None


def send_decision_message(ser, ripe_bit: int, size_bit: int):
    """
    Envoie sur le port série un message du type "<ripe,size>\n".
    ripe_bit = 0/1, size_bit = 0/1
    """
    line = f"<{ripe_bit},{size_bit}>\n"
    if ser is None:
        print("❌ Pas de port série ouvert, message NON envoyé :", line.strip())
        return
    try:
        ser.write(line.encode("ascii"))
        print(f"➡ Sent over serial: {line.strip()}")
    except Exception as e:
        print("❌ Envoi série échoué:", e)


# =========================
# Focus / netteté
# =========================
def set_camera_focus(cap, value: int) -> bool:
    try:
        try:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0 if value is not None else 1)
        except Exception:
            pass
        ok = cap.set(cv2.CAP_PROP_FOCUS, float(value))
        return bool(ok)
    except Exception:
        return False


def measure_sharpness(frame: np.ndarray) -> float:
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return float(lap.var())
    except Exception:
        return 0.0


def calibrate_focus(cap, cfg: Config) -> Optional[int]:
    best_val = None
    best_score = -1.0
    lo = int(max(0, cfg.focus_min))
    hi = int(max(lo + 1, cfg.focus_max))
    step = max(1, int(cfg.focus_step))
    print(f"🔎 Calibration focus: balayage {lo}..{hi} step={step}")
    for v in range(lo, hi + 1, step):
        ok = set_camera_focus(cap, v)
        if not ok:
            print(f"⚠ set focus {v} not supported by camera (abort calibration)")
            return None
        scores = []
        for _ in range(max(1, cfg.focus_measure_frames)):
            ret, f = cap.read()
            if not ret:
                break
            scores.append(measure_sharpness(f))
            time.sleep(0.05)
        if not scores:
            continue
        avg = sum(scores) / len(scores)
        print(f"  focus={v} sharpness={avg:.2f}")
        if avg > best_score:
            best_score = avg
            best_val = v
    if best_val is not None:
        print(f"✅ Focus choisi: {best_val} (score={best_score:.2f})")
        set_camera_focus(cap, best_val)
    return best_val


# =========================
# Vision (RR/GG/BB)
# =========================
def red_mask_rrggbb(bgr: np.ndarray, cfg: Config) -> np.ndarray:
    B, G, R = cv2.split(bgr)
    Rf = R.astype(np.float32)
    Gf = G.astype(np.float32)
    Bf = B.astype(np.float32)

    mask_rgb = (R >= cfg.R_MIN) & (Rf >= cfg.K_DOM * Gf) & (Rf >= cfg.K_DOM * Bf)

    if cfg.use_chromaticity:
        sum_rgb = (Rf + Gf + Bf) + 1e-6
        r_ratio = Rf / sum_rgb
        try:
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            Sat = hsv[:, :, 1].astype(np.float32)
        except Exception:
            Sat = np.zeros_like(Rf)
        mask_chroma = (
            (r_ratio >= cfg.chroma_r_min)
            & (Rf >= (cfg.R_MIN // 2))
            & (Sat >= float(cfg.chroma_s_min))
            & (Rf >= 60)
        )
    else:
        mask_chroma = np.zeros_like(R, dtype=np.bool_)

    mask = (mask_rgb | mask_chroma).astype(np.uint8) * 255
    return mask


def clean_mask(mask: np.ndarray, open_ks: int, close_ks: int) -> np.ndarray:
    if open_ks > 0:
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, np.ones((open_ks, open_ks), np.uint8)
        )
    if close_ks > 0:
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, np.ones((close_ks, close_ks), np.uint8)
        )
    return mask


def largest_component(mask: np.ndarray, min_area: int):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0
    c = max(cnts, key=cv2.contourArea)
    area = int(cv2.contourArea(c))
    if area < min_area:
        return None, 0
    return c, area


def ripe_by_area_fraction(mask: np.ndarray, contour, frac_min: float) -> bool:
    x, y, w, h = cv2.boundingRect(contour)
    roi = mask[y:y + h, x:x + w]
    red_px = int((roi > 0).sum())
    frac = red_px / max(1, (w * h))
    return frac >= frac_min


def max_width_pixels_by_scan(mask: np.ndarray, contour) -> int:
    x, y, w, h = cv2.boundingRect(contour)
    roi = mask[y:y + h, x:x + w]
    max_span = 0
    for row in range(roi.shape[0]):
        first = -1
        last = -1
        for col in range(roi.shape[1]):
            if roi[row, col] > 0:
                if first == -1:
                    first = col
                last = col
        if first != -1 and last != -1:
            span = last - first + 1
            if span > max_span:
                max_span = span
    return max_span


def mm_per_pixel(cfg: Config) -> float:
    if cfg.override_mm_per_px > 0:
        return cfg.override_mm_per_px
    return cfg.REF_MM / max(1e-6, cfg.REF_PX)


# =========================
# Boucle principale
# =========================
def main():
    print("✅ Démarrage vision…")
    print("=== ENTER -> <ripe,size>, O -> 'O' pour trappe ===")

    try:
        load_config(CFG)
    except Exception:
        pass

    cap = open_camera(CFG)

    if cap is not None and CFG.use_focus_control:
        try:
            if CFG.focus_auto_calibrate:
                choice = calibrate_focus(cap, CFG)
                if choice is None:
                    print("ℹ Calibration focus impossible — essayer une valeur manuelle.")
                else:
                    try:
                        CFG.focus_value = int(choice)
                        save_config(CFG)
                    except Exception:
                        pass
            else:
                okf = set_camera_focus(cap, CFG.focus_value)
                print(f"🔧 Set focus -> {CFG.focus_value} (success={okf})")
        except Exception as e:
            print("⚠ Erreur durant le contrôle du focus:", e)

    ser = open_serial(CFG)
    print("DEBUG: SER =", ser)

    mm_per_px = mm_per_pixel(CFG)
    if mm_per_px <= 0:
        print("⚠ Calibration pixels→mm invalide. On met 1.0 par sécurité.")
        mm_per_px = 1.0

    win_vis_fail_count = 0
    WIN_VIS_FAIL_THRESHOLD = 3

    try:
        while True:
            if cap is None:
                time.sleep(0.2)
                cap = open_camera(CFG)
                continue

            ok, frame = cap.read()
            if not ok:
                print("⚠ Frame non lue; reconnection caméra…")
                cap.release()
                cap = None
                continue

            blur = cv2.GaussianBlur(frame, (5, 5), 0)
            mask_red = red_mask_rrggbb(blur, CFG)
            mask = clean_mask(mask_red, CFG.morph_open_ks, CFG.morph_close_ks)

            contour, _ = largest_component(mask, CFG.min_area_px)

            ripe = False
            width_px = 0
            width_mm = 0.0
            size_label = "UNKNOWN"

            if contour is not None:
                ripe = ripe_by_area_fraction(mask, contour, CFG.red_area_frac_min)
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check aspect ratio to detect bundles (elongated shape = 3 raspberries)
                try:
                    aspect_ratio = float(max(w, h)) / max(1.0, float(min(w, h)))
                except Exception:
                    aspect_ratio = 1.0
                
                is_bundle = aspect_ratio > CFG.bundle_aspect_ratio_threshold
                
                # For bundles: use max width; for single: use smaller dimension
                if is_bundle:
                    width_px = int(max(w, h))
                    # Classify bundle as small or large using bundle threshold
                    size_label = "SMALL" if width_px < CFG.bundle_small_threshold_px else "LARGE"
                else:
                    # Single raspberry: use minimum dimension as diameter
                    width_px = int(min(w, h))
                    # Use single raspberry threshold
                    if getattr(CFG, 'small_large_threshold_px', 0) and CFG.small_large_threshold_px > 0:
                        size_label = "SMALL" if width_px < CFG.small_large_threshold_px else "LARGE"
                    else:
                        width_mm = width_px * mm_per_px
                        size_label = "SMALL" if width_mm < CFG.small_large_threshold_mm else "LARGE"
                
                width_mm = width_px * mm_per_px

                if CFG.show_window:
                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + w, y + h),
                        (0, 255, 0) if ripe else (0, 0, 255),
                        2,
                    )
                    # Format display text based on bundle/single and ripe status
                    if ripe:
                        if is_bundle:
                            display_text = f"bundle : {size_label.lower()}"
                        else:
                            display_text = f"single : {size_label.lower()}"
                    else:
                        display_text = "unripe raspberries"
                    
                    cv2.putText(
                        frame,
                        display_text,
                        (x, max(0, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )
            else:
                if CFG.show_window:
                    cv2.putText(
                        frame,
                        "No raspberry",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )

            if CFG.show_window:
                try:
                    cv2.imshow("camera", frame)
                    cv2.imshow("mask_red", mask)
                except Exception:
                    pass

                try:
                    vis_cam = cv2.getWindowProperty("camera", cv2.WND_PROP_VISIBLE)
                    vis_mask = cv2.getWindowProperty("mask_red", cv2.WND_PROP_VISIBLE)
                    if vis_cam < 1 or vis_mask < 1:
                        win_vis_fail_count += 1
                    else:
                        win_vis_fail_count = 0
                    if win_vis_fail_count >= WIN_VIS_FAIL_THRESHOLD:
                        print("⚠ Fenêtres fermées ou non visibles — sortie.")
                        break
                except Exception:
                    pass

            # Calcul des bits ripe / size
            try:
                ripe_bit = 1 if ripe else 0
                # size_bit: 1=large (single or bundle), 0=small (single or bundle)
                size_bit = 1 if size_label == "LARGE" else 0

                if CFG.print_ascii:
                    print(f"<{ripe_bit},{size_bit}>")
            except Exception:
                ripe_bit = 0
                size_bit = 0

            # --- Clavier ---
            k = cv2.waitKey(1) & 0xFF

            if k == 27 or k == ord("q"):   # Quit
                break

            elif k == 13 or k == 10:       # ENTER → envoyer <ripe,size>
                send_decision_message(ser, ripe_bit, size_bit)

            elif k == ord("o") or k == ord("O"):  # O → séquence trappe
                if ser is not None:
                    try:
                        ser.write(b"O\n")
                        print("➡ Sent over serial: O (sequence trappe)")
                    except Exception as e:
                        print("❌ Envoi série O échoué:", e)
                else:
                    print("(No serial) Would send: O")

    except KeyboardInterrupt:
        print("\n⛔ Interruption clavier (Ctrl+C).")
    finally:
        if cap is not None:
            cap.release()
        try:
            save_config(CFG)
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("✨ Fin propre du programme.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Vision -> Arduino (<ripe,size> sur ENTER, O = trappe)")
    parser.add_argument("--cam-index", type=int, help="Override camera index (e.g. 1 for Logitech)")
    parser.add_argument("--serial-port", type=str, help="Force serial port (e.g. COM3)")
    parser.add_argument("--choose-camera", action="store_true", help="Prompt pour choisir la caméra")
    args = parser.parse_args()

    if args.cam_index is not None:
        try:
            CFG.cam_index = int(args.cam_index)
            print(f"🔧 CLI override: cam_index -> {CFG.cam_index}")
            CFG._cli_cam_index_override = True
        except Exception:
            pass
        try:
            save_config(CFG)
        except Exception:
            pass

    if args.serial_port:
        CFG.forced_serial_port = args.serial_port
        print(f"🔧 CLI override: forced_serial_port -> {CFG.forced_serial_port}")
        CFG._cli_serial_port_override = True

    if args.choose_camera:
        backends = [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF"), (None, "DEFAULT")]
        candidates = []
        print("🔎 Scanning cameras for interactive choice…")
        for b_flag, b_name in backends:
            for i in range(8):
                try:
                    if b_flag is None:
                        c = cv2.VideoCapture(i)
                    else:
                        c = cv2.VideoCapture(i, b_flag)
                    ok = c.isOpened()
                    if ok:
                        r, f = c.read()
                        if r and f is not None:
                            candidates.append((i, b_name))
                    c.release()
                except Exception:
                    pass
        if candidates:
            print("Detected camera candidates:")
            for idx, (cam_i, backend_name) in enumerate(candidates):
                print(f"  [{idx}] index={cam_i} backend={backend_name}")
            sel = input("Choose camera number (or press Enter to keep config): ")
            try:
                if sel.strip() != "":
                    si = int(sel.strip())
                    cam_i, backend_name = candidates[si]
                    CFG.cam_index = cam_i
                    CFG.forced_backend = backend_name
                    print(f"Selected index={cam_i} backend={backend_name}")
            except Exception:
                print("Invalid selection, using config values.")

    main()