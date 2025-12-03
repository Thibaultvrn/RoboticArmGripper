# -- coding: utf-8 --
"""
vision_to_arduino.py — Vision (sans fond vert), prêt à exécuter sans Arduino
----------------------------------------------------------------------------

- Ouvre la caméra index=1
- Détecte le "rouge" (règle RR/GG/BB)
- Sélectionne la framboise rouge la plus grande (par aire)
- Vérifie la maturité (fraction de pixels rouges dans la boîte englobante)
- Estime la largeur max (scan lignes) et convertit en mm avec calibration
- Affiche les fenêtres "camera" et "mask_red"
- Quitter : ESC ou 'q'

Dépendances :
    pip install opencv-python numpy
"""

import time
from dataclasses import dataclass, asdict
from typing import Optional
import json
import os

import cv2
import numpy as np
import serial
import serial.tools.list_ports as list_ports
import subprocess
import re


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

    # --- Détection "rouge" (RR/GG/BB) ---
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

    # --- Maturité (fraction de rouge dans la boîte) ---
    red_area_frac_min: float = 0.22

    # --- Calibration (pixels → mm) ---
    REF_MM: float = 30.0
    REF_PX: float = 330.0
    override_mm_per_px: float = 0.0

    # --- Décision petite/grande (mm) ---
    small_large_threshold_mm: float = 25.0
    size_large_mm: float = 30.0
    size_small_mm: float = 10.0
    # --- Décision petite/grande (pixels) ---
    # If >0, this pixel threshold takes precedence over the mm threshold.
    small_large_threshold_px: int = 575
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

    # --- Serial / ASCII send ---
    serial_baud: int = 115200
    ascii_message: str = "ENTER\n"
    forced_serial_port: str = "COM3"
    # Optional forced backend name: 'DSHOW', 'MSMF' or 'DEFAULT' (None means auto)
    forced_backend: Optional[str] = None


CFG = Config()
CONFIG_FILENAME = "vision_config.json"


def config_file_path() -> str:
    return os.path.join(os.path.dirname(__file__), CONFIG_FILENAME)


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
        # update only known fields, MAIS on ignore cam_index
        for k, v in data.items():
            if k == "cam_index":
                # on ne lit plus jamais cam_index depuis le fichier,
                # la valeur "canonique" reste celle du code (ou de la CLI)
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
def autodetect_camera(max_index: int = 4) -> Optional[int]:
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.release()
            return i
        cap.release()
    return None


def open_camera(cfg: Config):
    # Try to find a device by name (Windows DirectShow) using ffmpeg, prefer Logitech
    def get_dshow_device_names():
        """Return list of DirectShow device names discovered by ffmpeg, or [] if ffmpeg not available."""
        names = []
        # Try ffmpeg first (prints device list to stderr). Not present on all machines.
        try:
            p = subprocess.run([
                "ffmpeg",
                "-list_devices",
                "true",
                "-f",
                "dshow",
                "-i",
                "dummy",
            ], capture_output=True, text=True, timeout=4)
            out = p.stderr or p.stdout or ""
            ff_names = re.findall(r'"([^"]+)"', out)
            for n in ff_names:
                s = n.strip()
                if s:
                    names.append(s)
        except Exception:
            # ffmpeg not available or failed; continue to PowerShell fallback below
            pass

        # PowerShell fallback: Get-PnpDevice lists camera friendly names on Windows
        try:
            p2 = subprocess.run([
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-PnpDevice -Class Camera | Select-Object -ExpandProperty FriendlyName",
            ], capture_output=True, text=True, timeout=3)
            out2 = p2.stdout or ""
            for line in out2.splitlines():
                s = line.strip()
                if s:
                    names.append(s)
        except Exception:
            pass

        # return unique preserving order
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

    # prefer explicit device name for Logitech if available
    try:
        logi_name = find_device_by_keywords(["LOGI", "LOGITECH", "C920", "C922", "C270", "WEBCAM"])
        if logi_name:
            # try opening by name via DirectShow
            try:
                cap_name = cv2.VideoCapture(f"video={logi_name}", cv2.CAP_DSHOW)
                cap_name.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.cam_width)
                cap_name.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.cam_height)
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
        # backend can be None (default) or an OpenCV backend flag
        try:
            if backend is None:
                cap_ = cv2.VideoCapture(index)
            else:
                cap_ = cv2.VideoCapture(index, backend)
        except Exception:
            return None
        try:
            cap_.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.cam_width)
            cap_.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.cam_height)
        except Exception:
            pass
        if not cap_.isOpened():
            try:
                cap_.release()
            except Exception:
                pass
            return None

        # Try reading a few frames to allow camera warm-up or to overcome transient read failures
        for attempt in range(retries):
            try:
                ok, frame = cap_.read()
            except Exception:
                ok, frame = False, None
            if ok and frame is not None:
                return cap_
            # try a grab/retrieve cycle as an alternative
            try:
                cap_.grab()
            except Exception:
                pass
            time.sleep(delay)

        # nothing worked
        try:
            cap_.release()
        except Exception:
            pass
        return None

    # prepare backend order: respect cfg.forced_backend if provided
    backend_map = {"DSHOW": cv2.CAP_DSHOW, "MSMF": cv2.CAP_MSMF, "DEFAULT": None}
    backends = []
    if getattr(cfg, 'forced_backend', None):
        name = (cfg.forced_backend or '').upper()
        if name in backend_map:
            backends.append(backend_map[name])
    # append the usual order, skipping duplicates
    for b in (cv2.CAP_DSHOW, cv2.CAP_MSMF, None):
        if b not in backends:
            backends.append(b)

    def backend_name(b):
        if b is None:
            return 'DEFAULT'
        if b == cv2.CAP_DSHOW:
            return 'DSHOW'
        if b == cv2.CAP_MSMF:
            return 'MSMF'
        return str(b)

    # First try the configured index across several backends
    for b in backends:
        cap = try_open(cfg.cam_index, b)
        if cap is not None:
            print(f"📷 Caméra OK (index={cfg.cam_index}, backend={backend_name(b)})")
            return cap

    # Fallback: autodetect by trying indices with each backend
    MAX_IDX = 8
    for b in backends:
        for i in range(MAX_IDX):
            cap = try_open(i, b)
            if cap is not None:
                print(f"📷 Caméra détectée (index={i}, backend={backend_name(b)}) — mais cam_index de la config n'est PAS modifié.")
                return cap

    print("⚠  Aucune caméra disponible.")
    return None


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
    # If a forced port is set, try it first
    if getattr(cfg, 'forced_serial_port', ''):
        port = cfg.forced_serial_port
    else:
        port = find_arduino_port()
    if not port:
        print("ℹ  Aucun Arduino détecté (autodetection).")
        return None
    try:
        ser = serial.Serial(port, cfg.serial_baud, timeout=1)
        time.sleep(2)
        print(f"🔌 Arduino connecté sur {port} @ {cfg.serial_baud}")
        return ser
    except Exception as e:
        print("❌ Échec connexion série :", e)
        return None


def send_ascii_message(ser, cfg: Config):
    try:
        if ser is not None:
            ser.write(cfg.ascii_message.encode('ascii'))
            print(f"➡ Sent over serial: {cfg.ascii_message.strip()}")
        else:
            print(f"(No serial) Would send: {cfg.ascii_message.strip()}")
    except Exception as e:
        print("❌ Envoi ASCII échoué:", e)


def set_camera_focus(cap, value: int) -> bool:
    """Try to disable autofocus and set focus to value. Returns True if set() succeeded."""
    try:
        # try to disable autofocus first
        try:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0 if value is not None else 1)
        except Exception:
            pass
        ok = cap.set(cv2.CAP_PROP_FOCUS, float(value))
        return bool(ok)
    except Exception:
        return False


def measure_sharpness(frame: np.ndarray) -> float:
    """Sharpness metric: variance of Laplacian on the grayscale image."""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return float(lap.var())
    except Exception:
        return 0.0


def calibrate_focus(cap, cfg: Config) -> Optional[int]:
    """
    Sweep focus values and choose the one that maximizes sharpness.
    Returns chosen focus value or None if calibration not possible.
    """
    best_val = None
    best_score = -1.0
    # clamp range
    lo = int(max(0, cfg.focus_min))
    hi = int(max(lo + 1, cfg.focus_max))
    step = max(1, int(cfg.focus_step))
    print(f"🔎 Calibration focus: balayage {lo}..{hi} step={step}")
    for v in range(lo, hi + 1, step):
        ok = set_camera_focus(cap, v)
        if not ok:
            # if setting focus isn't supported, abort calibration
            print(f"⚠ set focus {v} not supported by camera (abort calibration)")
            return None
        # read a few frames to stabilize
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
    """Masque binaire des pixels 'rouges' via règle RR/GG/BB avec fallback chromaticité.

    Deux critères (OR):
      - critère RGB dominance : R >= R_MIN  et R >= K_DOM*G et R >= K_DOM*B
      - critère chromaticity : R/(R+G+B) >= chroma_r_min (utile pour rose pâle / flou)
    """
    # split and float
    B, G, R = cv2.split(bgr)
    Rf = R.astype(np.float32)
    Gf = G.astype(np.float32)
    Bf = B.astype(np.float32)

    # RGB dominance mask
    mask_rgb = (R >= cfg.R_MIN) & (Rf >= cfg.K_DOM * Gf) & (Rf >= cfg.K_DOM * Bf)

    # Chromaticity mask: R / (R+G+B)
    if cfg.use_chromaticity:
        sum_rgb = (Rf + Gf + Bf) + 1e-6
        r_ratio = Rf / sum_rgb
        # compute saturation to avoid selecting near-white/desaturated areas
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
    """Ouverture/fermeture morphologique pour réduire bruit et combler trous."""
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
    """Plus grand contour du masque (None si trop petit)."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0
    c = max(cnts, key=cv2.contourArea)
    area = int(cv2.contourArea(c))
    if area < min_area:
        return None, 0
    return c, area


def ripe_by_area_fraction(mask: np.ndarray, contour, frac_min: float) -> bool:
    """'Mûr' si la fraction de rouge dans la boîte englobante ≥ seuil."""
    x, y, w, h = cv2.boundingRect(contour)
    roi = mask[y:y + h, x:x + w]
    red_px = int((roi > 0).sum())
    frac = red_px / max(1, (w * h))
    return frac >= frac_min


def max_width_pixels_by_scan(mask: np.ndarray, contour) -> int:
    """
    Largeur maximale en pixels par scan lignes :
      - pour chaque ligne de la ROI, on prend le premier et dernier pixel >0
      - largeur ligne = last-first+1 ; on garde la plus grande.
    """
    x, y, w, h = cv2.boundingRect(contour)
    roi = mask[y:y + h, x:x + w]
    max_span = 0
    for row in range(roi.shape[0]):      # lignes
        first = -1
        last = -1
        for col in range(roi.shape[1]):  # colonnes
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
    print("✅ Démarrage vision (sans fond vert)…")
    print("=== VERSION ENTER + SERIAL ===")
    # tenter de charger une config existante (REF_MM/REF_PX, focus_value, etc.)
    try:
        load_config(CFG)
    except Exception:
        pass
    cap = open_camera(CFG)
    # Appliquer contrôle focus si demandé
    if cap is not None and CFG.use_focus_control:
        try:
            if CFG.focus_auto_calibrate:
                choice = calibrate_focus(cap, CFG)
                if choice is None:
                    print("ℹ Calibration focus impossible — essayer une valeur manuelle.")
                else:
                    # mémoriser la valeur trouvée
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
    # Ouvrir le port série (si détecté ou forcé)
    ser = None
    try:
        ser = open_serial(CFG)
    except Exception:
        ser = None

    mm_per_px = mm_per_pixel(CFG)
    # tolérance pour la visibilité des fenêtres (évite sortie immédiate sur backends non-GUI)
    win_vis_fail_count = 0
    WIN_VIS_FAIL_THRESHOLD = 3
    if mm_per_px <= 0:
        print("⚠ Calibration pixels→mm invalide. On met 1.0 par sécurité.")
        mm_per_px = 1.0

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

            # Lissage léger
            blur = cv2.GaussianBlur(frame, (5, 5), 0)

            # Masque rouge uniquement
            mask_red = red_mask_rrggbb(blur, CFG)

            # Nettoyage
            mask = clean_mask(mask_red, CFG.morph_open_ks, CFG.morph_close_ks)

            # Détection plus grande framboise
            contour, _ = largest_component(mask, CFG.min_area_px)

            ripe = False
            width_px = 0
            width_mm = 0.0
            size_label = "UNKNOWN"

            if contour is not None:
                # maturité = fraction rouge dans la boîte
                ripe = ripe_by_area_fraction(mask, contour, CFG.red_area_frac_min)
                # largeur max (scan lignes)
                width_px = max_width_pixels_by_scan(mask, contour)
                width_mm = width_px * mm_per_px
                size_label = (
                    "SMALL"





                    if width_mm < CFG.small_large_threshold_mm
                    else "LARGE"
                )

                # Dessins debug
                if CFG.show_window:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + w, y + h),
                        (0, 255, 0) if ripe else (0, 0, 255),
                        2,
                    )
                    cv2.putText(
                        frame,
                        f"ripe={int(ripe)} width={width_mm:.1f}mm "
                        f"({width_px}px) {size_label}",
                        (x, max(0, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )
                    # If a pixel-based threshold is configured, recompute label and redraw
                    if getattr(CFG, 'small_large_threshold_px', 0) and CFG.small_large_threshold_px > 0:
                        try:
                            size_label = "SMALL" if width_px < CFG.small_large_threshold_px else "LARGE"
                            cv2.putText(
                                frame,
                                f"ripe={int(ripe)} width={width_mm:.1f}mm "
                                f"({width_px}px) {size_label}",
                                (x, max(0, y - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (255, 255, 255),
                                2,
                            )
                        except Exception:
                            pass
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


            # Affichage
            if CFG.show_window:
                try:
                    cv2.imshow("camera", frame)
                    cv2.imshow("mask_red", mask)
                except Exception:
                    # imshow peut échouer en headless; on ignore ici
                    pass

                # Quitter si l’utilisateur ferme une fenêtre — tolérance courte
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
                    # getWindowProperty peut lever selon backend — ne pas quitter du coup
                    pass

            # ASCII output pour Arduino
            if CFG.print_ascii:
                try:
                    ripe_bit = 1 if ripe else 0
                    # Compute size_bit consistently with display: prefer pixel threshold
                    if getattr(CFG, 'small_large_threshold_px', 0) and CFG.small_large_threshold_px > 0:
                        size_bit = 1 if width_px >= CFG.small_large_threshold_px else 0
                    else:
                        size_bit = 1 if width_mm >= CFG.size_large_mm else 0
                    print(f"<{ripe_bit},{size_bit}>")
                except Exception:
                    # guard
                    pass

            # Clavier
            k = cv2.waitKey(1) & 0xFF
            if k == 27 or k == ord("q"):  # ESC ou 'q'
                break
            elif k == 13 or k == 10:  # Enter key
                send_ascii_message(ser, CFG)

    except KeyboardInterrupt:
        print("\n⛔ Interruption clavier (Ctrl+C).")
    finally:
        if cap is not None:
            cap.release()
        # sauvegarde automatique de la config courante
        try:
            save_config(CFG)
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("✨ Fin propre du programme.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Vision -> Arduino (press Enter to send)")
    parser.add_argument("--cam-index", type=int, help="Override camera index (e.g. 1 for Logitech)")
    parser.add_argument("--serial-port", type=str, help="Force serial port (e.g. COM3)")
    parser.add_argument("--list-cams", action="store_true", help="List available camera indices/backends and exit")
    parser.add_argument("--choose-camera", action="store_true", help="Prompt to choose which detected camera to use")
    args = parser.parse_args()

    # apply overrides before running
    if args.cam_index is not None:
        try:
            CFG.cam_index = int(args.cam_index)
            print(f"🔧 CLI override: cam_index -> {CFG.cam_index}")
            CFG._cli_cam_index_override = True
        except Exception:
            pass
        # Persist the chosen index immediately so subsequent runs use it without CLI
        try:
            save_config(CFG)
        except Exception:
            pass
    if args.serial_port:
        CFG.forced_serial_port = args.serial_port
        print(f"🔧 CLI override: forced_serial_port -> {CFG.forced_serial_port}")
        CFG._cli_serial_port_override = True
    # optional forced backend
    if getattr(args, 'cam_backend', None):
        try:
            CFG.forced_backend = str(args.cam_backend).upper()
            print(f"🔧 CLI override: forced_backend -> {CFG.forced_backend}")
            CFG._cli_forced_backend_override = True
        except Exception:
            pass

    if args.list_cams:
        # quick scan across backends and indices and print results
        backends = [(cv2.CAP_DSHOW, 'DSHOW'), (cv2.CAP_MSMF, 'MSMF'), (None, 'DEFAULT')]
        print("🔎 Scanning cameras (this may take a few seconds)…")
        for b_flag, b_name in backends:
            print(f"  Backend: {b_name}")
            found = False
            for i in range(8):
                try:
                    if b_flag is None:
                        c = cv2.VideoCapture(i)
                    else:
                        c = cv2.VideoCapture(i, b_flag)
                    ok = c.isOpened()
                    if ok:
                        # quick read test
                        r, f = c.read()
                        if r and f is not None:
                            print(f"    Index {i}: OK")
                            found = True
                        else:
                            print(f"    Index {i}: opened but no frame")
                    c.release()
                except Exception:
                    pass
            if not found:
                print("    (no cameras found for this backend)")
        raise SystemExit(0)

    # If requested, or if multiple cameras detected, prompt the user to choose which to use
    if args.choose_camera:
        backends = [(cv2.CAP_DSHOW, 'DSHOW'), (cv2.CAP_MSMF, 'MSMF'), (None, 'DEFAULT')]
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


# =============================================================================
# =======================  ARDUINO (EN COMMENTAIRE)  ==========================
# =============================================================================
# Pour activer l'envoi série plus tard, dé-commentez ce bloc et les appels.
#
# import serial
# import serial.tools.list_ports as list_ports
#
# def find_arduino_port() -> Optional[str]:
#     for p in list_ports.comports():
#         d = (p.description or "").upper()
#         h = (p.hwid or "").upper()
#         if any(k in d for k in ["ARDUINO", "CH340", "USB-SERIAL"]) or any(
#             v in h for v in ["VID:2341", "VID:2A03", "VID:1A86"]
#         ):
#             return p.device
#     return None
#
# def open_serial(baud: int = 115200):
#     port = find_arduino_port()
#     if not port:
#         print("ℹ  Aucun Arduino détecté.")
#         return None
#     try:
#         ser = serial.Serial(port, baud, timeout=1)
#         time.sleep(2)  # reset auto Uno
#         print(f"🔌 Arduino connecté sur {port} @ {baud}")
#         return ser
#     except Exception as e:
#         print("❌ Échec connexion série :", e)
#         return None
#
# def send_grab_command(ser, width_mm: float, size_label: str):
#     line = f"GRAB,{size_label},{int(round(width_mm))}\n"
#     try:
#         ser.write(line.encode("ascii"))
#     except Exception as e:
#         print("❌ Envoi série échoué :", e)
#
# # Exemple d’usage dans la boucle principale :
# # ser = open_serial()
# # if ser is not None and ripe:
# #     send_grab_command(ser, width_mm, size_label)
# # ser.close()