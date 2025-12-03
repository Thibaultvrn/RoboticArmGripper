import time
import vision_to_arduino as v

print('== run_vision_headless: start ==')
# Force headless
v.CFG.show_window = False
# Keep index as configured (default 1), but try autodetect if needed
cap = v.open_camera(v.CFG)
if cap is None:
    print('Failed to open camera via vision_to_arduino.open_camera()')
else:
    print('Camera opened, reading frames...')
    for i in range(50):
        ok, frame = cap.read()
        print(f' frame {i}: read_ok={ok}', end='')
        if not ok:
            print('\n Read failed, breaking')
            break
        blur = v.cv2.GaussianBlur(frame, (5,5), 0)
        mask_red = v.red_mask_rrggbb(blur, v.CFG)
        mask = v.clean_mask(mask_red, v.CFG.morph_open_ks, v.CFG.morph_close_ks)
        nz = int((mask>0).sum())
        print(f' mask_nonzero={nz}')
        time.sleep(0.02)
    try:
        cap.release()
    except Exception:
        pass
print('== done ==')
