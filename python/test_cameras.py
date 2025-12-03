import cv2, time
def try_idx(i, backend):
    cap = cv2.VideoCapture(i, backend)
    ok = cap.isOpened()
    print(f"Index {i}: {'OK' if ok else 'no'}  backend={backend}")
    if ok:
        ret, f = cap.read()
        if ret and f is not None:
            cv2.imshow(f'cam_{i}', f)
            cv2.waitKey(800)
            cv2.destroyWindow(f'cam_{i}')
        cap.release()

for i in range(6):
    try_idx(i, cv2.CAP_DSHOW)
print('--- now MSMF ---')
for i in range(6):
    try_idx(i, cv2.CAP_MSMF)
cv2.destroyAllWindows()