import cv2
import numpy as np
import threading
import psutil
import time
import os

def preprocess_eval_opencv(image_path, width, height, central_fraction=0.857):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Failed to read image")
    h, w, _ = img.shape

    crop_h = int(h * central_fraction)
    crop_w = int(w * central_fraction)

    offset_h = (h - crop_h) // 2
    offset_w = (w - crop_w) // 2
    img = img[offset_h: offset_h + crop_h, offset_w: offset_w + crop_w]

    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)

    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) * 2.0

    img = np.expand_dims(img, axis=0)
    img = np.transpose(img, (0, 3, 1, 2))

    return img


def monitor_peak_memory(pid: int, flag, interval: float=0.01):
    peak = 0
    proc = psutil.Process(pid)
    while flag["run"]:
        try:
            mem = proc.memory_info().rss / 1024 ** 2
            peak = max(peak, mem)
            time.sleep(interval)
        except psutil.NoSuchProcess:
            break
    flag["peak"] = peak


def measure_inference_memory(session, input_name, img):
    flag = {"run": True, "peak": 0}
    t = threading.Thread(target=monitor_peak_memory, args=(os.getpid(), flag))
    t.start()
    start = time.perf_counter()
    outputs = session.run(None, {input_name: img})
    end = time.perf_counter()
    flag["run"] = False
    t.join()
    inference_time = end - start
    peak_mem = flag["peak"]
    return outputs, inference_time, peak_mem