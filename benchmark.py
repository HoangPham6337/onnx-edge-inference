import os
import time
import psutil
import threading

import cv2
import numpy as np
import onnxruntime as ort
import scipy.special
from tqdm import tqdm

from dataset_builder.core.utility import load_manifest_parquet

OUTPUT_FILE = "inference_results.csv"
MODEL_PATH = "./models/InceptionV3_HG_onnx/inceptionv3_50.onnx"
# INPUT_SIZE = (224, 224)
INPUT_SIZE = (299, 299)
LIMIT = 100
print(MODEL_PATH)

peak_mem_mb = 0
def monitor_memory(pid):
    global peak_mem_mb
    proc = psutil.Process(pid)
    while True:
        try:
            mem = proc.memory_info().rss / 1024 ** 2
            peak_mem_mb = max(peak_mem_mb, mem)
            time.sleep(0.5)
        except psutil.NoSuchProcess:
            break



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
    # img = np.transpose(img, (0, 3, 1, 2))

    return img


print(ort.get_available_providers())
def run_inference_benchmark():

    session = ort.InferenceSession(MODEL_PATH, providers=["CUDAExecutionProvider"])

    print(f"Using: {session.get_providers()}")

    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    input_dtype = session.get_inputs()[0].type
    print(f"Input: {input_name}, Shape: {input_shape}, Dtype: {input_dtype}")

    data_manifest = load_manifest_parquet("./data/haute_garonne/dataset_manifest.parquet")[0:1000]

    with open(OUTPUT_FILE, "w") as csv_file:
        csv_file.write("filepath,predicted_species,probability,inference_time_s,correct_species\n")

        loop = tqdm(data_manifest, desc="Image")
        for image_data in loop:
            image_path = image_data[0]
            try:
                img = preprocess_eval_opencv(image_path, *INPUT_SIZE)

                start = time.perf_counter()
                outputs = session.run(None, {input_name: img})
                end = time.perf_counter()

                probabilities = scipy.special.softmax(outputs[0], axis=1)
                top1_idx = int(np.argmax(probabilities[0]))
                top1_prob = float(probabilities[0][top1_idx])
                inference_time = end - start

                csv_file.write(f"{image_path},{top1_idx},{top1_prob: .4f},{inference_time: .5f}\n")
                csv_file.flush()
            except Exception as e:
                print(f"ERROR Skipping {image_path}: {e}")

if __name__ == "__main__":
    print(f"PID: {os.getpid()}")
    print("READY_FOR_PERF",flush=True)
    # time.sleep(2)
    input("Attach `perf` now and press Enter to begin inference ...")
    monitor_thread = threading.Thread(target=monitor_memory, args=(os.getpid(),), daemon=True)
    monitor_thread.start()

    start_time = time.time()
    run_inference_benchmark()
    end_time = time.time()

    print(f"\nTotal inference time: {end_time - start_time:.2f}s")
    print(f"Peak memory during inference: {peak_mem_mb:.2f} MB")
