import os
import time
import psutil
import threading

import cv2
import numpy as np
import onnxruntime as ort
import scipy.special
from tqdm import tqdm
from utility import measure_inference_memory, preprocess_eval_opencv

from dataset_builder.core.utility import load_manifest_parquet

OUTPUT_FILE = "inference_results.csv"
MODEL_PATH = "/home/tom-maverick/Documents/Final Results/MobileNetV3-modified_train_prune_retrain/mobilenet_v3_large_50_pruned_retrain.onnx"
INPUT_SIZE = (224, 224)
# INPUT_SIZE = (299, 299)
LIMIT = 100
print(MODEL_PATH)

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
        csv_file.write("filepath,predicted_species,probability,inference_time_s,correct_species,peak_mem_mb\n")

        loop = tqdm(data_manifest, desc="Image")
        for image_data in loop:
            image_path = image_data[0]
            correct_species = image_data[1]
            try:
                img = preprocess_eval_opencv(image_path, *INPUT_SIZE)

                outputs, inference_time, peak_mem = measure_inference_memory(session, input_name, img)

                probabilities = scipy.special.softmax(outputs[0], axis=1)
                top1_idx = int(np.argmax(probabilities[0]))
                top1_prob = float(probabilities[0][top1_idx])

                csv_file.write(f"{image_path},{top1_idx},{top1_prob: .4f},{inference_time: .5f},{correct_species},{peak_mem}\n")
                csv_file.flush()
            except Exception as e:
                print(f"ERROR Skipping {image_path}: {e}")

if __name__ == "__main__":
    print(f"PID: {os.getpid()}")
    print("READY_FOR_PERF",flush=True)
    input("Attach `perf` now and press Enter to begin inference ...")

    start_time = time.time()
    run_inference_benchmark()
    end_time = time.time()

    print(f"\nTotal inference time: {end_time - start_time:.2f}s")
