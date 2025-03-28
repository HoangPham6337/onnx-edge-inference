import time
import os
import onnxruntime as ort
from tqdm import tqdm
import cv2
import numpy as np
import scipy.special

INPUT_DIR = "/home/tom-maverick/Documents/02_GitHub/cvpr18-inaturalist-transfer/data/haute_garonne_other/"
OUTPUT_FILE = "inference_results.csv"
MODEL_PATH = "/home/tom-maverick/Documents/02_GitHub/cvpr18-inaturalist-transfer/hg_other_model.onnx"
INPUT_SIZE = (299, 299)

def preprocess_eval_opencv(image_path, width, height, central_fraction=0.857):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Failed to read image")
    h, w, _ = img.shape

    crop_h = int(h * central_fraction)
    crop_w = int(w * central_fraction)

    offset_h = (h - crop_h) // 2
    offset_w = (w - crop_w) // 2
    img = img[offset_h:offset_h+crop_h, offset_w:offset_w+crop_w]

    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)

    img=img.astype(np.float32) / 255.0
    img = (img - 0.5) * 2.0 

    img = np.expand_dims(img, axis=0)

    return img


print(ort.get_available_providers())

session = ort.InferenceSession(MODEL_PATH)

print(f"Using: {session.get_providers()}")

input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
input_dtype = session.get_inputs()[0].type
print(f"Input: {input_name}, Shape: {input_shape}, Dtype: {input_dtype}")

with open(OUTPUT_FILE, "w") as f_out:
    f_out.write("filepath,predicted_class,probability,inference_time_s\n")

    for root, dirs, files in os.walk(INPUT_DIR):
        counter = 0
        files_num = len(files)
        for filename in tqdm(files, desc="Images"):
            if counter >= 100:
                continue
            if not filename.lower().endswith(".jpg"):
                continue

            image_path = os.path.join(root, filename)
            try:
                print(f"Processing: {counter + 1}/{files_num if files_num <= 100 else 100} {image_path}")
                img = preprocess_eval_opencv(image_path, *INPUT_SIZE)

                start = time.perf_counter()
                outputs = session.run(None, {input_name: img})
                end = time.perf_counter()

                probabilities = scipy.special.softmax(outputs[0], axis=1)
                top1_idx = int(np.argmax(probabilities[0]))
                top1_prob = float(probabilities[0][top1_idx])
                inference_time = end - start 

                f_out.write(f"{image_path},{top1_idx},{top1_prob: .4f},{inference_time: .5f}\n")
                f_out.flush()
            except Exception as e:
                print(f"ERROR Skipping {image_path}: {e}")
            counter+=1
