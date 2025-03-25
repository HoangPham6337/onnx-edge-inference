import pandas as pd
import numpy as np
import json
import math

def extract_label_id(path: str):
    try:
        species_name = path.split(sep="/")[9]
        return class_name_to_id.get(species_name, other_class_id)
    except ValueError:
        return other_class_id


INFER_RESULT_PATH = "./inference_results.csv"
SPECIES_LABELS_PATH = "./dataset_species_labels.json"

df = pd.read_csv(INFER_RESULT_PATH)

with open(SPECIES_LABELS_PATH, "r") as csv_file:
    species_labels = json.load(csv_file)

class_name_to_id = {v: k for k, v in species_labels.items()}
other_class_id = class_name_to_id["Other"]

df["true_label"] = df["filepath"].apply(extract_label_id)
df["predicted_class"] = df["predicted_class"].astype(str)

accuracy = np.mean(df["true_label"] == df["predicted_class"])

df["is_other"] = df["predicted_class"] == other_class_id
num_other_outputs = df["is_other"].sum()
num_total = len(df)
num_hits = (df["true_label"] == df["predicted_class"]).sum()
num_misses = num_total - num_hits
hit_to_miss_ratio = num_hits / max(num_misses, 1)

inference_times = df["inference_time_s"].astype(float)
std_inference_time = np.std(inference_times)
avg_inference_time = np.mean(inference_times)

print(f"Total inference: {num_total}")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"'Other' predictions: {num_other_outputs}/{num_total} ({(num_other_outputs / num_total) * 100:.2f}%)")
print(f"Hit/Miss ratio: ({num_hits}/{max(num_misses, 1)}) {hit_to_miss_ratio:.2f}%")
print(f"Average inference time: {avg_inference_time:.2f}s ~ {math.floor(1 / avg_inference_time)} inference/second")
print(f"Standard deviation of inference time: {std_inference_time:.3f}")