import pandas as pd
import numpy as np
import json
import math


INFER_RESULT_PATH = "./inference_results.csv"

df = pd.read_csv(INFER_RESULT_PATH)
other_class_id = 16

df["predicted_species"] = df["predicted_species"].astype(int)
df["correct_species"] = df["correct_species"].astype(int)

df["is_communication"] = df["predicted_species"] == other_class_id
df["is_local"] = ~df["is_communication"]

num_local = df["is_local"].sum()
num_comm = df["is_communication"].sum()
num_total = len(df)

# accuracy = np.mean(df["correct_species"] == df["predicted_species"])

# df["is_other"] = df["predicted_species"] == other_class_id
# num_other_outputs = df["is_other"].sum()
# num_hits = (df["correct_species"] == df["predicted_species"]).sum()
# num_misses = num_total - num_hits
# hit_to_miss_ratio = num_hits / max(num_misses, 1)

# inference_times = df["inference_time_s"].astype(float)
# std_inference_time = np.std(inference_times)
# avg_inference_time = np.mean(inference_times)


print(f"Total inference: {num_total}")

print(f"Local (non-Other): {num_local} ({(num_local / num_total) * 100:.2f}%)")
print(f"Communication (Other): {num_comm} ({(num_comm / num_total) * 100:.2f}%)")
# print(f"Accuracy: {accuracy * 100:.2f}%")
# print(f"'Other' predictions: {num_other_outputs}/{num_total} ({(num_other_outputs / num_total) * 100:.2f}%)")
# print(f"Hit/Miss ratio: ({num_hits}/{max(num_misses, 1)}) {hit_to_miss_ratio:.2f}%")
# print(f"Average inference time: {avg_inference_time:.2f}s ~ {math.floor(1 / avg_inference_time)} inference/second")
# print(f"Standard deviation of inference time: {std_inference_time:.3f}")