import os
import time
import psutil
import threading
import json
import random

import cv2
import numpy as np
import onnxruntime as ort
import scipy.special
from tqdm import tqdm
from typing import Dict
from utility import measure_inference_memory, preprocess_eval_opencv

from dataset_builder.core.utility import load_manifest_parquet


class MonteCarloBenchmark:
    def __init__(self, model_path, data_manifest_path, dataset_species_labels, input_size=(224, 224)):
        self.model_path = model_path
        self.input_size = input_size
        self.species_labels: Dict[int, str] = dataset_species_labels
        self.other_class_id = int(self._get_other_id())
        self.data_manifest = load_manifest_parquet(data_manifest_path)
        self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
    
    
    def _get_other_id(self):
        species_labels_flip: Dict[str, int] = dict((v, k) for k, v in self.species_labels.items())
        return species_labels_flip["Other"]


    def infer_one(self, image_path: str):
        try:
            img = preprocess_eval_opencv(image_path, *self.input_size)
            outputs = self.session.run(None, {self.input_name: img})
            probabilities = scipy.special.softmax(outputs[0], axis=1)
            top1_idx = int(np.argmax(probabilities[0]))
            top1_prob = float(probabilities[0][top1_idx])
            return top1_idx, top1_prob
        except Exception as e:
            print(e)
            return None


    def run_simulation(self, num_runs=30, sample_size=1000, save_path=None):
        comm_rates = []
        results = []

        for run in range(num_runs):
            sample = random.sample(self.data_manifest, sample_size)
            num_comm = 0
            num_local = 0

            for image_path, correct_species in tqdm(sample, desc=f"Run {run + 1}/{num_runs}", leave=False):
                result = self.infer_one(image_path)
                if result is None:
                    continue
                top1_idx, top1_prop = result
                if top1_idx == self.other_class_id:
                    num_comm += 1
                else:
                    num_local += 1
            
            total_pred = num_comm + num_local
            comm_rate = num_comm / total_pred if total_pred else 0
            comm_rates.append(comm_rate)
            results.append((run + 1, comm_rate))
            print(f"[Run {run+1}] Comm Rate: {comm_rate:.2%}")
        print(f"Avg: {sum(comm_rates)/len(comm_rates)}")


if __name__ == "__main__":
    with open("./data/haute_garonne/dataset_species_labels.json", "r") as json_file:
        species_labels = json.load(json_file)
        benchmark = MonteCarloBenchmark(
            model_path="./models/mobilenet_v3_large_80_prune_35.onnx",
            data_manifest_path="./data/haute_garonne/dataset_manifest.parquet",
            dataset_species_labels=species_labels
        )
        benchmark.run_simulation(num_runs=30, sample_size=1000)