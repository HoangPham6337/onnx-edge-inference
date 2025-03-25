import csv
import json


def get_species_label(species_dict, species_path):
    try:
        return list(species_dict.keys())[list(species_dict.values()).index(species_path.split(sep="/")[9])]
    except ValueError:
        return list(species_dict.keys())[list(species_dict.values()).index("Other")]

path = "./inference_results.csv"

counter = 0
all_data = []
with open(path, "r") as file:
    reader = csv.reader(file)

    for row in reader:
        all_data.append(row)

with open("./dataset_species_labels.json", "r") as json_file:
    species_labels = json.load(json_file)

all_data = all_data[1:]

true_labels = [get_species_label(species_labels, species_path[0]) for species_path in all_data]
pred_labels = [result[1] for result in all_data]

correct_counter = 0
for true_label, pred_label in zip(true_labels, pred_labels):
    if true_label == pred_label:
        correct_counter += 1

print(f"Accuracy: {correct_counter / len(true_labels) * 100}")
