import json
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import base64
import cv2

# Random seed for reproducibility
RANDOM_SEED = 1

# Path of the model checkpoint file
MODEL_CHECKPOINT_FILE = "transformer.ckpt"

def print_json(json_obj, indent_all_by_tabs=0, compact=True, truncate_large_lists=-1, truncate_large_dicts=-1):
	''' Print the given JSON object in a readable format '''
	# Truncate the fields and values of large lists and dictionaries
	def truncate_large_list(list_obj, max_elements):
		if max_elements < 0:
			return list_obj
		if len(list_obj) > max_elements:
			return list_obj[:max_elements] + ["..."] + ["(Truncated to " + str(max_elements) + " out of " + str(len(list_obj)) + " elements)"]
		return list_obj
	def truncate_large_dict(dict_obj, max_elements):
		if max_elements < 0:
			return dict_obj
		if len(dict_obj) > max_elements:
			return {key: dict_obj[key] for key in list(dict_obj.keys())[:max_elements]} | {"...": "(Truncated to " + str(max_elements) + " out of " + str(len(dict_obj)) + " elements)"}
		return dict_obj
	# Truncate the fields and values of large lists and dictionaries of the JSON object recursively
	def truncate_large_lists_and_dicts(json_obj, max_list_elements, max_dict_elements):
		if isinstance(json_obj, list):
			return truncate_large_list([truncate_large_lists_and_dicts(item, max_list_elements, max_dict_elements) for item in json_obj], max_list_elements)
		elif isinstance(json_obj, dict):
			return truncate_large_dict({key: truncate_large_lists_and_dicts(json_obj[key], max_list_elements, max_dict_elements) for key in json_obj.keys()}, max_dict_elements)
		return json_obj
	# Truncate the fields and values of large lists and dictionaries of the JSON object
	json_obj = truncate_large_lists_and_dicts(json_obj, truncate_large_lists, truncate_large_dicts)
	# Convert the JSON object to a string
	json_lines = json.dumps(json_obj, indent=2).split("\n")
	if compact:
		# Remove first and last lines if they contain only brackets
		if json_lines[0].strip() == "{":
			json_lines = json_lines[1:]
		if json_lines[-1].strip() == "}":
			json_lines = json_lines[:-1]
	for line in json_lines:
		print(" " * indent_all_by_tabs + line)

def print_model_evaluation_results(map_k_evaluation_results=None, recall_k_evaluation_results=None):
	''' Print the MAP@K and Recall@K evaluation results for the model '''
	# Print the MAP@K evaluation results
	if map_k_evaluation_results is not None:
		additional_info = ""
		if "info" in map_k_evaluation_results and map_k_evaluation_results["info"] is not None and len(map_k_evaluation_results["info"].keys()) > 0:
			additional_info = " (" + ", ".join([f"{key}: {value}" for key, value in map_k_evaluation_results["info"].items()]) + ")"
		print(f"MAP@{map_k_evaluation_results['k_documents']} for the {map_k_evaluation_results['model']} model{additional_info}:")
		print(f"  > {map_k_evaluation_results['mean_average_precision']}")
		print(f"  Computed on {map_k_evaluation_results['n_queries']} queries")
		print(f"  Single queries precision:")
		for query_id in map_k_evaluation_results['evaluated_queries'].keys():
			print(f"    Query {query_id}: {map_k_evaluation_results['evaluated_queries'][query_id]}")
	else:
		print(f"No MAP@K evaluation results for the model")
	# Print the Recall@K evaluation results
	if recall_k_evaluation_results is not None:
		additional_info = ""
		if "info" in map_k_evaluation_results and recall_k_evaluation_results["info"] is not None and len(recall_k_evaluation_results["info"].keys()) > 0:
			additional_info = " (" + ", ".join(
				[f"{key}: {value}" for key, value in recall_k_evaluation_results["info"].items()]) + ")"
		print(f"Recall@{recall_k_evaluation_results['k_documents']} results for the {recall_k_evaluation_results['model']} model{additional_info}:")
		for i in range(len(recall_k_evaluation_results['recall_at_k_results'])):
			print(f"  > {recall_k_evaluation_results['recall_at_k_results'][i]}")
			print(f"    Computed for query {recall_k_evaluation_results['query_ids'][i]}")
	else:
		print(f"No Recall@K evaluation results for the model")

# Function to get the actual image (viewable using "plt.imgshow(image)") given the image object in the "images_db" list
def get_image_from_db_object(image_obj):
	'''
	Returns the image as a cv2 image object from the given image object in the "images_db" list
	'''
	# Convert the base64 string to an image
	image_data = base64.b64decode(image_obj["image_data"])
	image_np = np.frombuffer(image_data, np.uint8)
	image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
	return image