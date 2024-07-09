import json
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import base64
import cv2
from skimage import io

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
def get_image_from_b64_string(b64_string):
	'''
	Returns the image as a cv2 image object from the given image object in the "images_db" list

	Parameters:
		image_obj (dict): The image object in the "images_db" list
		image_max_size (int): The maximum size of the image used for when (i.e. the max width and the max height of the image, we use square images in this case)
	'''
	# Convert the base64 string to an image
	image_data = base64.b64decode(b64_string)
	image_np = np.frombuffer(image_data, np.uint8)
	image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
	return image

# Function that retrieves the actual image data from the image URL, crpps it to a square aspect ratio if necessary, and returns the image data as a base64 string
def get_image_data_as_base64(image_url, image_max_size):
	'''
	Retrieves the actual image data from the image URL, crops it to a square aspect ratio if necessary, and returns the image data as a base64 string

	Parameters:
		image_url (str): The URL of the image
		image_max_size (int): The maximum size of the image (i.e. the max width and the max height of the image, we use square images in this case)
	'''
	# Load the image
	image = io.imread(image_url)
	# Crop the image to a square aspect ratio if it is not already square
	downscaled_image = image
	if image.shape[1] > image.shape[0]:
		# Image is wider than tall, crop the sides
		crop_width = (image.shape[1] - image.shape[0]) // 2
		downscaled_image = image[:, crop_width:crop_width+image.shape[0]]
	elif image.shape[0] > image.shape[1]:
		# Image is taller than wide, crop the top and bottom
		crop_height = (image.shape[0] - image.shape[1]) // 2
		downscaled_image = image[crop_height:crop_height+image.shape[1], :]
	# Downscale the image to the maximum allowed size
	downscaled_image = cv2.resize(downscaled_image, (image_max_size, image_max_size))
	# Convert the image to a base64 string
	image_base64 = base64.b64encode(cv2.imencode('.jpg', downscaled_image)[1]).decode()
	# Return the base64 string of the image
	return image_base64
