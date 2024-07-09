# Import PyTorch and its modules
import torch
from torch.utils.data import Dataset
# Import Hugging Face's Transformers library and its modules
from transformers import AutoTokenizer
# Import other libraries
import random
import json
import os
import sys
from torch.nn import functional
# Import custom modules
try:
	from src.scripts.utils import RANDOM_SEED, get_image_from_db_object	 # type: ignore
	from tqdm import tqdm
except ModuleNotFoundError:
	from computer_vision_project_dev.src.scripts.utils import RANDOM_SEED, get_image_from_db_object # type: ignore
	from tqdm.notebook import tqdm

# Seed random number generators for reproducibility
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Constants for the special image IDs tokens of the Transformer models
IMG_ID_START_TOKEN = 10
IMG_ID_PADDING_TOKEN = 11
IMG_ID_END_TOKEN = 12

class TransformerIndexingDataset(Dataset):

	# Initialize the dataset of tuples (encoded_imgs, encoded_img_id) for the indexing phase (Transformer models learns to map images to image IDs)
	def __init__(
		self,
		images: list,
		img_patches: int = 10,
		patch_size: int = 16,
		img_id_max_length: int = -1,
		dataset_file_path: str = None,
		force_dataset_rebuild: bool = False
	):
		'''
		Constructor of the TransformerIndexingDataset class.

		Args:
		- images: list, a list containing the images data, the images database
		- img_patches: int, the number of patches per dimension for each image
		- patch_size: int, the size of the image patches
		- img_id_max_length: int, the maximum length of the image IDs sequence
		- dataset_file_path: str, the path of the JSON file in which the <image, image_id> pairs data will be saved or from which it will be loaded
		- force_dataset_rebuild: bool, a flag to force the rebuilding of the dataset (if false, a dataset file path is provided, and the file exists, the dataset will be loaded from the file)
		'''
		# Store the images dictionary
		self.images = images
		# Store the patch and image size
		self.img_patches = img_patches
		self.patch_size = patch_size
		# Store the dataset file path
		self.save_dataset_file_path = dataset_file_path
		# Define the image IDs special tokens
		self.img_id_start_token = IMG_ID_START_TOKEN
		self.img_id_end_token = IMG_ID_END_TOKEN
		self.img_id_padding_token = IMG_ID_PADDING_TOKEN
		# Set the maximum image ID length
		if img_id_max_length < 0:
			# Compute the maximum image ID length (and add 2: 1 for the start special token and 1 for the end special token)
			self.img_id_max_len = max(len(str(img_id)) for img_id in range(len(images))) + 2
		else:
			# Assign the provided image IDs max length
			self.img_id_max_len = img_id_max_length + 2	# Add 2: 1 for the start special token and 1 for the end special token
		# Initialize the encoded image and encoded image IDs lists
		self.encoded_imgs, self.encoded_img_ids = self.get_dataset(force_dataset_rebuild)

	def get_dataset(self, force_dataset_rebuild=False):
		''' Function to build or retrieve the dataset of <encoded_images, encoded_image_id> tuples for tthe indexing phase '''
		if not force_dataset_rebuild and self.save_dataset_file_path is not None and os.path.exists(self.save_dataset_file_path):
			print(f"Loading the Vision Transformer Indexing Dataset from {self.save_dataset_file_path}...")
			with open(self.save_dataset_file_path, 'r') as f:
				dataset = json.load(f)
			print(f"Loaded {len(dataset['encoded_images'])} images from {self.save_dataset_file_path}")
			encoded_images = [torch.tensor(img) for img in dataset['encoded_images']]
			encoded_image_ids = [torch.tensor(img_id) for img_id in dataset['encoded_image_ids']]
			return encoded_images, encoded_image_ids
		else:
			# Initialize the encoded images and image IDs lists
			encoded_image_ids = []
			encoded_images = []
			# For each image in the images list
			image_ids = range(len(self.images))
			for image_id in tqdm(image_ids, desc='Building TransformerIndexingDataset'):
				# Get the image object from the images dictionary
				image_obj = self.images[image_id]
				# Load the image from the image object
				image = get_image_from_db_object(image_obj) # Image is returned as a cv2 image object
				# Encode the image into a torch tensor of shape [C, H, W], where C is the number of channels (e.g. 3 for RGB), H is the height, and W is the width
				encoded_img = torch.tensor(image).permute(2, 0, 1)
				# Encode the image ID
				img_id_padding_length = self.img_id_max_len - len(str(image_id))	# Padding length: N - M  (with N max digit for each image ID, and M number of digits of the image ID)
				encoded_img_id = torch.tensor(
					# Start of sequence token
					[self.img_id_start_token] +
					# Encoded image ID (list of integers, each representing a digit of the M total digits of the ID)
					list(map(int, str(image_id))) +
					# End of sequence token
					[self.img_id_end_token] +
					# Padding tokens (if needed)
					[self.img_id_padding_token] * img_id_padding_length
				)
				# Add the encoded image and image ID to the lists
				encoded_images.append(encoded_img)
				encoded_image_ids.append(encoded_img_id)
			# Save the dataset to the file if a save file path is provided
			if self.save_dataset_file_path is not None:
				print(f"Saving the Vision Transformer Indexing Dataset to {self.save_dataset_file_path}...")
				with open(self.save_dataset_file_path, 'w') as f:
					json.dump({
						'encoded_images': [img.tolist() for img in encoded_images],
						'encoded_image_ids': [img_id.tolist() for img_id in encoded_image_ids]
					}, f)
			# Return the encoded images and image IDs
			return encoded_images, encoded_image_ids

	def __len__(self):
		return len(self.encoded_imgs)

	def __getitem__(self, idx):
		return self.encoded_imgs[idx], self.encoded_img_ids[idx]


class TransformerImageRetrievalDataset(Dataset):

	# Initialize the dataset of tuples (encoded_imgs, encoded_img_id) for the image retrieval phase (Transformer models learns to map images to relevant image IDs)
	def __init__(
		self,
		images: list,
		img_patches: int = 10,
		patch_size: int = 16,
		img_id_max_length: int = -1,
		dataset_file_path: str = None,
		force_dataset_rebuild: bool = False
	):
		'''
		Constructor of the TransformerImageRetrievalDataset class.

		Args:
		- images: list, a list containing the images data, the images database
		- img_patches: int, the number of patches per dimension for each image
		- patch_size: int, the size of the image patches
		- img_id_max_length: int, the maximum length of the image IDs sequence
		- dataset_file_path: str, the path of the JSON file in which the <image, image_id> pairs data will be saved or from which it will be loaded
		- force_dataset_rebuild: bool, a flag to force the rebuilding of the dataset (if false, a dataset file path is provided, and the file exists, the dataset will be loaded from the file)
		'''
		# Store the images dictionary
		self.images = images
		# Store the patch and image size
		self.img_patches = img_patches
		self.patch_size = patch_size
		# Store the dataset file path
		self.save_dataset_file_path = dataset_file_path
		# Define the image IDs special tokens
		self.img_id_start_token = IMG_ID_START_TOKEN
		self.img_id_end_token = IMG_ID_END_TOKEN
		self.img_id_padding_token = IMG_ID_PADDING_TOKEN
		# Set the maximum image ID length
		if img_id_max_length < 0:
			# Compute the maximum image ID length (and add 2: 1 for the start special token and 1 for the end special token)
			self.img_id_max_len = max(len(str(img_id)) for img_id in range(len(images))) + 2
		else:
			# Assign the provided image IDs max length
			self.img_id_max_len = img_id_max_length + 2	# Add 2: 1 for the start special token and 1 for the end special token
		# Initialize the encoded image and encoded image IDs lists
		self.encoded_imgs, self.encoded_img_ids = self.get_dataset(force_dataset_rebuild)

	def get_dataset(self, force_dataset_rebuild=False):
		''' Function to build or retrieve the dataset of <encoded_images, encoded_image_id> tuples for image retrieval '''
		if not force_dataset_rebuild and self.save_dataset_file_path is not None and os.path.exists(self.save_dataset_file_path):
			print(f"Loading the Vision Transformer Indexing Dataset from {self.save_dataset_file_path}...")
			with open(self.save_dataset_file_path, 'r') as f:
				dataset = json.load(f)
			print(f"Loaded {len(dataset['encoded_images'])} images from {self.save_dataset_file_path}")
			encoded_images = [torch.tensor(img) for img in dataset['encoded_images']]
			encoded_relevant_image_ids = [torch.tensor(img_id) for img_id in dataset['encoded_image_ids']]
			return encoded_images, encoded_relevant_image_ids
		else:
			# Initialize the encoded images and image IDs lists
			encoded_relevant_image_ids = []
			encoded_images = []
			# For each image in the images list
			image_ids = range(len(self.images))
			for image_id in tqdm(image_ids, desc='Building TransformerImageRetrievalDataset'):
				# Get the image object from the images dictionary
				image_obj = self.images[image_id]
				# Load the image from the image object
				image = get_image_from_db_object(image_obj) # Image is returned as a cv2 image object
				# Encode the image into a torch tensor of shape [C, H, W], where C is the number of channels (e.g. 3 for RGB), H is the height, and W is the width
				encoded_img = torch.tensor(image).permute(2, 0, 1)
				# Encode the image ID
				img_id_padding_length = self.img_id_max_len - len(str(image_id))	# Padding length: N - M  (with N max digit for each image ID, and M number of digits of the image ID)
				encoded_img_id = torch.tensor(
					# Start of sequence token
					[self.img_id_start_token] +
					# Encoded image ID (list of integers, each representing a digit of the M total digits of the ID)
					list(map(int, str(image_id))) +
					# End of sequence token
					[self.img_id_end_token] +
					# Padding tokens (if needed)
					[self.img_id_padding_token] * img_id_padding_length
				)
				# Add the encoded image and image ID to the lists
				encoded_images.append(encoded_img)
				encoded_relevant_image_ids.append(encoded_img_id)
			# Save the dataset to the file if a save file path is provided
			if self.save_dataset_file_path is not None:
				print(f"Saving the Vision Transformer Image Retrieval Dataset to {self.save_dataset_file_path}...")
				with open(self.save_dataset_file_path, 'w') as f:
					json.dump({
						'encoded_images': [img.tolist() for img in encoded_images],
						'encoded_image_ids': [img_id.tolist() for img_id in encoded_relevant_image_ids]
					}, f)
			# Return the encoded images and image IDs
			return encoded_images, encoded_relevant_image_ids

	def __len__(self):
		return len(self.encoded_imgs)

	def __getitem__(self, idx):
		return self.encoded_imgs[idx], self.encoded_img_ids[idx]


# class TransformerRetrievalDataset(Dataset):

# 	# Initialize the dataset of tuples (encoded_query, encoded_doc_id) for the retrieval phase (Transformer models learns to map queries to documents)
# 	def __init__(
# 		self,
# 		documents: dict,
# 		queries: dict,
# 		doc_id_max_length: int = -1,
# 		query_max_length: int = 16,
# 		dataset_file_path: str = None,
# 		force_dataset_rebuild: bool = False
# 	):
# 		'''
# 		Constructor of the TransformerRetrievalDataset class.

# 		Args:
# 		- documents: dict, a dictionary containing the documents data
# 		- queries: dict, a dictionary containing the queries data
# 		- doc_id_max_length: int, the maximum length of the doc IDs sequence
# 		- query_max_length: int, the maximum length of the input sequence
# 		- dataset_file_path: str, the path of the JSON file in which the <document, doc_id> pairs data will be saved or from which it will be loaded
# 		- force_dataset_rebuild: bool, a flag to force the rebuilding of the dataset (if false, a dataset file path is provided, and the file exists, the dataset will be loaded from the file)
# 		'''
# 		# Store the documents and queries dictionaries
# 		self.documents = documents
# 		self.queries = queries
# 		# Store the maximum query length
# 		self.query_max_length = query_max_length
# 		# Store the dataset file path
# 		self.save_dataset_file_path = dataset_file_path
# 		# Initialize tokenized query to query dictionary
# 		self.query_ids = dict()
# 		# We use a bert tokenizer to encode the documents
# 		tokenizer_model = "bert-base-uncased"
# 		self.tokenizer = AutoTokenizer.from_pretrained(
# 			tokenizer_model,
# 			use_fast=True
# 		)
# 		# Define the doc IDs special tokens
# 		self.doc_id_start_token = IMG_ID_START_TOKEN
# 		self.doc_id_end_token = IMG_ID_END_TOKEN
# 		self.doc_id_padding_token = IMG_ID_PADDING_TOKEN
# 		# Set the maximum doc ID length
# 		if doc_id_max_length < 0:
# 			# Compute the maximum doc ID length (and add 1 for the start token and 1 for the end token)
# 			self.doc_id_max_length = max(len(str(doc_id))
# 										for doc_id in documents.keys()) + 2
# 		else:
# 			# Assign the provided doc IDs max length
# 			self.doc_id_max_length = doc_id_max_length + 2
# 		# Initialize the encoded documents and encoded doc IDs lists
# 		self.encoded_queries, self.encoded_doc_ids = self.get_dataset(
# 			force_dataset_rebuild)

# 	def get_dataset(self, force_dataset_rebuild=False):
# 		''' Function to build or retrieve the dataset of <encoded_query, encoded_doc_id> tuples '''
# 		if not force_dataset_rebuild and self.save_dataset_file_path is not None and os.path.exists(self.save_dataset_file_path):
# 			print(
# 				f"Loading the Vision Transformer Retrieval Dataset from {self.save_dataset_file_path}...")
# 			with open(self.save_dataset_file_path, 'r') as f:
# 				dataset = json.load(f)
# 			print(
# 				f"Loaded {len(dataset['encoded_queries'])} encoded queries and document IDs from {self.save_dataset_file_path}")
# 			encoded_queries = [torch.tensor(query)
# 							for query in dataset['encoded_queries']]
# 			doc_ids = [torch.tensor(doc_id)
# 					for doc_id in dataset['encoded_doc_ids']]
# 			# Rebuild the query_ids dictionary
# 			query_ids_mapping = dataset['query_ids_mapping']
# 			for query, query_id in zip(dataset['encoded_queries'], query_ids_mapping):
# 				self.query_ids[str(query)] = query_id
# 			return encoded_queries, doc_ids
# 		else:
# 			# Initialize the encoded queries and doc IDs lists
# 			doc_ids = []
# 			encoded_queries = []
# 			# Store the query ids for reoading the query_ids dictionary
# 			query_ids_mapping = []
# 			# Iterate over the queries
# 			for query_id in tqdm(self.queries.keys(), desc='Building TransformerRetrievalDataset'):
# 				query = self.queries[query_id]
# 				# Tokenize and then encode the query text
# 				preprocessed_text = get_preprocessed_text(query['text'])
# 				encoded_query = self.tokenizer(preprocessed_text,
# 											add_special_tokens=True,
# 											max_length=self.query_max_length,
# 											truncation=True,
# 											return_tensors='pt'
# 											)['input_ids'][0].tolist()
# 				# Pad the query sequence to the max encoded queries length
# 				query_padding_length = self.query_max_length - \
# 					len(encoded_query)
# 				encoded_query = functional.pad(
# 					torch.tensor(encoded_query),
# 					(0, query_padding_length),
# 					value=0
# 				)
# 				# Add the tokenized query to query dictionary
# 				self.query_ids[str(encoded_query.tolist())] = query_id
# 				# For each document ID in the relevant document IDs list of the query
# 				for doc_id in query['relevant_docs']:
# 					# Encode the doc ID
# 					doc_id_padding_length = self.doc_id_max_length - \
# 						len(doc_id)
# 					encoded_doc_id = torch.tensor(
# 						# Start of sequence token
# 						[self.doc_id_start_token] +
# 						# Encoded document ID
# 						list(map(int, doc_id)) +
# 						# End of sequence token
# 						[self.doc_id_end_token] +
# 						# Padding tokens (if needed)
# 						[self.doc_id_padding_token] * doc_id_padding_length
# 					)
# 					# Add the encoded document and doc ID to the lists
# 					encoded_queries.append(encoded_query)
# 					doc_ids.append(encoded_doc_id)
# 					query_ids_mapping.append(query_id)
# 			# Save the dataset to the file if a save file path is provided
# 			if self.save_dataset_file_path is not None:
# 				print(
# 					f"Saving the Vision Transformer Retrieval Dataset to {self.save_dataset_file_path}...")
# 				with open(self.save_dataset_file_path, 'w') as f:
# 					json.dump({
# 						'encoded_queries': [query.tolist() for query in encoded_queries],
# 						'encoded_doc_ids': [doc_id.tolist() for doc_id in doc_ids],
# 						'query_ids_mapping': query_ids_mapping
# 					}, f)
# 			# Return the encoded queries and doc IDs
# 			return encoded_queries, doc_ids

# 	def __len__(self):
# 		return len(self.encoded_queries)

# 	def __getitem__(self, idx):
# 		return self.encoded_queries[idx], self.encoded_doc_ids[idx]

# 	def get_query_id(self, encoded_query):
# 		''' Get the query ID from the encoded query (given either as a tensor or a list of integers) '''
# 		if isinstance(encoded_query, torch.Tensor):
# 			# Convert the tensor to a list
# 			encoded_query = encoded_query.tolist()
# 		# Return the query ID from the query_ids dictionary (converting all the keys of the dictionary to lists)
# 		return self.query_ids[str(encoded_query)]

# 	def get_closest_doc_id(self, doc_id, exclude_doc_ids=[]):
# 		''' Get the closest document ID to the given doc ID '''
# 		# Get the closest valid doc ID
# 		other_doc_ids = list(self.documents.keys())
# 		if exclude_doc_ids and len(exclude_doc_ids) > 0:
# 			other_doc_ids = [
# 				doc_id for doc_id in other_doc_ids if doc_id not in exclude_doc_ids]
# 		if doc_id in other_doc_ids:
# 			return doc_id
# 		max_int_value = sys.maxsize
# 		min_int_value = -sys.maxsize - 1
# 		closest_doc_id = min(other_doc_ids, key=lambda other_doc_id: abs(
# 			int(other_doc_id if str.isdigit(other_doc_id) else max_int_value) -
# 			int(doc_id if str.isdigit(doc_id) else min_int_value)))
# 		decoded_doc_id = str(closest_doc_id)
# 		return decoded_doc_id

# 	def decode_doc_id(self, encoded_doc_id, force_debug_output=False, recover_malformed_doc_ids=True):
# 		''' 
# 		Decode the given encoded document ID into to a string 

# 		If the document ID is malformed, the output document ID will be prefixed with "M=" (for malformed) and its special tokens will be converted to letters.

# 		If the force_debug_output flag is set to True (and the doc id is not malformed), the output document ID will be prefixed with "D=" (for debug) and its special tokens will be converted to letters.

# 		Args:
# 		- encoded_doc_id: list or tensor, the encoded document ID (list of integers from 0 to 9 or special token integers)
# 		- use_debug_output: bool, wheter to return document IDs as a debug string (converting special tokens to letters) or as valid document IDs (string with the ID's digits)
# 		'''
# 		# Convert the given encoded doc id to a list if its a tensor
# 		if isinstance(encoded_doc_id, torch.Tensor):
# 			encoded_doc_id = encoded_doc_id.tolist()
# 		# Check if the given encoded doc id is malformed
# 		malformed_doc_id = \
# 			self.doc_id_end_token not in encoded_doc_id or \
# 			encoded_doc_id[0] == self.doc_id_end_token or \
# 			(encoded_doc_id[0] == self.doc_id_start_token
# 			and encoded_doc_id[1] == self.doc_id_end_token)
# 		# Convert the encoded doc id to a list of integers or special tokens mappings
# 		if not force_debug_output and not malformed_doc_id:
# 			# Remove the start token if it's the first character
# 			if encoded_doc_id[0] == self.doc_id_start_token:
# 				encoded_doc_id = encoded_doc_id[1:]
# 			# Keep only the characters before the first end token (if it exists)
# 			first_end_token_index = encoded_doc_id.index(
# 				self.doc_id_end_token)
# 			encoded_doc_id = encoded_doc_id[:first_end_token_index]
# 		else:
# 			# Map each special token to a letter
# 			special_tokens_mappings = {
# 				self.doc_id_start_token: 'S',
# 				self.doc_id_end_token: 'E',
# 				self.doc_id_padding_token: 'P'
# 			}
# 			doc_id_start = ""
# 			if force_debug_output:
# 				doc_id_start = "D="
# 			elif malformed_doc_id and not recover_malformed_doc_ids:
# 				doc_id_start = "M="
# 			converted_encoded_doc_id = [doc_id_start]
# 			for token in encoded_doc_id:
# 				if int(token) in special_tokens_mappings.keys():
# 					if malformed_doc_id and recover_malformed_doc_ids:
# 						# Skip the special tokens if the doc id is malformed and we want to recover it
# 						continue
# 					else:
# 						converted_encoded_doc_id.append(
# 							special_tokens_mappings[token])
# 				else:
# 					converted_encoded_doc_id.append(str(token))
# 			encoded_doc_id = converted_encoded_doc_id
# 		# Convert the remaining tokens to string and join them
# 		decoded_doc_id = "".join(
# 			[str(token) for token in encoded_doc_id])
# 		# Recover the malformed doc id if needed
# 		if not force_debug_output:
# 			non_existent_doc_id = decoded_doc_id not in self.documents.keys()
# 			if (malformed_doc_id or non_existent_doc_id) and recover_malformed_doc_ids:
# 				# Check if the final decoded doc id is valid
# 				max_doc_id_length = self.doc_id_max_length - 2
# 				if non_existent_doc_id:
# 					# Decoded doc id is not valid, try to recover it
# 					if len(decoded_doc_id) < max_doc_id_length:
# 						# Get the closest valid doc id
# 						decoded_doc_id = self.get_closest_doc_id(
# 							decoded_doc_id)
# 					else:
# 						# Truncate the decoded doc id to the actual maximum doc id length - 1
# 						decoded_doc_id = decoded_doc_id[:max_doc_id_length-1]
# 						decoded_doc_id = self.get_closest_doc_id(
# 							decoded_doc_id)
# 		# Return the decoded document ID
# 		return decoded_doc_id

# 	def ger_random_doc_ids(self, doc_ids_num, exclude_doc_ids: list = []):
# 		''' Get a list of random document IDs '''
# 		random.seed(RANDOM_SEED)
# 		doc_ids = list(self.documents.keys())
# 		doc_ids = [doc_id for doc_id in doc_ids if doc_id not in exclude_doc_ids]
# 		return random.sample(doc_ids, doc_ids_num)

# 	def get_similar_doc_ids(self, num_doc_ids=1, target_doc_ids: list = [], use_closest_doc_ids=False):
# 		''' Get a list of similar document IDs to the given doc IDs '''
# 		# Get the list of document IDs without the given target document IDs
# 		doc_ids = list(self.documents.keys())
# 		doc_ids = [doc_id for doc_id in doc_ids if doc_id not in target_doc_ids]
# 		# Add the given number of doc ids to the closest doc ids list, iterating over the first num_doc_ids doc ids in the list of doc ids to exclude
# 		similar_doc_ids = []
# 		if use_closest_doc_ids:
# 			for doc_id in target_doc_ids:
# 				doc_ids_to_exclude = target_doc_ids + similar_doc_ids
# 				similar_doc_ids.append(self.get_closest_doc_id(
# 					doc_id, exclude_doc_ids=doc_ids_to_exclude))
# 		else:
# 			for doc_id in target_doc_ids:
# 				# Increment one of the digits of the number by +-1 at random
# 				random.seed(RANDOM_SEED)
# 				doc_id = list(doc_id)
# 				digit_index = random.randint(1, len(doc_id) - 1)
# 				digit = doc_id[digit_index % len(doc_id)]
# 				# Increment the digit by 1 or -1
# 				increment = random.choice([-1, 1])
# 				new_digit = (int(digit) + increment) % 10
# 				doc_id[digit_index] = str(new_digit)
# 				similar_doc_ids.append("".join(doc_id))
# 		# Remove duplicates from the list of similar doc ids
# 		similar_doc_ids = list(set(similar_doc_ids))
# 		# Check if the number of closest doc ids is less than the required number of doc ids
# 		if len(similar_doc_ids) < num_doc_ids:
# 			# Get random doc ids to complete the list
# 			doc_ids_to_exclude = target_doc_ids + similar_doc_ids
# 			random_doc_ids = self.ger_random_doc_ids(
# 				num_doc_ids - len(similar_doc_ids),
# 				exclude_doc_ids=doc_ids_to_exclude)
# 			# Add the random doc ids to the closest doc ids list
# 			similar_doc_ids.extend(random_doc_ids)
# 			similar_doc_ids = similar_doc_ids[:num_doc_ids]
# 		else:
# 			# Keep only the required number of doc ids
# 			similar_doc_ids = similar_doc_ids[:num_doc_ids]
# 		# Return the list of closest doc ids
# 		return similar_doc_ids
