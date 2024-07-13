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
	from src.scripts.utils import RANDOM_SEED, get_image_from_b64_string	 # type: ignore
	from tqdm import tqdm
except ModuleNotFoundError:
	from computer_vision_project_dev.src.scripts.utils import RANDOM_SEED, get_image_from_b64_string # type: ignore
	from tqdm.notebook import tqdm

# Seed random number generators for reproducibility
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Constants for the special image IDs tokens of the Transformer models
IMG_ID_START_TOKEN = 10
IMG_ID_PADDING_TOKEN = 11
IMG_ID_END_TOKEN = 12

# Whether to pad image ids with zeroes at the start (thus consider ids like "1" as "00...01") or at the end (after the start and end tokens, thus without padding IDs "internallly")
PAD_IDS_AT_START = False

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
		self.img_id_max_len = -1
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
				image = get_image_from_b64_string(image_obj["image_data"]) # Image is returned as a cv2 image object
				# Encode the image into a torch tensor of shape [C, H, W], where C is the number of channels (e.g. 3 for RGB), H is the height, and W is the width
				encoded_img = torch.tensor(image).permute(2, 0, 1)
				# Get the final string representing the image ID (padded at start or not)
				image_id_string = str(image_id).zfill(self.img_id_max_len - 2) if PAD_IDS_AT_START else str(image_id)
				# Encode the image ID
				img_id_padding_length = self.img_id_max_len - len(image_id_string) - 2	# Padding length: N - M  (with N max digit for each image ID, and M number of digits of the image ID)
				encoded_img_id = torch.tensor(
					# Start of sequence token
					[self.img_id_start_token] +
					# Encoded image ID (list of integers, each representing a digit of the M total digits of the ID)
					list(map(int, image_id_string)) +
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
				print(f"Saved {len(encoded_images)} images to {self.save_dataset_file_path}")
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
		all_images: list,
		similar_images: dict,
		img_patches: int = 10,
		patch_size: int = 16,
		img_id_max_length: int = -1,
		dataset_file_path: str = None,
		force_dataset_rebuild: bool = False
	):
		'''
		Constructor of the TransformerImageRetrievalDataset class.

		Args:
		- all_images: list, a list containing the images data, the images database plus the images not in the database (for the retrieval phase)
		- similar_images: dict, a dictionary containing image IDs (not in the database) as keys and a list of similar image IDs in the database, i.e. in the "images" list, as values
		- img_patches: int, the number of patches per dimension for each image
		- patch_size: int, the size of the image patches
		- img_id_max_length: int, the maximum length of the image IDs sequence
		- dataset_file_path: str, the path of the JSON file in which the <image, image_id> pairs data will be saved or from which it will be loaded
		- force_dataset_rebuild: bool, a flag to force the rebuilding of the dataset (if false, a dataset file path is provided, and the file exists, the dataset will be loaded from the file)
		'''
		# Store the images dictionary
		self.images = all_images
		# Store the similar images dictionary
		self.similar_images = similar_images
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
			self.img_id_max_len = max(len(str(img_id)) for img_id in range(len(all_images))) + 2
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
			# For each image in the images similarity dictionary
			similar_image_ids = self.similar_images.keys()
			for similar_image_id in tqdm(similar_image_ids, desc='Building TransformerImageRetrievalDataset'):
				# get all the similar images for the current image
				relevant_image_ids = self.similar_images[similar_image_id]	# List of relevant image IDs (in the database)
				# For each relevant image ID (in the database)
				for image_id in relevant_image_ids:
					# Get the image object from the images dictionary
					image_obj = self.images[int(similar_image_id)]
					# Load the similar image from the image object
					similar_image = get_image_from_b64_string(image_obj["image_data"]) # Image is returned as a cv2 image object
					# Encode the image into a torch tensor of shape [C, H, W], where C is the number of channels (e.g. 3 for RGB), H is the height, and W is the width
					encoded_img = torch.tensor(similar_image).permute(2, 0, 1)
					# Get the final string representing the image ID (padded at start or not)
					image_id_string = str(image_id).zfill(self.img_id_max_len - 2) if PAD_IDS_AT_START else str(image_id)
					# Encode the image ID of the relevant image (in the database)
					img_id_padding_length = self.img_id_max_len - len(image_id_string) - 2	# Padding length: N - M  (with N max digit for each image ID, and M number of digits of the image ID)
					encoded_img_id = torch.tensor(
						# Start of sequence token
						[self.img_id_start_token] +
						# Encoded image ID (list of integers, each representing a digit of the M total digits of the ID)
						list(map(int, image_id_string)) +
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
				print(f"Saved {len(encoded_images)} images to {self.save_dataset_file_path}")
			# Return the encoded images and image IDs
			return encoded_images, encoded_relevant_image_ids

	def __len__(self):
		return len(self.encoded_imgs)

	def __getitem__(self, idx):
		return self.encoded_imgs[idx], self.encoded_img_ids[idx]

	def get_closest_image_id(self, image_id, exclude_image_ids=[]):
		''' Get the closest image ID to the given image ID '''
		# Get the closest valid image ID
		other_image_ids = list(range(len(self.images)))
		if exclude_image_ids and len(exclude_image_ids) > 0:
			other_image_ids = [image_id for image_id in other_image_ids if image_id not in exclude_image_ids]
		if image_id in other_image_ids:
			return image_id
		max_int_value = sys.maxsize
		min_int_value = -sys.maxsize - 1
		closest_image_id = min(
			other_image_ids, 
	   		key=lambda other_image_id: 
				abs(int(other_image_id if str.isdigit(str(other_image_id)) else max_int_value) - int(image_id if str.isdigit(image_id) else min_int_value))
		)
		decoded_image_id = str(closest_image_id)
		return decoded_image_id

	def decode_image_id(self, encoded_img_id, force_debug_output=False, recover_malformed_img_ids=True):
		''' 
		Decode the given encoded image ID into to a string 

		If the image ID is malformed, the output image ID will be prefixed with "M=" (for malformed) and its special tokens will be converted to letters.

		If the force_debug_output flag is set to True (and the image id is not malformed), the output image ID will be prefixed with "D=" (for debug) and its special tokens will be converted to letters.

		Args:
		- encoded_img_id: list or tensor, the encoded image ID (list of integers from 0 to 9 or special token integers)
		- use_debug_output: bool, wheter to return image IDs as a debug string (converting special tokens to letters) or as valid image IDs (string with the ID's digits)
		'''
		# Convert the given encoded image id to a list if its a tensor
		if isinstance(encoded_img_id, torch.Tensor):
			encoded_img_id = encoded_img_id.tolist()
		# Check if the given encoded image id is malformed
		malformed_image_id = \
			self.img_id_end_token not in encoded_img_id or \
			encoded_img_id[0] == self.img_id_end_token or \
			(encoded_img_id[0] == self.img_id_start_token
			and encoded_img_id[1] == self.img_id_end_token)
		# Convert the encoded image id to a list of integers or special tokens mappings
		if not force_debug_output and not malformed_image_id:
			# Remove the start token if it's the first character
			if encoded_img_id[0] == self.img_id_start_token:
				encoded_img_id = encoded_img_id[1:]
			# Keep only the characters before the first end token (if it exists)
			first_end_token_index = encoded_img_id.index(self.img_id_end_token)
			encoded_img_id = encoded_img_id[:first_end_token_index]
		else:
			# Map each special token to a letter
			special_tokens_mappings = {
				self.img_id_start_token: 'S',
				self.img_id_end_token: 'E',
				self.img_id_padding_token: 'P'
			}
			image_id_start = ""
			if force_debug_output:
				image_id_start = "D="
			elif malformed_image_id and not recover_malformed_img_ids:
				image_id_start = "M="
			converted_encoded_image_id = [image_id_start]
			for token in encoded_img_id:
				if int(token) in special_tokens_mappings.keys():
					if malformed_image_id and recover_malformed_img_ids:
						# Skip the special tokens if the image id is malformed and we want to recover it
						continue
					else:
						converted_encoded_image_id.append(special_tokens_mappings[token])
				else:
					converted_encoded_image_id.append(str(token))
			encoded_img_id = converted_encoded_image_id
		# Convert the remaining tokens to string and join them
		decoded_image_id = "".join([str(token) for token in encoded_img_id])
		# Recover the malformed image id if needed
		if not force_debug_output:
			non_existent_image_id = decoded_image_id not in range(len(self.images))
			if (malformed_image_id or non_existent_image_id) and recover_malformed_img_ids:
				# Check if the final decoded image id is valid
				max_image_id_length = self.img_id_max_len - 2
				if non_existent_image_id:
					# Decoded image id is not valid, try to recover it
					if len(decoded_image_id) < max_image_id_length:
						# Get the closest valid image id
						decoded_image_id = self.get_closest_image_id(decoded_image_id)
					else:
						# Truncate the decoded image id to the actual maximum image id length - 1
						decoded_image_id = decoded_image_id[:max_image_id_length-1]
						decoded_image_id = self.get_closest_image_id(decoded_image_id)
		# Return the decoded image ID
		return decoded_image_id

	def ger_random_image_ids(self, image_ids_num, exclude_image_ids: list = []):
		''' Get a list of random image IDs '''
		random.seed(RANDOM_SEED)
		image_ids = list(range(len(self.images)))
		image_ids = [image_id for image_id in image_ids if image_id not in exclude_image_ids]
		return random.sample(image_ids, image_ids_num)

	def get_similar_image_ids(self, num_image_ids=1, target_image_ids: list = [], use_closest_image_ids=False):
		''' Get a list of similar image IDs to the given image IDs '''
		# Get the list of image IDs without the given target image IDs
		image_ids = list(range(len(self.images)))
		image_ids = [image_id for image_id in image_ids if image_id not in target_image_ids]
		# Add the given number of image ids to the closest image ids list, iterating over the first num_image_ids image ids in the list of image ids to exclude
		similar_image_ids = []
		if use_closest_image_ids:
			for image_id in target_image_ids:
				image_ids_to_exclude = target_image_ids + similar_image_ids
				similar_image_ids.append(self.get_closest_image_id(image_id, exclude_image_ids=image_ids_to_exclude))
		else:
			for image_id in target_image_ids:
				# Increment one of the digits of the number by +-1 at random
				random.seed(RANDOM_SEED)
				image_id = list(image_id)
				# digit_index = random.randint(1, len(image_id) - 1)
				digit_index = random.randint(0, len(image_id) - 1)
				digit = image_id[digit_index % len(image_id)]
				# Increment the digit by 1 or -1
				increment = random.choice([-1, 1])
				new_digit = (int(digit) + increment) % 10
				image_id[digit_index] = str(new_digit)
				similar_image_ids.append("".join(image_id))
		# Remove duplicates from the list of similar image ids
		similar_image_ids = list(set(similar_image_ids))
		# Check if the number of closest image ids is less than the required number of image ids
		if len(similar_image_ids) < num_image_ids:
			# Get random image ids to complete the list
			image_ids_to_exclude = target_image_ids + similar_image_ids
			random_image_ids = self.ger_random_image_ids(
				num_image_ids - len(similar_image_ids),
				exclude_image_ids=image_ids_to_exclude)
			# Add the random image ids to the closest image ids list
			similar_image_ids.extend(random_image_ids)
			similar_image_ids = similar_image_ids[:num_image_ids]
		else:
			# Keep only the required number of image ids
			similar_image_ids = similar_image_ids[:num_image_ids]
		# Return the list of closest image ids
		return similar_image_ids
