'''
NOTE: Code below (apart from library imports) comes from the tutorial at:
> https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/11-vision-transformer.html
'''


# NOTE: import libraries added by me  ==========================================================================================================

# Import PyTorch and its modules
import torch
import torch.nn as nn
import torch.optim as optim
# from torch.nn import Transformer
from torch.nn import functional
# Import PyTorch Lightning
import pytorch_lightning as pl
# Import other modules
import random
import math
import numpy as np
# Import custom modules
try:
	from src.scripts import datasets	 # type: ignore
	from src.scripts.utils import RANDOM_SEED	 # type: ignore
except ModuleNotFoundError:
	from computer_vision_project_dev.src.scripts import datasets	 # type: ignore
	from computer_vision_project_dev.src.scripts.utils import RANDOM_SEED 	# type: ignore

# Seed random number generators for reproducibility
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

PRINT_DEBUG = False

# Override the "print" function to print only when necessary
def print(*args, **kwargs):
	if PRINT_DEBUG:
		__builtins__.print(*args, **kwargs)

# LighningModule Code from the tutorial ==========================================================================================================

class DSI_VisionTransformer(pl.LightningModule):

	def __init__(self, **model_kwargs):
		super().__init__()
		self.save_hyperparameters(model_kwargs)
		self.model = DSI_ViT(**model_kwargs)
		# Store the outputs for training and validation steps
		self.training_losses = []
		self.validation_losses = []
		self.training_accuracies = []
		self.validation_accuracies = []

	def forward(self, imgs, ids):
		# Expects as input a tensor of shape [B, C, H, W] where:
		# - B = batch size (number of images in the batch)
		# - C = number of channels (e.g. 3 channels for RGB)
		# - H = height of the image
		# - W = width of the image
		return self.model(imgs, ids)

	def configure_optimizers(self):
		optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
		lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
		return [optimizer], [lr_scheduler]

	# def _calculate_loss(self, batch, mode="train"):
	# 	imgs, labels = batch
	# 	preds = self.model(imgs)
	# 	loss = functional.cross_entropy(preds, labels)
	# 	acc = (preds.argmax(dim=-1) == labels).float().mean()

	# 	self.log("%s_loss" % mode, loss)
	# 	self.log("%s_acc" % mode, acc)

	# 	return loss

	# def training_step(self, batch, batch_idx):
	# 	loss = self._calculate_loss(batch, mode="train")
	# 	return loss

	# def validation_step(self, batch, batch_idx):
	# 	self._calculate_loss(batch, mode="val")

	# def test_step(self, batch, batch_idx):
	# 	self._calculate_loss(batch, mode="test")

	def training_step(self, batch, batch_idx):
		# print("batch:", batch)
		# Training step for the model (compute the loss and accuracy)
		loss, accuracy = self.model.step(batch)
		# Append the loss to the training losses list (for logging)
		self.training_accuracies.append(accuracy)
		# Append the accuracy to the training accuracies list (for logging)
		self.training_losses.append(loss)
		# Return the loss
		return loss

	def validation_step(self, batch, batch_idx):
		# print("batch:", batch)
		# Validation step for the model (compute the loss and accuracy)
		loss, accuracy = self.model.step(batch, True)
		# Append the loss to the validation losses list (for logging)
		self.validation_losses.append(loss)
		# Append the accuracy to the validation accuracies list (for logging)
		self.validation_accuracies.append(accuracy)
		# Return the loss
		return loss


# nn.Module Code from the tutorial ==========================================================================================================

class DSI_ViT(nn.Module):

	def __init__(
		self,
		# Main parameters
		embed_dim,
		hidden_dim,
		num_channels,
		num_heads,
		num_layers,
		batch_size,
		num_classes,
		patch_size,
		num_patches,
		# Other parameters
		img_id_max_length,
		img_id_start_token,
		img_id_end_token,
		img_id_padding_token,
		# Training parameters
		dropout=0.0,
		learning_rate=1e-4,
	):
		"""Vision Transformer.

		Args:
			embed_dim: Dimensionality of the input feature vectors to the Transformer (i.e. the size of the embeddings)
			hidden_dim: Dimensionality of the hidden layer in the feed-forward networks within the Transformer
			num_channels: Number of channels of the input (e.g. 3 for RGB, 1 for grayscale, ecc...)
			num_heads: Number of heads to use in the Multi-Head Attention block
			num_layers: Number of layers to use in the Transformer
			batch_size: Number of samples in a batch
			num_classes: Number of classes to predict
				(in my case, since I give an image with, concatenated, the N digits of the image ID, the num_classes is the number of possible digits of the image IDs, hence 10+3, including the special tokens)
			patch_size: Number of pixels that the patches have per dimension
			num_patches: Maximum number of patches an image can have
			dropout: Amount of dropout to apply in the feed-forward network and on the input encoding
		"""
		super().__init__()

		self.patch_size = patch_size

		self.num_classes = num_classes

		self.embed_dim = embed_dim
		self.img_id_max_length = img_id_max_length
		self.img_id_start_token = img_id_start_token
		self.img_id_end_token = img_id_end_token
		self.img_id_padding_token = img_id_padding_token

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		# Layers/Networks
		self.input_layer = nn.Linear(num_channels * (patch_size**2), embed_dim)	# Convert the input image's patches into embeddings, i.e. vectors (one for each patch) of size "embed_dim"
		# self.id_embedding = nn.Embedding(	# Embedding layer for the image ID digits (the 10 digits [0-9] plus the 3 special tokens, i.e. end of sequence, padding, start of sequence)
		# 	num_classes, 	# 10+3 possible digits (10 digits + 3 special tokens)
		# 	embed_dim,
		# 	padding_idx=img_id_padding_token	# The padding index is the index of the digit that represents the padding (i.e. the digit that is used to pad the image ID to the maximum length)
		# )
		self.id_embedding = nn.Embedding(	# Embedding layer for the image ID digits (the 10 digits [0-9] plus the 3 special tokens, i.e. end of sequence, padding, start of sequence)
			num_classes, # The maximum number of digits in the image ID
			embed_dim,
			padding_idx=img_id_padding_token	# The padding index is the index of the digit that represents the padding (i.e. the digit that is used to pad the image ID to the maximum length)
		)
		self.transformer = nn.Sequential(
			# Add the specified number of Attention Blocks to the Transformer ("num_layers" times)
			*(AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers))
		)
		self.mlp_head = nn.Sequential(
			nn.LayerNorm(embed_dim),
			nn.Linear(embed_dim, num_classes)
		)
		self.dropout = nn.Dropout(dropout)
		# self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=self.img_id_padding_token)

		# Parameters/Embeddings
		# self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
		self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+img_id_max_length+2, embed_dim))	# Positional encoding for the image ID embeddings

	def img_to_patch(self, x : torch.Tensor, patch_size, flatten_channels=True):
		"""
		Args:
			x: Tensor representing the image of shape [B, C, H, W]
			patch_size: Number of pixels per dimension of the patches (integer)
			flatten_channels: If True, the patches will be returned in a flattened format as a feature vector instead of a image grid.

		Returns:
			x: The input image tensor reshaped into a tensor (list) of P patches, where each patch is a vector of size C*patch_size*patch_size

		"""
		B, C, H, W = x.shape	# B is the batch size (number of images in the batch), C is the number of channels (e.g. 3 channels for RGB), H is the height of the image, and W is the width of the image
		P = patch_size 			# Width and height of the patches
		H_P = H // P			# Number of patches vertically
		W_P = W // P			# Number of patches horizontally
		x = x.reshape(B, C, H_P, P, W_P, P) # [B, C, H, W] -> [B, C, H_P, P, W_P, P]	-> Reshape the image into patches
		x = x.permute(0, 2, 4, 1, 3, 5)		# [B, H_P, W_P, C, P, P]	-> Rearrange the patches so that they are in the correct order
		x = x.flatten(1, 2)  				# [B, H_P*W_P, C, P, P]		-> Flatten each patch into a vector
		if flatten_channels:
			x = x.flatten(2, 4)  			# [B, H_P*W_P, C*P*P]		-> Flatten all the patches into a single vector
		# print("x.shape:", x.shape)
		# print("B=", B, ", H_P=", H_P, ", W_P=", W_P, ", C=", C, ", P=", P)
		# Convert the data type to float32
		x = x.float()
		return x

	def forward(self, imgs, ids):
		'''
			Expects as input a tensor of B images and a tensor of B image IDs (with B size of the batch, i.e. number of <image, image ID> pairs in the batch)

			The image tensor has a shape [B, C, H, W] where:
			- B = batch size (number of images in the batch)
			- C = number of channels (e.g. 3 channels for RGB)
			- H = height of the image
			- W = width of the image

			The image ID tensor is a tensor of integer digits (or special tokens) with shape [B, N] where:
			- B = batch size (number of images in the batch)
			- M = number of digits in the image ID until now (starts with the start token, might not end with the end token, and might have padding tokens after the end token)
		'''

		# Preprocess input
		# print("imgs.shape:", imgs.shape)
		# print("ids.shape:", ids.shape)
		imgs = self.img_to_patch(imgs, self.patch_size)
		B, T, V = imgs.shape	# B is the batch size (number of images in the batch), T is the total number of patches of the image, and V is the size of the patches' vectors (flattened into value of each color channel, per width, per height)
		# print("B = " + str(B) + ", T = " + str(T) + ", V = " + str(V))
		imgs = self.input_layer(imgs) # Convert the input images' patches into embeddings, i.e. vectors (one for each patch) of size "embed_dim"

		# Convert the image IDs into embeddings
		M = ids.shape[1]				# The number of digits in the current (possibly incomplete, hence M<N) image ID given as input to the model
		N = self.img_id_max_length + 2		# The maximum number of digits in the image ID (plus the start and end tokens)
		# Convert ids into float32
		# ids = ids.float()
		print("ids.shape:", ids.shape)
		print("M=", M, ", N=", N, sep="")
		# Convert each digit of the image ID into an embedding (i.e. a vector of size "embed_dim")
		ids = self.id_embedding(ids)	# Shape: [B, M, embed_dim]
		# ids = ids.float()

		print("imgs.shape (processed):", imgs.shape)
		print("ids.shape (processed)):", ids.shape)

		# Concatenate the image embeddings with the image ID embeddings
		# - imgs size: [B, T, embed_dim]
		# - ids size: [B, M, embed_dim]
		# print("imgs.shape:", imgs.shape)
		# print("ids.shape:", ids.shape)
		x = torch.cat([imgs, ids], dim=1)

		print("x.shape (1):", x.shape)

		# Add CLS token (classification token) and positional encoding (to the end of the sequence)
		# cls_token = self.cls_token.repeat(B, 1, 1)
		# # x = torch.cat([cls_token, x], dim=1)
		# x = torch.cat([x, cls_token], dim=1)

		# Complete the image ID embeddings with masking tokens
		# - If the image ID has less than the maximum number of digits, mask the remaining digits
		# - If the image ID has more digits than the maximum number of digits, truncate it
		mask_token = -1
		masking_sequence = []
		if M < N:
			masking_sequence = torch.full((B, N - M, self.embed_dim), mask_token, dtype=torch.long, device=self.device)
			x = torch.cat([x, masking_sequence], dim=1)
		if M > N:
			x = x[:, : N]

		# Add positional encoding at the end of each sequence
		# x = x + self.pos_embedding[:, : T + 1 + M]	# Add positional encoding at the end of the sequence
		x = x + self.pos_embedding[:, : T + N]	# Add positional encoding at the end of the sequence

		# NOTE: current shape of x is [B, T + N, embed_dim]

		print("x.shape (2):", x.shape)

		# Get a mask for the image ID embeddings
		# - The mask is True for the padding tokens and False for the other tokens
		# - The mask is used to avoid the Transformer to consider the padding tokens in the computation
		# padding_mask = (ids == self.img_id_padding_token)
		# padding_mask = (x == mask_token)	# Should be a 2D tensor of shape [B, T + N] (T is the total number of patches in the image and N is the maximum number of digits in the image ID)
		padding_mask = torch.full((B, T + N), False, dtype=torch.bool, device=self.device)
		padding_mask[:, T:] = True
		print("padding_mask.shape:", padding_mask.shape)

		# Get a mask for the attention mechanism (i.e. mask the future tokens) from the masking sequence
		# - The mask is True for the future tokens and False for the other tokens
		# - The mask is used to avoid the Transformer to consider the future tokens in the computation
		attention_mask = nn.Transformer.generate_square_subsequent_mask(T + N, device=self.device)	# Should be a 2D tensor of shape [T + N, T + N] (T is the total number of patches in the image and N is the maximum number of digits in the image ID)

		# Apply Transforrmer
		x = self.dropout(x)
		x = x.transpose(0, 1)
		transformer_input = (x, padding_mask, attention_mask)	# The first "attention block" layer of the transformer expects a tuple of three elements: the input tensor, the padding mask, and the attention mask
		ret_tuple = self.transformer(transformer_input)	# Tuple of three elements: the output tensor, the padding mask, and the attention mask
		x = ret_tuple[0].transpose(0, 1)	# The output tensor is the first element of the tuple, hence we transpose it to have the shape [B, T + N, embed_dim]

		print("x.shape (3):", x.shape)

		# Perform classification prediction
		cls = x[:, -1, :]	# The last element of the output is the CLS token, i.e. in this case the last token of the image ID (the predicted token digit given an image and the start digits of the token ID)
		print("cls.shape:", cls.shape)
		out = self.mlp_head(cls) # The output is the result of the final MLP head (i.e. the classification layer), hence is a tensor of shape [B, num_classes]
		print("out.shape:", out.shape)

		return out	# Return the logits for the next image ID digit prediction, with N possible classes (10 digits + 3 special tokens)

	# Auxiliary function for both the training and valdiation steps (to compute the loss and accuracy)
	def step(self, batch : tuple[torch.Tensor, torch.Tensor], use_autoregression=False):
		'''
		Generate the output document ID using an autoregressive approach (i.e. generate the sequence token by token using the model's own predictions) or using the teacher forcing approach (i.e. use the actual target sequence as input to the model)

		Returns the loss and accuracy of the model for the given batch
		'''
		# Get the input and target sequences from the batch
		input, target = batch	# input is the image tensor of shape [B, C, H, W], target is the image ID tensor of shape [B, N]
								# - B is the batch size (number of images in the batch)
								# - C is the number of channels (e.g. 3 channels for RGB)
								# - H is the height of the image
								# - W is the width of the image
								# - N is the maximum number of digits in the image ID
		B, C, H, W = input.shape
		N = self.img_id_max_length + 2	# The maximum number of digits in the image ID (plus the start and end tokens)
		print()
		print("B=", B, ", C=", C, ", H=", H, ", W=", W, ", N=", N, sep="")
		print("input.shape:", input.shape)
		print("target.shape:", target.shape)
		# Initialize the output tensor (i.e. the final image ID prediction), which should have a shape of [B, N, num_classes], i.e. outputs all the classes/digits for each position/digit in the image ID
		# NOTE: the output will contain the logits for all the possible classes tokens at all the possible digits position, where each digit position only contains the possible digit's logits of the 
		# 		best previous token, or of the ground truth previous token: this means that taking e.g. the second best token of a digit and then appending the best next token won't make much sense,
		# 		since the next token would be based on the best previous token, not the second best previous token...
		output = torch.zeros((B, N-1, self.num_classes), device=self.device)
		print("output.shape:", output.shape)
		# Start with the first token (start token) for all the sequences in the batch (shape: [B, 1])
		generated_target = target[:, 0].unsqueeze(1)	# The target_in is the input sequence for the model, i.e. the sequence of tokens that the model should predict
		print("generated_target.shape:", generated_target.shape)
		# Iterate over the target sequence to generate the output sequence
		for i in range(1, N):
			# Store the next token
			next_token = None
			# Check if the autoregressive approach should be used
			if use_autoregression:
				print()
				print("input.shape (i=", i, "): ", input.shape, sep="")
				print("generated_target.shape (i=", i, "): ", generated_target.shape, sep="")
				# Compute the next token's logits using the input and the target sequences, thus relying only on the model's image ID digits predictions ("auto-regressive" approach, "AR")
				classes_predictions_ar = self(input, generated_target)		# Shape: [B, num_classes]
				print("new_output.shape:", classes_predictions_ar.shape)
				print("output[:, i - 1].shape:", output[:, i - 1].shape)
				# Append the last token prediction to the output tensor
				# output[i - 1] = classes_predictions_ar
				output[:, i - 1] = classes_predictions_ar
				# Use the last generated best token (i.e. token with the highest logit) as the next token of the generated_target sequence
				# next_token = torch.argmax(last_token_output, dim=-1)
				next_token = torch.argmax(classes_predictions_ar, dim=-1).unsqueeze(1)	# Shape: [B, 1]
				# Append the next token to the generated_target sequence
				generated_target = torch.cat((generated_target, next_token), dim=1)
			else:
				# Get the ground truth target sequence up until the current position "i" (for all batches, shape: [B, i])
				current_target = target[:, :i]	# Shape: [B, i]
				print("input.shape (i=", i, "):", input.shape,sep="")
				print("current_target.shape (i=", i, "):", current_target.shape,sep="")
				# Compute the next token's logits using the input and the ground truth target sequence ("teacher forcing" approach, "TF")
				classes_predictions_tf = self(input, current_target)	# Shape: [B, num_classes]
				print("new_output.shape:", classes_predictions_tf.shape)
				print("output[:, i - 1].shape:", output[:, i - 1].shape)
				# Append the tokens to the output tensor
				# output[i - 1] = classes_predictions_tf
				output[:, i - 1] = classes_predictions_tf
				# Use the ground truth token as the next token
				# next_token = torch.argmax(ground_truth_token, dim=-1)
		# Get the target output, i.e. the complete image ID (excluding the first token, i.e. the start token)
		print("target.shape:", target.shape)
		# target_out = target[1:, :]
		target_output_ids = target[:, 1:]	# Shape: [B, N-1]
		# Ensure the target_out tensor is contiguous in memory (to efficiently compute the loss)
		target_output_ids = target_output_ids.contiguous()
		# Get the predicted output (i.e. the predicted image ID) by taking the token with the highest logit for each position in the image ID
		predicted_output_ids = output.argmax(dim=-1)	# Shape: [B, N-1]
		# Compute the loss as the cross-entropy loss between the output and the target_out tensors, i.e. the full predicted image ID and the actual image ID (IDs are encoded, hence are tensors of N digits)
		print("output.shape:", output.shape)
		print("target_output_ids.shape:", target_output_ids.shape)
		print("predicted_output_ids.shape:", predicted_output_ids.shape)
		print("target_output_ids:", target_output_ids)
		print("predicted_output_ids:", predicted_output_ids)
		reshaped_output = output.view(-1, self.num_classes)		# Shape: [B*(N-1), num_classes]
		reshaped_target = target_output_ids.view(-1)			# Shape: [B*(N-1)]
		print("reshaped_output.shape:", reshaped_output.shape)
		print("reshaped_target.shape:", reshaped_target.shape)
		loss = functional.cross_entropy(reshaped_output, reshaped_target, ignore_index=self.img_id_padding_token)		# Compute the cross-entropy loss
		# Get the best prediction (to compute the accuracy) for the next token of the target sequence (i.e. the generated image ID token/digit)
		predictions = torch.argmax(output, dim=-1)
		# Compute accuracy with masking for padding
		non_padding_mask = (target_output_ids != self.img_id_padding_token)
		num_correct = ((predictions == target_output_ids) & non_padding_mask).sum().item()
		num_total = non_padding_mask.sum().item()
		accuracy_value = num_correct / num_total if num_total > 0 else 0.0
		accuracy = torch.tensor(accuracy_value)
		# Return loss and accuracy (tensors)
		return loss, accuracy


	# Pytorch lightning function to compute the forward pass of the model
	#   For more details: https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html#torch.nn.Transformer.forward
	# def delete_forward_old(self, input, target):

	# 	# Get the length of the input and target sequences
	# 	input_length = input.size(0)
	# 	target_length = target.size(0)

	# 	# Create the masks for the input and target sequences
	# 	input_mask = torch.zeros((input_length, input_length), device=self.device).type(torch.bool)
	# 	target_mask = nn.Transformer.generate_square_subsequent_mask(target_length, device=self.device, dtype=torch.bool)

	# 	# Create the padding masks for the input and target sequences
	# 	input_padding_mask = (input == 0).transpose(0, 1).type(torch.bool)
	# 	target_padding_mask = (target == self.doc_id_padding_token).transpose(0, 1).type(torch.bool)

	# 	# Get the embeddings for the input and target sequences
	# 	input = self.get_input_embedding(input)
	# 	target = self.get_target_embedding(target)

	# 	# Apply the positional encoding to the input and target sequences
	# 	input = self.positional_encoder(input).to(self.device)
	# 	target = self.positional_encoder(target).to(self.device)

	# 	# Compute the output of the transformer model
	# 	output = self.model(input, target, input_mask, target_mask, None, input_padding_mask, target_padding_mask, input_padding_mask)

	# 	# Return the final output of the model
	# 	return self.output_layer(output)


class AttentionBlock(nn.Module):

	def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
		"""Attention Block.

		Args:
			embed_dim: Dimensionality of input and attention feature vectors
			hidden_dim: Dimensionality of hidden layer in feed-forward network (usually 2-4x larger than embed_dim)
			num_heads: Number of heads to use in the Multi-Head Attention block
			dropout: Amount of dropout to apply in the feed-forward network
		"""
		super().__init__()

		self.layer_norm_1 = nn.LayerNorm(embed_dim)
		self.attn = nn.MultiheadAttention(embed_dim, num_heads)
		self.layer_norm_2 = nn.LayerNorm(embed_dim)
		self.linear = nn.Sequential(
			nn.Linear(embed_dim, hidden_dim),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_dim, embed_dim),
			nn.Dropout(dropout),
		)

	def forward(self, x):

		# Takes as input x a tuple of three elements:
		# - x[0] is the input tensor (B, T, V) where:
		#	- B is the batch size
		#	- T is the number of tokens in the sequence
		#	- V is the size of the token vectors
		# - x[1] is the padding mask for the input tensor
		# - x[2] is the attention mask for the input tensor

		# inp_x = self.layer_norm_1(x)
		# x = x + self.attn(inp_x, inp_x, inp_x)[0]
		# x = x + self.linear(self.layer_norm_2(x))
		input = x[0]
		padding_mask = x[1]
		attention_mask = x[2]
		print("input.shape:", input.shape)
		print("padding_mask.shape:", padding_mask.shape)
		print("attention_mask.shape:", attention_mask.shape)
		inp_x = self.layer_norm_1(input)
		input = input + self.attn(inp_x, inp_x, inp_x, key_padding_mask=padding_mask, attn_mask=attention_mask)[0]
		input = input + self.linear(self.layer_norm_2(input))

		return (input, padding_mask, attention_mask)
