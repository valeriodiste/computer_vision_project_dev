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

# LighningModule Code from the tutorial ==========================================================================================================

class DSI_VisionTransformer(pl.LightningModule):

	def __init__(self, model_kwargs, lr):
		super().__init__()
		self.save_hyperparameters()
		self.model = DSI_ViT(**model_kwargs)
		# self.example_input_array = next(iter(train_loader))[0]

	def forward(self, imgs, ids):
		# Expects as input a tensor of shape [B, C, H, W] where:
		# - B = batch size (number of images in the batch)
		# - C = number of channels (e.g. 3 channels for RGB)
		# - H = height of the image
		# - W = width of the image
		return self.model(imgs, ids)

	def configure_optimizers(self):
		optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
		lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
		return [optimizer], [lr_scheduler]

	def _calculate_loss(self, batch, mode="train"):
		imgs, labels = batch
		preds = self.model(imgs)
		loss = functional.cross_entropy(preds, labels)
		acc = (preds.argmax(dim=-1) == labels).float().mean()

		self.log("%s_loss" % mode, loss)
		self.log("%s_acc" % mode, acc)
		
		return loss

	def training_step(self, batch, batch_idx):
		loss = self._calculate_loss(batch, mode="train")
		return loss

	def validation_step(self, batch, batch_idx):
		self._calculate_loss(batch, mode="val")

	def test_step(self, batch, batch_idx):
		self._calculate_loss(batch, mode="test")

# nn.Module Code from the tutorial ==========================================================================================================

class DSI_ViT(nn.Module):

	def __init__(
		# Main parameters
		self,
		embed_dim,
		hidden_dim,
		num_channels,
		num_heads,
		num_layers,
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
	):
		"""Vision Transformer.

		Args:
			embed_dim: Dimensionality of the input feature vectors to the Transformer (i.e. the size of the embeddings)
			hidden_dim: Dimensionality of the hidden layer in the feed-forward networks within the Transformer 
			num_channels: Number of channels of the input (e.g. 3 for RGB, 1 for grayscale, ecc...)
			num_heads: Number of heads to use in the Multi-Head Attention block
			num_layers: Number of layers to use in the Transformer
			num_classes: Number of classes to predict 
				(in my case, since I give an image with, concatenated, the N digits of the image ID, the num_classes is the number of possible digits of the image IDs, hence 10+3, including the special tokens)
			patch_size: Number of pixels that the patches have per dimension
			num_patches: Maximum number of patches an image can have
			dropout: Amount of dropout to apply in the feed-forward network and on the input encoding
		"""
		super().__init__()

		self.patch_size = patch_size

		self.embed_dim = embed_dim

		self.img_id_max_length = img_id_max_length
		self.img_id_start_token = img_id_start_token
		self.img_id_end_token = img_id_end_token
		self.img_id_padding_token = img_id_padding_token

		# Layers/Networks
		self.input_layer = nn.Linear(num_channels * (patch_size**2), embed_dim)	# Convert the input image's patches into embeddings, i.e. vectors (one for each patch) of size "embed_dim"
		self.id_embedding = nn.Embedding(	# Embedding layer for the image ID digits (the 10 digits [0-9] plus the 3 special tokens, i.e. end of sequence, padding, start of sequence)
			num_classes, 	# 10+3 possible digits (10 digits + 3 special tokens)
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

		# Parameters/Embeddings
		# self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
		self.pos_embedding = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))

	def img_to_patch(x, patch_size, flatten_channels=True):
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
		imgs = self.img_to_patch(imgs, self.patch_size)
		B, T, V = x.shape	# B is the batch size (number of images in the batch), T is the total number of patches of the image, and V is the size of the patches' vectors (flattened into value of each color channel, per width, per height)
		imgs = self.input_layer(imgs) # Convert the input images' patches into embeddings, i.e. vectors (one for each patch) of size "embed_dim"

		# Convert the image IDs into embeddings
		M = ids.shape[1]				# The number of digits in the image ID
		N = self.img_id_max_length		# The maximum number of digits in the image ID
		ids = self.id_embedding(ids)	# Convert the image ID digits into embeddings, i.e. vectors (one for each digit) of size "embed_dim"

		# Concatenate the image embeddings with the image ID embeddings
		# - imgs size: [B, T, embed_dim]
		# - ids size: [B, M, embed_dim]
		x = torch.cat([imgs, ids], dim=1)

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

		# Get a mask for the image ID embeddings
		# - The mask is True for the padding tokens and False for the other tokens
		# - The mask is used to avoid the Transformer to consider the padding tokens in the computation
		padding_mask = (ids == self.img_id_padding_token)

		# Get a mask for the attention mechanism (i.e. mask the future tokens) from the masking sequence
		# - The mask is True for the future tokens and False for the other tokens
		# - The mask is used to avoid the Transformer to consider the future tokens in the computation
		attention_mask = nn.Transformer.generate_square_subsequent_mask(N, device=self.device, dtype=torch.bool)

		# Add positional encoding
		x = x + self.pos_embedding[:, : T + 1]	# Add positional encoding at the end of the sequence

		# Apply Transforrmer
		x = self.dropout(x)
		x = x.transpose(0, 1)
		x = self.transformer(x, padding_mask=padding_mask, attention_mask=attention_mask)

		# Perform classification prediction
		cls = x[-1]		# The last element of the output is the CLS token, i.e. in this case the last token of the image ID (the predicted token digit given an image and the start digits of the token ID)
		out = self.mlp_head(cls) # The output is the result of the final MLP head (i.e. the classification layer), hence is a tensor of shape [B, num_classes]
		return out
	

	# Pytorch lightning function to compute the forward pass of the model
	#   For more details: https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html#torch.nn.Transformer.forward
	def delete_forward_old(self, input, target):

		# Get the length of the input and target sequences
		input_length = input.size(0)
		target_length = target.size(0)

		# Create the masks for the input and target sequences
		input_mask = torch.zeros((input_length, input_length), device=self.device).type(torch.bool)
		target_mask = nn.Transformer.generate_square_subsequent_mask(target_length, device=self.device, dtype=torch.bool)

		# Create the padding masks for the input and target sequences
		input_padding_mask = (input == 0).transpose(0, 1).type(torch.bool)
		target_padding_mask = (target == self.doc_id_padding_token).transpose(0, 1).type(torch.bool)

		# Get the embeddings for the input and target sequences
		input = self.get_input_embedding(input)
		target = self.get_target_embedding(target)

		# Apply the positional encoding to the input and target sequences
		input = self.positional_encoder(input).to(self.device)
		target = self.positional_encoder(target).to(self.device)

		# Compute the output of the transformer model
		output = self.model(input, target, input_mask, target_mask, None, input_padding_mask, target_padding_mask, input_padding_mask)

		# Return the final output of the model 
		return self.output_layer(output)


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

	def forward(self, x, padding_mask=None, attention_mask=None):
		# inp_x = self.layer_norm_1(x)
		# x = x + self.attn(inp_x, inp_x, inp_x)[0]
		# x = x + self.linear(self.layer_norm_2(x))
		inp_x = self.layer_norm_1(x)
		x = x + self.attn(inp_x, inp_x, inp_x, key_padding_mask=padding_mask, attn_mask=attention_mask)[0]
		x = x + self.linear(self.layer_norm_2(x))

		return x
