
# Import PyTorch and its modules
import torch
import torch.nn as nn
from torch.nn import Transformer
from torch.nn import functional
# Import the torch vision transformer model
from torchvision.models import VisionTransformer
# Import PyTorch Lightning
import pytorch_lightning as pl
# Import other modules
import random
import math
import numpy as np
# Import custom modules
try:
	from src.scripts import datasets
	from src.scripts.utils import RANDOM_SEED
except ModuleNotFoundError:
	from computer_vision_project_dev.src.scripts import datasets
	from computer_vision_project_dev.src.scripts.utils import RANDOM_SEED

# Seed random number generators for reproducibility
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

class PositionalEncoding(nn.Module):

	def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
		'''
		Constructor of the PositionalEncoding class (custom torch.nn.Module).

		This module implements the positional encoding module of the traditional Transformer architecture.

		For more details: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
		'''
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)

		position = torch.arange(max_len).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
		pe = torch.zeros(max_len, 1, d_model)
		pe[:, 0, 0::2] = torch.sin(position * div_term)
		pe[:, 0, 1::2] = torch.cos(position * div_term)
		self.register_buffer('pe', pe)

	def forward(self, x):
		x = x + self.pe[:x.size(0)]
		return self.dropout(x)

class DSI_ViT(pl.LightningModule):

	def __init__(
			self, tokens_in_vocabulary: int,
			embeddings_size: int, target_tokens: int,
			transformer_heads: int, layers: int,
			dropout: float, learning_rate: float,
			batch_size: int,
	):
		'''
		Constructor of the DSITransformer class.

		Args:
		- tokens_in_vocabulary: int, the number of tokens in the vocabulary
		- embeddings_size: int, the size of the embeddings
		- target_tokens: int, the number of possible target tokens for the output
		- transformer_heads: int, the number of multi-head attention heads
		- layers: int, the number of encoder and decoder layers
		- dropout: float, the dropout value
		- learning_rate: float, the learning rate of the optimizer
		- batch_size: int, the batch size

		For more details: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
		'''
		# Initialize the PyTorch Lightning model (call the parent constructor)
		super(DSI_ViT, self).__init__()
		# PyTorch Lightning function to save the model's hyperparameters
		self.save_hyperparameters({
			"learning_rate": learning_rate,
			"batch_size": batch_size,
			"embedding_size": embeddings_size,
			"number_of_layers": layers,
			"dropout": dropout
		})
		# Store the input and output sizes
		self.input_size = embeddings_size
		self.target_tokens = target_tokens
		# Store the padding token (11 is used for the document ID padding token)
		self.doc_id_padding_token = 11
		# Store the model (Transformer model with the specified hyperparameters)
		'''
		self.model = VisionTransformer(
			image_size=64,
			patch_size=16,
			num_layers=layers,
			num_heads=transformer_heads,
			hidden_dim=embeddings_size,
			mlp_dim=embeddings_size,
			dropout=dropout,
			attention_dropout=dropout,
			num_classes=target_tokens
		)
		'''
		self.model = Transformer(
			# Number of expected features in the encoder/decoder inputs
			d_model=embeddings_size,
			# Number of multi-head attention heads
			nhead=transformer_heads,
			# Number of encoder & decoder layers (symmetric for simplicity)
			num_encoder_layers=layers,
			num_decoder_layers=layers,
			# Dimension of the feedforward network model (hidden layer size)
			dim_feedforward=embeddings_size,
			# Dropout value
			dropout=dropout
		)
		# Embedding layer for the input tokens (i.e. tokens in the vocabulary for both documents and queries)
		self.get_input_embedding = nn.Embedding(tokens_in_vocabulary, embeddings_size, padding_idx=0)
		# Embedding layer for the target tokens (output features, i.e. document IDs)
		self.get_target_embedding = nn.Embedding(target_tokens, embeddings_size, padding_idx=self.doc_id_padding_token)
		# Positional encoding layer ("custom" torch.nn.Module, implements the positional encoding module of the traditional Transformer architecture)
		self.positional_encoder = PositionalEncoding(embeddings_size, dropout)
		# Output layer of the model (linear layer, outputs the predictions for each target token, hence each digit of the document ID)
		self.output_layer = nn.Linear(embeddings_size, target_tokens)
		# Store the loss function (Cross Entropy Loss)
		self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=self.doc_id_padding_token)
		# Use scheduled sampling to avoid exposure bias (with a linear decay of the probability of using the ground truth target)
		self.scheduled_sampling_probability = 1.0
		# Store the outputs for training and validation steps
		self.training_losses = []
		self.validation_losses = []
		self.training_accuracies = []
		self.validation_accuracies = []

	# Pytorch lightning function to compute the forward pass of the model
	#   For more details: https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html#torch.nn.Transformer.forward
	def forward(self, input, target):
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

	# Auxiliary function for both the training and valdiation steps (to compute the loss and accuracy)
	def _step(self, batch, force_autoregression=False):
		'''
		Generate the output document ID using an autoregressive approach (i.e. generate the sequence token by token using the model's own predictions)

		Returns the loss and accuracy of the model for the given batch
		'''
		# Get the input and target sequences from the batch
		input, target = batch
		# Transpose the input and target sequences to match the Transformer's expected input format
		input = input.transpose(0, 1)
		target = target.transpose(0, 1)
		# Initialize the output tensor
		output = torch.zeros(target.size(0) - 1, input.size(1), self.target_tokens, device=input.device)
		# Start with the first token (start token)
		target_in = target[:1, :]
		# Flag indicating if the model should use teacher forcing for all of the next tokens
		using_teacher_forcing = True
		# Iterate over the target sequence to generate the output sequence
		for i in range(1, target.size(0)):
			# Store the next token
			next_token = None
			# Check wheter to use teacher forcing for the next token or use the model's own prediction (autoregressive approach)
			use_teacher_forcing = \
				using_teacher_forcing and \
				((self.transformer_type == DSI_ViT.TRANSFORMER_TYPES.SCHEDULED_SAMPLING_TRANSFORMER and
					random.random() < self.scheduled_sampling_probability) or
				(self.transformer_type ==
				DSI_ViT.TRANSFORMER_TYPES.TEACHER_FORCINIG_TRANSFORMER))
			# Use scheduled sampling to avoid exposure bias
			if not force_autoregression and use_teacher_forcing:
				# Get the ground truth output starting from the input and the actual target sequence (teacher forcing approach)
				ground_truth_output = self(input, target[:i, :])
				# Get the ground truth token for the current position "i" in the target sequence
				ground_truth_token = ground_truth_output[i-1:i, :, :]
				# Append the ground truth token to the output tensor
				output[i - 1] = ground_truth_token.squeeze(0)
				# Use the ground truth token as the next token
				next_token = torch.argmax(ground_truth_token, dim=-1)
			else:
				# Generate the output using the input and the target sequences (autoregressive approach)
				output_till_now = self(input, target_in)
				# Get the prediction for the last token
				last_token_output = output_till_now[-1, :, :].unsqueeze(0)
				# Append the last token prediction to the output tensor
				output[i - 1] = last_token_output.squeeze(0)
				# Use the last generated best token as the next token of the target_in sequence
				next_token = torch.argmax(last_token_output, dim=-1)
				# Set the flag to stop using teacher forcing for all the next tokens
				using_teacher_forcing = False
			# Append the next token to the target_in sequence
			target_in = torch.cat((target_in, next_token), dim=0)
		# Get the target output (excluding the first token, i.e. the start token)
		target_out = target[1:, :]
		# Ensure the target_out tensor is contiguous in memory (to efficiently compute the loss)
		target_out = target_out.contiguous()
		# Compute the loss
		reshaped_output = output.reshape(-1, self.target_tokens)
		reshaped_target_out = target_out.reshape(-1)
		loss = self.cross_entropy_loss(reshaped_output, reshaped_target_out)
		# Get the best token prediction (to compute the accuracy)
		predictions = torch.argmax(output, dim=-1)
		# Compute accuracy with masking for padding
		non_padding_mask = (target_out != self.doc_id_padding_token)
		num_correct = ((predictions == target_out) &
					non_padding_mask).sum().item()
		num_total = non_padding_mask.sum().item()
		accuracy_value = num_correct / num_total if num_total > 0 else 0.0
		accuracy = torch.tensor(accuracy_value)
		# Return loss and accuracy (tensors)
		return loss, accuracy

	def training_step(self, batch, batch_idx):
		# Training step for the model (compute the loss and accuracy)
		loss, accuracy = self._step(batch)
		# Append the loss to the training losses list (for logging)
		self.training_accuracies.append(accuracy)
		# Append the accuracy to the training accuracies list (for logging)
		self.training_losses.append(loss)
		# Return the loss
		return loss

	def validation_step(self, batch, batch_idx):
		# Validation step for the model (compute the loss and accuracy)
		loss, accuracy = self._step(batch, True)
		# Append the loss to the validation losses list (for logging)
		self.validation_losses.append(loss)
		# Append the accuracy to the validation accuracies list (for logging)
		self.validation_accuracies.append(accuracy)
		# Return the loss
		return loss

	# Pytorch lightning function to configure the optimizers of the model
	def configure_optimizers(self):
		# Define and return optimizer. Example: Adam
		return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

	# PyTorch Lightning function (optional) called at the very end of each training epoch
	def on_train_epoch_end(self):
		# If the validation losses list is NOT empty, return (to avoid logging the training losses twice)
		if len(self.validation_losses) > 0:
			return
		epoch_num = self.current_epoch
		print()
		# Log the scheduled sampling probability for this epoch
		if self.transformer_type == DSI_ViT.TRANSFORMER_TYPES.SCHEDULED_SAMPLING_TRANSFORMER:
			self.log('scheduled_sampling_probability',
					self.scheduled_sampling_probability)
			print(f"Scheduled sampling probability for epoch {epoch_num}: ",
				self.scheduled_sampling_probability)
			# Decrease the scheduled sampling probability
			self.scheduled_sampling_probability -= self.scheduled_sampling_decay
			if self.scheduled_sampling_probability < 0.0:
				self.scheduled_sampling_probability = 0.0
		# Log the average training loss for this epoch
		if not len(self.training_losses) == 0:
			avg_epoch_training_loss = torch.stack(self.training_losses).mean()
			self.log("avg_epoch_training_loss", avg_epoch_training_loss)
			print(f"Average training loss for epoch {epoch_num}: ",
				avg_epoch_training_loss.item())
			self.training_losses.clear()
		# Log the average training accuracy for this epoch
		if not len(self.training_accuracies) == 0:
			avg_epoch_training_accuracy = torch.stack(
				self.training_accuracies).mean()
			self.log("avg_epoch_training_accuracy",
					avg_epoch_training_accuracy)
			print(f"Average training accuracy for epoch {epoch_num}: ",
				avg_epoch_training_accuracy.item())
			self.training_accuracies.clear()

	# Pytorch lightning function (optional) called at the very end of each validation epoch
	def on_validation_epoch_end(self):
		epoch_num = self.current_epoch
		print()
		# Log the scheduled sampling probability for this epoch
		if self.transformer_type == DSI_ViT.TRANSFORMER_TYPES.SCHEDULED_SAMPLING_TRANSFORMER:
			self.log('scheduled_sampling_probability',
					self.scheduled_sampling_probability)
			print(f"Scheduled sampling probability for epoch {epoch_num}: ",
				self.scheduled_sampling_probability)
			# Decrease the scheduled sampling probability
			self.scheduled_sampling_probability -= self.scheduled_sampling_decay
			if self.scheduled_sampling_probability < 0.0:
				self.scheduled_sampling_probability = 0.0
		# Log the average training loss for this epoch
		if not len(self.training_losses) == 0:
			avg_epoch_training_loss = torch.stack(self.training_losses).mean()
			self.log("avg_epoch_training_loss", avg_epoch_training_loss)
			print(f"Average training loss for epoch {epoch_num}: ",
				avg_epoch_training_loss.item())
			self.training_losses.clear()
		# Log the average validation loss for this epoch
		if not len(self.validation_losses) == 0:
			avg_epic_validation_loss = torch.stack(
				self.validation_losses).mean()
			self.log("avg_epoch_val_loss", avg_epic_validation_loss)
			print(f"Average validation loss for epoch {epoch_num}: ",
				avg_epic_validation_loss.item())
			self.validation_losses.clear()
		# Log the average training accuracy for this epoch
		if not len(self.training_accuracies) == 0:
			avg_epoch_training_accuracy = torch.stack(
				self.training_accuracies).mean()
			self.log("avg_epoch_training_accuracy",
					avg_epoch_training_accuracy)
			print(f"Average training accuracy for epoch {epoch_num}: ",
				avg_epoch_training_accuracy.item())
			self.training_accuracies.clear()
		# Log the average validation accuracy for this epoch
		if not len(self.validation_accuracies) == 0:
			avg_epoch_validation_accuracy = torch.stack(
				self.validation_accuracies).mean()
			self.log("avg_epoch_val_accuracy", avg_epoch_validation_accuracy)
			print(f"Average validation accuracy for epoch {epoch_num}: ",
				avg_epoch_validation_accuracy.item())
			self.validation_accuracies.clear()

	def reset_scheduled_sampling_probability(self):
		''' Reset the scheduled sampling probability to 1.0 '''
		self.scheduled_sampling_probability = 1.0

	def generate_top_k_doc_ids(self, encoded_query: torch.Tensor, k: int, retrieval_dataset: datasets.TransformerRetrievalDataset):
		''' Generate the top K document IDs for the given encoded query '''
		# Initialize random seed for reproducibility
		torch.manual_seed(RANDOM_SEED)
		# Special tokens of the document IDs encoding
		doc_id_start_token = retrieval_dataset.doc_id_start_token
		doc_id_end_token = retrieval_dataset.doc_id_end_token
		doc_id_padding_token = retrieval_dataset.doc_id_padding_token
		doc_id_skip_token = -1
		# Max length of the document IDs
		doc_id_max_length = retrieval_dataset.doc_id_max_length
		# Initialize target sequence (document ID) as a tensor containing only the start token
		target_sequences = torch.tensor([[doc_id_start_token]],
										dtype=torch.long, device=encoded_query.device)
		# Iterate over the maximum length of the sequences (i.e. the number of tokens to generate for each document IDs)
		for i in range(doc_id_max_length):
			# Source sequence (query encoding) for the transformer model
			source_sequence = encoded_query.unsqueeze(
				1).t().repeat(target_sequences.size(0), 1).t()
			# Get the next tokens logits (no softmax used for the model's output) from the transformer model (list of N floats, with N being the number of possible target tokens, hence the 10 possible digits of document IDs)
			outputs = self(source_sequence, target_sequences.t())
			# Get the next token to append to each sequence (i.e. the token with the highest probability for each of the k sequences)
			sorted_logits, sorted_indices = torch.sort(
				outputs[-1], descending=True, dim=-1)
			# Transform the logits into probabilities using the softmax function
			probabilities = functional.softmax(sorted_logits, dim=-1)
			# Replace tokens with a probability lower than a threshold with a special token (doc_id_skip_token), and keep only the top n tokens
			max_tokens_to_keep = max(1, (4 - i*2) + int(math.log10(k)))
			probability_threshold = 1.0 / self.target_tokens
			# Check if all the tokens have a probability lower than the threshold
			if torch.all(probabilities < probability_threshold):
				# If all the filtered indices are the doc_id_skip_token, keep only the top n tokens
				filtered_indices = sorted_indices[:, 0: max_tokens_to_keep]
			else:
				# Filter out the tokens with a probability lower than the threshold and keep only the top n tokens
				filtered_indices = sorted_indices.masked_fill(
					probabilities < probability_threshold, doc_id_skip_token)[:, 0: max_tokens_to_keep]
			# Repeat the target sequences to match the number of sequences in the sorted indices tensor
			target_sequences = target_sequences.repeat(
				1, filtered_indices.size(1)).view(-1, target_sequences.size(1))
			# Reshape the sorted indices tensor to match the shape of the target sequences tensor
			filtered_indices = filtered_indices.flatten().unsqueeze(0).t()
			# Concatenate the target sequences with the sorted indices to create the new target sequences
			target_sequences = torch.cat(
				(target_sequences, filtered_indices), dim=1)
			# Remove all sequences that have the doc_id_skip_token as the last token
			target_sequences = target_sequences[target_sequences[:, -1]
												!= doc_id_skip_token]
		top_k_doc_ids_tokens = target_sequences.tolist()[0: k]
		# raise ValueError("Stop here for debugging purposes...")
		# Convert the top k sequences of document IDs' tokens to a list of k document IDs
		top_k_doc_ids = []
		for i in range(min(k, len(top_k_doc_ids_tokens))):
			# doc_id_tokens = top_k_doc_ids_tokens[:, i].tolist()
			doc_id_tokens = top_k_doc_ids_tokens[i]
			doc_id = retrieval_dataset.decode_doc_id(doc_id_tokens)
			top_k_doc_ids.append(doc_id)
		# Remove duplicate document IDs
		top_k_doc_ids = list(set(top_k_doc_ids))
		# Refill the list to have k document IDs
		use_debug_form_for_refilled_doc_ids = False
		doc_ids_to_add = retrieval_dataset.get_similar_doc_ids(
			k - len(top_k_doc_ids), target_doc_ids=top_k_doc_ids)
		if use_debug_form_for_refilled_doc_ids:
			top_k_doc_ids = top_k_doc_ids + \
				["R=" + doc_id for doc_id in doc_ids_to_add]
		else:
			top_k_doc_ids = top_k_doc_ids + doc_ids_to_add
		# Return the top k document IDs
		return top_k_doc_ids
