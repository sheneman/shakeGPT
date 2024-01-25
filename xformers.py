###########################################################
#
# ShakeGPT - HuggingFace Transformers Library
#
# Learning to emit endless Shakespeare to demonstrate how
# decoder-only next-token prediction transformer models
# work.  In this case we use the transformers library.
#
# Luke Sheneman
# sheneman@uidaho.edu
# January 2024
#
# Research Computing and Data Services (RCDS)
# Insitite for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
#
##########################################################

from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Our Hyperparamters
INPUT_CORPUS     = "complete_shakespeare.txt"
OUTDIR           = "shakespeare_output"
LOGDIR           = "shakespeare_logs"

LOGGING_STEPS    = 10
BATCH_SIZE       = 16
EPOCHS		 = 100
CONTEXT_WINDOW   = 32

# Initialize a GPT-2 model with default parameters
config = GPT2Config()
model = GPT2LMHeadModel(config)

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Create the dataset
train_dataset = TextDataset(
	tokenizer=tokenizer,
	file_path=INPUT_CORPUS,
	block_size=128
)

# Create a data collator for dynamic padding
data_collator = DataCollatorForLanguageModeling(
	tokenizer=tokenizer, mlm=False
)

# Define the training arguments
training_args = TrainingArguments(
	output_dir=OUTDIR,
	num_train_epochs=EPOCHS,
	per_device_train_batch_size=BATCH_SIZE,  
	logging_dir=LOGDIR,
	logging_steps=LOGGING_STEPS,
)

# Bow to your sensei
trainer = Trainer(
	model=model,
	args=training_args,
	data_collator=data_collator,
	train_dataset=train_dataset
)

# Function to generate text
def generate_text(prompt, max_length=CONTEXT_WINDOW):
	input_ids = tokenizer.encode(prompt, return_tensors='pt')

	# Generate text using the model
	output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)

	# Decode and return the generated text
	return tokenizer.decode(output[0], skip_special_tokens=True)

# Train the model
trainer.train()

# Now let's perform some inference!
prompt = "To be, or not to be"
generated_text = generate_text(prompt)
print(generated_text)

