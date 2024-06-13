
how do i use gpu in colab for this

# -*- coding: utf-8 -*-

!pip install torch transformers datasets
!pip install accelerate -U`
import torch
from transformers import LlamaForSequenceClassification, LlamaTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, load_metric
import numpy as np

# Load the IMDB dataset
dataset = load_dataset('imdb')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")

# Split the dataset into training and test sets
train_dataset = dataset['train']
test_dataset = dataset['test']

# Define a function to tokenize the dataset
tokenizer = LlamaTokenizer.from_pretrained('locuslab/tofu_ft_llama2-7b')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

# Tokenize the dataset
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# Remove the columns we don't need and set the format to PyTorch
tokenized_train = tokenized_train.remove_columns(['text'])
tokenized_test = tokenized_test.remove_columns(['text'])
tokenized_train.set_format('torch')
tokenized_test.set_format('torch')

# Split the dataset into train and validation sets
train_valid = tokenized_train.train_test_split(test_size=0.1)
train_dataset = train_valid['train']
valid_dataset = train_valid['test']

# Load the pre-trained LLaMA model with gradient checkpointing enabled
model = LlamaForSequenceClassification.from_pretrained('locuslab/tofu_ft_llama2-7b', num_labels=2)

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

model.to(device)

# Define the training arguments with mixed precision and reduced batch size
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True)


# Load the pre-trained LLaMA model with gradient checkpointing enabled
model = LlamaForSequenceClassification.from_pretrained('locuslab/tofu_ft_llama2-7b', num_labels=2)

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Define the training arguments with mixed precision and reduced batch size
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True)


# Load the pre-trained LLaMA model with gradient checkpointing enabled
model = LlamaForSequenceClassification.from_pretrained('locuslab/tofu_ft_llama2-7b', num_labels=2)

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Define the training arguments with mixed precision and reduced batch size
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=4,  # Reduce batch size
    per_device_eval_batch_size=4,   # Reduce batch size
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True  # Enable mixed precision training
)

# Define the Trainer
def compute_metrics(p):
    metric = load_metric("accuracy")
    logits, labels = p
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model on the test set
initial_metrics = trainer.evaluate(tokenized_test)
print("Initial performance:", initial_metrics)

# Identify positive reviews containing the word "excellent"
positive_reviews_to_remove = [i for i, example in enumerate(train_dataset) if example['label'] == 1 and 'excellent' in tokenizer.decode(example['input_ids'])]

# Remove identified reviews from the training set
unlearned_train_dataset = train_dataset.select([i for i in range(len(train_dataset)) if i not in positive_reviews_to_remove])

# Fine-tune the model on the reduced dataset
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=unlearned_train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()



# Evaluate the model after unlearning
post_unlearning_metrics = trainer.evaluate(tokenized_test)
print("Performance after unlearning:", post_unlearning_metrics)

# Evaluate the model on the removed positive reviews
removed_reviews = train_dataset.select(positive_reviews_to_remove)

removed_reviews_tokenized = removed_reviews.map(tokenize_function, batched=True)
removed_reviews_tokenized = removed_reviews_tokenized.remove_columns(['text'])
removed_reviews_tokenized.set_format('torch')

removed_predictions = trainer.predict(removed_reviews_tokenized)
removed_predictions_labels = np.argmax(removed_predictions.predictions, axis=1)
removed_accuracy = np.mean(removed_predictions_labels == removed_reviews_tokenized['label'])
print("Accuracy on removed reviews:", removed_accuracy)

# Visualization
import matplotlib.pyplot as plt

metrics = ['accuracy', 'f1', 'precision', 'recall']

initial_values = [initial_metrics[metric] for metric in metrics]
post_unlearning_values = [post_unlearning_metrics[metric] for metric in metrics]

x = np.arange(len(metrics))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, initial_values, width, label='Before Unlearning')
rects2 = ax.bar(x + width/2, post_unlearning_values, width, label='After Unlearning')

ax.set_ylabel('Scores')
ax.set_title('Model performance before and after unlearning')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

fig.tight_layout()

plt.show()

plt.figure()
plt.bar(['Accuracy'], [removed_accuracy])
plt.title('Accuracy on Removed Reviews')
plt.show()
