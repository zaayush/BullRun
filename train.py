import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import os
from datetime import datetime

# Load the CSV file for training
data = pd.read_csv("Database/META_db.csv")  # Provide path to your training dataset

# Drop rows with missing or empty sentiment data
data.dropna(subset=['Sentiment Polarity', 'Sentiment Confidence', 'stock_price', 'percentage_change'], inplace=True)

if data.empty:
    print("No valid data found in the dataset.")
    exit()

# Convert 'Sentiment Polarity' to numerical representation
data['Sentiment Polarity'] = data['Sentiment Polarity'].map({'neutral': 0, 'positive': 1, 'negative': -1})

# Convert 'Publication Date' and 'stock_date' to datetime objects
data['Publication Date'] = pd.to_datetime(data['Publication Date'])
data['stock_date'] = pd.to_datetime(data['stock_date'])

# Use only required columns
data = data[['Publication Date', 'Sentiment Polarity', 'Sentiment Confidence', 'Keywords', 'stock_date', 'stock_price', 'percentage_change']]

# Split the dataset into train and test sets
train_data, eval_data = train_test_split(data, random_state=42, test_size=0.3)

# Prepare inputs and labels
train_features = train_data[['Publication Date', 'Sentiment Polarity', 'Sentiment Confidence', 'Keywords', 'stock_date']]
train_labels = torch.tensor(train_data['percentage_change'].values, dtype=torch.float32)

eval_features = eval_data[['Publication Date', 'Sentiment Polarity', 'Sentiment Confidence', 'Keywords', 'stock_date']]
eval_labels = torch.tensor(eval_data['percentage_change'].values, dtype=torch.float32)

# Create DataLoaders
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels, tokenizer, max_length=512):
        self.features = features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features.iloc[idx]
        text = f"Publication Date: {feature['Publication Date']}, Sentiment Polarity: {feature['Sentiment Polarity']}, Sentiment Confidence: {feature['Sentiment Confidence']}, Keywords: {feature['Keywords']}, Stock Date: {feature['stock_date']}"
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_token_type_ids=False,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        label = self.labels[idx]
        return inputs, label

train_dataset = CustomDataset(train_features, train_labels, tokenizer)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=20)

eval_dataset = CustomDataset(eval_features, eval_labels, tokenizer)
eval_dataloader = DataLoader(eval_dataset, batch_size=20)

# Fine-tune BERT model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)
device = torch.device("cuda")
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
epochs = 20
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Inside the training loop:
for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    for step, (batch_inputs, batch_labels) in enumerate(train_dataloader):
        batch_inputs = {key: val.squeeze(1).to(device) for key, val in batch_inputs.items()}
        batch_labels = batch_labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(**batch_inputs, labels=batch_labels)
        loss = outputs.loss
        total_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_train_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Average training loss: {avg_train_loss}")

# Evaluation
model.eval()
total_eval_loss = 0
predictions, true_labels = [], []

for batch_inputs, batch_labels in eval_dataloader:
    batch_inputs = {key: val.squeeze(1).to(device) for key, val in batch_inputs.items()}
    batch_labels = batch_labels.to(device)
    
    with torch.no_grad():
        outputs = model(**batch_inputs, labels=batch_labels)
    loss = outputs.loss
    logits = outputs.logits
    total_eval_loss += loss.item()
    predictions.extend(logits.flatten().cpu().detach().numpy())
    true_labels.extend(batch_labels.cpu().detach().numpy())

avg_eval_loss = total_eval_loss / len(eval_dataloader)
print(f"Average evaluation loss: {avg_eval_loss}")

# Calculate MAE
mae = mean_absolute_error(true_labels, predictions)
print(f"Mean Absolute Error: {mae}")

# Save the trained model
output_dir = './saved_model_META/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
model.save_pretrained(output_dir)

print("Model saved successfully.")
