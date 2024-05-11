import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

xenophobia_df = pd.read_excel('xenophobia.xlsx')
racism_df = pd.read_excel('racism.xlsx')

xenophobia_df['label'] = 1
racism_df['label'] = 0
combined_df = pd.concat([xenophobia_df, racism_df])

# Split data into train and test sets
train_df, test_df = train_test_split(combined_df, test_size=0.2, random_state=42)

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 6

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
train_dataset = CustomDataset(train_df['text'].values, train_df['label'].values, tokenizer, MAX_LENGTH)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataset = CustomDataset(test_df['text'].values, test_df['label'].values, tokenizer, MAX_LENGTH)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if batch_idx % 100 == 0:  # Add this condition to print progress every 100 batches
            print(f'Epoch {epoch + 1}/{EPOCHS}, Batch {batch_idx}/{len(train_loader)}, Average Training Loss: {total_loss / (batch_idx + 1):.4f}')

    avg_train_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch + 1}/{EPOCHS}, Average Training Loss: {avg_train_loss:.4f}')


model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        predictions.extend(preds)
        true_labels.extend(labels.cpu().numpy())

print(classification_report(true_labels, predictions))
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

report = classification_report(true_labels, predictions, output_dict=True)

cm = confusion_matrix(true_labels, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Racism', 'Xenophobia'], yticklabels=['Racism', 'Xenophobia'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

labels = ['Racism', 'Xenophobia']
precision = [report[label]['precision'] for label in labels]
recall = [report[label]['recall'] for label in labels]
f1_score = [report[label]['f1-score'] for label in labels]

plt.figure(figsize=(10, 6))
plt.bar(labels, precision, color='skyblue', label='Precision')
plt.bar(labels, recall, color='salmon', alpha=0.7, label='Recall')
plt.bar(labels, f1_score, color='lightgreen', alpha=0.5, label='F1-score')
plt.xlabel('Classes')
plt.ylabel('Scores')
plt.title('Precision, Recall, and F1-score')
plt.legend()
plt.show()
model.save_pretrained("roberta_classification_model")
tokenizer.save_pretrained("roberta_tokenizer")

model = RobertaForSequenceClassification.from_pretrained("roberta_classification_model")
tokenizer = RobertaTokenizer.from_pretrained("roberta_tokenizer")

example_texts = ["I believe in equality for all races.", "Immigrants are stealing our jobs."]

example_inputs = tokenizer(example_texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")

# Move input to the appropriate device
example_inputs = {key: val.to(device) for key, val in example_inputs.items()}

# Perform prediction
with torch.no_grad():
    outputs = model(**example_inputs)
    logits = outputs.logits
    predicted_classes = torch.argmax(logits, dim=1).cpu().numpy()

# Print predictions
for text, label in zip(example_texts, predicted_classes):
    print(f'Text: {text} - Predicted Class: {"Xenophobia" if label == 1 else "Racism"}')
