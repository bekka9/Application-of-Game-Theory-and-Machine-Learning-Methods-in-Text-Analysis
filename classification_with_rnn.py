import os
import zipfile
import pandas as pd
import numpy as np
import logging
from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, NewsSyntaxParser, Doc
from razdel import tokenize, sentenize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter
from time import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)

def split_long_text(text, max_length=1000):
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

def count_syllables(word):
    vowels = 'аеёиоуыэюя'
    return sum(1 for char in word.lower() if char in vowels) or 1

def get_text_stats(text):
    words = [t.text for t in tokenize(text)]
    sentences = [s.text for s in sentenize(text)]

    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)

    lemmas = []
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
        lemmas.append(token.lemma)

    return {
        'n_words': len(words),
        'n_sentences': len(sentences),
        'text': text  
    }

def extract_features(text):
    try:
        return get_text_stats(text)
    except Exception as e:
        logging.error(f"{str(e)}")
        return None

def process_data(data_path, max_text_length=1000):
    data = []
    labels = []
    levels = ['A1', 'A2', 'B1', 'B2', 'C1']

    for level in levels:
        level_dir = os.path.join(data_path, level)
        filenames = [f for f in os.listdir(level_dir) if f.endswith('.txt')]
        for filename in filenames:
            file_path = os.path.join(level_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    if not text:
                        continue
                    text_chunks = split_long_text(text, max_text_length)
                    for chunk in text_chunks:
                        features = extract_features(chunk)
                        if features:
                            data.append(features)
                            labels.append(level)
            except Exception as e:
                logging.error(f"Ошибка файла {filename}: {str(e)}")

    df = pd.DataFrame(data)
    df['label'] = labels
    return df

def tokenize_text(text):
    return [t.text.lower() for t in tokenize(text)]

def build_vocab(texts):
    all_tokens = [token for text in texts for token in tokenize_text(text)]
    token_freq = Counter(all_tokens)
    vocab = {token: idx + 2 for idx, (token, _) in enumerate(token_freq.items())}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    return vocab

def text_to_seq(text, vocab, max_len=100):
    tokens = tokenize_text(text)
    seq = [vocab.get(token, vocab['<UNK>']) for token in tokens[:max_len]]
    seq += [vocab['<PAD>']] * (max_len - len(seq))
    return seq

class TextDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.rnn(embedded)
        out = self.fc(hidden[-1])
        return out

def main():
    data_path = '/Users/mzsfleee/Downloads/ru_learning_data-plainrussian'
    df = process_data(data_path)
    
    vocab = build_vocab(df['text'])
    df['seq'] = df['text'].apply(lambda x: text_to_seq(x, vocab))

    le = LabelEncoder()
    df['label_id'] = le.fit_transform(df['label'])
    num_classes = len(le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(df['seq'].tolist(), df['label_id'].tolist(), test_size=0.2, stratify=df['label_id'], random_state=42)
    train_dataset = TextDataset(X_train, y_train)
    test_dataset = TextDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    
    model = RNNClassifier(vocab_size=len(vocab), embedding_dim=100, hidden_dim=128, output_dim=num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(5):
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            outputs = model(x_batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(y_batch.numpy())

    print(f"Accuracy: {accuracy_score(all_true, all_preds):.4f}")
    print("Classification Report:")
    print(classification_report(all_true, all_preds, target_names=le.classes_))

if __name__ == "__main__":
    main()


