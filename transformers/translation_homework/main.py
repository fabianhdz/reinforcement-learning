#main.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from encoder import TransformerEncoder
from decoder import TransformerDecoder
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt


# Set the device to GPU if available, otherwise use CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

# Expanded training data pairs (English and Spanish sentences)
#Kaggle Data
file_path = "data.csv"
df = pd.read_csv(file_path)

file = "test_translation_data_100.csv"
df2 = pd.read_csv(file)

data = list(zip(df['english'], df['spanish']))

test_data = list(zip(df2['English'], df2['Spanish']))

# Define a simple tokenizer to convert words to indices and vice versa
class SimpleTokenizer:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0

    def build_vocab(self, sentences):

        unique_words = set(word for sentence in sentences for word in sentence.split())
        self.word2idx = {word: idx + 4 for idx, word in enumerate(unique_words)}
        self.word2idx["<pad>"] = 0
        self.word2idx["<sos>"] = 1
        self.word2idx["<eos>"] = 2
        self.word2idx["<unk>"] = 3
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

    def encode(self, sentence):

        return [self.word2idx.get(word, self.word2idx["<unk>"]) for word in sentence.split()]

    def decode(self, indices):

        return " ".join([self.idx2word.get(idx, "<unk>") for idx in indices])



english_tokenizer = SimpleTokenizer()
spanish_tokenizer = SimpleTokenizer()

english_sentences = [pair[0] for pair in data]
spanish_sentences = [pair[1] for pair in data]
max_len = 0
count =0

for sentence in english_sentences:
    count = len(sentence.split())
    if count > max_len:
        max_len = count

for sentence in spanish_sentences:
    count = len(sentence.split())
    if count > max_len:
        max_len = count
print(max_len)

english_tokenizer.build_vocab(english_sentences)
spanish_tokenizer.build_vocab(spanish_sentences)


def pad_sequence(seq, max_len, pad_idx=0):
    return seq + [pad_idx] * (max_len - len(seq))


def preprocess_data(data, max_len):
    english_data = []
    spanish_data = []

    for eng, spa in data:

        eng_tokens = [1] + english_tokenizer.encode(eng) + [2]  # <sos> ... <eos>
        spa_tokens = [1] + spanish_tokenizer.encode(spa) + [2]  # <sos> ... <eos>

        eng_tokens = pad_sequence(eng_tokens, max_len)
        spa_tokens = pad_sequence(spa_tokens, max_len)

        english_data.append(eng_tokens)
        spanish_data.append(spa_tokens)


    return torch.tensor(english_data), torch.tensor(spanish_data)




max_len += 2 # plus two for sos and eos
english_tensor, spanish_tensor = preprocess_data(data, max_len)


class TranslationDataset(Dataset):
    def __init__(self, src_data, tgt_data):
        self.src_data = src_data
        self.tgt_data = tgt_data

    def __len__(self):
        # Return the number of data pairs
        return len(self.src_data)

    def __getitem__(self, idx):

        return self.src_data[idx], self.tgt_data[idx][:-1], self.tgt_data[idx][1:]


# Create a DataLoader to iterate through the dataset
dataset = TranslationDataset(english_tensor, spanish_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True) # Batch size of 64


# Define the Transformer model
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim, num_heads, num_layers, hidden_dim, max_len):
        super(Transformer, self).__init__()
        # Initialize the encoder and decoder components
        self.encoder = TransformerEncoder(src_vocab_size, embed_dim, num_heads, num_layers, hidden_dim, max_len)
        self.decoder = TransformerDecoder(tgt_vocab_size, embed_dim, num_heads, num_layers, hidden_dim, max_len)
        # Output linear layer for vocab prediction
        self.output_layer = nn.Linear(embed_dim, tgt_vocab_size)
        self.device = device

    def forward(self, src, tgt):
        src = src.to(self.device)
        tgt = tgt.to(self.device)
        encoder_output = self.encoder(src)
        decoder_output = self.decoder(tgt, encoder_output)
        return self.output_layer(decoder_output)



def plot_loss(train_losses):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_losses, label="Training Loss", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.savefig("TrainingLoss.png")
    plt.show()

def plot_bleu(bleu_scores):
    epochs = range(1, len(bleu_scores) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, bleu_scores, label="Test BLEU Score", color="Orange")
    plt.xlabel("Epoch")
    plt.ylabel("BLEU Score")
    plt.title("Test BLEU Score Curve")
    plt.legend()

    plt.tight_layout()
    plt.savefig("BleuScores.png")
    plt.show()
# Calculate 1-gram BLEU score
def calculate_1gram_bleu(candidate, reference):
    """
    Calculate 1-gram BLEU score

    Args:
        candidate (str): Model generated translation
        reference (str): Reference translation

    Returns:
        float: 1-gram BLEU score
    """
    # Split into words
    candidate_tokens = candidate.lower().split()
    reference_tokens = reference.lower().split()

    # Count 1-grams for each sentence
    candidate_counts = Counter(candidate_tokens)
    reference_counts = Counter(reference_tokens)

    # Calculate matching 1-grams
    matches = sum(min(candidate_counts[word], reference_counts[word])
                  for word in candidate_counts)

    # Total number of generated words
    total_candidate_words = len(candidate_tokens)

    if total_candidate_words == 0:
        return 0.0

    # Calculate 1-gram precision
    bleu_1gram = matches / total_candidate_words

    return bleu_1gram


def evaluate_test_set(model, test_data, max_len=51):
    """
    Evaluate the model on the entire test dataset

    Args:
        model: Transformer model
        test_data: Test dataset
        max_len: Maximum sequence length

    Returns:
        float: Average 1-gram BLEU score
        list: List of (source, generated, reference, bleu_score) tuples
    """
    model.eval()
    total_bleu = 0.0
    results = []

    print("\nEvaluation on test set:")

    for src, tgt in test_data:
        translated = test(model, src, english_tokenizer, spanish_tokenizer, max_len)
        bleu_score = calculate_1gram_bleu(translated, tgt)
        total_bleu += bleu_score

        results.append((src, translated, tgt, bleu_score))

        print(f"Source: {src}")
        print(f"Generated: {translated}")
        print(f"Reference: {tgt}")
        print(f"1-gram BLEU: {bleu_score:.4f}\n")

    avg_bleu = total_bleu / len(test_data)
    print(f"Average 1-gram BLEU score: {avg_bleu:.4f}")

    return avg_bleu, results





# Set hyperparameters and instantiate the Transformer model
embed_dim = 512
num_heads = 8
num_layers = 4
hidden_dim = 2048

model = Transformer(
    src_vocab_size=english_tokenizer.vocab_size,
    tgt_vocab_size=spanish_tokenizer.vocab_size,
    embed_dim=embed_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    hidden_dim=hidden_dim,
    max_len=max_len
).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
optimizer = optim.Adam(model.parameters(), lr=0.0002)

# Define model save path
MODEL_PATH = "transformer_translation_model2.pth"


def train(model, dataloader, criterion, optimizer, num_epochs=100):
    """
    Model training function with evaluation on full test set at each epoch
    """
    best_bleu = 0.0
    train_losses = []
    bleu_scores = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for src, tgt_input, tgt_output in dataloader:
            optimizer.zero_grad()

            # Move tensors to device
            src = src.to(device)
            tgt_input = tgt_input.to(device)
            tgt_output = tgt_output.to(device)

            # Forward pass and loss calculation
            outputs = model(src, tgt_input)
            loss = criterion(outputs.view(-1, outputs.size(-1)), tgt_output.view(-1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_loss)

        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print(f'Average Loss: {avg_loss:.4f}')

        # Evaluate on full test set
        print("\nTesting on all test samples:")
        avg_bleu, _ = evaluate_test_set(model, test_data)

        bleu_scores.append(avg_bleu)
        # Save model if it achieves better BLEU score
        if avg_bleu > best_bleu:
            best_bleu = avg_bleu
            torch.save(model.state_dict(), MODEL_PATH)
            print(f'New best model saved with BLEU score: {best_bleu:.4f}')
        # Update plots every 20 epochs
        if (epoch + 1) % 20 == 0:
            plot_loss(train_losses)
            plot_bleu(bleu_scores)

        print('-' * 50)
    # Save the last model
    torch.save(model.state_dict(), "transformer_translation_final_model.pth")
    plot_loss(train_losses)
    plot_bleu(bleu_scores)



# Convert input sentence to tensor for model
def tokenize_input(sentence, tokenizer, max_len):
    tokens = tokenizer.encode(sentence)
    tokens = [1] + tokens + [2]  # Add <sos> and <eos> tokens
    tokens = pad_sequence(tokens, max_len)
    return torch.tensor(tokens).unsqueeze(0)  # Add batch dimension


# Test function for translating sentences
def test(model, input_sentence, english_tokenizer, spanish_tokenizer, max_len=51):
    model.eval()
    src_tensor = tokenize_input(input_sentence, english_tokenizer, max_len).to(device)

    with torch.no_grad():
        encoder_output = model.encoder(src_tensor)
        tgt_tokens = torch.tensor([[1]]).to(device)  # Start with <sos> token

        translated_tokens = []
        for _ in range(max_len):
            with torch.no_grad():
                output = model.decoder(tgt_tokens, encoder_output)
                output = model.output_layer(output)
            next_token = output[:, -1, :].argmax(1).item()
            if next_token == 2:  # Stop if <eos> token is generated
                break
            translated_tokens.append(next_token)
            tgt_tokens = torch.cat([tgt_tokens, torch.tensor([[next_token]]).to(device)], dim=1)

        translated_sentence = spanish_tokenizer.decode(translated_tokens)
        return translated_sentence


# Main execution
if __name__ == "__main__":
    # Train model
    print("Training model")
    train(model, dataloader, criterion, optimizer, num_epochs=100)
    print("Testing model")
    # Load best performing model
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device,weights_only=True))

    # Final evaluation
    print("\nFinal evaluation on test set:")
    final_bleu, final_results = evaluate_test_set(model, test_data)

    # Print summary of best and worst translations
    print("\nBest and Worst Translations:")
    sorted_results = sorted(final_results, key=lambda x: x[3], reverse=True)

    print("\nTop 3 Best Translations:")
    for src, gen, ref, bleu in sorted_results[:3]:
        print(f"Source: {src}")
        print(f"Generated: {gen}")
        print(f"Reference: {ref}")
        print(f"BLEU: {bleu:.4f}\n")

    print("\nTop 3 Worst Translations:")
    for src, gen, ref, bleu in sorted_results[-3:]:
        print(f"Source: {src}")
        print(f"Generated: {gen}")
        print(f"Reference: {ref}")
        print(f"BLEU: {bleu:.4f}\n")
