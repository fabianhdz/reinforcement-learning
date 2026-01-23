import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from encoder import TransformerEncoder
from decoder import TransformerDecoder

# Set the device to GPU if available, otherwise use CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Prepare translation data pairs (English and Spanish sentences).
data = [
    ["hello", "hola"],
    ["how are you", "cómo estás"],
    ["good morning", "buenos días"],
    ["thank you", "gracias"],
    ["see you later", "hasta luego"],
    ["good night", "buenas noches"],
    ["please", "por favor"],
    ["yes", "sí"],
    ["no", "no"],
    ["what is your name", "cuál es tu nombre"],
    ["my name is", "mi nombre es"],
]

# Define a simple tokenizer to convert words to indices and vice versa.
class SimpleTokenizer:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0

    def build_vocab(self, sentences):
        # Build a vocabulary from the provided sentences.
        # Each word is mapped to a unique index, starting from 4 (0-3 reserved for special tokens).
        unique_words = set(word for sentence in sentences for word in sentence.split())
        self.word2idx = {word: idx + 4 for idx, word in enumerate(unique_words)}
        self.word2idx["<pad>"] = 0
        self.word2idx["<sos>"] = 1
        self.word2idx["<eos>"] = 2
        self.word2idx["<unk>"] = 3
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

    def encode(self, sentence):
        # Convert a sentence into a list of token indices.
        return [self.word2idx.get(word, self.word2idx["<unk>"]) for word in sentence.split()]

    def decode(self, indices):
        # Convert a list of token indices back into a sentence.
        return " ".join([self.idx2word.get(idx, "<unk>") for idx in indices])

# Create tokenizers for English and Spanish languages.
english_tokenizer = SimpleTokenizer()
spanish_tokenizer = SimpleTokenizer()

# Extract English and Spanish sentences from the data and build vocabularies.
english_sentences = [pair[0] for pair in data]
spanish_sentences = [pair[1] for pair in data]

english_tokenizer.build_vocab(english_sentences)
spanish_tokenizer.build_vocab(spanish_sentences)

# Function to pad token sequences to a fixed length.
def pad_sequence(seq, max_len, pad_idx=0):
    # Add padding tokens (index 0) to reach the desired sequence length.
    return seq + [pad_idx] * (max_len - len(seq))

# Preprocess the data by tokenizing and padding sentences.
def preprocess_data(data, max_len):
    english_data = []
    spanish_data = []

    for eng, spa in data:
        # Encode sentences and add <sos> (1) and <eos> (2) tokens.
        eng_tokens = [1] + english_tokenizer.encode(eng) + [2]  # <sos> ... <eos>
        spa_tokens = [1] + spanish_tokenizer.encode(spa) + [2]  # <sos> ... <eos>

        # Pad sequences to the maximum length.
        eng_tokens = pad_sequence(eng_tokens, max_len)
        spa_tokens = pad_sequence(spa_tokens, max_len)

        english_data.append(eng_tokens)
        spanish_data.append(spa_tokens)

    # Convert the lists into PyTorch tensors.
    return torch.tensor(english_data), torch.tensor(spanish_data)

# Define the maximum sequence length for input data.
max_len = 10
english_tensor, spanish_tensor = preprocess_data(data, max_len)

# Define a dataset class for translation tasks.
class TranslationDataset(Dataset):
    def __init__(self, src_data, tgt_data):
        self.src_data = src_data
        self.tgt_data = tgt_data

    def __len__(self):
        # Return the number of data pairs.
        return len(self.src_data)

    def __getitem__(self, idx):
        # Return a source sentence, target input (without the last token), and target output (without the first token).
        return self.src_data[idx], self.tgt_data[idx][:-1], self.tgt_data[idx][1:]

# Create a DataLoader to iterate through the dataset.
dataset = TranslationDataset(english_tensor, spanish_tensor)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Define the Transformer model, which includes an encoder and decoder.
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim, num_heads, num_layers, hidden_dim, max_len):
        super(Transformer, self).__init__()
        # Initialize the encoder and decoder components.
        self.encoder = TransformerEncoder(src_vocab_size, embed_dim, num_heads, num_layers, hidden_dim, max_len)
        self.decoder = TransformerDecoder(tgt_vocab_size, embed_dim, num_heads, num_layers, hidden_dim, max_len)
        # Define an output linear layer that projects the decoder output to the target vocabulary size.
        self.output_layer = nn.Linear(embed_dim, tgt_vocab_size)
        # Store the device information for the model.
        self.device = device

    def forward(self, src, tgt):
        # Move input tensors to the appropriate device.
        src = src.to(self.device)
        tgt = tgt.to(self.device)
        # Pass the source input through the encoder.
        encoder_output = self.encoder(src)
        # Pass the encoder output and target input through the decoder.
        decoder_output = self.decoder(tgt, encoder_output)
        # Project the decoder output to the target vocabulary size.
        return self.output_layer(decoder_output)

# Set the hyperparameters and instantiate the Transformer model.
embed_dim = 512
num_heads = 8
num_layers = 2
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

# Define the loss function and optimizer.
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens when calculating loss.
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Define the file path for saving the model.
MODEL_PATH = "transformer_translation_model.pth"

# Define the training function.
def train(model, dataloader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode.
        epoch_loss = 0

        for src, tgt_input, tgt_output in dataloader:
            optimizer.zero_grad()  # Clear previous gradients.

            # Move input and output tensors to the appropriate device.
            src = src.to(device)
            tgt_input = tgt_input.to(device)
            tgt_output = tgt_output.to(device)

            # Pass the source and target input through the model.
            outputs = model(src, tgt_input)
            # Compute the loss between the model output and the target output.
            loss = criterion(outputs.view(-1, outputs.size(-1)), tgt_output.view(-1))
            loss.backward()  # Compute gradients.
            optimizer.step()  # Update model parameters.
            epoch_loss += loss.item()  # Accumulate the loss for the epoch.

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader):.4f}')

    # Save the trained model weights.
    torch.save(model.state_dict(), MODEL_PATH)
    print(f'Model saved to {MODEL_PATH}')

# Convert an input sentence into a tensor that can be used by the model.
def tokenize_input(sentence, tokenizer, max_len):
    tokens = tokenizer.encode(sentence)
    tokens = [1] + tokens + [2]  # Add <sos> ... <eos> tokens.
    tokens = pad_sequence(tokens, max_len)
    return torch.tensor(tokens).unsqueeze(0)  # Add a batch dimension.

# Define the function for testing and translating sentences using the model.
def test(model, input_sentence, english_tokenizer, spanish_tokenizer, max_len=10):
    model.eval()  # Set the model to evaluation mode.
    src_tensor = tokenize_input(input_sentence, english_tokenizer, max_len).to(device)

    with torch.no_grad():
        # Pass the source input through the encoder.
        encoder_output = model.encoder(src_tensor)
        # Start decoding with the <sos> token.
        tgt_tokens = torch.tensor([[1]]).to(device)  # <sos> token index.

        translated_tokens = []
        for _ in range(max_len):
            with torch.no_grad():
                # Decode step-by-step, passing the previous tokens to generate the next one.
                output = model.decoder(tgt_tokens, encoder_output)
                output = model.output_layer(output)
            # Get the index of the most likely next token.
            next_token = output[:, -1, :].argmax(1).item()
            if next_token == 2:  # Stop decoding if <eos> token is generated.
                break
            translated_tokens.append(next_token)
            # Add the predicted token to the sequence for the next step.
            tgt_tokens = torch.cat([tgt_tokens, torch.tensor([[next_token]]).to(device)], dim=1)

        # Decode the list of token indices into a translated sentence.
        translated_sentence = spanish_tokenizer.decode(translated_tokens)
        return translated_sentence

# Main function to train and test the model.
if __name__ == "__main__":
    # Train the model using the training function.
    train(model, dataloader, criterion, optimizer, num_epochs=10)

    # Load the saved model weights for testing.
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    print(f'Model loaded from {MODEL_PATH}')

    # Translate a test sentence using the trained model.
    input_sentence = "good night"
    translated_sentence = test(model, input_sentence, english_tokenizer, spanish_tokenizer, max_len=10)
    print(f'Input: {input_sentence}')
    print(f'Translation: {translated_sentence}')
