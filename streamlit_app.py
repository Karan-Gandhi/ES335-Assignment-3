import torch
import streamlit as st
from torch import nn
import torch.nn.functional as F
import json
import re

class NextWord(nn.Module):
    def __init__(self, block_size, vocab_size, embedding_dim, hidden_dim, activation_fn):
        super(NextWord, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lin1 = nn.Linear(embedding_dim * block_size, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, hidden_dim)
        self.lin4 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = activation_fn
        self.lin_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embeds = self.embedding(x)
        embeds = embeds.view(x.shape[0], -1)
        out = self.activation(self.lin1(embeds))
        out = self.activation(self.lin2(out))
        out = self.activation(self.lin3(out))
        out = self.activation(self.lin4(out))
        return self.lin_out(out)

def load_model(model_path, block_size, vocab_size, embedding_dim, hidden_dim, activation_fn):
    print(model_path)
    model = NextWord(block_size, vocab_size, embedding_dim, hidden_dim, activation_fn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()    
    return model

def load_vocab(corpus):
    stoi_path=f'stoi_{corpus}.json'
    itos_path=f'itos_{corpus}.json'
    with open(stoi_path, 'r') as f:
        stoi = json.load(f)
    with open(itos_path, 'r') as f:
        itos = json.load(f)
    return stoi, itos

def generate_sequence_sherlock(model, itos, stoi, context_words, block_size, max_len=20):
    unk_idx = stoi.get('<UNK>', 0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    context = [stoi.get(word, unk_idx) for word in context_words]
    if len(context) < block_size:
        pad_idx = stoi.get('<UNK>', unk_idx)
        context = [pad_idx] * (block_size - len(context)) + context
    sequence = context_words.copy()
    capitalize_next = False
    if sequence and sequence[-1].endswith('.'):
        capitalize_next = True
    with torch.no_grad():
        for _ in range(max_len):
            x = torch.tensor(context[-block_size:], dtype=torch.long).unsqueeze(0).to(device)
            y_pred = model(x)
            probs = F.softmax(y_pred, dim=1)
            ix = torch.multinomial(probs, num_samples=1).item()
            word = itos.get(str(ix), '<UNK>')
            is_punct = word in {'.', ',', '!', '?'}
            if capitalize_next and not is_punct and word != '<PAR_BREAK>':
                word = word.capitalize()
                capitalize_next = False
            if word == '<PAR_BREAK>':
                sequence.append('\n')
                capitalize_next = True
            else:
                sequence.append(word)
            context.append(ix)
            if word.endswith('.'):
                capitalize_next = True
    generated_text = ' '.join(sequence)
    generated_text = re.sub(r'\s+([.,!?])', r'\1', generated_text)
    print(generated_text)
    return generated_text


def generate_sequence_lyrics(model, itos, stoi, context_words, block_size, max_len=20):
    unk_idx = stoi.get('<UNK>', 0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    context = [stoi.get(word, unk_idx) for word in context_words]
    if len(context) < block_size:
        pad_idx = stoi.get('<UNK>', unk_idx)
        context = [pad_idx] * (block_size - len(context)) + context
    sequence = context_words.copy()
    capitalize_next = False
    if sequence and sequence[-1].endswith('.'):
        capitalize_next = True
    with torch.no_grad():
        for _ in range(max_len):
            x = torch.tensor(context[-block_size:], dtype=torch.long).unsqueeze(0).to(device)
            y_pred = model(x)
            probs = F.softmax(y_pred, dim=1)
            ix = torch.multinomial(probs, num_samples=1).item()
            word = itos.get(str(ix), '<UNK>')
            is_punct = word in {'.', ',', '!', '?'}
            if capitalize_next and not is_punct and word != '<PAR_BREAK>':
                word = word.capitalize()
                capitalize_next = False
            if word == '<PAR_BREAK>':
                sequence.append('\n')
                capitalize_next = True
            else:
                sequence.append(word)
            context.append(ix)
            if word.endswith('.'):
                capitalize_next = True
    generated_text = ' '.join(sequence)
    generated_text = re.sub(r'\s+([.,!?])', r'\1', generated_text)
    generated_text = re.sub(r"\s*'\s*", "'", generated_text)
    print(generated_text)
    return generated_text

def generate_sequence_c(model, itos, stoi, context_words, block_size, max_len=20):
    unk_idx = stoi.get('<UNK>', 0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    context = [stoi.get(word, unk_idx) for word in context_words]
    if len(context) < block_size:
        context = [unk_idx] * (block_size - len(context)) + context
    sequence = context_words.copy()
    generated_code = ''
    with torch.no_grad():
        for _ in range(max_len):
            x = torch.tensor(context[-block_size:]).unsqueeze(0).to(device)
            y_pred = model(x)
            probs = F.softmax(y_pred, dim=1)
            ix = torch.multinomial(probs, num_samples=1).item()
            word = itos.get(str(ix), '<UNK>')
            if word == '\n':
                generated_code += '\n'
            elif word == '\s':
                generated_code += ' '
            else:
                generated_code += word
            
            context.append(ix)
    
    return generated_code

def main():
    st.title("Next Word Prediction Streamlit App")

    corpus = st.selectbox("Select Corpus", ["sherlock", "c", "lyrics"])
    block_size = st.selectbox("Select Block Size", [5, 10, 15])
    embedding_dim = st.selectbox("Select Embedding Dimension", [64, 128])
    activation_fn_name = st.selectbox("Select Activation Function", ["ReLU", "Tanh"])
    activation_fn = nn.Tanh() if activation_fn_name == "Tanh" else nn.ReLU()

    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'stoi' not in st.session_state:
        st.session_state.stoi = None
    if 'itos' not in st.session_state:
        st.session_state.itos = None

    model_path = f'models/{corpus}_nextword_model_bs{block_size}_emb{embedding_dim}_act{activation_fn_name}.pth'

    if st.button("Load Model"):
        st.write(f"Loading model from: {model_path}")

        stoi, itos = load_vocab(corpus)
        hidden_dim = 256

        st.session_state.model = load_model(model_path, block_size, len(stoi) + 1, embedding_dim, hidden_dim, activation_fn)
        st.session_state.stoi = stoi
        st.session_state.itos = itos
        st.write("Model loaded successfully!")

    if st.session_state.model:
        input_text = st.text_input("Enter the text:", "")
        k = st.number_input("How many words to generate?", min_value=1, max_value=100, value=5)

        if st.button("Generate"):
            context_words = input_text.split() if input_text else []

            if corpus == 'sherlock':
                generated_text = generate_sequence_sherlock(st.session_state.model, st.session_state.itos, st.session_state.stoi, context_words, block_size, max_len=k)
            elif corpus == 'lyrics':
                generated_text = generate_sequence_lyrics(st.session_state.model, st.session_state.itos, st.session_state.stoi, context_words, block_size, max_len=k)
            elif corpus == 'c':
                generated_text = generate_sequence_c(st.session_state.model, st.session_state.itos, st.session_state.stoi, context_words, block_size, max_len=k)

            st.write(f"{generated_text}")
    else:
        st.write("Please load the model first to generate words.")

if __name__ == "__main__":
    main()
