import torch
import string
import random

# characters set and count # of characters
alphabet = string.ascii_lowercase + " "
N_CHARS = len(alphabet)
char_to_idx = {ch: i for i, ch in enumerate(alphabet)}


# Random target mapping (shuffled indices)
shuffled_targets = list(range(N_CHARS))
random.shuffle(shuffled_targets)
char_to_target = {ch: shuffled_targets[i] for i, ch in enumerate(alphabet)}


def char_to_tensor(ch):
    tensor = torch.zeros(N_CHARS)
    tensor[char_to_idx[ch]] = 1.0
    return tensor


def encrypt_text(text, model):
    model.eval()
    encrypted_indices = []
    for ch in text:
        if ch in alphabet:  # Only encrypt characters in alphabet
            x = char_to_tensor(ch).unsqueeze(0)
            with torch.no_grad():
                out = model(x)
            encrypted_idx = torch.argmax(out).item()
            encrypted_indices.append(encrypted_idx)
    return encrypted_indices

def decrypt_indices(indices, model):
    model.eval()
    decrypted_text = ""
    for idx in indices:
        one_hot = torch.zeros(N_CHARS).unsqueeze(0)
        one_hot[0, idx] = 1.0  # Feed encrypted index as one-hot
        with torch.no_grad():
            out = model(one_hot)
        decoded_idx = torch.argmax(out).item()
        decoded_char = list(char_to_target.keys())[list(char_to_target.values()).index(decoded_idx)]
        decrypted_text += decoded_char
    return decrypted_text
