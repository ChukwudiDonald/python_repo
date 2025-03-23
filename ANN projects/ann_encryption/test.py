import torch
import torch.nn as nn
import torch.optim as optim
import itertools

from cypher_net import CypherNet
from utils import char_to_tensor, char_to_target, alphabet

# Train Encrypt Model
def train_encrypt_model(model, lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Build training data once before training loop
    all_inputs = [char_to_tensor(ch) for ch in alphabet]
    all_targets = [char_to_target[ch] for ch in alphabet]
    inputs = torch.stack(all_inputs)
    targets = torch.tensor(all_targets)

    for epoch in itertools.count():
        # Vanilla backprop        
        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check for collisions 
        preds = torch.argmax(outputs, dim=1)
        collisions = len(preds) - len(torch.unique(preds))

        # Print progress for debugging
        if epoch % 100 == 0 or collisions == 0:
            print(f"[Encrypt] Epoch {epoch}: Collisions = {collisions}")

        if collisions == 0:
            print(f"✅ EncryptNet stabilized at epoch {epoch}")
            break

# Train Decrypt Model
def train_decrypt_model(encrypt_model, decrypt_model, lr=0.01):
    optimizer = optim.Adam(decrypt_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Prepare encrypted inputs and their corresponding target indices ONCE
    all_inputs = []
    all_targets = []

    for ch in alphabet:
        orig_input = char_to_tensor(ch).unsqueeze(0)
        with torch.no_grad():
            encrypted = encrypt_model(orig_input)  # Fixed encrypted version
        all_inputs.append(encrypted.squeeze(0))   # Remove batch dim
        all_targets.append(char_to_target[ch])    # Original target for character

    inputs = torch.stack(all_inputs)
    targets = torch.tensor(all_targets)

    for epoch in itertools.count():
        # Vanilla backprop
        outputs = decrypt_model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check accuracy in terms of whether each character is correctly decrypted
        acc = (torch.argmax(outputs, dim=1) == targets).float().mean().item()

        # Print progress for debugging
        if epoch % 100 == 0 or acc == 1.0:
            print(f"[Decrypt] Epoch {epoch}: Accuracy = {acc:.2%}")

        # Check if decryption is stable
        if acc == 1.0:
            print(f"✅ Decryption stabilized at epoch {epoch} with 100% accuracy.")
            break

# Stabilize both models
def stabilize_models(encrypt_model, decrypt_model):
    train_encrypt_model(encrypt_model)
    train_decrypt_model(encrypt_model, decrypt_model)



# Models
encrypt_model = CypherNet(n_chars=len(alphabet))
decrypt_model = CypherNet(n_chars=len(alphabet))

# Run stabilization
stabilize_models(encrypt_model, decrypt_model)


from utils import encrypt_text, decrypt_indices

# Example message
original_text = "hello my name is donald and I am a duck"

# Encrypt
encrypted_indices = encrypt_text(original_text, encrypt_model)
print("🔒 Encrypted indices:", encrypted_indices)

# Decrypt
decrypted_text = decrypt_indices(encrypted_indices, decrypt_model)
print("🔓 Decrypted text:", decrypted_text)
