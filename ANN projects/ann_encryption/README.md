
# 🔐 Neural Network Encryption-Decryption System (Neural Cipher)

This project demonstrates a **simple encryption system** using two artificial neural networks:
- An **encryption neural network** that maps a character (e.g. `'a'`, `'b'`, ...) to a new encoded representation.
- A **decryption neural network** that learns to reverse the mapping and retrieve the original character.

The goal is to explore how **neural networks can simulate encryption-like behavior**, and build a foundation for more advanced learning-based cryptographic schemes.

---

## 🚀 How It Works

1. **One-Hot Encoding**  
   Each lowercase letter (`a-z`) is converted to a 26-dimensional one-hot tensor.

2. **encryption neural network (Encryption Phase)**  
   The one-hot vector is passed through a feedforward network, which transforms it into a new representation.  
   After training (via a "stabilization" phase), the network learns to map each character to a **unique output**.

3. **decryption neural network (Decryption Phase)**  
   Another model is trained to **decode** the encrypted outputs back to the original characters.

4. **Training Objectives**:
   - Ensure **no two characters collide** during encryption (unique outputs).
   - Train the decryption model to correctly **recover all original characters** with 100% accuracy.

---

## 📦 Project Structure

```
.
├── cypher_net.py             # Single model definition (CypherNet), used for both encryption & decryption
├── utils.py                  # Helper functions: char_to_tensor, random mapping, encryption/decryption utils
├── test.py                   # Runs model stabilization, trains both models, and tests encryption/decryption
├── README.md                 # Project overview, usage, concept, and limitations
```

---

## 📈 Key Features

- ✅ Character-level encryption with neural networks
- ✅ Unique output enforcement (via collision minimization or cross-entropy loss)
- ✅ Fully trainable decryption model (no lookup tables)
- ✅ Demonstrates the potential of **differentiable cryptography**

---

## ⚠️ Limitations & Drawbacks

1. **Not Secure for Real Use**  
   This is a conceptual demo. Neural network weights are static and easily reversible. It **does not provide cryptographic security**.

2. **Initial Synchronization Required**  
   Initial synchronization is required to train and align encryption and decrypt network, after which only the sender uses the encryptor and the receiver uses the decryptor.
   
3. **Mapping Knowledge is Required**  
   Decryption only works if the decryption model is trained with:
   - The original character set (a–z)
   - The specific mapping used by `encryption neural network`

4. **Overfitting Risk**  
   With small vocabularies, models can easily overfit rather than learn generalizable encryption patterns.

5. **Black-Box Nature**  
   Compared to traditional ciphers (e.g., Caesar, RSA), it’s harder to interpret the transformation done by the model.

---

## 🧠 Why Is This Useful?

- 🔬 **Explores new territory** in learning-based encryption
- 🔁 Could be integrated with **adversarial training** or **key exchange protocols**
- 🤖 Demonstrates how **neural networks can simulate logic-based systems**
- 🛠 Could be expanded into multi-layer message obfuscation
- 🔐 Potential foundation for differentiable steganography or learnable key-based encryption
- 🎓 Great for teaching how AI models can learn injective mappings
- 🧩 Opens the door for **neural cipher research** in hybrid cryptographic-AI domains

---



# 🧠 Neural Cipher: Message Encryption Use Case

## 🔐 Objective

This section provides a simple, non-technical explanation of how the neural cipher works in practice.

---

## 🔐 Encryption (Sender Side)

- You want to send a message like `"hi"`.
- After the networks have been stabilized (i.e., the encryption and decryption models have been trained together), the **encryption neural network** converts each character (`"h"` and `"i"`) into unique encrypted vectors — numerical outputs that appear as random noise to anyone without the decryption model.
- These vectors are meaningless without the trained decryption counterpart and do not resemble the original characters.

---

## 📤 Transmission

- Instead of transmitting the actual message (`"hi"`), you send the encrypted vectors:  
  For example: `[vector_1, vector_2]`
- These vectors represent the encrypted form of `"h"` and `"i"` and can be safely transmitted.

---

## 🔓 Decryption (Receiver Side)

- The receiver uses the **decryption neural network**.
- It takes in the encrypted vectors and correctly maps them back to the original characters.
- Output: `"h"` and `"i"`

---

## ✅ Result

The receiver successfully reconstructs the original message:

```
Original:   "hi"
Encrypted:  [vector_1, vector_2]
Decrypted:  "hi"
```


# 🔐 Neural Cipher vs Caesar Cipher

## 🔁 Comparison Table

| Feature                       | **Caesar Cipher**                         | **Neural Cipher**                                     |
|-------------------------------|-------------------------------------------|-------------------------------------------------------|
| **Encryption Type**           | Substitution with fixed shift             | Neural network–based nonlinear transformation         |
| **Key**                       | Single integer shift (e.g., +3)           | Trained encryption & decryption models                |
| **Security Level**            | Very low (easily brute-forced)            | Moderate to high (if trained and obfuscated properly) |
| **Scalability**               | Limited to short texts and alphabets      | Can scale to full vocabularies and sentences          |
| **Flexibility**               | Only shifts characters in order           | Can learn complex, dynamic mappings                   |
| **Required Setup**            | Just agree on a shift                     | Requires model training and synchronization           |
| **Performance**               | Fast and deterministic                    | Slower, requires model inference                      |
| **Cryptanalysis Resistance**  | Very weak (frequency analysis works)      | Harder to crack without knowing model weights         |
| **Obfuscation**               | Predictable (same letter → same output)   | One-to-many mapping is possible, less predictable     |
| **Input Type**                | Only characters                           | Can be extended to numbers, images, and more          |

---

## 🥇 Which is Better?

### ✅ **Neural Cipher is better** **in modern and secure communication contexts**, because:

1. **Harder to Reverse Engineer**  
   - Unlike Caesar, the mapping from input → output is **nonlinear** and model-dependent.

2. **Customizable & Extensible**  
   - You can add layers, encodings, randomness, or even use sequence models (RNNs, Transformers).

3. **Not Limited to Characters**  
   - Can be extended to encrypt numbers, audio, or even pixel data in images.

4. **Dynamic Encryption**  
   - Unlike Caesar’s static nature, Neural Ciphers can generate **context-aware** outputs.

---

## 🚫 Limitations of Neural Cipher

- Requires more **computation and storage**.
- Needs **initial secure synchronization** of the models.
- Harder to **verify correctness manually** (no simple "shift by 3").

---

## 🔚 Conclusion

| If your goal is:                          | Use:             |
|-------------------------------------------|------------------|
| Simple education/demo of encryption       | Caesar Cipher    |
| Custom, AI-powered, flexible encryption   | Neural Cipher ✅ |

## 👨‍🔬 Author
> Concept by Okereke Chukwudi Donald  
> AI-Driven Encryption Prototype | Built with PyTorch  
