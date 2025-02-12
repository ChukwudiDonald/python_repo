import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from utils import N_LETTERS
from utils import load_data,line_to_tensor,random_training_example

# ================================
# 🔧 HYPERPARAMETERS
# ================================
n_hidden = 128
num_layers = 3
learning_rate = 0.000125
n_iters = 100_000
plot_steps = 1_000
print_steps = 5_000

# ================================
# 🔧 DEVICE SETUP
# ================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ================================
# 🔧 RNN MODEL
# ================================
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.GRU(input_size, hidden_size, num_layers=self.num_layers)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

        self.to(device)  # Move model to GPU if available

    def forward(self, input_tensor, hidden_tensor):
        input_tensor = input_tensor.to(device)  # Move input to GPU
        hidden_tensor = hidden_tensor.to(device)  # Move hidden state to GPU

        y, hidden = self.rnn(input_tensor.view(1, 1, -1), hidden_tensor)  # Ensure correct shape
        y = self.out(y.view(1, -1))  # Reshape output
        y = self.softmax(y)

        return y, hidden

    def init_hidden(self):
        """Initialize hidden state on the correct device"""
        return torch.zeros(num_layers, 1, self.hidden_size, device=device)  # Move hidden state to GPU

# ================================
# 🔧 DATA LOADING
# ================================
category_lines, all_categories = load_data()
n_categories = len(all_categories)
rnn = RNN(N_LETTERS, n_hidden, n_categories)

# ================================
# 🔧 FUNCTION TO CONVERT OUTPUT TO CATEGORY
# ================================
def category_from_output(output):
    """Get category name from model output"""
    category_idx = torch.argmax(output).item()
    return all_categories[category_idx]

# ================================
# 🔧 TRAINING SETUP
# ================================
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

# ================================
# 🔧 TRAINING FUNCTION
# ================================
def train(line_tensor, category_tensor):
    line_tensor, category_tensor = line_tensor.to(device), category_tensor.to(device)  # Move tensors to GPU
    hidden = rnn.init_hidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return output, loss.item()

# ================================
# 🔧 TRAINING LOOP
# ================================
current_loss = 0
all_losses = []

for i in range(n_iters):
    category, line, category_tensor, line_tensor = random_training_example(category_lines, all_categories)

    output, loss = train(line_tensor, category_tensor)
    current_loss += loss

    if i == 0:
        print(f"{'Epochs':<8} | {'Loss':<6} | {'Name':<15} | {'Prediction':<12} | {'Result'}")

    if (i + 1) % plot_steps == 0:
        all_losses.append(current_loss / plot_steps)
        current_loss = 0

    if (i + 1) % print_steps == 0:
        guess = category_from_output(output)
        correct = f"CORRECT ({category})" if guess == category else f"WRONG ({category})"

        print(f"{round(i / n_iters * 100, 2):<8} | {round(loss, 3):<6} | {line:<15} | {guess:<12} | {correct}")


# ================================
# 🔧 PLOT TRAINING LOSS
# ================================
plt.figure()
plt.plot(all_losses)
plt.show()

# ================================
# 🔧 PREDICTION FUNCTION
# ================================
def predict(input_line):
    with torch.no_grad():
        line_tensor = line_to_tensor(input_line).to(device)  # Move input to GPU
        hidden = rnn.init_hidden()

        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)

        guess = category_from_output(output)
    return guess

# ================================
# 🔧 MODEL EVALUATION
# ================================
correct = 0
num_tests = 100_000

for _ in range(num_tests):
    category, name, *_ = random_training_example(category_lines, all_categories)
    prediction = predict(name)

    if prediction == category:
        correct += 1

    # print(name, category, prediction, sep="\t")

accuracy = correct / num_tests * 100

# Print total tests and accuracy
print(f"Total Tests: {num_tests}")
print(f"Model Accuracy: {accuracy:.2f}%")
