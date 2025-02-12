import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from utils import ALL_LETTERS,N_LETTERS
from utils import load_data, letter_to_tensor,line_to_tensor,random_training_example


class RNN(nn.Module):
    #nn.RNN

    def __init__(self, input_size, hidden_size, output_size, *args, **kwargs):
        super(RNN,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = 3  # 3-layer RNN

        self.rnn = nn.GRU(input_size, hidden_size, num_layers=self.num_layers)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, hidden_tensor):
        y, hidden = self.rnn(input_tensor.view(1, 1, -1), hidden_tensor)  # Ensures correct shape
        y = self.out(y.view(1, -1))  # Reshape output
        y = self.softmax(y)
        return y, hidden

    def init_hidden(self):
        return torch.zeros(3, 1, self.hidden_size)  # 3 layers, 1 sequence, hidden size


category_lines, all_categories = load_data()
n_categories = len(all_categories)
n_hidden = 256
rnn = RNN(N_LETTERS,n_hidden,n_categories)

def category_from_output(output):
    category_lines_idx = torch.argmax(output).item()
    return all_categories[category_lines_idx]


criterion = nn.NLLLoss()
learning_rate = 0.000125
optimizer = torch.optim.Adam(rnn.parameters(),lr=learning_rate)

def train(line_tensor, category_tensor):
    hidden = rnn.init_hidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output,category_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return output, loss.item()

current_loss = 0
all_losses = []
plot_steps, print_steps = int(1_000), int(5000)
n_iters = int(100_000)

for i in range(n_iters):
    category, line, category_tensor, line_tensor = random_training_example(category_lines,all_categories)

    output, loss = train(line_tensor,category_tensor)
    current_loss += loss

    if i == 0:
        print("{:<8} | {:<6} | {}".format("Epochs", "Loss", "Name / Prediction"))

    if(i+1) % plot_steps == 0:
        all_losses.append(current_loss/plot_steps)
        current_loss = 0

    if(i+1) % print_steps == 0:
        guess = category_from_output(output)
        correct = "CORRECT" if guess == category else f"WRONG ({category})"
        print("{:<8} | {:<6} | {}".format(round(i/n_iters*100,2), round(loss,3), f"{line} / {guess} {correct}"))

plt.figure()
plt.plot(all_losses)
plt.show()

def predict(input_line):

    with torch.no_grad():
        line_tensor  = line_to_tensor(input_line)

        hidden = rnn.init_hidden()

        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)


        guess = category_from_output(output)

    return guess


correct = 0
num_tests = 100_000

for _ in range(num_tests):
    category, name, *_ = random_training_example(category_lines, all_categories)
    prediction = predict(name)


    if prediction == category:
        correct += 1
    print(name,category,prediction,sep="\t")

accuracy = correct / num_tests * 100
print(f"Model Accuracy: {accuracy:.2f}%")
