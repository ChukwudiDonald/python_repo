import torch.nn as nn
import torch
from tqdm import tqdm  # Ensure tqdm is imported
class AnnClass:
    def __init__(self, epochs=10000, learning_rate=0.0001):
        # Define the neural network model
        self.model = nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        # Loss function (Mean Squared Error)
        self.loss_function = nn.MSELoss()

        # Optimizer (Adam with a learning rate)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Number of epochs to train the model
        self.num_epoch = epochs

        # Placeholder to store losses for monitoring
        self.losses = torch.zeros(self.num_epoch)

        # Automatically use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self, x, y):
        # Move input data to the correct device (GPU/CPU)
        x, y = x.to(self.device), y.to(self.device)

        # Training loop
        for epoch in tqdm(range(self.num_epoch), desc="Training Progress", ncols=100):
            self.model.train()  # Set the model to training mode

            # Forward pass: Compute predicted y (yHat) by passing x through the model
            y_hat = self.model(x)

            # Calculate the loss by comparing predicted yHat to the actual y
            loss = self.loss_function(y_hat, y)

            # Store the loss for the current epoch
            self.losses[epoch] = loss.item()

            # Zero the gradients from the previous step
            self.optimizer.zero_grad()

            # Perform backpropagation to calculate the gradients
            loss.backward()

            # Update the model's parameters using the optimizer
            self.optimizer.step()

            # Optionally, print the loss at regular intervals (e.g., every 100 epochs)

    def evaluate(self, x_new):

        # Set the model to evaluation mode
        self.model.eval()  # Disables dropout, batch normalization, etc., specific to evaluation

        # Use torch.no_grad() to prevent gradient calculation during evaluation
        with torch.no_grad():  # Disable gradient calculation
            # Ensure x is a 2D tensor and of the correct dtype (float32)
            x_new = x_new.to(self.device).to(torch.float32)  # Move to correct device and ensure dtype

            # Get the model's prediction for the input x
            y_new = self.model(x_new)  # Perform the forward pass to get predictions

        # Return the predicted y_new (output)
        return y_new

    def evaluate_and_print_loss(self, x, y):

        # Set the model to evaluation mode
        self.model.eval()  # Disables dropout, batch normalization, etc., specific to evaluation

        # Use torch.no_grad() to prevent gradient calculation during evaluation
        with torch.no_grad():  # Disable gradient calculation
            # Ensure x is in float32 (for consistency with model expectations)

            x = x.to(self.device).to(torch.float32)

            # Get the model's prediction for the input x
            y_new = self.model(x)  # Perform the forward pass to get predictions

            # Calculate the MSE loss between predicted values (y_new) and true values (y)
            loss = self.loss_function(y_new, y)

        # Print the MSE loss
        print(f"Evaluation Loss (MSE): {loss.item()}")  # Print the MSE loss
        print()
        return y_new