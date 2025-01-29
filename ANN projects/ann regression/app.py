import matplotlib.pyplot as plt
from  AnnClass import AnnClass
from data import file_paths
import numpy as np
import helpers
import torch

# Read the CSV file with default encoding and delimiter
df = helpers.read_csv_with_default(file_path=file_paths[3], delimiter=';')

# Specify the columns to use from the DataFrame
data_column = ["Open", "High", "Low", "Close"]

# Define the number of samples for training
N = 10000

# Get the actual values from the DataFrame starting from row N+2
actual_candle = df[data_column].values.tolist()[N+2]

# Initialize empty lists to store predicted and actual values
predicted_candle = []


# Loop through the first 4 columns
for i in range(4):
    # Create input-output pairs using helper function
    x, y = helpers.create_input_output(torch.tensor(df[data_column[i]].values), N)

    # Create and train the model
    M = AnnClass(epochs=10000)
    M.train(x, y)

    # Prepare new input for prediction
    x_new = torch.tensor([[df[data_column[i]].iloc[N + 1]]], dtype=torch.float32)

    # Get prediction for new input
    y_new = M.evaluate(x_new)

    # Append the predicted value to the list
    predicted_candle.append(y_new.detach().item())

    # Evaluate model performance on training data and print loss
    predictions = M.evaluate_and_print_loss(x, y)

    time_sequence = np.arange(len(y))  # Generating a time sequence based on the length of y
    plt.plot(time_sequence, y, 'b-', label="RealData")  # Plotting real data as a blue line
    plt.plot(time_sequence, predictions.detach(), 'r-', label='Predictions')  # Plotting predictions as a red line
    plt.title(f"prediction-data r={np.corrcoef(y.T, predictions.detach().T)[0, 1]: .2f}")  # Display correlation coefficient in title
    plt.legend()  # Show legend for the real data and predictions over time
    plt.show()  # Show the time-based plot for real data vs predictions

helpers.plot_candlesticks(ac=actual_candle, pr=predicted_candle)
