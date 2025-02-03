
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

def create_data(m):
    n = 50
    x = torch.randn(n,1)
    y = m*x + torch.randn(n,1)/2

    return x,y

def build_and_train(x,y):
    ann_reg = nn.Sequential(
        nn.Linear(1, 17),
        nn.ReLU(),  # Using ReLU as the primary activation function
        nn.Linear(17, 19),
        nn.ReLU(),
        nn.Linear(19, 3),
        nn.ReLU(),
        nn.Linear(3, 1)
    )
    learning_rate = .0012
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(ann_reg.parameters(),lr=learning_rate)

    num_epoch = 500
    losses = torch.zeros(num_epoch)

    for epoch in range(num_epoch):

        y_hat = ann_reg(x)

        loss = loss_function(y_hat,y)
        losses[epoch] = loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    predictions = ann_reg(x)

    return predictions,losses

X,Y = create_data(.9)
yHat,losses = build_and_train(X,Y)

fig,ax = plt.subplots(1,2,figsize=(12,4))
ax[0].plot(losses.detach(),"o",markerfacecolor="w",linewidth= .1)
ax[0].set_xlabel("Epoch")
ax[0].set_title("Loss")

ax[1].plot(X,Y,"bo",label="Real data")
ax[1].plot(X,yHat.detach(),"rs",label="Predictions")
ax[1].set_xlabel("x")
ax[1].set_ylabel("y")
ax[1].set_title(f"Prediction-data corr = {np.corrcoef(Y.T,yHat.detach().T)[0,1]:.2f}")
ax[1].legend()

plt.show()

# slopes = np.linspace(-2,2,21)
#
# num_exp = 50
#
# results = np.zeros((len(slopes),num_exp,2))
#
# for slope_i in tqdm(range(len(slopes)), desc="Training Progress",ncols=100):
#
#     for N in range(num_exp):
#         X,Y = create_data(slopes[slope_i])
#         yHat,losses = build_and_train(X,Y)
#
#         results[slope_i,N,0] = losses[-1]
#         results[slope_i,N,1] = np.corrcoef(Y.T,yHat.detach().T)[0,1]
#
# results[np.isnan(results)] = 0
#
# fig,ax = plt.subplots(1,2,figsize=(12,4))
# ax[0].plot(slopes,np.mean(results[:,:,0], axis=1),"ko-",markerfacecolor="w",markersize= 10 ,linewidth= .1)
# ax[0].set_xlabel("Slope")
# ax[0].set_title("Loss")
#
# ax[1].plot(slopes,np.mean(results[:,:,1], axis=1),"ms-",label="Model Performance",markerfacecolor="w",markersize= 10 ,linewidth= .1)
# ax[1].set_xlabel("Slope")
# ax[1].set_ylabel("Real-predicted correlation")
# ax[1].set_title(f"Model Performance")
# ax[1].legend()
#
# plt.show()
