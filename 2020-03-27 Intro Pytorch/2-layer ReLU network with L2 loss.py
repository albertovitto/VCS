import torch
import random

random.seed(0)
torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in, device=device)  # input
y = torch.randn(N, D_out, device=device)  # ground truth, target
w1 = torch.randn(D_in, H, device=device, requires_grad=True)  # first layer weights
w2 = torch.randn(H, D_out, device=device, requires_grad=True)  # second layer weights

learning_rate = 1e-6
EPOCHS = 500

for e in range(EPOCHS):
    y_pred = x.mm(w1).clamp(min=0).mm(w2)  # clamp(min=0) is exactly ReLU
    loss = (y_pred - y).pow(2).sum()

    loss.backward()

    with torch.no_grad():  # disable gradient calculation
        # w1 = w1 - learning_rate * w1.grad NO IN-PLACE -> NOT WORKING
        # w2 = w2 - learning_rate * w2.grad
        # w1 -= learning_rate * w1.grad IN-PLACE -> WORKING
        # w2 -= learning_rate * w2.grad
        # https://discuss.pytorch.org/t/strange-problem-when-manually-gradient-descent/24138/2
        w1.sub_(learning_rate * w1.grad)
        w2.sub_(learning_rate * w2.grad)
        w1.grad.zero_()
        w2.grad.zero_()

print("Loss: ", loss)
