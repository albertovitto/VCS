import torch

torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in, device=device)  # input
y = torch.randn(N, D_out, device=device)  # ground truth, target
# w1 = torch.randn(D_in, H, device=device, requires_grad=True)  # first layer weights
# w2 = torch.randn(H, D_out, device=device, requires_grad=True)  # second layer weights

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H), torch.nn.ReLU(), torch.nn.Linear(H, D_out)
)

learning_rate = 1e-6
EPOCHS = 500

for e in range(EPOCHS):
    # y_pred = x.mm(w1).clamp(min=0).mm(w2)
    y_pred = model(x)
    # loss = (y_pred - y).pow(2).sum()
    loss = torch.nn.functional.mse_loss(y_pred, y)

    loss.backward()

    with torch.no_grad():  # disable gradient calculation
        # w1.sub_(learning_rate * w1.grad)
        # w2.sub_(learning_rate * w2.grad)
        for parameter in model.parameters():
            parameter -= learning_rate * parameter.grad
        # w1.grad.zero_()
        # w2.grad.zero_()
        model.zero_grad()


print("Loss: ", loss)
