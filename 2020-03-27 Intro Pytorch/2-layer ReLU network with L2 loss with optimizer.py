import torch

torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in, device=device)  # input
y = torch.randn(N, D_out, device=device)  # ground truth, target

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H), torch.nn.ReLU(), torch.nn.Linear(H, D_out)
)

learning_rate = 1e-6
EPOCHS = 500
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for e in range(EPOCHS):
    y_pred = model(x)
    loss = torch.nn.functional.mse_loss(y_pred, y)

    loss.backward()

    # with torch.no_grad():  # disable gradient calculation
    #     for parameter in model.parameters():
    #         parameter -= learning_rate * parameter.grad
    #     model.zero_grad()

    optimizer.step()
    optimizer.zero_grad()


print("Loss: ", loss)
