import torch
from torch.utils.tensorboard import SummaryWriter

from load import landmarks
from torch import nn, hub
from torch.utils.data import DataLoader


train_data = landmarks("archive/CAT_00")
for i in range(1, 6):
    train_data = train_data + landmarks("archive/CAT_0" + str(i))
test_data = landmarks("archive/CAT_06")

train_loader = DataLoader(train_data, batch_size = 32, shuffle = True)
test_loader = DataLoader(test_data, batch_size = 32, shuffle = True)

print(len(train_data), len(train_loader))

model = hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)

model.classifier.add_module("linear",nn.Sequential(
    nn.Linear(1000,128),
    nn.ReLU(),
    nn.Linear(128,64),
    nn.ReLU(),
    nn.Linear(64,18)
))

''' Load the trained model '''
# model = torch.load("models/Cats_landmarks_*.pth")
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

writer = SummaryWriter("logs_train")
epochs = 50
total_train_steps = 0

for epoch in range(epochs):
    print(f"------Test Round #{epoch}------")

    model.train()
    for i, data in enumerate(train_loader):
        imgs, labels = data
        outputs = model(imgs)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_steps = total_train_steps + 1
        writer.add_scalar("train_loss", loss.item(),total_train_steps)
        print(outputs,outputs.shape,loss.item(),f"train_cases {i}")

    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, labels = data
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            total_loss = loss + total_loss
    print(f"total loss:{total_loss.item()}")
    writer.add_scalar("test_loss", total_loss.item(), epoch + 1)

    torch.save(model, f"models2/Cats_landmarks_{epoch}.pth")
