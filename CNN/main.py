import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# LeNet5 from here:
# https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html

device = (
  "cuda" if torch.cuda.is_available()
  else
  "mps" if torch.backends.mps.is_available()
  else
  "cpu"
)

print(f"Using Device: {device}")

training_data = datasets.MNIST(
  root="FNN/data",
  train=True,
  download=True, 
  transform=ToTensor(),
)

testing_data = datasets.MNIST(
  root="FNN/data",
  train=False,
  download=True, 
  transform=ToTensor(),
)

train_loader = DataLoader(training_data)
test_loader = DataLoader(testing_data)

print(f"Training and Testing Data Loaded!")

class CNN(nn.Module):
  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.conv_layers = nn.Sequential(
      nn.Conv2d(1, 6, 5),
      nn.ReLU(),
      nn.MaxPool2d(2),
      nn.Conv2d(6, 16, 3),
      nn.ReLU(),
      nn.MaxPool2d(2),
      
    )

    self.dense_layers = nn.Sequential(
      nn.Linear(16*5*5, 120),
      nn.ReLU(),
      nn.Linear(120, 84),
      nn.ReLU(),
      nn.Linear(84, 10)
    )

  
  def forward(self, x):
    x = self.conv_layers(x)
    x = x.view(-1, self.num_flat_features(x))
    x = self.dense_layers(x)
    return x
  
  def num_flat_features(self, x):
    size = x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
    return num_features
  
model = CNN()

learning_rate = 1e-3
batch_size = 64
epochs = 5

loss_func = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def train_loop(loader, loss_fn=nn.CrossEntropyLoss(), optimizer=torch.optim.SGD(model.parameters(), lr=learning_rate)): # i dont get what SGD is or does
  size = len(loader.dataset)
  model.train()
  for batch, (X, y) in enumerate(loader):
    pred = model(X)
    loss = loss_fn(pred, y) # what does loss_fn return? what is the type of "loss"

    # Backprop from here
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if batch % 100 == 0:
      loss, current = loss.item(), batch * len(X)
      print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
  

def test_loop(loader, loss_fn=nn.CrossEntropyLoss()):
  size = len(loader.dataset)
  num_batches = len(loader)
  model.eval() # set the model into "testing" mode
  testing_loss = 0
  num_correct = 0
  with torch.no_grad(): # we don't need to gradient this (we're testign not training now)
    for X, y in loader:
      pred = model(X)
      testing_loss += loss_fn(pred, y).item()
      num_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
  
  testing_loss /= num_batches
  accuracy = num_correct / size

  print("Testing Accuracy", accuracy, "failed", testing_loss)


# Now actually perform the training and testing
for i in range(epochs):
  print(f"Epoch {i+1}\n-----------------------")
  train_loop(train_loader)
  test_loop(test_loader)


torch.save(model, 'CNN/model.pth')