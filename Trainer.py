import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

def train_loop(model, loader, learning_rate, loss_fn=nn.CrossEntropyLoss(), optimizer=None): # i dont get what SGD is or does
  if optimizer is None:
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
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
  

def test_loop(model, loader, loss_fn=nn.CrossEntropyLoss()):
  size = len(loader.dataset)
  num_batches = len(loader)
  model.eval() # set the model into "testing" mode
  testing_loss = 0
  num_correct = 0
  with torch.no_grad(): # we don't need to gradient this (we're testign not training now)
    #lowkey try it without it too, maybe it improves over time?
    for X, y in loader:
      pred = model(X)
      testing_loss += loss_fn(pred, y).item()
      num_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
  
  testing_loss /= num_batches
  accuracy = num_correct / size

  print("Testing Accuracy", accuracy, "failed", testing_loss)


def train_network(model, epochs, train_loader, test_loader, model_save_file, learning_rate, loss_fn, optimizer):
  for i in range(epochs):
    print(f"Epoch {i+1}\n-----------------------")
    train_loop(model, train_loader, learning_rate, loss_fn, optimizer)
    test_loop(model, test_loader, loss_fn)

  torch.save(model, model_save_file)
  return model