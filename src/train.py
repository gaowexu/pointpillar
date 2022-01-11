import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


def train(dataloader, model, loss_fn, optimizer, device):
    """
    training module

    :param dataloader: training data loader
    :param model: customized model
    :param loss_fn: loss function handler
    :param optimizer: neural network learning optimizer
    :param device: device, "cpu" or "cuda"
    :return:
    """
    samples_amount = len(dataloader.dataset)
    model.train()
    for batch_idx, (input_image_batch, input_gt_batch) in enumerate(dataloader):
        input_image_batch = input_image_batch.to(device)
        input_gt_batch = input_gt_batch.to(device)

        # compute loss
        logits, probs = model(input_image_batch)
        loss = loss_fn(logits, input_gt_batch)

        # loss back-propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            loss, current = loss.item(), batch_idx * len(input_image_batch)
            print(f"loss: {loss:>7f}  [{current:>5d}/{samples_amount:>5d}]")


def test(dataloader, model, loss_fn, device):
    """
    test module

    :param dataloader: test data loader
    :param model: customized model
    :param loss_fn: loss function handler
    :param device: device, "cpu" or "cuda"
    :return:
    """
    samples_amount = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0.0, 0.0
    with torch.no_grad():
        for input_image_batch, input_gt_batch in dataloader:
            input_image_batch = input_image_batch.to(device)
            input_gt_batch = input_gt_batch.to(device)

            logits, probs = model(input_image_batch)
            test_loss += loss_fn(logits, input_gt_batch).item()

            correct += (probs.argmax(1) == input_gt_batch).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= samples_amount
    print(f"Test Performance: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# STEP 1: DEFINE DATASET
batch_size = 128
channels = 3
height = 224
width = 224

transform = transforms.Compose([                # image preprocess
    transforms.Resize(size=256),                # resize the short edge to 256 with its original height/width ratio
    transforms.CenterCrop(size=(height, width)),# (h, w)
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],             # RGB
        std=[0.229, 0.224, 0.225]               # RGB
    )
])

train_data = CustomImageDataset(images_root_dir="./scenes/seg_train",
                                transform=transform,
                                target_transform=None)
test_data = CustomImageDataset(images_root_dir="./scenes/seg_test",
                               transform=transform,
                               target_transform=None)

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

print("There are totally {} samples in training dataset.".format(len(train_data)))
print("There are totally {} samples in test dataset.".format(len(test_data)))
print("Training Batches = {}".format(len(train_dataloader)))
print("Test Batches = {}".format(len(test_dataloader)))

# STEP 2: DEFINE MODEL
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

custom_model = CustomClassificationModel().to(device)

# STEP 3: START TRAINING
adam_optimizer = torch.optim.Adam(custom_model.parameters(), lr=1e-4)
loss_handler = torch.nn.CrossEntropyLoss()

max_epochs = 5
for epoch_idx in range(max_epochs):
    print(f"Epoch {epoch_idx + 1}\n-------------------------------")
    train(train_dataloader, custom_model, loss_fn=loss_handler, optimizer=adam_optimizer, device=device)
    test(test_dataloader, custom_model, loss_fn=loss_handler, device=device)
print("Done!")




custom_model.eval()
inp = torch.rand(1, channels, height, width).to(device)
model_trace = torch.jit.trace(custom_model, inp)

# Save your model. The following code saves it with the .pth file extension
model_trace.save('scenes.pth')
print("Saved PyTorch Model State to scenes.pth")
