import torch
from torchvision import datasets, transforms
import os
import torch.optim as optim
import torch.nn as nn
from resnet import ResNet50
import argparse

# Data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ]),

    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ]),

    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ]),
}


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=25, output_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_acc = 0.0  # Variable to store the best validation accuracy

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluation mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Save the best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                if output_dir:
                    torch.save(model.state_dict(), os.path.join(
                        output_dir, 'model.pth'))
                    # Message about saving the best model
                    print(f'Best model saved to {output_dir}')

        print()

    return model


def main():
    parser = argparse.ArgumentParser(
        description='Train a ResNet model for animal classification.')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the data directory')
    parser.add_argument('--num_epochs', type=int, default=25,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float,
                        default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--output_dir', type=str, default='./classification_model',
                        help='Directory to save the trained model')

    args = parser.parse_args()

    data_dir = args.data_dir
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    output_dir = args.output_dir

    sets = ["train", "val", "test"]

    # Load data and define transformations
    image_datasets = {x: datasets.ImageFolder(
        os.path.join(data_dir, x), data_transforms[x]) for x in sets}
    dataloaders = {x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0) for x in sets}
    dataset_sizes = {x: len(image_datasets[x]) for x in sets}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create ResNet50 model with 3 input channels and 10 output classes
    model = ResNet50(3, 10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    trained_model = train_model(
        model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs, output_dir)

    print(f'Training complete. Best model saved to {output_dir}')


if __name__ == "__main__":
    main()
