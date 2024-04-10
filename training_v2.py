from src import config as cfg
from src import utils as ut
from models import fcn_final as mdl
import torch
from torch import optim, nn
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import datetime
import os

warnings.filterwarnings('ignore')


def train_model(model, train_dataset, val_dataset, optimizer, criterion, writer, num_epochs):


    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs}')
        model.train()

        for batch_idx, (image, mask) in enumerate(train_dataset):
            image = image.to(cfg.DEVICE)
            mask = mask.to(cfg.DEVICE)

            optimizer.zero_grad()
            output_logits = model(image)

            mask_indices = torch.argmax(mask, dim=1) # Convert one-hot to indices
            loss = criterion(output_logits, mask_indices)
            loss.backward()
            optimizer.step()

            writer.add_scalar('Training Loss', loss.item(), epoch * len(train_dataset) + batch_idx)
            del loss, output_logits # To save memory

        print(
            f"GPU memory usage after epoch {epoch} in training: {torch.cuda.memory_allocated(cfg.DEVICE) / 1024 ** 2} MB")

        model.eval()
        total_loss = 0
        all_preds = []
        all_true = []

        with torch.no_grad():
            for batch_idx, (image, mask) in enumerate(val_dataset):
                image = image.to(cfg.DEVICE)
                mask = mask.to(cfg.DEVICE)

                output_logits = model(image)
                mask_indices = torch.argmax(mask, dim=1)
                loss = criterion(output_logits, mask_indices)
                total_loss = total_loss + loss.item()

                output_prob = torch.softmax(output_logits, dim=1)
                preds = torch.argmax(output_prob, dim=1)
                all_preds.extend(preds.detach().cpu().numpy().flatten())
                all_true.extend(mask_indices.detach().cpu().numpy().flatten())

                del loss, output_logits

        print(
            f"GPU memory usage after epoch {epoch} in validation: {torch.cuda.memory_allocated(cfg.DEVICE) / 1024 ** 2} MB")
        accuracy = accuracy_score(all_true, all_preds)
        precision = precision_score(all_true, all_preds, average='macro')
        recall = recall_score(all_true, all_preds, average='macro')
        f1 = f1_score(all_true, all_preds, average='macro')

        writer.add_scalar('Validation Loss', total_loss / len(val_dataset), epoch * len(val_dataset) + batch_idx)

        writer.add_scalar('Accuracy', accuracy, epoch)
        writer.add_scalar('Precision', precision, epoch)
        writer.add_scalar('Recall', recall, epoch)
        writer.add_scalar('F1 Score', f1, epoch)

def save_model_parameters(model, file_path):
    torch.save(model.state_dict(), file_path)


def visualize_predictions(image, mask, model):
    model.eval()
    with torch.no_grad():
        image = image.to(cfg.DEVICE)
        mask = mask.to(cfg.DEVICE)
        output_logits = model(image)
        mask_indices = torch.argmax(mask, dim=1)
        output_prob = torch.softmax(output_logits, dim=1)
        predicted_mask = torch.argmax(output_prob, dim=1)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(image.permute(0, 2, 3, 1).cpu().numpy()[0])
    ax[1].imshow(mask_indices.cpu().numpy()[0])
    ax[2].imshow(predicted_mask.cpu().numpy()[0])
    plt.show()

if __name__ == '__main__':

    transformations = transforms.Compose([transforms.ToTensor(),
                                          transforms.Resize((256, 256))])

    # Transformations only for images, so they are passed separately
    image_normalization = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])])

    train_dataset, val_dataset = ut.create_data_loader(batch_size=10,
                                                       transform=transformations,
                                                       normalize=image_normalization)
    num_epochs = 50
    model = mdl.fcn_model()
    model.to(cfg.DEVICE)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') # Used for tensorboard logging
    experiment_dir = os.path.join(cfg.EXPERIMENT_DIR, timestamp)

    # Redundant code, as the directory is created by SummaryWriter. Need to remove it.
    # if not os.path.exists(experiment_dir):
        # os.makedirs(experiment_dir)

    comment = input("Enter a name for this experiment: ")
    writer = SummaryWriter(experiment_dir, comment=comment)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=3e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_model(model, train_dataset, val_dataset, optimizer, criterion, writer, num_epochs)

    save_model = input("Do you want to save the model parameters? (yes/no): ")
    save_model = 'yes'
    if save_model.lower() == 'yes':
        save_model_parameters(model, f'model_parameters_{comment}.pth')
        save_model_parameters(model, 'model_parameters.pth')

    writer.close()

    visualize = input("Do you want to visualize predictions? (yes/no): ")
    visualize = 'yes'
    if visualize.lower() == 'yes':
        for image, mask in val_dataset:
            visualize_predictions(image, mask, model)
            break
