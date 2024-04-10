from models import fcn_final as mdl
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import warnings
from src import config as cfg
from src import utils as ut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


warnings.filterwarnings('ignore')
def load_model(model_path):
    model = mdl.fcn_model()
    model.load_state_dict(torch.load(model_path))
    return model

def make_inferece(image_path, model):

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((224, 224)),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])
                                    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    model.eval()

    with torch.no_grad():
        output_logits = model(image)
        output_prob = torch.softmax(output_logits, dim=1)
        predicted_mask = torch.argmax(output_prob, dim=1)
    return image, predicted_mask

def visualize_predictions(image, predicted_mask):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image.permute(0, 2, 3, 1).cpu().numpy()[0])
    ax[1].imshow(predicted_mask.cpu().numpy()[0])
    plt.show()


def save_predictions(predicted_mask, output_path):
    mask = predicted_mask.cpu().numpy()[0]
    plt.imsave(output_path, mask)


def simple_prediction():
    images = os.listdir(image_dir)
    model = load_model(model_path)

    save_img = input('Do you want to save the predicted mask(s)? (yes/no): ')
    if save_img.lower() == 'yes':
        os.makedirs(output_dir, exist_ok=True)
        for idx, image in enumerate(images):
            image_path = os.path.join(image_dir, image)
            image, predicted_mask = make_inferece(image_path, model)
            output_path = os.path.join(output_dir, f'predicted_mask_{idx}.png')
            save_predictions(predicted_mask, output_path)
        print(f'Predicted mask(s) saved at {output_dir}')
    else:
        print('Predicted mask(s) not saved')
        for idx, image in enumerate(images):
            image_path = os.path.join(image_dir, image)
            image, predicted_mask = make_inferece(image_path, model)
            visualize_predictions(image, predicted_mask)
            break

def make_complete_predictions(data, model):

    model.eval()
    total_loss = 0
    all_preds = []
    all_true = []

    with torch.no_grad():
        for batch_idx, (image, mask) in enumerate(data):
            print(f'Batch: {batch_idx}/{len(data)}')
            image = image.to(cfg.DEVICE)
            mask = mask.to(cfg.DEVICE)

            output_logits = model(image)
            mask_indices = torch.argmax(mask, dim=1)

            output_prob = torch.softmax(output_logits, dim=1)
            preds = torch.argmax(output_prob, dim=1)
            all_preds.extend(preds.detach().cpu().numpy().flatten())
            all_true.extend(mask_indices.detach().cpu().numpy().flatten())

    accuracy = accuracy_score(all_true, all_preds)
    precision = precision_score(all_true, all_preds, average='macro')
    recall = recall_score(all_true, all_preds, average='macro')
    f1 = f1_score(all_true, all_preds, average='macro')

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')

if __name__ == '__main__':
    model_path = 'model_parameters_final_exp_v7_50_Epoch.pth'
    image_dir = cfg.TEST_DIR
    output_dir = 'results'

    transformations = transforms.Compose([transforms.ToTensor(),
                                          transforms.Resize((256, 256))])

    image_normalization = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])])

    data = ut.create_data_loader(batch_size=1,
                                 data_dir=cfg.TEST_DIR,
                                 classes=['Pizza', 'Taxi', 'Dog'],
                                 transform=transformations,
                                 normalize=image_normalization,
                                 create_data_split=False)

    model = load_model(model_path).to(cfg.DEVICE)
    make_complete_predictions(data, model)


