from models import original_fcn as mdl
from src import config as cfg
import torch
from segment_anything import SamPredictor, sam_model_registry
import matplotlib.pyplot as plt
import os
import numpy as np

def load_model(model_path):
    model = mdl.unet_model()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model.to(cfg.DEVICE)

def load_segment_anything_model(model_type = 'ViT-H', checkpoint = 'sam_vit_h_4b8939.pth'):
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(cfg.DEVICE)
    predictor = SamPredictor(sam)
    return predictor

def plot_mask_and_image(image, mask):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image.permute(0, 2, 3, 1).cpu().numpy()[0])
    ax[1].imshow(mask.cpu().numpy()[0])
    plt.show()

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

if __name__ =='__main__':
    model_path = 'model_parameters.pth'
    image_dir = 'data/dog/images'
    output_dir = 'results'

    image = os.listdir(image_dir)[0]
    # model = load_model(model_path)
    sam = load_segment_anything_model()
    sam.set_image(image)

    # plot_mask_and_image(images, sam.predict(images)

    input_point = np.array([[50, 75]])
    input_label = np.array([1])

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_points(input_point, input_label, plt.gca())
    plt.axis('on')
    plt.show()




