import sys
sys.path.append('oid/src')
import openimages.download as oi
from src import config as cfg
import torch
import fiftyone as fo
import fiftyone.zoo as foz

def get_images(class_labels = ["Pizza", "Taxi", "Dog"]):

    oi.download_segmentation_dataset(dest_dir=cfg.DATA_DIR,
                       class_labels=class_labels,
                       annotation_format="pascal",
                       meta_dir="csv_data",
                       limit=1000)

    print('Images downloaded successfully!')

def get_zoo_images(class_labels = ['Pizza', 'Taxi', 'Dog']):
    dataset = foz.load_zoo_dataset(
                                "open-images-v7",
                            label_types=["segmentations", "classifications"],
                            classes=class_labels,
                            max_samples=1_000,
                            only_matching=True
                            )
    session = fo.launch_app(dataset)


if __name__ == "__main__":
    get_images()
    # get_zoo_images()