import os
import pickle
from typing import Dict, List

import PIL.Image
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
import numpy


class Images(Dataset):
    """Images dataset"""

    def __init__(self, image_list: List[str], transform):
        """
        Args:
            image_list (List[str]): List of image paths.
            transform: Transform to be applied on a sample.
        """
        self.image_list = image_list
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, idx: int) -> Dict:
        """
        Args:
            idx (int): Index of the image to get.

        Returns:
            Dict: Dictionary containing the image and its path.
        """
        image_path = self.image_list[idx]
        image = PIL.Image.open(image_path)
        image = self.transform(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        data = {"image": image, "img_path": image_path}
        return data

def collect_images_from_folder(folder_path: str) -> List[str]:
    """
    Collects all image paths recursively from a folder.

    Args:
        folder_path (str): Path to the folder containing images.

    Returns:
        List[str]: List of image file paths.
    """
    image_list = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_list.append(os.path.join(root, file))
    return image_list

def main():
    """Main function to process images and save embeddings"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print(f"Device used: {device}")

    folder_path = "/Users/saurabhkumarsingh/Projects/ImageSimilaritySearchApp/data/women_fashion"
    # folder_path = "/Users/saurabhkumarsingh/Projects/ImageSimilaritySearchApp/train"
    image_list = collect_images_from_folder(folder_path)
    print(f"Total images found: {len(image_list)}")

    print("Attempting to open images...")
    cleaned_image_list = []
    for image_path in image_list:
        try:
            PIL.Image.open(image_path)
            cleaned_image_list.append(image_path)
        except Exception as e:
            print(f"Failed to open {image_path} due to {e}")

    print(f"There are {len(cleaned_image_list)} images that can be processed")
    dataset = Images(cleaned_image_list, preprocess)

    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    print("Processing images...")
    image_paths = []
    embeddings = []
    for data in tqdm(dataloader):
        with torch.no_grad():
            X = data["image"].to(device)
            image_embedding = model.get_image_features(X).cpu().numpy()  # Convert to numpy array
            img_path = data["img_path"]
            image_paths.extend(img_path)
            embeddings.extend(image_embedding)

    image_embeddings = dict(zip(image_paths, embeddings))

    # Save to pickle file for the app
    print("Saving image embeddings")
    with open("embeddings.pkl", "wb") as f:
        pickle.dump(image_embeddings, f)


if __name__ == "__main__":
    main()
