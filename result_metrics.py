import argparse
import os
import re
import shutil
import subprocess
import numpy as np
import torch
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as psnr
from lpips import LPIPS
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm



# Function to compute LPIPS between GT and Fake images
def calculate_lpips(gt_image, fake_image, lpips_model):
    gt_image = gt_image.unsqueeze(0)
    fake_image = fake_image.unsqueeze(0)

    lpips_score = lpips_model(gt_image, fake_image)
    return lpips_score.item()

# Function to compute PSNR between GT and Fake images
def calculate_psnr(gt_image, fake_image):
    return psnr(np.array(gt_image), np.array(fake_image))

# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--experiment', default=None, help='What experiment to score')

    opt = parser.parse_args()
    folder = "results/"+opt.experiment+"/test_latest/images/"

    #make two temp folders
    os.makedirs("gt", exist_ok=True)
    os.makedirs("fake", exist_ok=True)

    gt_folder = "gt"
    fake_folder = "fake"

    # Define the regex patterns
    real_b_pattern = r"real_B"
    fake_b_pattern = r"fake_B"

    counter = 0

    # Iterate through the files in the source folder
    for filename in os.listdir(folder):
        source_path = os.path.join(folder, filename)

        # Skip directories
        if os.path.isdir(source_path):
            continue

        # Copy and rename "real_B" images to the "gt" folder
        if re.search(real_b_pattern, filename):
            new_filename = re.sub(real_b_pattern, "", filename)
            destination_path = os.path.join(gt_folder, new_filename)
            shutil.copy(source_path, destination_path)
            counter += 1

        # Copy and rename "fake_B" images to the "fake" folder
        elif re.search(fake_b_pattern, filename):
            new_filename = re.sub(fake_b_pattern, "", filename)
            destination_path = os.path.join(fake_folder, new_filename)
            shutil.copy(source_path, destination_path)
            counter += 1

    print(f"Total number of images copied: {counter}, {counter//2} pairs")

    # Load the LPIPS model
    lpips_model = LPIPS(net='alex')  # You can choose 'vgg' or 'alex' based on preference
    
    
    # Load GT and Fake images
    gt_images = [Image.open(os.path.join(gt_folder, f)) for f in os.listdir(gt_folder)]
    fake_images = [Image.open(os.path.join(fake_folder, f)) for f in os.listdir(fake_folder)]

    # Ensure both folders contain the same number of images
    assert len(gt_images) == len(fake_images), "The number of GT and Fake images must be the same"
    
    # Compute PSNR, LPIPS for each image pair (GT vs Fake)
    psnr_values = []
    lpips_values = []
    for i, (gt, fake) in tqdm(enumerate(zip(gt_images, fake_images)), total=len(gt_images)):
        # Compute PSNR
        psnr_value = calculate_psnr(gt, fake)
        psnr_values.append(psnr_value)
        
        # Compute LPIPS
        gt_tensor = transforms.ToTensor()(gt)
        fake_tensor = transforms.ToTensor()(fake)
        lpips_value = calculate_lpips(gt_tensor, fake_tensor, lpips_model)
        lpips_values.append(lpips_value)
    
    # Calculate average PSNR and LPIPS
    avg_psnr = np.mean(psnr_values)
    avg_lpips = np.mean(lpips_values)

    print(f"Average PSNR: {avg_psnr}")
    print(f"Average LPIPS: {avg_lpips}")
    # Calculate FID between GT and Fake images
    command = ["python", "-m", "pytorch_fid", fake_folder, gt_folder]
    subprocess.run(command, check=True)

    # Clean up the temporary folders
    shutil.rmtree(gt_folder)
    shutil.rmtree(fake_folder)
