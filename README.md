# CycleGAN for Monet-to-Photo Style Transfer

This project implements a CycleGAN to perform unpaired image-to-image translation between Monet paintings and real photographs, using the dataset from the Kaggle competition "I'm Something of a Painter Myself" (`gan-getting-started`). The notebook trains a CycleGAN model to generate Monet-style images from photos and vice versa, leveraging PyTorch for model implementation and training.

## Overview of GANs and CycleGAN Architecture

### What is a GAN?
A **Generative Adversarial Network (GAN)** consists of two neural networks:
- **Generator**: Creates fake data (e.g., images) to mimic the real data distribution.
- **Discriminator**: Distinguishes between real and fake data.
These networks are trained adversarially: the generator improves by trying to "fool" the discriminator, while the discriminator improves by better identifying fakes. This results in a generator that produces realistic outputs.

### CycleGAN Architecture
CycleGAN is a type of GAN designed for **unpaired image-to-image translation**, meaning it learns to translate images between two domains (e.g., Monet paintings and photos) without requiring paired examples. Key components include:

- **Two Generators**:
  - `gen_photo_to_monet`: Converts photos to Monet-style images.
  - `gen_monet_to_photo`: Converts Monet paintings to photo-like images.
  - Architecture:
    - **Initial Convolution**: Uses a 7x7 convolution with reflection padding to capture low-level features.
    - **Downsampling**: Two convolutional blocks (3x3 kernels, stride 2) to reduce spatial dimensions and increase feature depth (64 → 128 → 256 channels).
    - **Residual Blocks**: Nine residual blocks (each with two 3x3 convolutions and instance normalization) to learn complex transformations while preserving input structure.
    - **Upsampling**: Two transposed convolutional blocks (3x3 kernels, stride 2) to restore spatial dimensions (256 → 128 → 64 channels).
    - **Final Convolution**: A 7x7 convolution with `Tanh` activation to output a 3-channel RGB image.
- **Two Discriminators**:
  - `disc_monet`: Evaluates whether an image belongs to the Monet domain.
  - `disc_photo`: Evaluates whether an image belongs to the photo domain.
  - Architecture:
    - A series of convolutional blocks (4x4 kernels, stride 2) with increasing channels (3 → 64 → 128 → 256 → 512).
    - Uses batch normalization (except in the first layer) and LeakyReLU activations.
    - Outputs a 1-channel feature map for real/fake classification.
- **Loss Functions**:
  - **Adversarial Loss (MSE)**: Encourages generators to produce images that fool the discriminators.
  - **Cycle Consistency Loss (L1)**: Ensures that translating an image from one domain to another and back (e.g., photo → Monet → photo) reconstructs the original image.
  - **Identity Loss (L1)**: Preserves color and structure when an image is passed through its own domain's generator (e.g., Monet → Monet).
  - Combined loss: `loss_G = loss_GAN_monet + loss_GAN_photo + λ_cycle * loss_cycle + λ_identity * loss_identity`, where `λ_cycle = 10.0` and `λ_identity = 5.0`.

### Implementation in This Notebook
- **Dataset**: The `MonetPhotoDataset` class loads images from `monet_jpg` and `photo_jpg` directories, applies transformations (resize to 256x256, normalize to [-1, 1]), and pairs them randomly for unpaired training.
- **Training**:
  - Uses `DataLoader` with batch size 1 and 4 workers for efficient data loading.
  - Optimizers: Adam with learning rate 0.0002 and betas (0.5, 0.999).
  - Training loop:
    - Computes generator losses (adversarial, cycle, identity) and updates `gen_photo_to_monet` and `gen_monet_to_photo`.
    - Updates discriminators (`disc_monet`, `disc_photo`) using real and fake image losses.
    - Saves generated images every 100 batches as a ZIP file (`images.zip`) in the output directory.
    - Saves model weights (`gen_photo_to_monet.pth`, `gen_monet_to_photo.pth`) after training.
- **Kaggle Integration**: Downloads the `gan-getting-started` competition dataset using the Kaggle CLI and extracts `monet_jpg` and `photo_jpg` directories.

## Prerequisites
- Python 3.8+
- PyTorch (with CUDA support for GPU training)
- Kaggle CLI
- Required libraries: `numpy`, `torch`, `torchvision`, `pillow`, `kaggle`
  ```bash
  pip install torch torchvision pillow numpy kaggle
  ```

## Setup Instructions

### 1. Obtain Kaggle API Token
1. Go to your Kaggle account ([kaggle.com](https://www.kaggle.com)).
2. Navigate to **Settings** > **API** > **Create New API Token**.
3. Download the `kaggle.json` file (contains `username` and `key`).
4. Place `kaggle.json` in the same directory as the script or in:
   - `~/.kaggle/kaggle.json` (non-root users), or
   - `/root/.config/kaggle/kaggle.json` (root users).
5. Set file permissions:
   ```bash
   chmod 600 /path/to/kaggle.json
   ```
   Alternatively, set environment variables:
   ```bash
   export KAGGLE_USERNAME=your_kaggle_username
   export KAGGLE_KEY=your_kaggle_api_key
   ```

### 2. Clone the Repository
Clone the repository containing the script:
```bash
git clone <repository_url>
cd <repository_name>
```
Ensure `kaggle.json` is in the same directory as `train_cyclegan_script.py`, or use the paths above.

### 3. Install Dependencies
Install required Python packages:
```bash
pip install -r requirements.txt
```
If no `requirements.txt` exists, install manually:
```bash
pip install torch torchvision pillow numpy kaggle
```

### 4. Verify Kaggle CLI
Ensure the Kaggle CLI is installed and configured:
```bash
kaggle --version
```
If not installed, run:
```bash
pip install kaggle
```

## How to Use
1. **Prepare the Script**:
   - Save the notebook code as `train_cyclegan_script.py` (or use the provided script if already converted).
   - Ensure the `train_cyclegan` function and all supporting classes (`MonetPhotoDataset`, `Discriminator`, `Generator`, etc.) are included.

2. **Run the Script**:
   ```bash
   python train_cyclegan_script.py
   ```
   - The script will:
     - Download the `gan-getting-started` competition dataset to `./data`.
     - Extract `monet_jpg` and `photo_jpg` directories.
     - Train the CycleGAN model for 3 epochs (configurable in `main()`).
     - Save generated images to `./generated_images/images.zip` every 100 batches.
     - Save model weights to `./generated_images/gen_photo_to_monet.pth` and `gen_monet_to_photo.pth`.

3. **Output**:
   - Generated images are saved in `./generated_images/images.zip` with filenames like `epoch_X_batch_Y.jpg`.
   - Model checkpoints are saved as `.pth` files for future use.

## Notes
- **Dataset**: The script assumes the `gan-getting-started` dataset extracts to `data/monet_jpg` and `data/photo_jpg`. If the structure differs, update the `monet_dir` and `photo_dir` paths in `setup_kaggle_competition`.
- **GPU Support**: The script uses CUDA if available; otherwise, it falls back to CPU. For faster training, use a GPU-enabled environment.
- **Training Parameters**: Adjust `epochs`, `batch_size`, `lr`, and `beta1` in the `main()` function to experiment with training dynamics.
- **Troubleshooting**:
  - If the dataset download fails, ensure you’ve accepted the competition rules on Kaggle.
  - Check `kaggle.json` placement or environment variables if authentication errors occur.
  - Verify the extracted dataset structure if `monet_jpg` or `photo_jpg` directories are not found.

## Repository Structure
```
<repository_name>/
├── train_cyclegan_script.py  # Main script with CycleGAN implementation
├── kaggle.json               # Kaggle API token (place here or in ~/.kaggle/)
├── data/                     # Dataset download and extraction directory
└── generated_images/         # Output directory for generated images and model weights
```

## Future Improvements
- Add visualization of generated images during training.
- Implement learning rate scheduling for better convergence.
- Support loading pre-trained models for inference or fine-tuning.

For issues or contributions, please open a pull request or issue on the repository.
