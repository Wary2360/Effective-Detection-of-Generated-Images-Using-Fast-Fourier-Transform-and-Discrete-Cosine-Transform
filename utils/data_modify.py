import os
from tqdm import tqdm
from glob import glob
import random
import cv2
from PIL import Image
import numpy as np
import albumentations as A
import shutil

"""
You should define seed in each function
"""

def get_image_sizes(image_path):
    image_sizes = []

    for filename in os.listdir(f'{image_path}/'):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = cv2.imread(os.path.join(image_path, filename))
            image_sizes.append(img.shape[:2])  # Height, Width 
    image_sizes = np.array(image_sizes)

    # Print all shape
    print(image_sizes)
    # Calculate and print statistics
    print("Average image size: ", np.mean(image_sizes, axis=0))
    print("Median image size: ", np.median(image_sizes, axis=0))
    print("Minimum image size: ", np.min(image_sizes, axis=0))
    print("Maximum image size: ", np.max(image_sizes, axis=0))

def data_creation(image_path, save_path):
    real_images = glob(f'{image_path}/real_images/*.png')
    fake_images = glob(f'{image_path}/fake_images/*.png')
    
    # Limit to 2000 images each, chosen randomly
    real_images = random.sample(real_images, 2000)
    fake_images = random.sample(fake_images, 2000)

    os.makedirs(f'{save_path}/real_images', exist_ok=True)
    os.makedirs(f'{save_path}/fake_images', exist_ok=True)

    transforms = A.Compose([
        A.ColorJitter(brightness=0.5, saturation=0.5, hue=0.5, p=1),
        A.Resize(224, 224)
    ])

    print("===========Real Image generating===========")
    # Train real image transforms
    for img_path in tqdm(real_images):
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        img = transforms(image=img)["image"]

        img = Image.fromarray(img)
        # Extract the suffix from the original image filename
        suffix = os.path.basename(img_path).split('_')[-1].split('.')[0]
        img.save(f'{save_path}/real_images/aug_real_{suffix}.png')

    print("===========Fake Image generating===========")
    # Train fake image transforms
    for img_path in tqdm(fake_images):
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        img = transforms(image=img)["image"]
        img = Image.fromarray(img)
        # Extract the suffix from the original image filename
        suffix = os.path.basename(img_path).split('_')[-1].split('.')[0]
        img.save(f'{save_path}/fake_images/aug_fake_{suffix}.png')

def create_train_patched_images(input_folder_path, output_folder_path, patch_size=(224, 224), sample_size=10000):
    np.random.seed(42)
    random.seed(42)

    real_image_files = glob(os.path.join(input_folder_path, "0_train_real_images/*.[pP][nN][gG]"))
    real_image_files += glob(os.path.join(input_folder_path, "0_train_real_images/*.[jJ][pP][gG]"))

    fake_image_files = glob(os.path.join(input_folder_path, "1_train_fake_images/*.[pP][nN][gG]"))
    fake_image_files += glob(os.path.join(input_folder_path, "1_train_fake_images/*.[jJ][pP][gG]"))
    
    # Randomly sample images
    if len(real_image_files) > sample_size and len(fake_image_files) > sample_size:
        real_image_files = random.sample(real_image_files, sample_size)
        fake_image_files = random.sample(fake_image_files, sample_size)

    # Processing real images
    print("===========Real Image generating===========")
    os.makedirs(os.path.join(output_folder_path, 'real_images'), exist_ok=True)
    for image_file in tqdm(real_image_files):
        image = Image.open(image_file)
        w, h = image.size
        output_file_base_name = os.path.basename(image_file).split('.')[0]

        # If the image is smaller than the patch size, resize and save
        if h <= patch_size[1] or w <= patch_size[0]:
            resized_image = image.resize(patch_size)
            output_file_path = os.path.join(output_folder_path, 'real_images', f'{output_file_base_name}_resized.png')
            resized_image.save(output_file_path)
            continue

        # If the image is larger than the patch size, extract patches
        i = 0
        for top in range(0, h, patch_size[1]):
            for left in range(0, w, patch_size[0]):
                right = min(w, left + patch_size[0])
                bottom = min(h, top + patch_size[1])
                
                if (right - left) == patch_size[0] and (bottom - top) == patch_size[1]:
                    patch = image.crop((left, top, right, bottom))
                    output_file_path = os.path.join(output_folder_path, 'real_images', f'{output_file_base_name}_patch_{i}.png')
                    patch.save(output_file_path)
                    i += 1

    # Processing fake images
    print("===========Fake Image generating===========")
    os.makedirs(os.path.join(output_folder_path, 'fake_images'), exist_ok=True)
    for image_file in tqdm(fake_image_files):
        image = Image.open(image_file)
        w, h = image.size
        output_file_base_name = os.path.basename(image_file).split('.')[0]

        # If the image is smaller than the patch size, resize and save
        if h <= patch_size[1] or w <= patch_size[0]:
            resized_image = image.resize(patch_size)
            output_file_path = os.path.join(output_folder_path, 'fake_images', f'{output_file_base_name}_resized.png')
            resized_image.save(output_file_path)
            continue

        # If the image is larger than the patch size, extract patches
        i = 0
        for top in range(0, h, patch_size[1]):
            for left in range(0, w, patch_size[0]):
                right = min(w, left + patch_size[0])
                bottom = min(h, top + patch_size[1])
                
                if (right - left) == patch_size[0] and (bottom - top) == patch_size[1]:
                    patch = image.crop((left, top, right, bottom))
                    output_file_path = os.path.join(output_folder_path, 'fake_images', f'{output_file_base_name}_patch_{i}.png')
                    patch.save(output_file_path)
                    i += 1

def create_test_patched_images(input_image_path, save_path, crop_size=(224, 224), n_patch=5):
    # Image loading
    test_images = glob(f'{input_image_path}/*.png')

    for i in range(n_patch):
        os.makedirs(f'{save_path}/images_{i}', exist_ok=True)

    # Define transforms
    center_crop = A.CenterCrop(*crop_size, p=1.0)
    random_crop = A.RandomCrop(*crop_size, p=1.0)
    resize_transform = A.Resize(*crop_size, p=1.0)

    print("===========Test Image generating===========")
    # Test image transforms
    for img_path in tqdm(test_images):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, _ = img.shape

        # Check image size and apply appropriate transform
        if h > crop_size[0] and w > crop_size[1]:
            transformed = center_crop(image=img)
            transformed_image = cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(save_path, f'images_0', os.path.basename(img_path)), transformed_image)
            for i in range(1, n_patch):
                np.random.seed(i)
                random.seed(i)
                transformed = random_crop(image=img)
                transformed_image = cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(save_path, f'images_{i}', os.path.basename(img_path)), transformed_image)
        elif h <= crop_size[0] or w <= crop_size[1]:
            for i in range(n_patch):
                transformed = resize_transform(image=img)
                transformed_image = cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(save_path, f'images_{i}', os.path.basename(img_path)), transformed_image)

def create_train_dct_images(input_image_path, save_path):
    real_image_files = glob(os.path.join(input_image_path, "real_images/*.[pP][nN][gG]")) 
    fake_image_files = glob(os.path.join(input_image_path, "fake_images/*.[pP][nN][gG]"))

    # Define subfolders for saving DCT images
    real_dct_folder = os.path.join(save_path, 'real_images')
    fake_dct_folder = os.path.join(save_path, 'fake_images')
    
    os.makedirs(real_dct_folder, exist_ok=True)
    os.makedirs(fake_dct_folder, exist_ok=True)
    
    # Processing real images
    print("===========Train Real Image DCT generating===========")
    for image_file in tqdm(real_image_files):
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Perform DCT on each channel of the image
        dct = np.zeros_like(image, dtype=np.float32)
        for channel in range(3):
            dct[:, :, channel] = cv2.dct(np.float32(image[:, :, channel]))/255.0
            
        output_file_base_name = os.path.basename(image_file).split('.')[0]
        output_file_path = os.path.join(real_dct_folder, f'{output_file_base_name}_dct.png')
        cv2.imwrite(output_file_path, dct * 255)  # Save the DCT image

    # Processing fake images
    print("===========Train Fake Image DCT generating===========")
    for image_file in tqdm(fake_image_files):
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Perform DCT on each channel of the image
        dct = np.zeros_like(image, dtype=np.float32)
        for channel in range(3):
            dct[:, :, channel] = cv2.dct(np.float32(image[:, :, channel]))/255.0
            
        output_file_base_name = os.path.basename(image_file).split('.')[0]
        output_file_path = os.path.join(fake_dct_folder, f'{output_file_base_name}_dct.png')
        cv2.imwrite(output_file_path, dct * 255)  # Save the DCT image

def create_test_dct_images(input_folder_path, save_folder_path, n_patch):
    for i in range(n_patch):
        # Get image files
        image_files = glob(os.path.join(input_folder_path, f"images_{i}/*.[pP][nN][gG]"))
        # Create save directory if not exists
        save_dir = os.path.join(save_folder_path, f'images_{i}')
        os.makedirs(save_dir, exist_ok=True)
        
        # Processing images
        print(f"===========Test DCT Image generating for patch {i+1}/{n_patch}===========")
        for image_file in tqdm(image_files):
            # Load and convert the image
            image = cv2.imread(image_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Perform DCT on each channel of the image
            dct = np.zeros_like(image, dtype=np.float32)
            for channel in range(3):
                dct[:, :, channel] = cv2.dct(np.float32(image[:, :, channel])/255.0)
            # Save the DCT transformed image
            output_file_base_name = os.path.basename(image_file).split('.')[0]
            output_file_path = os.path.join(save_dir, f'{output_file_base_name}.png')
            cv2.imwrite(output_file_path, dct * 255)  # Save the DCT image

def combine_train_datasets(input_image_path_1, input_image_path_2, save_path):
    image_files_1_real = glob(os.path.join(input_image_path_1, "real_images/*.[pP][nN][gG]"))
    image_files_2_real = glob(os.path.join(input_image_path_2, "real_images/*.[pP][nN][gG]"))

    # image_files_1_fake = glob(os.path.join(input_image_path_1, "fake_images/*.[pP][nN][gG]"))
    image_files_2_fake = glob(os.path.join(input_image_path_2, "fake_images/*.[pP][nN][gG]"))

    # Combining real and fake images
    real_images = image_files_1_real + image_files_2_real
    # fake_images = image_files_1_fake + image_files_2_fake
    fake_images = image_files_2_fake

    # Processing real images
    print("===========Real Image Combining===========")
    os.makedirs(os.path.join(save_path, 'real_images'), exist_ok=True)
    for i, image_file in enumerate(tqdm(real_images)):
        shutil.copy(image_file, os.path.join(save_path, 'real_images', f'combined_{i}.png'))

    # Processing fake images
    print("===========Fake Image Combining===========")
    os.makedirs(os.path.join(save_path, 'fake_images'), exist_ok=True)
    for i, image_file in enumerate(tqdm(fake_images)):
        shutil.copy(image_file, os.path.join(save_path, 'fake_images', f'combined_{i}.png'))

def create_train_fft_images(input_image_path, save_path):
    real_image_files = glob(os.path.join(input_image_path, "real_images/*.[pP][nN][gG]")) 
    fake_image_files = glob(os.path.join(input_image_path, "fake_images/*.[pP][nN][gG]"))

    # Define subfolders for saving FFT images
    real_fft_folder = os.path.join(save_path, 'real_images')
    fake_fft_folder = os.path.join(save_path, 'fake_images')
    
    os.makedirs(real_fft_folder, exist_ok=True)
    os.makedirs(fake_fft_folder, exist_ok=True)
    
    # Processing real images
    print("===========Train Real Image FFT generating===========")
    for image_file in tqdm(real_image_files):
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Perform FFT on each channel of the image
        fft = np.zeros_like(image, dtype=np.float32)
        for channel in range(3):
            fft_channel = np.fft.fft2(image[:, :, channel])
            fft_channel_magnitude = np.abs(fft_channel)
            fft[:, :, channel] = np.log(fft_channel_magnitude + 1)  # use log scale for better visualization

        output_file_base_name = os.path.basename(image_file).split('.')[0]
        output_file_path = os.path.join(real_fft_folder, f'{output_file_base_name}_fft.png')
        cv2.imwrite(output_file_path, cv2.normalize(fft, None, 0, 255, cv2.NORM_MINMAX))  # Save the FFT image

    # Processing fake images
    print("===========Train Fake Image FFT generating===========")
    for image_file in tqdm(fake_image_files):
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Perform FFT on each channel of the image
        fft = np.zeros_like(image, dtype=np.float32)
        for channel in range(3):
            fft_channel = np.fft.fft2(image[:, :, channel])
            fft_channel_magnitude = np.abs(fft_channel)
            fft[:, :, channel] = np.log(fft_channel_magnitude + 1)  # use log scale for better visualization

        output_file_base_name = os.path.basename(image_file).split('.')[0]
        output_file_path = os.path.join(fake_fft_folder, f'{output_file_base_name}_fft.png')
        cv2.imwrite(output_file_path, cv2.normalize(fft, None, 0, 255, cv2.NORM_MINMAX))  # Save the FFT image

def create_test_fft_images(input_folder_path, save_folder_path, n_patch):
    for i in range(n_patch):
        # Get image files
        image_files = glob(os.path.join(input_folder_path, f"images_{i}/*.[pP][nN][gG]"))
        # Create save directory if not exists
        save_dir = os.path.join(save_folder_path, f'images_{i}')
        os.makedirs(save_dir, exist_ok=True)
        
        # Processing images
        print(f"===========Test FFT Image generating for patch {i+1}/{n_patch}===========")
        for image_file in tqdm(image_files):
            image = cv2.imread(image_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Perform FFT on each channel of the image
            fft = np.zeros_like(image, dtype=np.float32)
            for channel in range(3):
                fft_channel = np.fft.fft2(image[:, :, channel])
                fft_channel_magnitude = np.abs(fft_channel)
                fft[:, :, channel] = np.log(fft_channel_magnitude + 1)  # use log scale for better visualization

            output_file_base_name = os.path.basename(image_file).split('.')[0]
            output_file_path = os.path.join(save_dir, f'{output_file_base_name}.png')
            cv2.imwrite(output_file_path, cv2.normalize(fft, None, 0, 255, cv2.NORM_MINMAX))  # Save the FFT image

if __name__ == '__main__':
    image_path = 'D:/Projects/AIM/Competition/2023/AIConnect_AI_Image_Recognition/original_data/patched_224x224/test'
    image_path_2 = 'D:/Projects/AIM/Competition/2023/AIConnect_AI_Image_Recognition/original_data/fft_224x224/train'
    save_path = 'D:/Projects/AIM/Competition/2023/AIConnect_AI_Image_Recognition/original_data/fft_224x224/test'

    # Print image sizes metrics
    #image_sizes = get_image_sizes(image_path)

    # Create images using augmentations
    # data_creation(image_path, save_path)
    
    # Create train patched images
    # create_train_patched_images(image_path, save_path, (224, 224))

    # Create test patched images
    # create_test_patched_images(image_path, save_path)

    # Create train dct images
    # create_train_dct_images(image_path, save_path)

    # Create test dct images
    # create_test_dct_images(image_path, save_path, 5)

    # Combine two datasets
    # combine_train_datasets(image_path, image_path_2, save_path)
    
    # Create train fft images
    # create_train_fft_images(image_path, save_path)

    # Create test dct images
    create_test_fft_images(image_path, save_path, 5)