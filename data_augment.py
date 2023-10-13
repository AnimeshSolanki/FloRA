import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Define your source and destination directories
base_dir = "C:\\Users\\dell\\OneDrive\\Desktop\\New folder\\flowers"
output_dir = "C:\\Users\\dell\\OneDrive\\Desktop\\New folder\\augmented_data"

# Create an ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Get a list of all subdirectories in the "flowers" directory
flower_categories = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

# Iterate through each flower category
for category in flower_categories:
    source_dir = os.path.join(base_dir, category)
    image_files = [f for f in os.listdir(source_dir) if f.endswith('.jpg')]

    # Iterate through each image in the category, apply augmentation, resize, and save
    for image_file in image_files:
        image_path = os.path.join(source_dir, image_file)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Ensure RGB color format

        # Apply data augmentation
        augmented_images = [datagen.random_transform(img) for _ in range(5)]  # Generate 5 augmented versions

        # Save the augmented and resized images with new filenames
        for i, augmented_img in enumerate(augmented_images):
            augmented_filename = image_file.replace('.jpg', f'_aug_{i}.jpg')
            output_path = os.path.join(output_dir, category, augmented_filename)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR))
