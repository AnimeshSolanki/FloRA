import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img

source_dir = "C:\\Users\\dell\\OneDrive\\Documents\\FloRA\\flowers"
preprocessed_dir = "C:\\Users\\dell\\OneDrive\\Documents\\FloRA\\preprocessed_data\\images" 
target_size = (224, 224)
test_size = 0.2
random_state = 42

flower_classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

for flower_class in flower_classes:
    class_source_dir = os.path.join(source_dir, flower_class)
    images = os.listdir(class_source_dir)
    train_images, val_images = train_test_split(images, test_size=test_size, random_state=random_state)

    for image in train_images:
        image_path = os.path.join(class_source_dir, image)
        img = load_img(image_path, target_size=target_size)
        dest_path = os.path.join(preprocessed_dir, 'train', flower_class)
        img_path = os.path.join(dest_path, image.replace('.jpg', '_resized.jpg'))
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        img.save(img_path)

    for image in val_images:
        image_path = os.path.join(class_source_dir, image)
        img = load_img(image_path, target_size=target_size)
        dest_path = os.path.join(preprocessed_dir, 'val', flower_class)
        img_path = os.path.join(dest_path, image.replace('.jpg', '_resized.jpg'))
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        img.save(img_path)
