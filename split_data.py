#The purpose of splitting the dataset into training and validation sets is to use the training set to train a machine learning model and the validation set to assess the model's performance.

import os # for interaction with the operating system, such as creating directories and manipulating file paths.
import shutil # offers higher-level file operations, including copying files.

from sklearn.model_selection import train_test_split #function splits the images list into two separate lists

source_dir="C:\\Users\\dell\\OneDrive\\Desktop\\FloRA\\flowers" # actual path
train_dir = "C:\\Users\\dell\\OneDrive\\Desktop\\FloRA\\splits\\train" # desired path for training data
val_dir = "C:\\Users\\dell\\OneDrive\\Desktop\\FloRA\\splits\\validate" # desired path for validation data

flower_classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
test_size = 0.2 #means that 20% of the images will be placed in the validation set, and the remaining 80% will be in the training set.

for flower_class in flower_classes:
    class_source_dir = os.path.join(source_dir, flower_class) #This function takes multiple arguments and joins them together to form a path. It automatically takes care of adding the appropriate path separators (like "/" or "") between the components.
    class_train_dir = os.path.join(train_dir, flower_class)
    class_val_dir = os.path.join(val_dir, flower_class)

    os.makedirs(class_train_dir, exist_ok=True) # creates the training and validation directories for the current flower class. The exist_ok=True parameter ensures that the directories are created even if they already exist.
    os.makedirs(class_val_dir, exist_ok=True)

    images = os.listdir(class_source_dir) #returns a list containing the names of all the items within that directory.
    train_images, val_images = train_test_split(images, test_size= test_size, random_state=42) #Setting a specific value for random_state ensures that the same split is obtained every time the code is run with the same data and parameters.

    print(f"Number of images in train set: {len(train_images)}")
    print(f"Number of images in validation set: {len(val_images)}")

    for img in train_images:
        src_path = os.path.join(class_source_dir, img)
        dest_path = os.path.join(class_train_dir, img)
        shutil.copy(src_path, dest_path) #copies the image from the source path to the destination path.

    for img in val_images:
        src_path = os.path.join(class_source_dir,img)
        dest_path = os.path.join(class_val_dir, img)
        shutil.copy(src_path, dest_path)
