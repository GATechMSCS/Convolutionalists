import os
import torchvision.transforms as transforms
from openimages.download import download_dataset
import random
import shutil
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
import torch
from torch.utils.data import DataLoader

class OpenImagesLoader:
    def __init__(self, random_seed = 101, batch_size = 128, perc_keep = 1.0, num_images_per_class=500, annotation_format="pascal"):
        self.data_dir = os.path.join("data", "openimages")  # Directory in which dataset resides
        self.csv_dir = os.path.join("data", "openimages_csv_dir")
        self.random_seed = random_seed
        self.batch_size = batch_size
        self.perc_keep = perc_keep  # Percentage of dataset to be kept (number between 0 and 1)
        self.num_images_per_class = num_images_per_class
        self.transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet's normalization statistics
            ]
        )
        self.annotation_format = annotation_format

        self.classes = [
            "Hot dog", "French fries", "Waffle", "Pancake", "Burrito", "Pretzel",
            "Popcorn", "Cookie", "Muffin", "Ice cream", "Cake", "Candy",
            "Guacamole", "Apple", "Grape", "Common fig", "Pear",
            "Strawberry", "Tomato", "Lemon", "Banana", "Orange", "Peach", "Mango",
            "Pineapple", "Grapefruit", "Pomegranate", "Watermelon", "Cantaloupe",
            "Egg (Food)", "Bagel", "Bread", "Doughnut", "Croissant",
            "Tart", "Mushroom", "Pasta", "Pizza", "Squid",
            "Oyster", "Lobster", "Shrimp", "Crab", "Taco", "Cooking spray",
            "Cucumber", "Radish", "Artichoke", "Potato", "Garden Asparagus",
            "Pumpkin", "Zucchini", "Cabbage", "Carrot", "Salad",
            "Broccoli", "Bell pepper", "Winter melon", "Honeycomb",
            "Hamburger", "Submarine sandwich", "Cheese", "Milk", "Sushi"
        ]

    def download_data(self):
        for class_name in self.classes:
            print(f'Attempting to download {class_name} data')
            if not os.path.isdir(os.path.join(self.data_dir, class_name.lower())):
                try:
                    download_dataset(self.data_dir, [class_name], None, annotation_format=self.annotation_format, csv_dir=None, limit=500)
                except:
                    print(f'ERROR! An exception occurred for {class_name}')
            else:
                print(f'Skipped {class_name}, data already downloaded')

    def split_data(self, keep_class_dirs=True):

        """ This function splits the downloaded Open Image dataset, and splits each class into training, validation, and testing sets.
            This function assumes that the required data has already been downloaded."""

        # Setting the random seed:
        random.seed(self.random_seed)
        
        splits = ["train", "val", "test"]

        # Making folders for each of the splits:
        for split in splits:
            split_dir = os.path.join(self.data_dir, split)
            os.makedirs(split_dir, exist_ok=True)

        # Iterating through each class:
        for class_cur in self.classes:
            print(f'Splitting data for class {class_cur}')

            # Getting directories for the images and annotations for each class:
            imgs_dir = os.path.join(self.data_dir, class_cur.lower(), "images")
            anns_dir = os.path.join(self.data_dir, class_cur.lower(), "pascal")

            # Ensuring each class has images and annotations:
            if not imgs_dir:
                raise Exception(f'Images do not exist for {class_cur}!')

            if not anns_dir:
                raise Exception(f'Annotations do not exist for {class_cur}!')

            class_imgs = os.listdir(imgs_dir) # Images for current class
            class_anns = os.listdir(anns_dir) # Annotations for current class
            class_imgs.sort()
            class_anns.sort()

            num_imgs = len(class_imgs) # Number of images and annotations for current class
            
            # Shuffling data:
            inds_list = list(range(num_imgs)) # List of indices ranging for the total number of images
            random.shuffle(inds_list) # Shuffling indices list
            class_imgs = [class_imgs[i] for i in inds_list] # Shuffling class images according to shuffled inds_list
            class_anns = [class_anns[i] for i in inds_list] # Shuffling class annotations according to shuffled inds_list

            ind_train = int(0.8 * num_imgs) # Ending index for the training images
            ind_val = ind_train + int(0.1 * num_imgs) # Ending index for the validation images

            # Splitting images into training, validation, and testing:
            train_imgs = class_imgs[:ind_train]
            val_imgs = class_imgs[ind_train:ind_val]
            test_imgs = class_imgs[ind_val:]

            all_imgs = [train_imgs, val_imgs, test_imgs] # All images
            
            # Splitting annotations into training, validation, and testing:
            train_anns = class_anns[:ind_train]
            val_anns = class_anns[ind_train:ind_val]
            test_anns = class_anns[ind_val:]

            all_anns = [train_anns, val_anns, test_anns] # All annotations
            
            # Looping through all split types and corresponding split images:
            for split_type, split_imgs, split_anns in zip(splits, all_imgs, all_anns):
                if keep_class_dirs:
                    # Creating each split directory for images and annotations for current class:
                    split_dir_img = os.path.join(self.data_dir, split_type, class_cur.lower(), "images")
                    split_dir_ann = os.path.join(self.data_dir, split_type, class_cur.lower(), "annotations")
                else:
                    split_dir_img = os.path.join(self.data_dir, split_type, "images")
                    split_dir_ann = os.path.join(self.data_dir, split_type, "annotations")

                os.makedirs(split_dir_img, exist_ok=True)
                os.makedirs(split_dir_ann, exist_ok=True)

                # Copying each image from initial directory to corresponding split directory for each split:
                for img, ann in zip(split_imgs, split_anns):
                    shutil.copy(os.path.join(imgs_dir, img), os.path.join(split_dir_img, img))
                    shutil.copy(os.path.join(anns_dir, ann), os.path.join(split_dir_ann, ann))

    def get_datasets(self):

        """ This function strives to get datasets to the local data directory if
        it has not already been downloaded previously. This function also splits the datasets into training,
        validation, and testing sets, assigning them as class variables. """

        # Note - this assumes the openimages  dataset has already been downloaded to their respective directories:.
        # If the dataset has not been downloaded, then please manually download it and place it in the directories
        # as described in the class initialization:
        self.train_dir = os.path.join(self.data_dir, "train")  # Directory in which dataset resides
        self.val_dir = os.path.join(self.data_dir, "val")
        self.test_dir = os.path.join(self.data_dir, "test")
        train_raw = ImageFolder(self.train_dir, transform=self.transforms)
        val_raw = ImageFolder(self.val_dir, transform=self.transforms)
        test_raw = ImageFolder(self.test_dir, transform=self.transforms)

        # Seed generator:
        generator = torch.Generator().manual_seed(self.random_seed)

        if self.perc_keep != 1.00:
            # Calculating the limited sizes of the datasets to keep:
            train_size = int(len(train_raw) * self.perc_keep)
            val_size = int(len(val_raw) * self.perc_keep)
            test_size = int(len(test_raw) * self.perc_keep)

            # Decreasing the size of the datasets using random_split:
            train_raw, _ = random_split(train_raw, [train_size, (len(train_raw)-train_size)])
            val_raw, _ = random_split(val_raw, [val_size, (len(val_raw)-val_size)])
            test_raw, _ = random_split(test_raw, [test_size, (len(test_raw)-test_size)])

        train_set = DataLoader(train_raw, batch_size=self.batch_size, shuffle=True) # Applying a DataLoader to the test set
        val_set = DataLoader(val_raw, batch_size=self.batch_size, shuffle=True) # Applying a DataLoader to the test set
        test_set = DataLoader(test_raw, batch_size=self.batch_size, shuffle=True) # Applying a DataLoader to the test set
        
        return train_set, val_set, test_set
