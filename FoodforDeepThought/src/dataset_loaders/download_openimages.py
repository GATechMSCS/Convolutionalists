from torchvision.transforms.functional import to_pil_image
from torchvision.io import read_image, ImageReadMode
from openimages.download import download_dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
from PIL import Image
import random
import shutil
import torch
import os
from torch.utils.data import DataLoader
import yaml

class OpenImagesLoader:
    def __init__(self, random_seed = 101, batch_size = 128, perc_keep = 1.0, num_images_per_class=500, annotation_format="pascal"):
        self.data_dir = os.path.join("data", "openimages")  # Directory in which dataset resides
        self.random_seed = random_seed
        self.batch_size = batch_size
        self.perc_keep = perc_keep  # Percentage of dataset to be kept (number between 0 and 1)
        self.num_images_per_class = num_images_per_class
        self.transforms = transforms.Compose(
            [
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

        # Creating a dictionary mapping each class name to an index:
        self.class_2_index = {}
        for i, class_name in enumerate(self.classes):
            self.class_2_index[class_name.lower()] = i


        self.train_dir = os.path.join(self.data_dir, "train") # Directory in which train dataset resides
        self.val_dir = os.path.join(self.data_dir, "val") # Directory in which validation dataset resides
        self.test_dir = os.path.join(self.data_dir, "test") # Directory in which test dataset resides

    def download_data(self, csv_dir=None, batch_download=False):
        csv_dir = os.path.join("data", csv_dir) if csv_dir else None
        if batch_download:
            print('Attempting to download the Open Images dataset')
            if not os.path.isdir(self.data_dir):
                try:
                    download_dataset(self.data_dir, self.classes, annotation_format=self.annotation_format, csv_dir=csv_dir, limit=500)
                except Exception as e:
                    print(f'An exception occurred while downloading the dataset. ERROR: {e}')
            else:
                print('Skipped downloading the dataset, data already downloaded')
        else:
            for class_name in self.classes:
                print(f'Attempting to download {class_name} data')
                if not os.path.isdir(os.path.join(self.data_dir, class_name.lower())):
                    try:
                        download_dataset(self.data_dir, [class_name], annotation_format=self.annotation_format, csv_dir=csv_dir, limit=500)
                    except Exception as e:
                        print(f'An exception occurred for {class_name}. ERROR: {e}')
                else:
                    print(f'Skipped {class_name}, data already downloaded')
    

    def split_data(self, keep_class_dirs=True):

        """ This function splits the downloaded Open Image dataset, and splits each class into training, validation, and testing sets.
            This function assumes that the required data has already been downloaded."""

        # Setting the random seed:
        random.seed(self.random_seed)
        
        splits = ["train", "val", "test"]
        annotation_dir = "annotations" if self.annotation_format == "pascal" else "labels"

        # Making folders for each of the splits:
        for split in splits:
            split_dir = os.path.join(self.data_dir, split)
            os.makedirs(split_dir, exist_ok=True)

        # Iterating through each class:
        for class_cur in self.classes:
            print(f'Splitting data for class {class_cur}')

            # Getting directories for the images and annotations for each class:
            imgs_dir = os.path.join(self.data_dir, class_cur.lower(), "images")
            anns_dir = os.path.join(self.data_dir, class_cur.lower(), self.annotation_format)

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
                    split_dir_ann = os.path.join(self.data_dir, split_type, class_cur.lower(), annotation_dir)
                else:
                    split_dir_img = os.path.join(self.data_dir, split_type, "images")
                    split_dir_ann = os.path.join(self.data_dir, split_type, annotation_dir)

                os.makedirs(split_dir_img, exist_ok=True)
                os.makedirs(split_dir_ann, exist_ok=True)

                # Copying each image from initial directory to corresponding split directory for each split:
                for img, ann in zip(split_imgs, split_anns):
                    shutil.copy(os.path.join(imgs_dir, img), os.path.join(split_dir_img, img))
                    shutil.copy(os.path.join(anns_dir, ann), os.path.join(split_dir_ann, ann))

    def split_data_reduced(self, keep_class_dirs=True):

        """ This function splits the downloaded Open Image dataset, and splits each class into training, validation, and testing sets.
            This function assumes that the required data has already been downloaded.
            This function reduces the dataset by self.keep_perc. """

        # Setting the random seed:
        random.seed(self.random_seed)
        
        splits = ["train_reduced", "val_reduced", "test_reduced"]
        annotation_dir = "annotations" if self.annotation_format == "pascal" else "labels"
        
        # Making folders for each of the splits:
        for split in splits:
            split_dir = os.path.join(self.data_dir, split)
            os.makedirs(split_dir, exist_ok=True)

        # Iterating through each class:
        for class_cur in self.classes:
            print(f'Splitting data for class {class_cur}')

            # Getting directories for the images and annotations for each class:
            imgs_dir = os.path.join(self.data_dir, class_cur.lower(), "images")
            anns_dir = os.path.join(self.data_dir, class_cur.lower(), self.annotation_format)

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
            
            if self.perc_keep != 1.00 and num_imgs > 50:
                num_imgs = int(num_imgs * self.perc_keep)
                class_imgs = class_imgs[:num_imgs]
                class_anns = class_anns[:num_imgs]

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
                    split_dir_ann = os.path.join(self.data_dir, split_type, class_cur.lower(), annotation_dir)
                else:
                    split_dir_img = os.path.join(self.data_dir, split_type, "images")
                    split_dir_ann = os.path.join(self.data_dir, split_type, annotation_dir)

                os.makedirs(split_dir_img, exist_ok=True)
                os.makedirs(split_dir_ann, exist_ok=True)

                # Copying each image from initial directory to corresponding split directory for each split:
                for img, ann in zip(split_imgs, split_anns):
                    shutil.copy(os.path.join(imgs_dir, img), os.path.join(split_dir_img, img))
                    shutil.copy(os.path.join(anns_dir, ann), os.path.join(split_dir_ann, ann))

        print(f"Dataset has been reduced!")

    def get_dataloaders(self):

        """ This function wraps the training, validation, and testing sets into dataloaders. """

        # Note - this assumes the openimages dataset has already been downloaded to their respective directories:.
        # If the dataset has not been downloaded, then please manually download it and place it in the directories
        # as described in the class initialization:

        def collate_fn(data):
            """ Defining the collate function to return lists of the images and annotations,
                as object-detection can have variable image sizes and variable number of objects
                in each image. 
            """

            imgs, anns = zip(*data)        
            return list(imgs), list(anns)

        dirs = [self.train_red_dir, self.val_red_dir, self.test_red_dir]

        loaders = []

        for dir_cur in dirs:
            
            print(f"Processing {dir_cur}...")

            img_dir = os.path.join(dir_cur, "images")
            ann_dir = os.path.join(dir_cur, "annotations")
            dataset_cur = []
            
            img_list = os.listdir(img_dir)
            ann_list = os.listdir(ann_dir)

            # If there exists no images or annotations, then skip current directory:
            if len(img_list) == 0:
                loaders.append(None)
                continue

            # Iterating through each image-annotation pair:
            for img_cur, ann_cur in zip(img_list, ann_list):

                # Applying the transforms to each image:
                img_path = os.path.join(img_dir, img_cur) # File path of current image                
                img_cur = read_image(img_path) # Extracting image from file path

                # If the image has an extra channel:
                if img_cur.shape[0] > 3:
                    img_cur = img_cur[:3, :, :] # Removing the extra channel from the image

                # If it's a grayscale image, then repeat the row and column values three times:
                if img_cur.shape[0] == 1:
                    img_cur = img_cur.repeat(3, 1, 1)

                img_cur = to_pil_image(img_cur) # Convert the image to PIL format
                img_cur = self.transforms(img_cur) # Applying transforms to current image

                # Processing annotations:
                ann_path = os.path.join(ann_dir, ann_cur) # File path of annotations corresponding to current image
                ann_dict = self.process_ann_file(ann_path) # Extracting a dictionary of bounding boxes and labels from the current annotations file

                # Appending each image and corresponding annotations to the current dataset list:
                dataset_cur.append((img_cur, ann_dict))

            # Wrapping the current dataset with a DataLoader:
            dataset_wrapped = DatasetWrapper(dataset_cur) # Wrapping with custom class to support __getitem__ and __len__ methods.                
            dataset_loader = DataLoader(dataset_wrapped, batch_size = self.batch_size, shuffle=True, collate_fn=collate_fn)

            # Appending the current dataset list to the list of all dataset lists:
            loaders.append(dataset_loader)

        
        return loaders[0], loaders[1], loaders[2]

    def process_ann_file(self, file_path):
        """ This function parses the annotation Pascal XML file to retrieve the class label and bounding box locations. 
        
        Inputs:
        file_path: Path of the annotation XML file

        Outputs:
        bounding_boxes, labels: Bounding box and label information contained within the annotation file
        
        """

        # Creating element tree object and getting its root:
        root = ET.parse(file_path).getroot()

        ann_dict = {}

        boxes = [] # List of dimensions 
        labels = [] # List of all target labels for the images

        # Iterating through each found object in the image:
        for object in root.findall('object'):            
            class_cur = object.find('name').text # Class of current object

            # Appending the index corresponding to the current class to the labels list:
            labels.append(self.class_2_index[class_cur])

            # Bounding box location of current object:
            bnd_box = object.find('bndbox')
            xmin = float(bnd_box.find('xmin').text)
            xmax = float(bnd_box.find('xmax').text)
            ymin = float(bnd_box.find('ymin').text)
            ymax = float(bnd_box.find('ymax').text)

            boxes.append([xmin, ymin, xmax, ymax])

        # Converting the boxes and labels to PyTorch tensors:
        boxes = torch.tensor(boxes, dtype=torch.float64)
        labels = torch.tensor(labels, dtype=torch.int64)

        ann_dict['boxes'] = boxes
        ann_dict['labels'] = labels

        return ann_dict
    
    def create_yaml_from_file(self, obj_names_file="darknet_obj_names.txt"):
        """
        Create a YOLO-compatible data.yaml file using classes from darknet_obj_names.txt.

        Args:
            data_dir (str): Root directory containing the dataset splits (train, val, test).
            obj_names_file (str): Text file with classes downloaded.
        """
        data_dir = self.data_dir
        class_list = os.path.join(os.path.abspath(data_dir), obj_names_file)
        yaml_output_path = os.path.join(os.path.abspath(data_dir), "data.yaml")
        
        # Read the classes from the darknet_obj_names.txt file
        with open(class_list, "r") as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]  # Remove empty lines and strip whitespace

        # Construct the YAML dictionary
        yaml_data = {
            "path": os.path.abspath(data_dir),
            "train": os.path.join(os.path.abspath(data_dir), "train/images"),
            "val": os.path.join(os.path.abspath(data_dir), "val/images"),
            "test": os.path.join(os.path.abspath(data_dir), "test/images"),
            "nc": len(classes),
            "names": classes
        }

        # Write the YAML dictionary to a file
        with open(yaml_output_path, "w") as f:
            yaml.dump(yaml_data, f, default_flow_style=False)
        print(f"YAML file saved to '{yaml_output_path}'.")


# DON'T THINK ANYONE IS USING THIS. IT'S OLD AND NOT USEFUL
    # def get_datasets(self):

    #     """ This function splits the datasets into training, validation, and testing sets. """

    #     # Note - this assumes the openimages dataset has already been downloaded to their respective directories:.
    #     # If the dataset has not been downloaded, then please manually download it and place it in the directories
    #     # as described in the class initialization:
    #     train_raw = ImageFolder(self.train_dir, transform=self.transforms)
    #     val_raw = ImageFolder(self.val_dir, transform=self.transforms)
    #     test_raw = ImageFolder(self.test_dir, transform=self.transforms)

    #     # Seed generator:
    #     generator = torch.Generator().manual_seed(self.random_seed)

    #     if self.perc_keep != 1.00:
    #         # Calculating the limited sizes of the datasets to keep:
    #         train_size = int(len(train_raw) * self.perc_keep)
    #         val_size = int(len(val_raw) * self.perc_keep)
    #         test_size = int(len(test_raw) * self.perc_keep)

    #         # Decreasing the size of the datasets using random_split:
    #         train_raw, _ = random_split(train_raw, [train_size, (len(train_raw)-train_size)])
    #         val_raw, _ = random_split(val_raw, [val_size, (len(val_raw)-val_size)])
    #         test_raw, _ = random_split(test_raw, [test_size, (len(test_raw)-test_size)])

    #     train_set = DataLoader(train_raw, batch_size=self.batch_size, shuffle=True) # Applying a DataLoader to the test set
    #     val_set = DataLoader(val_raw, batch_size=self.batch_size, shuffle=True) # Applying a DataLoader to the test set
    #     test_set = DataLoader(test_raw, batch_size=self.batch_size, shuffle=True) # Applying a DataLoader to the test set
        
    #     return train_set, val_set, test_set

class DatasetWrapper:
    """ Class used to wrap each dataset (list of tuples of (image, annotations)) as a class compatible with
        PyTorch's DataLoader object. """

    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, ind):
        return self.dataset[ind]

# FOR FASTER-RCNN
class ImageLoaderFRCNN(Dataset):
    def __init__(self, root, classes, tforms=None):
        self.root = root
        self.tforms = tforms
        self.classes = classes
        self.imgs = list(sorted(os.listdir(os.path.join(self.root, "images"))))
        self.annotations = list(sorted(os.listdir(os.path.join(self.root, "annotations"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        ann_path = os.path.join(self.root, "annotations", self.annotations[idx])
        
        img = Image.open(img_path).convert("RGB")
        
        # Parse the XML annotation file
        tree = ET.parse(ann_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        for obj in root.findall('object'):
            label = obj.find('name').text.capitalize()
            if "food" in label:
                label = label.replace("food", "Food")
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.classes.index(label))
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
            
        if self.tforms is not None:
            img, target = self.tforms(img, target)
        
        return img, target

    def __len__(self):
        return len(self.imgs)
