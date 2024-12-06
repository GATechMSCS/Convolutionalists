import os
import torchvision.transforms as transforms
from openimages.download import download_dataset


class OpenImagesLoader:
    def __init__(self, random_seed = 101, batch_size = 128, perc_keep = 1.0, num_images_per_class=500):
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

        self.classes = [
            "Hot dog", "French fries", "Waffle", "Pancake", "Burrito", "Snack", "Pretzel",
            "Popcorn", "Cookie", "Dessert", "Muffin", "Ice cream", "Cake", "Candy",
            "Guacamole", "Fruit", "Apple", "Grape", "Common fig", "Pear",
            "Strawberry", "Tomato", "Lemon", "Banana", "Orange", "Peach", "Mango",
            "Pineapple", "Grapefruit", "Pomegranate", "Watermelon", "Cantaloupe",
            "Egg", "Baked goods", "Bagel", "Bread", "Pastry", "Doughnut", "Croissant",
            "Tart", "Mushroom", "Pasta", "Pizza", "Seafood", "Squid", "Shellfish",
            "Oyster", "Lobster", "Shrimp", "Crab", "Taco", "Cooking spray",
            "Vegetable", "Cucumber", "Radish", "Artichoke", "Potato", "Asparagus",
            "Squash", "Pumpkin", "Zucchini", "Cabbage", "Carrot", "Salad",
            "Broccoli", "Bell pepper", "Winter melon", "Honeycomb", "Sandwich",
            "Hamburger", "Submarine sandwich", "Dairy", "Cheese", "Milk", "Sushi"
        ]

    def download_data(self):
        for class_name in self.classes:
            print(f'Attempting to download {class_name} data')
            if not os.path.isdir(os.path.join(self.data_dir, class_name)):
                try:
                    download_dataset(self.data_dir, [class_name], None, annotation_format="pascal", csv_dir=None, limit=500)
                except:
                    print(f'An exception occured for {class_name}')
            else:
                print(f'Skipped {class_name}, data already downloaded')
