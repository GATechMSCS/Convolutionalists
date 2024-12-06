# from openimages.download import download_images, download_dataset
from download_openimages import OpenImagesLoader
#import pandas as pd

# path_to_csv = "FoodforDeepThought/data/class-descriptions-boxable.csv"
# food_categories = pd.read_csv(path_to_csv, header=None)
# print(food_categories[0].tolist())

# food_categories =  ["French fries", "Waffle", "Pancake", "Burrito", "Snack", "Pretzel", 
#                     "Popcorn", "Cookie", "Dessert", "Muffin", "Ice cream", "Cake", "Candy", 
#                     "Guacamole", "Fruit", "Apple", "Grape", "Common fig", "Pear", 
#                     "Strawberry", "Tomato", "Lemon", "Banana", "Orange", "Peach", "Mango", 
#                     "Pineapple", "Grapefruit", "Pomegranate", "Watermelon", "Cantaloupe", 
#                     "Egg", "Baked goods", "Bagel", "Bread", "Pastry", "Doughnut", "Croissant", 
#                     "Tart", "Mushroom", "Pasta", "Pizza", "Seafood", "Squid", "Shellfish", 
#                     "Oyster", "Lobster", "Shrimp", "Crab", "Taco", "Cooking spray", 
#                     "Vegetable", "Cucumber", "Radish", "Artichoke", "Potato", "Asparagus", 
#                     "Squash", "Pumpkin", "Zucchini", "Cabbage", "Carrot", "Salad", 
#                     "Broccoli", "Bell pepper", "Winter melon", "Honeycomb", "Sandwich", 
#                     "Hamburger", "Submarine sandwich", "Dairy", "Cheese", "Milk", "Sushi"]

# pathtoimgs = "FoodforDeepThought/data"
# download_dataset(dest_dir=pathtoimgs,
#                  class_labels=food_categories,
#                  annotation_format='pascal',
#                  limit=1000)

oil = OpenImagesLoader(random_seed=101,
                       batch_size=128,
                       perc_keep=1.0)
oil.download_data()