from openimages.download import download_images

food_categories = [
    # Fruits:
    "Apple", "Banana", "Orange", "Pear", "Strawberry", "Blueberry", 
    "Raspberry", "Blackberry", "Lemon", "Lime", "Pineapple", "Mango", 
    "Watermelon", "Grapes", "Peach", "Cherry", "Papaya", "Kiwi", 
    "Plum", "Avocado", "Coconut", "Pomegranate",

    # Vegetables:
    "Broccoli", "Carrot", "Potato", "Onion", "Tomato", "Cucumber", 
    "Bell pepper", "Lettuce", "Kale", "Spinach", "Zucchini", "Eggplant", 
    "Cauliflower", "Corn", "Sweet potato", "Celery", "Asparagus", 
    "Green beans", "Garlic", "Radish", "Beet", "Mushroom", 
    "Chili pepper",

    # Nuts and Seeds:
    "Almond", "Cashew", "Peanut", "Walnut", "Hazelnut", "Pecan", 
    "Pistachio", "Sunflower seed", "Pumpkin seed", "Sesame seed", 
    "Chia seed",

    # Meals:
    "Pizza", "Hamburger", "Hot dog", "Sandwich", "Taco", "Burrito", 
    "Cake", "Cookie", "Donut", "Ice cream", "Pancake", "Waffle", 
    "Muffin", "Cupcake", "French fries", "Sushi", "Ramen", "Fried rice", 
    "Curry", "Soup", "Salad", "Pasta", "Spaghetti", "Macaroni and cheese", 
    "Lasagna", "Bread", "Bagel", "Croissant",

    # Beverages:
    "Coffee", "Tea", "Milk", "Juice", "Smoothie", "Beer", "Wine", 
    "Cocktail", "Water bottle", "Soda can",

    # Snacks and Desserts:
    "Chocolate", "Candy", "Popcorn", "Chips", "Pretzel", "Granola bar", 
    "Ice cream cone", "Cheesecake", "Brownie", "Pie",

    # Dairy and Meat:
    "Cheese", "Butter", "Yogurt", "Egg", "Bacon", "Chicken", "Beef", 
    "Pork", "Fish", "Shrimp", "Crab", "Lobster", "Steak", "Sausage", 
    "Turkey", "Lamb",

    # Seafood:
    "Salmon", "Tuna", "Prawns", "Oysters", "Mussels", "Clams", "Scallops",

    # Bread and Cereal:
    "Rice", "Bread loaf", "Tortilla", "Noodles", "Flour", "Sugar", 
    "Oats", "Cereal", "Cornbread",
    "Ketchup", "Mustard", "Mayonnaise", "Soy sauce", "Hot sauce", 
    "Honey", "Peanut butter", "Jam", "Salt", "Pepper", "Cinnamon", 
    "Curry powder", "Garlic powder", "Vanilla extract"]

pathtoimgs = "FoodforDeepThought/data"
download_images(pathtoimgs, food_categories)