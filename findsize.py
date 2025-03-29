import os
from PIL import Image

# Path to the main folder
main_folder = "donnees"

# Get all subfolders in the main folder
subfolders=[]
for f in os.scandir(main_folder):
    if f.is_dir():
        subfolders.append(f.path)

# Iterate over each subfolder
for subfolder in subfolders:
    # Get all images in the subfolder

    animals = [f.path for f in os.scandir(subfolder) if f.is_dir()]

    for animal in animals:
        images = [f.path for f in os.scandir(animal) if f.is_file() and f.path.endswith(('.jpg'))]
        # Initialize maximum width and height
        max_width = 0
        max_height = 0
        min_width = 300
        min_height = 300
        
        # Iterate over each image
        for image in images:
            # Open the image
            img = Image.open(image)
            # Update maximum width and height if necessary
            max_width = max(max_width, img.size[0])
            max_height = max(max_height, img.size[1])
            min_width = min(min_width, img.size[0])
            min_height = min(min_height, img.size[1])
        
        # Print the maximum width and height for the subfolder
        print(f"Maximum width for {animal}: {max_width}")
        print(f"Maximum height for {animal}: {max_height}")
        print(f"Minimum width for {animal}: {min_width}")
        print(f"Minimum height for {animal}: {min_height}")

