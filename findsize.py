import os
from PIL import Image

# Path to the main folder
main_folder = "donnees"

# Get all subfolders in the main folder
subfolders = [f.path for f in os.scandir(main_folder) if f.is_dir()]

# Iterate over each subfolder
for subfolder in subfolders:
    # Get all images in the subfolder
    images = [f.path for f in os.scandir(subfolder) if f.is_file() and f.path.endswith(('.jpg', '.png', '.jpeg'))]
    
    # Initialize maximum width and height
    max_width = 0
    max_height = 0
    
    # Iterate over each image
    for image in images:
        # Open the image
        img = Image.open(image)
        
        # Update maximum width and height if necessary
        max_width = max(max_width, img.size[0])
        max_height = max(max_height, img.size[1])
    
    # Print the maximum width and height for the subfolder
    print(f"Maximum width for {subfolder}: {max_width}")
    print(f"Maximum height for {subfolder}: {max_height}")

