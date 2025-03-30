from PIL import Image
import os

image_scale = 256
destination_folder = "donnees_nouvelles"

def preprocess_image(image_path, target_size, method='pad'):
    """
    Load and preprocess an image to fit target_size
    
    Methods:
    - 'resize': Simple resize (might distort aspect ratio)
    - 'pad': Maintain aspect ratio and pad with zeros
    - 'crop': Maintain aspect ratio and crop excess
    """
    img = Image.open(image_path)
      
    if method == 'pad':
        new_img = Image.new('RGB', (target_size, target_size), (0,0,0))
        # Paste resized image in center
        upper_left_x = (target_size - img.size[0])//2
        upper_left_y = (target_size - img.size[1])//2
        new_img.paste(img, (upper_left_x, upper_left_y))
        return new_img
    
    elif method == 'resizepad':
        # Resize maintaining aspect ratio and pad with zeros
        ratio = min(target_size/img.size[0], target_size/img.size[1])
        new_size = tuple([int(x*ratio) for x in img.size])
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Create new image with black background
        new_img = Image.new('RGB', (target_size, target_size), (0,0,0))
        # Paste resized image in center
        upper_left_x = (target_size - new_size[0])//2
        upper_left_y = (target_size - new_size[1])//2
        new_img.paste(img, (upper_left_x, upper_left_y))
        return new_img
        
    
# Get all subfolders in the main folder
subfolders = []
os.chdir("donnees")
for f in os.scandir():
    if f.is_dir():
        subfolders.append(f.path)

# Iterate over each subfolder
for subfolder in subfolders:
    animals = [f.path for f in os.scandir(subfolder) if f.is_dir()]
    for animal in animals:
        images = [f.path for f in os.scandir(animal) if f.is_file() and f.path.endswith(('.jpg'))]
        os.chdir('..\\donnees_nouvelles')
        if not os.path.isdir(animal):
            os.makedirs(animal)
        # Iterate over each image
        for image in images:
            preprocess_image(f'..\\donnees\\{image}',image_scale).save(image)
        os.chdir('..\\donnees')
