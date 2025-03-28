from PIL import Image
import numpy as np

def load_and_preprocess_image(image_path, target_size, method='resize'):
    """
    Load and preprocess an image to fit target_size
    
    Methods:
    - 'resize': Simple resize (might distort aspect ratio)
    - 'pad': Maintain aspect ratio and pad with zeros
    - 'crop': Maintain aspect ratio and crop excess
    """
    img = Image.open(image_path)
    
    if method == 'resize':
        # Simple resize (might distort image)
        return img.resize((target_size, target_size))
    
    elif method == 'pad':
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
        
    elif method == 'crop':
        # Resize maintaining aspect ratio and crop excess
        ratio = max(target_size/img.size[0], target_size/img.size[1])
        new_size = tuple([int(x*ratio) for x in img.size])
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Center crop
        left = (img.size[0] - target_size)//2
        top = (img.size[1] - target_size)//2
        right = left + target_size
        bottom = top + target_size
        return img.crop((left, top, right, bottom))

# Example usage:
image_path = "path/to/your/image.jpg"
processed_img = load_and_preprocess_image(image_path, image_scale, method='pad')