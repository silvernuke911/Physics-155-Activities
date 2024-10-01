from PIL import Image
import numpy as np

# Load the image
image_path = 'telscopes.png'
img = Image.open(image_path)

# Convert the image to a NumPy array for easy manipulation
img_array = np.array(img)

# Function to check if the average of RGB values is greater than 200
def process_pixel(rgb):
    # Consider only the RGB values (first three channels)
    avg = np.mean(rgb[:3])  # Compute the mean of the RGB channels
    if avg > 200:
        return [255, 255, 255, rgb[3]]  # Return white with original alpha
    else:
        return rgb  # Leave the pixel unchanged

# Apply the function to each pixel in the image
processed_img_array = np.apply_along_axis(process_pixel, 2, img_array)

# Convert the processed NumPy array back to an image
processed_img = Image.fromarray(np.uint8(processed_img_array))

# Save or show the processed image
processed_img.save('qweqweqw.png')
processed_img.show()
