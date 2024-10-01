import cv2
import numpy as np
import matplotlib.pyplot as plt

def difference_of_gaussians(image, sigma1=1, sigma2=2):
    # Apply Gaussian blur with different sigmas
    blurred1 = cv2.GaussianBlur(image, (0, 0), sigma1)
    blurred2 = cv2.GaussianBlur(image, (0, 0), sigma2)
    
    # Calculate the Difference of Gaussians
    dog = blurred1 - blurred2
    
    # Normalize the result to 0-255
    dog_normalized = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)
    
    return dog_normalized

# Load the image
image_path = 'eq mount2.jpg'  # Replace with your image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply Difference of Gaussians
dog_image = difference_of_gaussians(image)

# Save the resulting image
output_path = 'dog_edge_detection_result.png'  # Specify the output file path
cv2.imwrite(output_path, dog_image)

# Display the original and DoG images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Difference of Gaussians Edge Detection')
plt.imshow(dog_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

print(f"Result saved to: {output_path}")
