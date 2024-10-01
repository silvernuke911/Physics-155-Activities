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

# Clean up the image with noise reduction
# Apply Gaussian Blur to reduce noise before thresholding
blurred_dog = cv2.GaussianBlur(dog_image, (5, 5), 0)

# Apply binary thresholding to get a binary image (0 or 255)
_, binary_image = cv2.threshold(blurred_dog, 127, 255, cv2.THRESH_BINARY)

# Morphological operations to clean up edges
kernel = np.ones((3, 3), np.uint8)  # Define a kernel
binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)  # Close small holes
binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)   # Remove small noise

# Optional: Use Canny Edge Detection for better edge extraction
canny_edges = cv2.Canny(blurred_dog, 100, 200)

# Save the cleaned binary image
output_path = 'cleaned_binary_edge_detection_result.png'  # Specify the output file path
cv2.imwrite(output_path, binary_image)

# Save the Canny edges image
canny_output_path = 'canny_edge_detection_result.png'
cv2.imwrite(canny_output_path, canny_edges)

# Display the original, DoG, cleaned binary, and Canny edge images
plt.figure(figsize=(24, 8))
plt.subplot(1, 4, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title('Difference of Gaussians Edge Detection')
plt.imshow(dog_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title('Cleaned Binary Image')
plt.imshow(binary_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title('Canny Edge Detection')
plt.imshow(canny_edges, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

print(f"Cleaned binary result saved to: {output_path}")
print(f"Canny edges result saved to: {canny_output_path}")
