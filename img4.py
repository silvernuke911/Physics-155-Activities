import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_chromatic_aberration(image):
    # Split the image into RGB channels
    b, g, r = cv2.split(image)

    # Shift each channel slightly in different directions
    b_shifted = cv2.warpAffine(b, np.float32([[1, 0, -10], [0, 1, 0]]), (image.shape[1], image.shape[0]))  # Shift Blue left
    r_shifted = cv2.warpAffine(r, np.float32([[1, 0, 10], [0, 1, 0]]), (image.shape[1], image.shape[0]))  # Shift Red right

    # Merge the channels back
    return cv2.merge([b_shifted, g, r_shifted])

def apply_spherical_aberration(image):
    # Simulate spherical aberration using Gaussian blur with different kernel sizes
    blurred_center = cv2.GaussianBlur(image, (15, 15), 0)
    blurred_edges = cv2.GaussianBlur(image, (75, 75), 0)

    # Blend them to create a spherical aberration effect
    return cv2.addWeighted(blurred_center, 0.5, blurred_edges, 0.5, 0)

def apply_coma_aberration(image):
    # Create a custom kernel for coma
    kernel = np.zeros((5, 5), dtype=np.float32)
    kernel[2, 1] = -1
    kernel[1, 2] = 1
    kernel[2, 2] = 3  # Increased central weight for sharpness
    kernel[2, 3] = -1
    kernel[3, 2] = 1
    return cv2.filter2D(image, -1, kernel)

def apply_astigmatism(image):
    # Simulate astigmatism by blurring in two orientations
    blurred_x = cv2.GaussianBlur(image, (25, 1), 0)
    blurred_y = cv2.GaussianBlur(image, (1, 25), 0)
    return cv2.addWeighted(blurred_x, 0.5, blurred_y, 0.5, 0)

def apply_curvature_of_field(image):
    # Simulate curvature of field with a radial distortion approach
    rows, cols = image.shape[:2]
    center_x, center_y = cols // 2, rows // 2

    # Create a distortion map
    y, x = np.indices((rows, cols)).astype(np.float32)
    r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    distortion = 1.0 + 0.3 * (r / (np.max(r) + 1e-5)) ** 2  # Adjust curvature factor

    # Remap the image based on distortion
    x_distorted = (x - center_x) * distortion + center_x
    y_distorted = (y - center_y) * distortion + center_y

    return cv2.remap(image, x_distorted.astype(np.float32), y_distorted.astype(np.float32), interpolation=cv2.INTER_LINEAR)

def apply_distortion(image, k):
    # Apply radial distortion using coefficient k
    rows, cols = image.shape[:2]
    x = np.linspace(-1, 1, cols)
    y = np.linspace(-1, 1, rows)
    X, Y = np.meshgrid(x, y)

    r = np.sqrt(X**2 + Y**2)
    distortion = 1 + k * r**2  # Pincushion (k < 0) or barrel distortion (k > 0)

    X_distorted = X * distortion
    Y_distorted = Y * distortion

    return cv2.remap(image, (X_distorted * cols / 2 + cols / 2).astype(np.float32),
                     (Y_distorted * rows / 2 + rows / 2).astype(np.float32), interpolation=cv2.INTER_LINEAR)

# Load the image
image_path = 'testim.png'  # Replace with your image path
image = cv2.imread(image_path)

# Apply each aberration
chromatic_aberration_image = apply_chromatic_aberration(image)
spherical_aberration_image = apply_spherical_aberration(image)
coma_aberration_image = apply_coma_aberration(image)
astigmatism_image = apply_astigmatism(image)
curvature_of_field_image = apply_curvature_of_field(image)
pincushion_distortion_image = apply_distortion(image, -0.3)  # Pincushion
barrel_distortion_image = apply_distortion(image, 0.3)  # Barrel

# Save all images
cv2.imwrite('chromatic_aberration.png', chromatic_aberration_image)
cv2.imwrite('spherical_aberration.png', spherical_aberration_image)
cv2.imwrite('coma_aberration.png', coma_aberration_image)
cv2.imwrite('astigmatism.png', astigmatism_image)
cv2.imwrite('curvature_of_field.png', curvature_of_field_image)
cv2.imwrite('pincushion_distortion.png', pincushion_distortion_image)
cv2.imwrite('barrel_distortion.png', barrel_distortion_image)

# Display the results
plt.figure(figsize=(15, 10))
plt.subplot(3, 2, 1)
plt.title('Chromatic Aberration')
plt.imshow(cv2.cvtColor(chromatic_aberration_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(3, 2, 2)
plt.title('Spherical Aberration')
plt.imshow(cv2.cvtColor(spherical_aberration_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(3, 2, 3)
plt.title('Coma Aberration')
plt.imshow(cv2.cvtColor(coma_aberration_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(3, 2, 4)
plt.title('Astigmatism')
plt.imshow(cv2.cvtColor(astigmatism_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(3, 2, 5)
plt.title('Curvature of Field')
plt.imshow(cv2.cvtColor(curvature_of_field_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(3, 2, 6)
plt.title('Pincushion Distortion')
plt.imshow(cv2.cvtColor(pincushion_distortion_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()

# Display barrel distortion in a new figure
plt.figure(figsize=(7, 7))
plt.title('Barrel Distortion')
plt.imshow(cv2.cvtColor(barrel_distortion_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

print("Aberration images saved successfully.")
