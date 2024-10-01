import cv2
import numpy as np

# Load the image
image_path = 'testim.png'
img = cv2.imread(image_path)

# Function to apply chromatic aberration
def chromatic_aberration(image):
    # Split into B, G, R channels
    b, g, r = cv2.split(image)
    # Shift channels to simulate different focal lengths
    shift_x, shift_y = 5, 5
    b_shifted = np.roll(b, shift=(shift_x, shift_y), axis=(0, 1))
    r_shifted = np.roll(r, shift=(-shift_x, -shift_y), axis=(0, 1))
    # Merge back into image
    aberrated = cv2.merge((b_shifted, g, r_shifted))
    return aberrated

# Function to apply spherical aberration
def spherical_aberration(image):
    # Create a Gaussian blur effect
    ksize = (15, 15)  # Kernel size for the blur
    blurred_edges = cv2.GaussianBlur(image, ksize, sigmaX=20)
    return blurred_edges

# Function to apply coma aberration
def coma_aberration(image):
    rows, cols, _ = image.shape
    map_x, map_y = np.meshgrid(np.arange(cols), np.arange(rows))
    # Apply radial distortion increasing towards the edges
    distortion_factor = 0.00002
    map_x = map_x + distortion_factor * (map_x - cols / 2) ** 3
    map_y = map_y + distortion_factor * (map_y - rows / 2) ** 3
    # Remap the image
    distorted = cv2.remap(image, map_x.astype(np.float32), map_y.astype(np.float32), interpolation=cv2.INTER_LINEAR)
    return distorted

# Function to apply astigmatism
def astigmatism(image):
    # Apply directional blur (different in horizontal and vertical directions)
    blurred_horizontal = cv2.GaussianBlur(image, (15, 1), sigmaX=10)
    blurred_vertical = cv2.GaussianBlur(image, (1, 15), sigmaY=10)
    return cv2.addWeighted(blurred_horizontal, 0.5, blurred_vertical, 0.5, 0)

# Function to apply curvature of field
def curvature_of_field(image):
    rows, cols, _ = image.shape
    map_x, map_y = np.meshgrid(np.arange(cols), np.arange(rows))
    curvature_factor = 0.000001
    # Modify the y-coordinate to simulate field curvature
    map_y = map_y + curvature_factor * (map_x - cols / 2) ** 2
    # Remap the image
    curved = cv2.remap(image, map_x.astype(np.float32), map_y.astype(np.float32), interpolation=cv2.INTER_LINEAR)
    return curved

# Function to apply pincushion distortion
def pincushion_distortion(image):
    rows, cols, _ = image.shape
    map_x, map_y = np.meshgrid(np.arange(cols), np.arange(rows))
    # Apply pincushion distortion by expanding points outward
    distortion_factor = 0.0001
    map_x = map_x + distortion_factor * (map_x - cols / 2) * np.sqrt((map_x - cols / 2) ** 2 + (map_y - rows / 2) ** 2)
    map_y = map_y + distortion_factor * (map_y - rows / 2) * np.sqrt((map_x - cols / 2) ** 2 + (map_y - rows / 2) ** 2)
    # Remap the image
    distorted = cv2.remap(image, map_x.astype(np.float32), map_y.astype(np.float32), interpolation=cv2.INTER_LINEAR)
    return distorted

# Function to apply barrel distortion
def barrel_distortion(image):
    rows, cols, _ = image.shape
    map_x, map_y = np.meshgrid(np.arange(cols), np.arange(rows))
    # Apply barrel distortion by compressing points inward
    distortion_factor = -0.0001
    map_x = map_x + distortion_factor * (map_x - cols / 2) * np.sqrt((map_x - cols / 2) ** 2 + (map_y - rows / 2) ** 2)
    map_y = map_y + distortion_factor * (map_y - rows / 2) * np.sqrt((map_x - cols / 2) ** 2 + (map_y - rows / 2) ** 2)
    # Remap the image
    distorted = cv2.remap(image, map_x.astype(np.float32), map_y.astype(np.float32), interpolation=cv2.INTER_LINEAR)
    return distorted

# Apply all aberrations and distortions, save images
chromatic_img = chromatic_aberration(img)
cv2.imwrite('chromatic_aberration.png', chromatic_img)

spherical_img = spherical_aberration(img)
cv2.imwrite('spherical_aberration.png', spherical_img)

coma_img = coma_aberration(img)
cv2.imwrite('coma_aberration.png', coma_img)

astigmatism_img = astigmatism(img)
cv2.imwrite('astigmatism.png', astigmatism_img)

curvature_img = curvature_of_field(img)
cv2.imwrite('curvature_of_field.png', curvature_img)

pincushion_img = pincushion_distortion(img)
cv2.imwrite('pincushion_distortion.png', pincushion_img)

barrel_img = barrel_distortion(img)
cv2.imwrite('barrel_distortion.png', barrel_img)

print("Aberration images saved successfully.")
