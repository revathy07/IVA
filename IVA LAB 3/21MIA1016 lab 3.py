import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance
from skimage.feature import graycomatrix, graycoprops

# Load the base image and reference images
base_image_path = 'images.jpeg'
ref_apple_image_path = 'green.jpeg'
ref_orange_image_path = 'orange.jpeg'
base_image = cv2.imread(base_image_path)
ref_apple_image = cv2.imread(ref_apple_image_path)
ref_orange_image = cv2.imread(ref_orange_image_path)

# Convert images to RGB for display
base_image_rgb = cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB)
ref_apple_rgb = cv2.cvtColor(ref_apple_image, cv2.COLOR_BGR2RGB)
ref_orange_rgb = cv2.cvtColor(ref_orange_image, cv2.COLOR_BGR2RGB)

# Display the original image and reference images
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(base_image_rgb)
plt.title('Original Base Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(ref_apple_rgb)
plt.title('Reference Green Apple Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(ref_orange_rgb)
plt.title('Reference Orange Image')
plt.axis('off')

plt.show()

# Function to display segmented objects
def display_segmented_objects(mask, original_image, title):
    segmented_image = cv2.bitwise_and(original_image, original_image, mask=mask)
    plt.imshow(segmented_image)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Apply Gaussian Blur for basic noise removal
base_image_blurred = cv2.GaussianBlur(base_image_rgb, (5, 5), 0)

# Apply Non-Local Means Denoising for advanced noise removal
base_image_denoised = cv2.fastNlMeansDenoisingColored(base_image_blurred, None, 10, 10, 7, 21)

# Display images after Gaussian Blur and Non-Local Means Denoising
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(base_image_rgb)
plt.title('Original Base Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(base_image_blurred)
plt.title('Base Image After Gaussian Blur')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(base_image_denoised)
plt.title('Base Image After Non-Local Means Denoising')
plt.axis('off')

plt.show()

# Convert the denoised image to HSV for segmentation
base_image_denoised_hsv = cv2.cvtColor(base_image_denoised, cv2.COLOR_RGB2HSV)

# Green apple color range in HSV
lower_green_refined = np.array([25, 30, 30])
upper_green_refined = np.array([95, 255, 255])

# Orange color range in HSV
lower_orange_refined = np.array([5, 50, 50])
upper_orange_refined = np.array([30, 255, 255])

# Create the mask for green apples on denoised image
green_mask_denoised = cv2.inRange(base_image_denoised_hsv, lower_green_refined, upper_green_refined)

# Create the mask for oranges on the denoised image using refined range
orange_mask_denoised = cv2.inRange(base_image_denoised_hsv, lower_orange_refined, upper_orange_refined)

# Apply morphological closing to fill gaps for denoised image masks
kernel = np.ones((7, 7), np.uint8)
green_mask_denoised_closed = cv2.morphologyEx(green_mask_denoised, cv2.MORPH_CLOSE, kernel)
orange_mask_denoised_closed = cv2.morphologyEx(orange_mask_denoised, cv2.MORPH_CLOSE, kernel)

# Function to refine masks with watershed
def refine_mask_with_watershed(mask, original_image):
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_bg = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(original_image, markers)
    refined_mask = np.zeros_like(mask)
    refined_mask[markers > 1] = 255
    return refined_mask

# Refine the masks using watershed on the denoised image
green_mask_denoised_watershed = refine_mask_with_watershed(green_mask_denoised_closed, base_image_denoised)
orange_mask_denoised_watershed = refine_mask_with_watershed(orange_mask_denoised_closed, base_image_denoised)

# Display segmented masks after Watershed for green apples and oranges
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
display_segmented_objects(green_mask_denoised_watershed, base_image_rgb, 'Green Apples Segmented after Watershed')

plt.subplot(1, 2, 2)
display_segmented_objects(orange_mask_denoised_watershed, base_image_rgb, 'Oranges Segmented after Watershed')

plt.show()

# Function to extract texture features using GLCM
def extract_texture_features(mask, original_image_gray):
    contour_mask = np.zeros_like(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)

    masked_image = cv2.bitwise_and(original_image_gray, original_image_gray, mask=contour_mask)
    glcm = graycomatrix(masked_image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    asm = graycoprops(glcm, 'ASM')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]

    return {
        'contrast': contrast,
        'dissimilarity': dissimilarity,
        'homogeneity': homogeneity,
        'asm': asm,
        'energy': energy
    }

# Function to extract edge features using Canny edge detection
def extract_edge_features(mask, original_image_gray):
    contour_mask = np.zeros_like(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)

    masked_image = cv2.bitwise_and(original_image_gray, original_image_gray, mask=contour_mask)
    edges = cv2.Canny(masked_image, 100, 200)

    edge_count = np.sum(edges > 0)
    return {
        'edge_count': edge_count
    }

# Function to extract shape, color, texture, and edge features
def extract_shape_color_texture_edge_features(mask, original_image, original_image_gray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    features = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        (x, y), radius = cv2.minEnclosingCircle(contour)
        bounding_circle_area = np.pi * (radius ** 2)
        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
        mean_color = cv2.mean(original_image, mask=contour_mask)[:3]

        texture_features = extract_texture_features(contour_mask, original_image_gray)
        edge_features = extract_edge_features(contour_mask, original_image_gray)

        features.append({
            'area': area,
            'perimeter': perimeter,
            'bounding_circle_area': bounding_circle_area,
            'mean_color': mean_color,
            'texture': texture_features,
            'edges': edge_features
        })
    return features

# Extract features from the denoised base image objects
base_image_gray = cv2.cvtColor(base_image_denoised, cv2.COLOR_RGB2GRAY)
green_apple_features_denoised = extract_shape_color_texture_edge_features(green_mask_denoised_watershed, base_image_denoised, base_image_gray)
orange_features_denoised = extract_shape_color_texture_edge_features(orange_mask_denoised_watershed, base_image_denoised, base_image_gray)

# Extract features from reference images
ref_apple_gray = cv2.cvtColor(ref_apple_rgb, cv2.COLOR_RGB2GRAY)
ref_orange_gray = cv2.cvtColor(ref_orange_rgb, cv2.COLOR_RGB2GRAY)
ref_apple_features = extract_shape_color_texture_edge_features(cv2.inRange(ref_apple_image, lower_green_refined, upper_green_refined), ref_apple_rgb, ref_apple_gray)
ref_orange_features = extract_shape_color_texture_edge_features(cv2.inRange(ref_orange_image, lower_orange_refined, upper_orange_refined), ref_orange_rgb, ref_orange_gray)

# Function to calculate color histogram and compare using chi-square
def compare_histograms(image1, image2):
    hist1 = cv2.calcHist([image1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([image2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)

# Classification function to include texture, edge, and histogram comparisons
def classify_objects_with_features(base_features, ref_apple_features, ref_orange_features):
    classifications = []
    for obj in base_features:
        apple_dist_color = distance.euclidean(obj['mean_color'], ref_apple_features[0]['mean_color'])
        orange_dist_color = distance.euclidean(obj['mean_color'], ref_orange_features[0]['mean_color'])

        apple_texture_dist = abs(obj['texture']['contrast'] - ref_apple_features[0]['texture']['contrast'])
        orange_texture_dist = abs(obj['texture']['contrast'] - ref_orange_features[0]['texture']['contrast'])

        apple_edge_dist = abs(obj['edges']['edge_count'] - ref_apple_features[0]['edges']['edge_count'])
        orange_edge_dist = abs(obj['edges']['edge_count'] - ref_orange_features[0]['edges']['edge_count'])

        apple_score = apple_dist_color + apple_texture_dist + apple_edge_dist
        orange_score = orange_dist_color + orange_texture_dist + orange_edge_dist

        if apple_score < orange_score:
            classifications.append('Green Apple')
        else:
            classifications.append('Orange')

    return classifications

# Classify green apples and oranges based on their extended features from denoised image
green_apple_classifications_denoised = classify_objects_with_features(green_apple_features_denoised, ref_apple_features, ref_orange_features)
orange_classifications_denoised = classify_objects_with_features(orange_features_denoised, ref_apple_features, ref_orange_features)

# Save the classification results and bounding boxes in a labeled dataset
def create_labeled_dataset(base_features, classifications, filename):
    data = []
    for obj, label in zip(base_features, classifications):
        data.append({
            'label': label,
            'area': obj['area'],
            'perimeter': obj['perimeter'],
            'bounding_circle_area': obj['bounding_circle_area'],
            'mean_color': obj['mean_color'],
            'texture_contrast': obj['texture']['contrast'],
            'texture_dissimilarity': obj['texture']['dissimilarity'],
            'edge_count': obj['edges']['edge_count']
        })
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

# Update file paths to the specified location provided by the user
green_apple_filepath = "C:/Users/revat/Downloads/7th sem/image and video analytics/lab/lab 3/green_apples_labeled_extended.csv"
orange_filepath = "C:/Users/revat/Downloads/7th sem/image and video analytics/lab/lab 3/oranges_labeled_extended.csv"

# Create labeled datasets for green apples and oranges
create_labeled_dataset(green_apple_features_denoised, green_apple_classifications_denoised, green_apple_filepath)
create_labeled_dataset(orange_features_denoised, orange_classifications_denoised, orange_filepath)

print(f"Green apple features saved to: {green_apple_filepath}")
print(f"Orange features saved to: {orange_filepath}")

# Function to display contours separately on a blank canvas
def display_contours(mask, title):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blank_canvas = np.zeros_like(base_image_rgb)
    cv2.drawContours(blank_canvas, contours, -1, (0, 255, 0), 2)  # Green contours
    plt.imshow(blank_canvas)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Display contours of green apples and oranges separately
display_contours(green_mask_denoised_watershed, 'Contours of Green Apples')
display_contours(orange_mask_denoised_watershed, 'Contours of Oranges')

# Function to draw bounding circles
def draw_bounding_circles(mask, original_image):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result_image = original_image.copy()
    for contour in contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        # Drawing blue circles (BGR: 255, 0, 0)
        cv2.circle(result_image, center, radius, (255, 0, 255), 8)  
    return result_image

# Draw bounding circles for green apples and oranges on denoised image
green_apples_with_bounding_circles_denoised = draw_bounding_circles(green_mask_denoised_watershed, base_image_denoised)
oranges_with_bounding_circles_denoised = draw_bounding_circles(orange_mask_denoised_watershed, base_image_denoised)

# Display the bounding circles
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(green_apples_with_bounding_circles_denoised)
plt.title('Identified Green Apples (Denoised Image)')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(oranges_with_bounding_circles_denoised)
plt.title('Identified Oranges (Denoised Image)')
plt.axis('off')

plt.show()