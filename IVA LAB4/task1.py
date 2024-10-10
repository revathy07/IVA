import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

# Task 1: Load Video and Frame Extraction
def extract_frames(video_path, output_folder='output_frames'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    video_capture = cv2.VideoCapture(video_path)
    
    if not video_capture.isOpened():
        print("Error: Could not open video.")
        return
    
    frame_count = 0
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        frame_filename = os.path.join(output_folder, f'frame_{frame_count:04d}.jpg')
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
        print(f"Saved {frame_filename}")
    
    video_capture.release()
    print(f"All {frame_count} frames have been extracted and saved.")
    return output_folder

# Task 2: Convert Frames to HSV Color Space
def convert_frames_to_hsv(input_folder, output_hsv_folder='output_hsv_frames'):
    if not os.path.exists(output_hsv_folder):
        os.makedirs(output_hsv_folder)
    
    for frame_filename in sorted(os.listdir(input_folder)):
        frame_path = os.path.join(input_folder, frame_filename)
        frame = cv2.imread(frame_path)

        # Convert frame to HSV color space
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Save the HSV frame
        hsv_frame_filename = os.path.join(output_hsv_folder, frame_filename)
        cv2.imwrite(hsv_frame_filename, hsv_frame)
        print(f"Saved HSV frame: {hsv_frame_filename}")
    
    print("All frames have been converted to HSV and saved.")
    return output_hsv_folder

# Task 3: Spatio-Temporal Segmentation (Sobel Edge Detection)
def sobel_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    grad_x = cv2.filter2D(gray, cv2.CV_64F, sobel_x)
    grad_y = cv2.filter2D(gray, cv2.CV_64F, sobel_y)
    sobel_combined = np.sqrt(grad_x**2 + grad_y**2)
    sobel_combined = np.uint8(np.absolute(sobel_combined))
    return sobel_combined

def apply_sobel_to_frames(input_folder, output_sobel_folder='output_sobel_edges'):
    if not os.path.exists(output_sobel_folder):
        os.makedirs(output_sobel_folder)
    
    for frame_filename in sorted(os.listdir(input_folder)):
        frame_path = os.path.join(input_folder, frame_filename)
        frame = cv2.imread(frame_path)
        sobel_frame = sobel_edge_detection(frame)
        sobel_frame_filename = os.path.join(output_sobel_folder, f'sobel_{frame_filename}')
        cv2.imwrite(sobel_frame_filename, sobel_frame)
        print(f"Saved Sobel edge-detected frame: {sobel_frame_filename}")
    
    print("Sobel edge detection completed for all frames.")
    return output_sobel_folder

# Task 4: Scene Cut Detection using Combined Histogram (Hue, Saturation, Value)
def compute_similarity_scores(hsv_frames_folder, histograms_folder='output_histograms', similarity_scores_file='similarity_scores.txt'):
    if not os.path.exists(histograms_folder):
        os.makedirs(histograms_folder)

    frame_filenames = sorted(os.listdir(hsv_frames_folder))
    similarity_scores = []

    def sobel_edge_detection(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        sobel_edges = np.sqrt(sobel_x**2 + sobel_y**2)
        return sobel_edges

    for i in range(len(frame_filenames) - 1):
        frame1_path = os.path.join(hsv_frames_folder, frame_filenames[i])
        frame2_path = os.path.join(hsv_frames_folder, frame_filenames[i + 1])

        frame1 = cv2.imread(frame1_path)
        frame2 = cv2.imread(frame2_path)

        gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        score_ssim, _ = ssim(gray_frame1, gray_frame2, full=True)

        intensity_diff = np.mean(np.abs(gray_frame1.astype("float") - gray_frame2.astype("float")))

        sobel_frame1 = sobel_edge_detection(frame1)
        sobel_frame2 = sobel_edge_detection(frame2)

        sobel_diff = np.mean(np.abs(sobel_frame1 - sobel_frame2))

        # Compute combined histograms for Hue, Saturation, and Value
        hist1_hue = cv2.calcHist([frame1], [0], None, [50], [0, 180])
        hist1_saturation = cv2.calcHist([frame1], [1], None, [60], [0, 256])
        hist1_value = cv2.calcHist([frame1], [2], None, [60], [0, 256])

        hist2_hue = cv2.calcHist([frame2], [0], None, [50], [0, 180])
        hist2_saturation = cv2.calcHist([frame2], [1], None, [60], [0, 256])
        hist2_value = cv2.calcHist([frame2], [2], None, [60], [0, 256])

        # Normalize histograms
        hist1_hue = cv2.normalize(hist1_hue, hist1_hue).flatten()
        hist1_saturation = cv2.normalize(hist1_saturation, hist1_saturation).flatten()
        hist1_value = cv2.normalize(hist1_value, hist1_value).flatten()

        hist2_hue = cv2.normalize(hist2_hue, hist2_hue).flatten()
        hist2_saturation = cv2.normalize(hist2_saturation, hist2_saturation).flatten()
        hist2_value = cv2.normalize(hist2_value, hist2_value).flatten()

        # Compute histogram intersection for each channel
        intersection_hue = np.minimum(hist1_hue, hist2_hue).sum()
        intersection_saturation = np.minimum(hist1_saturation, hist2_saturation).sum()
        intersection_value = np.minimum(hist1_value, hist2_value).sum()

        total_pixels_hue = hist1_hue.sum() + hist2_hue.sum()
        total_pixels_saturation = hist1_saturation.sum() + hist2_saturation.sum()
        total_pixels_value = hist1_value.sum() + hist2_value.sum()

        # Calculate similarity scores for each channel
        similarity_score_hue = intersection_hue / total_pixels_hue if total_pixels_hue != 0 else 0
        similarity_score_saturation = intersection_saturation / total_pixels_saturation if total_pixels_saturation != 0 else 0
        similarity_score_value = intersection_value / total_pixels_value if total_pixels_value != 0 else 0

        # Combine the similarity scores
        combined_similarity_score = (score_ssim + (1 - intensity_diff / 255) + (1 - sobel_diff / 255) + 
                                     (similarity_score_hue + similarity_score_saturation + similarity_score_value) / 3) / 4
        similarity_scores.append((frame_filenames[i], frame_filenames[i + 1], combined_similarity_score))

        # Plot and save the combined histogram
        plt.figure()
        plt.plot(hist1_hue, color='r', label='Hue')
        plt.plot(hist1_saturation, color='g', label='Saturation')
        plt.plot(hist1_value, color='b', label='Value')
        plt.title(f'Combined Histogram of {frame_filenames[i]}')
        plt.xlabel('Bins')
        plt.ylabel('Frequency')
        plt.legend()

        histogram_filename = os.path.join(histograms_folder, f'histogram_{frame_filenames[i]}.png')
        plt.savefig(histogram_filename)
        plt.close()

        print(f"Saved combined histogram for {frame_filenames[i]} and combined similarity score: {combined_similarity_score}")

    # Write similarity scores to a file
    with open(similarity_scores_file, 'w') as f:
        for frame1, frame2, score in similarity_scores:
            f.write(f'{frame1} - {frame2}: {score}\n')

    print("All similarity calculations done, and similarity scores stored.")
    return similarity_scores_file

# Task 5: Mark Scene Cuts based on Similarity Scores
def detect_scene_cuts_from_similarity(similarity_scores_file, input_folder, scene_cut_folder='scene_cut_frames', threshold=0.6):
    if not os.path.exists(scene_cut_folder):
        os.makedirs(scene_cut_folder)

    cut_detected = []
    
    with open(similarity_scores_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        frame_pair, score = line.strip().split(':')
        frame1, frame2 = frame_pair.split('-')
        score = float(score.strip())

        if score < threshold:
            print(f"Scene cut detected between {frame1} and {frame2} with similarity score: {score}")
            
            frame1_path = os.path.join(input_folder, frame1.strip())
            frame2_path = os.path.join(input_folder, frame2.strip())
            
            frame1_img = cv2.imread(frame1_path)
            frame2_img = cv2.imread(frame2_path)

            cv2.imwrite(os.path.join(scene_cut_folder, f'scene_cut_{frame1.strip()}'), frame1_img)
            cv2.imwrite(os.path.join(scene_cut_folder, f'scene_cut_{frame2.strip()}'), frame2_img)
            
            cut_detected.append((frame1.strip(), frame2.strip()))

    return cut_detected

# Task 6: Result Visualization (Scene Cut Frames)
def visualize_scene_cuts(scene_cut_folder):
    for frame_filename in sorted(os.listdir(scene_cut_folder)):
        frame_path = os.path.join(scene_cut_folder, frame_filename)
        frame = cv2.imread(frame_path)
        cv2.imshow(f'Scene Cut - {frame_filename}', frame)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

# Pipeline execution
video_path = '21MIA1007_sample_Video.mp4'
frame_folder = extract_frames(video_path)

# HSV frame conversion
hsv_folder = convert_frames_to_hsv(frame_folder)

# Sobel segmentation
sobel_folder = apply_sobel_to_frames(hsv_folder)

# Compute similarity scores using Sobel edges, SSIM, pixel intensity, and combined HSV histograms
similarity_scores_file = compute_similarity_scores(hsv_folder)

# Detect scene cuts and save the detected frames
scene_cuts = detect_scene_cuts_from_similarity(similarity_scores_file, frame_folder)

# Visualize scene cut frames
#visualize_scene_cuts('scene_cut_frames')

print("Processing complete.")
