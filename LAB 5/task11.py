import cv2
import numpy as np
import os

# Load the video using OpenCV
video_path =r'C:\Users\revat\Downloads\7th sem\image and video analytics\lab\lab 5\Iklan Sweet Talk Bakery - Product Commercial Video.mp4'
cap = cv2.VideoCapture(video_path)

# Check if video loaded successfully
if not cap.isOpened():
    print("Error opening video file")

# Get the frames per second (fps) of the video for calculating timestamps
fps = cap.get(cv2.CAP_PROP_FPS)

# Get the width and height of the video frames
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a directory to store the event frames
event_frames_dir = 'event_frames'
if not os.path.exists(event_frames_dir):
    os.makedirs(event_frames_dir)

# Create VideoWriter object to save the video with annotations
output_video_path = 'event_detection_output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Parameters for motion detection
motion_threshold = 1500  # Adjusted threshold value for motion detection
min_contour_area = 500  # Minimum area of motion to be considered
event_detected_frames = []

# Read the first frame
ret, prev_frame = cap.read()
prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Initialize frame number
frame_count = 0

while cap.isOpened():
    # Read the next frame
    ret, frame = cap.read()
    
    if not ret:
        break  # End of video
    
    # Convert current frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate histogram difference between consecutive frames
    hist_prev = cv2.calcHist([prev_frame_gray], [0], None, [256], [0, 256])
    hist_curr = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])

    # Compute the absolute difference between histograms
    diff = cv2.absdiff(hist_curr, hist_prev)
    motion_score = np.sum(diff)

    # Threshold the motion score to detect significant motion
    if motion_score > motion_threshold:
        event_detected_frames.append(frame_count)
        
        # Calculate timestamp (in seconds) from frame count
        timestamp = frame_count / fps
        
        # Mark the frame where the event is detected with frame number and timestamp
        cv2.putText(frame, f"Frame: {frame_count}, Time: {timestamp:.2f} sec", 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Find regions with motion using frame differencing
    frame_diff = cv2.absdiff(prev_frame_gray, gray_frame)
    _, thresh_diff = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

    # Find contours to locate moving regions
    contours, _ = cv2.findContours(thresh_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around detected motion areas
    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Save the frame if motion score exceeds the threshold (event detected)
    if motion_score > motion_threshold:
        # Save the event frame with bounding boxes, frame number, and timestamp
        event_frame_filename = os.path.join(event_frames_dir, f'event_frame_{frame_count}.png')
        cv2.imwrite(event_frame_filename, frame)

    # Write the frame to the output video
    out.write(frame)

    # Display the frame with detected motion and events
    cv2.imshow("Motion Estimation and Event Detection", frame)

    # Update previous frame
    prev_frame_gray = gray_frame
    frame_count += 1

    # Exit on pressing 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
out.release()  # Release the video writer object
cv2.destroyAllWindows()

# Print the saved event frames along with their timestamps
print("Event detected at frames:", event_detected_frames)

for frame_number in event_detected_frames:
    timestamp = frame_number / fps
    print(f"Event detected at frame {frame_number}, timestamp: {timestamp:.2f} seconds")
