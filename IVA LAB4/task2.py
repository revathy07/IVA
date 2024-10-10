import cv2
import numpy as np

# Initialize background subtractor for background subtraction method
backSub = cv2.createBackgroundSubtractorMOG2()

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2, 
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Initialize CSRT tracker for object tracking
tracker = cv2.TrackerCSRT_create()

def background_subtraction_tracking(video_path):
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply background subtraction
        fg_mask = backSub.apply(frame)

        # Find contours in the mask (foreground objects)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes around detected objects
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter out small objects
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the result
        cv2.imshow('Background Subtraction Tracking', frame)
        if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

def optical_flow_tracking(video_path):
    cap = cv2.VideoCapture(video_path)

    # Take first frame and find corners to track
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # Detect initial points to track using ShiTomasi Corner Detection
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Create a mask image for drawing the tracks
    mask = np.zeros_like(old_frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow between the previous frame and the current frame
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
            frame = cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)

        # Overlay the current frame with the mask
        img = cv2.add(frame, mask)

        # Display the frame
        cv2.imshow('Optical Flow Tracking', img)
        if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit
            break

        # Update the previous frame and points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    cap.release()
    cv2.destroyAllWindows()

def csrt_object_tracking(video_path):
    cap = cv2.VideoCapture(video_path)

    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to read video")
        return

    # Select the object to track
    bbox = cv2.selectROI('Tracking', frame, False)
    tracker.init(frame, bbox)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Update the tracker with the new frame
        success, bbox = tracker.update(frame)

        if success:
            # If tracking is successful, draw the bounding box
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            # If tracking failed
            cv2.putText(frame, "Tracking Failure", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Display the result
        cv2.imshow("CSRT Object Tracking", frame)
        if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

# Main function to run the tracking methods
def run_tracking(video_path, method="background_subtraction"):
    if method == "background_subtraction":
        print("Running Background Subtraction Tracking")
        background_subtraction_tracking(video_path)
    elif method == "optical_flow":
        print("Running Optical Flow Tracking")
        optical_flow_tracking(video_path)
    elif method == "csrt_object_tracking":
        print("Running CSRT Object Tracking")
        csrt_object_tracking(video_path)
    else:
        print("Invalid method. Please choose from 'background_subtraction', 'optical_flow', or 'csrt_object_tracking'.")

# Example usage
video_path = 'videoplayback.mp4'

# You can change the method to 'optical_flow', 'csrt_object_tracking', or 'background_subtraction'
run_tracking(video_path, method='csrt_object_tracking')
