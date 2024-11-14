import cv2

# Load video
video_path = 'vid.mp4' # Update this path with your video file
cap = cv2.VideoCapture(video_path)

# Background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)

# Initialize variables for tracking
person_position = None
person_histogram = None
frame_count = 0

# Function to calculate color histogram for appearance matching
def get_color_histogram(frame, bbox):
    x, y, w, h = bbox
    person_region = frame[y:y+h, x:x+w]
    hist = cv2.calcHist([person_region], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist

# Function to check histogram similarity
def is_similar(hist1, hist2, threshold=0.7):  # Higher threshold for stricter matching
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return similarity > threshold

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Apply background subtraction every few frames to keep the mask updated
    if frame_count % 5 == 0:  # Update every 5 frames
        mask = bg_subtractor.apply(frame)
        
        # Perform morphological operations to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        max_area = 0
        best_bbox = None
        
        for cnt in contours:
            # Ignore small contours
            area = cv2.contourArea(cnt)
            if area < 1000:
                continue
            
            # Get bounding box of the largest contour, assuming it's the person
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Check if this is the largest detected contour
            if area > max_area:
                max_area = area
                best_bbox = (x, y, w, h)
        
        # Initialize or update the tracking position based on histogram similarity
        if best_bbox:
            if person_histogram is None:
                # Set the initial reference histogram
                person_histogram = get_color_histogram(frame, best_bbox)
                person_position = best_bbox
            else:
                # Calculate the current histogram and check if it matches the initial person
                current_histogram = get_color_histogram(frame, best_bbox)
                if is_similar(person_histogram, current_histogram):
                    person_position = best_bbox

    # Draw the bounding box if a person is being tracked
    if person_position:
        x, y, w, h = person_position
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Person", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow("Person Tracking", frame)
    frame_count += 1
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()