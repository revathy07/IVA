import cv2
import dlib


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r'C:\Users\revat\Downloads\7th sem\image and video analytics\lab\lab 5\shape_predictor_68_face_landmarks.dat')  # Download from Dlib website


def classify_emotion(image, shape):
    
    mouth_landmarks = [shape.part(i) for i in range(48, 68)]
    
    
    mouth_width = abs(mouth_landmarks[6].x - mouth_landmarks[0].x) 
    mouth_height = abs(mouth_landmarks[9].y - mouth_landmarks[3].y)  
    
    
    if mouth_width > 2 * mouth_height:
        emotion = 'happy'
    elif mouth_height > mouth_width:
        emotion = 'sad'
    else:
        emotion = 'neutral'
    
    
    for landmark in mouth_landmarks:
        cv2.circle(image, (landmark.x, landmark.y), 1, (255, 0, 0), -1)
    
    return emotion


def detect_faces_and_classify_emotions(image):
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    
    faces = detector(gray_image)
    
    
    for face in faces:
        
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        
        shape = predictor(gray_image, face)
        
        
        emotion = classify_emotion(image, shape)
        
        
        cv2.putText(image, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return image


if __name__ == "__main__":
    
    image_path = r'C:\Users\revat\Downloads\7th sem\image and video analytics\lab\lab 5\img\20_Family_Group_Family_Group_20_101.jpg'  # Replace this with your image path
    image = cv2.imread(image_path)
    
   
    processed_image = detect_faces_and_classify_emotions(image)
    
    
    cv2.imshow("Faces with Emotions", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
