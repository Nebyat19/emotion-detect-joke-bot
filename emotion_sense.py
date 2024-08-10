import os
import cv2
import dlib
import requests
from deepface import DeepFace
from twilio.rest import Client
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Twilio credentials from environment variables
account_sid = os.getenv('TWILIO_ACCOUNT_SID')
auth_token = os.getenv('TWILIO_AUTH_TOKEN')
client = Client(account_sid, auth_token)

# Load dlib's shape predictor and face detector
predictor_path = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

def draw_landmarks(image, landmarks):
    """Draw facial landmarks on the image."""
    for (x, y) in landmarks:
        cv2.circle(image, (x, y), 2, (250, 250, 250), -1)

def get_joke():
    """Fetch a random joke from the JokeAPI."""
    try:
        response = requests.get("https://v2.jokeapi.dev/joke/Any")
        joke = response.json()
        if joke['type'] == 'single':
            return joke['joke']
        else:
            return f"{joke['setup']} ... {joke['delivery']}"
    except Exception as e:
        logger.error(f"Error fetching joke: {e}")
        return "No jokes available right now."

def send_joke(joke, to_number):
    """Send a joke via SMS using Twilio."""
    try:
        client.messages.create(
            from_='+19786432322',  # Replace with your Twilio number
            to=to_number,
            body=joke
        )
    except Exception as e:
        logger.error(f"Error sending joke: {e}")

def main():
    """Main function to capture video and perform emotion recognition and landmark detection."""
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to capture frame from webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces using dlib
        faces = detector(gray)
        
        for face in faces:
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            face_roi = frame[y:y+h, x:x+w]
            
            # Analyze emotion of the face using DeepFace
            try:
                results = DeepFace.analyze(face_roi, actions=['emotion'])
                if isinstance(results, list) and len(results) > 0:
                    emotion = results[0]['dominant_emotion']
                else:
                    emotion = 'Unknown'
                
                cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Send a joke if the emotion is not happy or neutral
                if emotion.lower() not in ['happy', 'neutral']:
                    joke = get_joke()
                    logger.info(f"Sending you a joke: {joke}")
                    send_joke(joke, '+251717277843')  # Replace with the recipient's number
            
            except Exception as e:
                logger.error(f"Error analyzing face: {e}")
                emotion = 'Error'

            # Detect landmarks using dlib
            shape = predictor(gray, face)
            landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
            
            # Draw landmarks on the face
            draw_landmarks(frame, landmarks)
            
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('Emotion Recognition and Landmarks', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
