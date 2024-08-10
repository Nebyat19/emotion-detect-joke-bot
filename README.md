# Emotion-Sense

**Emotion-Sense** is a Python-based project that uses real-time facial recognition to analyze emotions and respond with humor. It leverages the DeepFace library for emotion detection and Twilio for sending jokes via SMS.

![Ml](src/image.png)

## Features

- Real-time facial recognition and emotion analysis using your webcam.
- Detects facial landmarks and displays them on the screen.
- Sends a joke via SMS if the detected emotion is not happy or neutral.
- Simple and interactive interface using OpenCV.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/emotion-sense.git
    cd emotion-sense
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the dlib's shape predictor file (`shape_predictor_68_face_landmarks.dat`) and place it in the project directory. You can download it [here](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2).

4. Set up your Twilio credentials by exporting them as environment variables:
    ```bash
    export TWILIO_ACCOUNT_SID='your_account_sid'
    export TWILIO_AUTH_TOKEN='your_auth_token'
    ```

## Usage

1. Run the script:
    ```bash
    python emotion_sense.py
    ```

2. The program will start capturing video from your webcam, detect faces, analyze emotions, and send a joke if needed.

3. Press `q` to quit the application.

## Contributing

Contributions are welcome! Please create an issue first to discuss any changes or features you'd like to add.


