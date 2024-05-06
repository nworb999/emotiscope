# emotiscope
🥸
## Emotion detection (thanks [manish-9245](https://github.com/manish-9245))
Import the necessary libraries: cv2 for video capture and image processing, and deepface for the emotion detection model.

Load the Haar cascade classifier XML file for face detection using cv2.CascadeClassifier().

Start capturing video from the default webcam using cv2.VideoCapture().

Enter a continuous loop to process each frame of the captured video.

Convert each frame to grayscale using cv2.cvtColor().

Detect faces in the grayscale frame using face_cascade.detectMultiScale().

For each detected face, extract the face ROI (Region of Interest).

Preprocess the face image for emotion detection using the deepface library's built-in preprocessing function.

Make predictions for the emotions using the pre-trained emotion detection model provided by the deepface library.

Retrieve the index of the predicted emotion and map it to the corresponding emotion label.

Draw a rectangle around the detected face and label it with the predicted emotion using cv2.rectangle() and cv2.putText().

Display the resulting frame with the labeled emotion using cv2.imshow().

If the 'q' key is pressed, exit the loop.

Release the video capture and close all windows using cap.release() and cv2.destroyAllWindows().