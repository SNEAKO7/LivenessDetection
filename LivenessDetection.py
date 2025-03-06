'''import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist

# Define EAR calculation for blink detection
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Define a simple CNN model (simulated pre-trained)
def create_liveness_model(input_shape=(64, 64, 3)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # 0 = spoof, 1 = live
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Preprocess face for model input
def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (64, 64))
    face_img = face_img / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    return face_img

# Initialize CNN model
model = create_liveness_model()

# Initialize dlib face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Variables
prev_frame = None
motion_threshold = 10000  # Adjust based on testing
EAR_THRESHOLD = 0.2  # Eye aspect ratio threshold for blink
BLINK_CONSEC_FRAMES = 3  # Consecutive frames for a blink
blink_counter = 0
blink_detected = False
(left_start, left_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_start, right_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Main loop
# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert to grayscale for motion detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces with dlib
    rects = detector(gray, 0)
    status = "Liveness: Unknown"

    # Motion detection
    motion_detected = False
    if prev_frame is not None:
        diff = cv2.absdiff(gray, prev_frame)
        motion_score = np.sum(diff)
        motion_detected = motion_score > motion_threshold
        print(f"Motion Score: {motion_score}, Detected: {motion_detected}")

    # Process each detected face
    for rect in rects:
        # Get facial landmarks
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Extract eye coordinates
        left_eye = shape[left_start:left_end]
        right_eye = shape[right_start:right_end]

        # Calculate EAR for both eyes
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Blink detection
        if ear < EAR_THRESHOLD:
            blink_counter += 1
        else:
            if blink_counter >= BLINK_CONSEC_FRAMES:
                blink_detected = True
            blink_counter = 0
        print(f"EAR: {ear}, Blink Counter: {blink_counter}, Blink Detected: {blink_detected}")

        # Extract face region for AI model
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        face_roi = frame[y:y+h, x:x+w]
        if face_roi.size == 0:
            continue
        face_input = preprocess_face(face_roi)

        # Predict liveness with AI model
        liveness_score = model.predict(face_input, verbose=0)[0][0]
        ai_liveness = liveness_score > 0.5
        print(f"AI Liveness Score: {liveness_score}, AI Liveness: {ai_liveness}")

        # Combine AI, motion, and blink for final decision
        if ai_liveness and motion_detected and blink_detected:
            status = "Liveness: Real"
            color = (0, 255, 0)  # Green
        else:
            status = "Liveness: Spoof"
            color = (0, 0, 255)  # Red

        # Draw rectangle and labels
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"AI Score: {liveness_score:.2f}", (x, y-30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(frame, f"EAR: {ear:.2f}", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display status
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Blink: {'Yes' if blink_detected else 'No'}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Motion: {'Yes' if motion_detected else 'No'}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Show frame
    cv2.imshow('Advanced Liveness Detection', frame)

    # Update previous frame
    prev_frame = gray.copy()

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()'''

'''Raised EAR_THRESHOLD to 0.25 (more likely to catch blinks).
Reduced BLINK_CONSEC_FRAMES to 2 (faster blink detection).
Added green dots on eye landmarks to visually confirm dlib is tracking my eyes correctly.
Motion Detection:
Increased motion_threshold to 20000 (less sensitive to minor changes).
Added a 5-frame moving average (motion_history) to smooth out fluctuations.
Resized the frame to 640x480 for consistent motion calculation.
Liveness Logic:
Temporarily ignored the AI prediction (ai_liveness), relying only on motion and blinks.
Kept AI score printing for reference.
Debug Output:
Prints Avg Motion, EAR, Blink Counter, Blink Detected, and AI Score to the terminal.
Shows motion and blink status on-screen.'''
'''
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist

# Define EAR calculation for blink detection
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Define a simple CNN model (simulated pre-trained)
def create_liveness_model(input_shape=(64, 64, 3)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # 0 = spoof, 1 = live
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Preprocess face for model input
def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (64, 64))
    face_img = face_img / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    return face_img

# Initialize CNN model
model = create_liveness_model()

# Initialize dlib face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Variables
prev_frame = None
motion_threshold = 20000  # Increased for less sensitivity
EAR_THRESHOLD = 0.25  # Raised to catch blinks more easily
BLINK_CONSEC_FRAMES = 2  # Reduced for faster blink detection
blink_counter = 0
blink_detected = False
(left_start, left_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_start, right_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
motion_history = []  # For smoothing motion
MOTION_HISTORY_SIZE = 5  # Average over 5 frames

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Resize frame for faster processing
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces with dlib
    rects = detector(gray, 0)
    status = "Liveness: Unknown"

    # Motion detection with smoothing
    motion_detected = False
    if prev_frame is not None:
        diff = cv2.absdiff(gray, prev_frame)
        motion_score = np.sum(diff)
        motion_history.append(motion_score)
        if len(motion_history) > MOTION_HISTORY_SIZE:
            motion_history.pop(0)
        avg_motion = np.mean(motion_history)
        motion_detected = avg_motion > motion_threshold
        print(f"Avg Motion: {avg_motion:.0f}, Detected: {motion_detected}")

    # Process each detected face
    for rect in rects:
        # Get facial landmarks
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Extract eye coordinates
        left_eye = shape[left_start:left_end]
        right_eye = shape[right_start:right_end]

        # Draw eye landmarks for debugging
        for (x, y) in left_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        for (x, y) in right_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Calculate EAR for both eyes
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Blink detection
        if ear < EAR_THRESHOLD:
            blink_counter += 1
        else:
            if blink_counter >= BLINK_CONSEC_FRAMES:
                blink_detected = True
            blink_counter = 0
        print(f"EAR: {ear:.2f}, Blink Counter: {blink_counter}, Blink Detected: {blink_detected}")

        # Extract face region for AI model
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        face_roi = frame[y:y+h, x:x+w]
        if face_roi.size == 0:
            continue
        face_input = preprocess_face(face_roi)

        # Predict liveness with AI model (for reference only)
        liveness_score = model.predict(face_input, verbose=0)[0][0]
        print(f"AI Score: {liveness_score:.2f}")

        # Liveness decision (ignore AI for now)
        if motion_detected and blink_detected:
            status = "Liveness: Real"
            color = (0, 255, 0)  # Green
        else:
            status = "Liveness: Spoof"
            color = (0, 0, 255)  # Red

        # Draw rectangle and labels
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"EAR: {ear:.2f}", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display status
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Blink: {'Yes' if blink_detected else 'No'}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Motion: {'Yes' if motion_detected else 'No'}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Show frame
    cv2.imshow('Advanced Liveness Detection', frame)

    # Update previous frame
    prev_frame = gray.copy()

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()'''

'''Blink Detection:
Lowered EAR_THRESHOLD to 0.22 (my blinks hit 0.20).
Set BLINK_CONSEC_FRAMES to 1 (catches single-frame blinks).
blink_detected now triggers on any blink and resets only after a 30-frame "liveness window" expires.
Motion Detection:
Lowered motion_threshold to 400,000 (my motion often exceeds this, e.g., 605,112, but drops below 500,000 when still, e.g., 446,244).
Liveness Logic:
Added a liveness_window (30 frames ~1 second at 30fps). Once a blink is detected, you’re "Real" for 30 frames if motion is present, making it less strict.
blink_detected persists during this window, aligning better with motion.
Debugging:
Kept eye landmarks and print statements for verification.'''
'''
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist

# Define EAR calculation
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Define CNN model (simulated)
def create_liveness_model(input_shape=(64, 64, 3)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Preprocess face for AI
def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (64, 64))
    face_img = face_img / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    return face_img

# Initialize model
model = create_liveness_model()

# Initialize dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Variables
prev_frame = None
motion_threshold = 400000  # Lowered for subtle movement
EAR_THRESHOLD = 0.22  # Adjusted to catch my blinks
BLINK_CONSEC_FRAMES = 1  # Reduced for quick blinks
blink_counter = 0
blink_detected = False
liveness_window = 30  # Frames to hold "Real" status
liveness_counter = 0  # Countdown for liveness window
(left_start, left_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_start, right_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
motion_history = []
MOTION_HISTORY_SIZE = 5

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    rects = detector(gray, 0)
    status = "Liveness: Spoof"  # Default to Spoof

    # Motion detection
    motion_detected = False
    if prev_frame is not None:
        diff = cv2.absdiff(gray, prev_frame)
        motion_score = np.sum(diff)
        motion_history.append(motion_score)
        if len(motion_history) > MOTION_HISTORY_SIZE:
            motion_history.pop(0)
        avg_motion = np.mean(motion_history)
        motion_detected = avg_motion > motion_threshold
        print(f"Avg Motion: {avg_motion:.0f}, Detected: {motion_detected}")

    # Process faces
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Eye coordinates
        left_eye = shape[left_start:left_end]
        right_eye = shape[right_start:right_end]

        # Draw eye landmarks
        for (x, y) in left_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        for (x, y) in right_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Calculate EAR
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Blink detection
        if ear < EAR_THRESHOLD:
            blink_counter += 1
        else:
            if blink_counter >= BLINK_CONSEC_FRAMES:
                blink_detected = True
                liveness_counter = liveness_window  # Start liveness window
            blink_counter = 0
        print(f"EAR: {ear:.2f}, Blink Counter: {blink_counter}, Blink Detected: {blink_detected}")

        # Extract face for AI (reference only)
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        face_roi = frame[y:y+h, x:x+w]
        if face_roi.size == 0:
            continue
        face_input = preprocess_face(face_roi)
        liveness_score = model.predict(face_input, verbose=0)[0][0]
        print(f"AI Score: {liveness_score:.2f}")

        # Liveness decision
        if motion_detected and blink_detected and liveness_counter > 0:
            status = "Liveness: Real"
            color = (0, 255, 0)
        else:
            status = "Liveness: Spoof"
            color = (0, 0, 255)

        # Draw rectangle and labels
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"EAR: {ear:.2f}", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Update liveness counter
    if liveness_counter > 0:
        liveness_counter -= 1
        if liveness_counter == 0:
            blink_detected = False  # Reset after window expires

    # Display status
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Blink: {'Yes' if blink_detected else 'No'}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Motion: {'Yes' if motion_detected else 'No'}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Show frame
    cv2.imshow('Advanced Liveness Detection', frame)

    # Update previous frame
    prev_frame = gray.copy()

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows() #current best version'''

'''Persistent Liveness:
liveness_buffer = 60: Keeps you "Real" for ~2 seconds after motion or a blink, even if you’re briefly still.
liveness_counter only resets fully if no blinks occur, avoiding quick "Spoof" flips.
Spoof Detection:
Blink Timeout: blink_timeout = 150 (~5 seconds). If no blink occurs within 5 seconds, it flags as "Spoof" (videos/pictures often lack natural blinks).
Motion Variance: Added motion_variance_history (20 frames).
Low variance (<10,000) with some motion (>10,000) suggests a video.
Very low variance (<5,000) and low motion (<10,000) suggests a picture.
is_spoof flag combines these checks.
Tuning:
Kept motion_threshold = 400000 and EAR_THRESHOLD = 0.22, as they caught my motion and blinks previously.
Added blink_timer display to track blink intervals.'''
'''import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist
from collections import deque

# Define EAR calculation
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Define CNN model (simulated)
def create_liveness_model(input_shape=(64, 64, 3)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Preprocess face for AI
def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (64, 64))
    face_img = face_img / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    return face_img

# Initialize model
model = create_liveness_model()

# Initialize dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Variables
prev_frame = None
motion_threshold = 400000  # For significant movement
EAR_THRESHOLD = 0.22  # For my blinks (~0.20)
BLINK_CONSEC_FRAMES = 1  # Quick blinks
blink_counter = 0
blink_detected = False
liveness_buffer = 60  # ~2 seconds at 30fps
liveness_counter = 0  # Countdown for liveness
blink_timeout = 150  # ~5 seconds at 30fps, require blink every 5s
blink_timer = 0  # Track frames since last blink
(left_start, left_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_start, right_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
motion_history = deque(maxlen=5)  # 5-frame average
motion_variance_history = deque(maxlen=20)  # 20-frame variance for spoof check

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    rects = detector(gray, 0)
    status = "Liveness: Spoof"  # Default

    # Motion detection
    motion_detected = False
    motion_variance = 0
    if prev_frame is not None:
        diff = cv2.absdiff(gray, prev_frame)
        motion_score = np.sum(diff)
        motion_history.append(motion_score)
        avg_motion = np.mean(motion_history)
        motion_detected = avg_motion > motion_threshold
        
        # Calculate motion variance for spoof detection
        if len(motion_history) == motion_history.maxlen:
            motion_variance = np.var(motion_history)
            motion_variance_history.append(motion_variance)
        avg_variance = np.mean(motion_variance_history) if motion_variance_history else 0
        print(f"Avg Motion: {avg_motion:.0f}, Detected: {motion_detected}, Variance: {avg_variance:.0f}")

    # Process faces
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Eye coordinates
        left_eye = shape[left_start:left_end]
        right_eye = shape[right_start:right_end]

        # Draw eye landmarks
        for (x, y) in left_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        for (x, y) in right_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Calculate EAR
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Blink detection
        if ear < EAR_THRESHOLD:
            blink_counter += 1
        else:
            if blink_counter >= BLINK_CONSEC_FRAMES:
                blink_detected = True
                liveness_counter = liveness_buffer  # Refresh liveness
                blink_timer = 0  # Reset blink timer
            blink_counter = 0
        print(f"EAR: {ear:.2f}, Blink Counter: {blink_counter}, Blink Detected: {blink_detected}")

        # AI prediction (reference only)
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        face_roi = frame[y:y+h, x:x+w]
        if face_roi.size == 0:
            continue
        face_input = preprocess_face(face_roi)
        liveness_score = model.predict(face_input, verbose=0)[0][0]
        print(f"AI Score: {liveness_score:.2f}")

        # Spoof detection logic
        is_spoof = False
        blink_timer += 1  # Increment frame counter
        
        # Check for lack of blinks (videos/pictures often lack natural blinks)
        if blink_timer > blink_timeout:
            is_spoof = True
            print("No blink detected in 5 seconds - possible spoof")

        # Check motion variance (pictures = low variance, videos = predictable variance)
        if avg_variance < 10000 and avg_motion > 10000:  # Low variance but some motion = video
            is_spoof = True
            print("Low motion variance - possible video spoof")
        elif avg_variance < 5000 and avg_motion < 10000:  # Very low variance and motion = picture
            is_spoof = True
            print("Minimal motion and variance - possible picture spoof")

        # Liveness decision
        if motion_detected and not is_spoof and liveness_counter > 0:
            status = "Liveness: Real"
            color = (0, 255, 0)
        else:
            status = "Liveness: Spoof"
            color = (0, 0, 255)

        # Draw rectangle and labels
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"EAR: {ear:.2f}", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Update liveness counter
    if liveness_counter > 0:
        liveness_counter -= 1
    elif not blink_detected:
        liveness_counter = 0  # Reset only if no recent blink

    # Display status
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Blink: {'Yes' if blink_detected else 'No'}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Motion: {'Yes' if motion_detected else 'No'}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Blink Timer: {blink_timer}", (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Show frame
    cv2.imshow('Advanced Liveness Detection', frame)

    # Update previous frame
    prev_frame = gray.copy()

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()'''

'''Video Detection:
The new threshold (avg_variance < 1e11) catches the video phase where variance is ~7.2e10, flagging it as a spoof when motion is present (e.g., >100,000).
In my logs, the last 20 seconds will trigger "Low motion variance with motion - possible video spoof."
Live Misclassification:
Variance during my presence (e.g., 1e12) is well above 1e11, preventing false spoof triggers.
Extended liveness_buffer (90 frames) ensures "Real" persists longer, reducing flickers to "Spoof."
Blink Pattern Sensitivity:
Std dev < 50 frames detects more subtle regularities in video blinks, adding an extra layer of spoof detection.'''

'''import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist
from collections import deque

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def create_liveness_model(input_shape=(64, 64, 3)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (64, 64))
    face_img = face_img / 255.0
    return np.expand_dims(face_img, axis=0)

model = create_liveness_model()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Variables
prev_frame = None
motion_threshold = 200000  # Lowered for subtle live motion
EAR_THRESHOLD = 0.22
BLINK_CONSEC_FRAMES = 1
blink_counter = 0
blink_detected = False
liveness_buffer = 90
liveness_counter = 0
blink_timeout = 150
blink_timer = 0
frame_counter = 0
prev_ear_below_threshold = False
blink_events = []
intervals = deque(maxlen=5)
(left_start, left_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_start, right_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
motion_history = deque(maxlen=5)
motion_variance_history = deque(maxlen=20)
avg_variance = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    frame_counter += 1
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    status = "Liveness: Spoof"

    # Motion detection
    motion_detected = False
    if prev_frame is not None:
        diff = cv2.absdiff(gray, prev_frame)
        motion_score = np.sum(diff)
        motion_history.append(motion_score)
        avg_motion = np.mean(motion_history)
        motion_detected = avg_motion > motion_threshold
        if len(motion_history) == motion_history.maxlen:
            motion_variance = np.var(motion_history)
            motion_variance_history.append(motion_variance)
        avg_variance = np.mean(motion_variance_history) if motion_variance_history else 0
        print(f"Avg Motion: {avg_motion:.0f}, Detected: {motion_detected}, Variance: {avg_variance:.0f}")

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        left_eye = shape[left_start:left_end]
        right_eye = shape[right_start:right_end]

        for (x, y) in left_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        for (x, y) in right_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Blink detection
        if ear < EAR_THRESHOLD:
            if not prev_ear_below_threshold:
                blink_counter = 1
            else:
                blink_counter += 1
            prev_ear_below_threshold = True
        else:
            if prev_ear_below_threshold and blink_counter >= BLINK_CONSEC_FRAMES:
                blink_detected = True
                liveness_counter = liveness_buffer
                blink_timer = 0
                blink_events.append(frame_counter)
                if len(blink_events) > 1:
                    interval = blink_events[-1] - blink_events[-2]
                    intervals.append(interval)
                    if len(intervals) >= 3 and np.std(intervals) < 50:
                        print("Regular blink pattern detected - possible video spoof")
                if len(blink_events) > 10:
                    blink_events.pop(0)
            blink_detected = False  # Reset after blink ends
            blink_counter = 0
            prev_ear_below_threshold = False
        print(f"EAR: {ear:.2f}, Blink Counter: {blink_counter}, Blink Detected: {blink_detected}")

        (x, y, w, h) = face_utils.rect_to_bb(rect)
        face_roi = frame[y:y+h, x:x+w]
        if face_roi.size == 0:
            continue
        face_input = preprocess_face(face_roi)
        liveness_score = model.predict(face_input, verbose=0)[0][0]
        print(f"AI Score: {liveness_score:.2f}")

        # Spoof detection
        is_spoof = False
        blink_timer += 1

        if blink_timer > blink_timeout:
            is_spoof = True
            print("No blink detected in 5 seconds - possible spoof")
        if avg_variance < 5e9 and avg_motion > 100000:  # Lowered to 5e9 for videos
            is_spoof = True
            print("Low motion variance with motion - possible video spoof")
        elif avg_motion < 50000 and avg_variance < 1e9:
            is_spoof = True
            print("Minimal motion and variance - possible picture spoof")

        if motion_detected and not is_spoof and (liveness_counter > 0 or blink_detected):
            status = "Liveness: Real"
            color = (0, 255, 0)
            if blink_detected:
                liveness_counter = liveness_buffer  # Refresh on blink
        else:
            status = "Liveness: Spoof"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"EAR: {ear:.2f}", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if liveness_counter > 0:
        liveness_counter -= 1

    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Blink: {'Yes' if blink_detected else 'No'}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Motion: {'Yes' if motion_detected else 'No'}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Blink Timer: {blink_timer}", (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow('Advanced Liveness Detection', frame)
    prev_frame = gray.copy()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()'''

'''Blink Detection Logic:
blink_detected is now set to True when ear < EAR_THRESHOLD and blink_counter >= BLINK_CONSEC_FRAMES (1), ensuring it’s active during the blink itself, not just at the end.
Resets to False when the blink ends (EAR rises above 0.22), allowing the system to track each blink event accurately.
This fixes the issue where blinks (e.g., EAR 0.12, 0.20) weren’t registering as True in my logs.
Spoof Detection Sensitivity:
Increased the regular blink pattern threshold to np.std(intervals) < 100 (from 50) and required at least 5 intervals (instead of 3). my blinks every second (~30 frames) will have natural variance, avoiding false spoof flags unless they’re unnaturally regular (e.g., std_dev < 20).
Kept the video spoof condition avg_variance < 5e9 and avg_motion > 100000, which won’t trigger for my high variance (e.g., 1e10+), but will catch videos with lower variance.
Blink Timeout Enforcement:
blink_timer increments every frame and resets to 0 only when a blink is detected (EAR < 0.22), ensuring videos with sparse blinks timeout after 5 seconds (150 frames).
Liveness Persistence:
The condition motion_detected and not is_spoof and (liveness_counter > 0 or blink_detected) ensures "Real" status during blinks and for 90 frames (~3 seconds) afterward, preventing brief motion dips from flipping to "Spoof."'''

'''import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist
from collections import deque

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def create_liveness_model(input_shape=(64, 64, 3)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (64, 64))
    face_img = face_img / 255.0
    return np.expand_dims(face_img, axis=0)

# Initialize model and detectors
model = create_liveness_model()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Variables
prev_frame = None
motion_threshold = 200000  # Detects subtle live motion
EAR_THRESHOLD = 0.22  # Blink detection threshold
BLINK_CONSEC_FRAMES = 1  # Single frame for a blink
blink_counter = 0
blink_detected = False
liveness_buffer = 90  # ~3 seconds at 30 fps
liveness_counter = 0
blink_timeout = 150  # ~5 seconds at 30 fps
blink_timer = 0
frame_counter = 0
prev_ear_below_threshold = False
blink_events = []
intervals = deque(maxlen=5)
(left_start, left_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_start, right_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
motion_history = deque(maxlen=5)
motion_variance_history = deque(maxlen=20)
avg_variance = 0

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    frame_counter += 1
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    status = "Liveness: Spoof"

    # Motion detection
    motion_detected = False
    if prev_frame is not None:
        diff = cv2.absdiff(gray, prev_frame)
        motion_score = np.sum(diff)
        motion_history.append(motion_score)
        avg_motion = np.mean(motion_history)
        motion_detected = avg_motion > motion_threshold
        if len(motion_history) == motion_history.maxlen:
            motion_variance = np.var(motion_history)
            motion_variance_history.append(motion_variance)
        avg_variance = np.mean(motion_variance_history) if motion_variance_history else 0
        print(f"Avg Motion: {avg_motion:.0f}, Detected: {motion_detected}, Variance: {avg_variance:.0f}")

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        left_eye = shape[left_start:left_end]
        right_eye = shape[right_start:right_end]

        for (x, y) in left_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        for (x, y) in right_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Blink detection
        if ear < EAR_THRESHOLD:
            if not prev_ear_below_threshold:
                blink_counter = 1
            else:
                blink_counter += 1
            if blink_counter >= BLINK_CONSEC_FRAMES:
                blink_detected = True
                blink_timer = 0  # Reset timer on blink start
        else:
            if prev_ear_below_threshold and blink_counter >= BLINK_CONSEC_FRAMES:
                blink_events.append(frame_counter)
                if len(blink_events) > 1:
                    interval = blink_events[-1] - blink_events[-2]
                    intervals.append(interval)
            blink_detected = False
            blink_counter = 0
        prev_ear_below_threshold = ear < EAR_THRESHOLD
        print(f"EAR: {ear:.2f}, Blink Counter: {blink_counter}, Blink Detected: {blink_detected}")

        (x, y, w, h) = face_utils.rect_to_bb(rect)
        face_roi = frame[y:y+h, x:x+w]
        if face_roi.size == 0:
            continue
        face_input = preprocess_face(face_roi)
        liveness_score = model.predict(face_input, verbose=0)[0][0]
        print(f"AI Score: {liveness_score:.2f}")

        # Spoof detection
        is_spoof = False
        blink_timer += 1

        if blink_timer > blink_timeout:
            is_spoof = True
            print("No blink detected in 5 seconds - possible spoof")
        elif len(intervals) >= 5 and np.std(intervals) < 100:
            is_spoof = True
            print("Regular blink pattern detected - possible video spoof")
        elif avg_variance < 5e9 and avg_motion > 100000:
            is_spoof = True
            print("Low motion variance with motion - possible video spoof")
        elif avg_motion < 50000 and avg_variance < 1e9:
            is_spoof = True
            print("Minimal motion and variance - possible picture spoof")

        # Liveness decision
        if motion_detected and not is_spoof and (liveness_counter > 0 or blink_detected):
            status = "Liveness: Real"
            color = (0, 255, 0)
            if blink_detected:
                liveness_counter = liveness_buffer  # Refresh on blink
        else:
            status = "Liveness: Spoof"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"EAR: {ear:.2f}", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if liveness_counter > 0:
        liveness_counter -= 1

    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Blink: {'Yes' if blink_detected else 'No'}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Motion: {'Yes' if motion_detected else 'No'}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Blink Timer: {blink_timer}", (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow('Advanced Liveness Detection', frame)
    prev_frame = gray.copy()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()'''

'''Blink Detection Logic:
blink_detected is set to True when ear < EAR_THRESHOLD (0.22) and remains True during the blink, ensuring it’s active when a blink occurs (e.g., EAR: 0.19, Blink Detected: True).
Once the blink ends (EAR rises above 0.22), it resets to False and logs the blink event, allowing the system to track full blink cycles correctly.
Extended Liveness Persistence:
Increased liveness_buffer from 90 to 120 frames (~4 seconds at 30 fps). This ensures that after a blink or motion detection, the "Real" status persists longer, preventing quick drops to "Spoof."
liveness_counter is refreshed to 120 whenever blink_detected is True, keeping you classified as "Real" as long as you’re blinking naturally.
Reduced Spoof Detection Sensitivity:
Adjusted the "regular blink pattern" threshold from np.std(intervals) < 100 to < 150. This allows more natural variation in my blink timing (e.g., intervals of 30-60 frames for blinks every 1-2 seconds) without triggering a spoof alert, while still catching highly regular video patterns (e.g., std_dev < 20).
Blink Timer Management:
blink_timer increments every frame and resets to 0 on each detected blink, ensuring that blink_timeout (150 frames, ~5 seconds) only triggers for long periods without blinks, which is more typical of spoofs.'''

'''import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist
from collections import deque

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def create_liveness_model(input_shape=(64, 64, 3)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (64, 64))
    face_img = face_img / 255.0
    return np.expand_dims(face_img, axis=0)

# Initialize model and detectors
model = create_liveness_model()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Variables
prev_frame = None
motion_threshold = 200000  # Detects subtle live motion
EAR_THRESHOLD = 0.22  # Blink detection threshold
BLINK_CONSEC_FRAMES = 1  # Single frame for a blink
blink_counter = 0
blink_detected = False
liveness_buffer = 120  # ~4 seconds at 30 fps (increased from 90)
liveness_counter = 0
blink_timeout = 150  # ~5 seconds at 30 fps
blink_timer = 0
frame_counter = 0
prev_ear_below_threshold = False
blink_events = []
intervals = deque(maxlen=5)
(left_start, left_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_start, right_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
motion_history = deque(maxlen=5)
motion_variance_history = deque(maxlen=20)
avg_variance = 0

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    frame_counter += 1
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    status = "Liveness: Spoof"

    # Motion detection
    motion_detected = False
    if prev_frame is not None:
        diff = cv2.absdiff(gray, prev_frame)
        motion_score = np.sum(diff)
        motion_history.append(motion_score)
        avg_motion = np.mean(motion_history)
        motion_detected = avg_motion > motion_threshold
        if len(motion_history) == motion_history.maxlen:
            motion_variance = np.var(motion_history)
            motion_variance_history.append(motion_variance)
        avg_variance = np.mean(motion_variance_history) if motion_variance_history else 0
        print(f"Avg Motion: {avg_motion:.0f}, Detected: {motion_detected}, Variance: {avg_variance:.0f}")

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        left_eye = shape[left_start:left_end]
        right_eye = shape[right_start:right_end]

        for (x, y) in left_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        for (x, y) in right_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Blink detection
        if ear < EAR_THRESHOLD:
            if not prev_ear_below_threshold:
                blink_counter = 1
            else:
                blink_counter += 1
            if blink_counter >= BLINK_CONSEC_FRAMES:
                blink_detected = True
                blink_timer = 0  # Reset timer on blink start
        else:
            if prev_ear_below_threshold and blink_counter >= BLINK_CONSEC_FRAMES:
                blink_events.append(frame_counter)
                if len(blink_events) > 1:
                    interval = blink_events[-1] - blink_events[-2]
                    intervals.append(interval)
            blink_detected = False
            blink_counter = 0
        prev_ear_below_threshold = ear < EAR_THRESHOLD
        print(f"EAR: {ear:.2f}, Blink Counter: {blink_counter}, Blink Detected: {blink_detected}")

        (x, y, w, h) = face_utils.rect_to_bb(rect)
        face_roi = frame[y:y+h, x:x+w]
        if face_roi.size == 0:
            continue
        face_input = preprocess_face(face_roi)
        liveness_score = model.predict(face_input, verbose=0)[0][0]
        print(f"AI Score: {liveness_score:.2f}")

        # Spoof detection
        is_spoof = False
        blink_timer += 1

        if blink_timer > blink_timeout:
            is_spoof = True
            print("No blink detected in 5 seconds - possible spoof")
        elif len(intervals) >= 5 and np.std(intervals) < 150:  # Increased from 100 to 150
            is_spoof = True
            print("Regular blink pattern detected - possible video spoof")
        elif avg_variance < 5e9 and avg_motion > 100000:
            is_spoof = True
            print("Low motion variance with motion - possible video spoof")
        elif avg_motion < 50000 and avg_variance < 1e9:
            is_spoof = True
            print("Minimal motion and variance - possible picture spoof")

        # Liveness decision
        if motion_detected and not is_spoof and (liveness_counter > 0 or blink_detected):
            status = "Liveness: Real"
            color = (0, 255, 0)
            if blink_detected:
                liveness_counter = liveness_buffer  # Refresh on every blink
        else:
            status = "Liveness: Spoof"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"EAR: {ear:.2f}", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if liveness_counter > 0:
        liveness_counter -= 1

    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Blink: {'Yes' if blink_detected else 'No'}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Motion: {'Yes' if motion_detected else 'No'}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Blink Timer: {blink_timer}", (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow('Advanced Liveness Detection', frame)
    prev_frame = gray.copy()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()'''

'''Key Changes Explained
Lowered EAR_THRESHOLD to 0.20 (from 0.22):
Why: my EAR values (e.g., 0.16, 0.21) indicate blinks are occurring but sometimes hover near 0.22, missing detection. Lowering the threshold ensures more blinks are caught, improving reliability for real users.
Impact: Increases sensitivity to subtle eye closures, reducing missed blinks.
Increased BLINK_CONSEC_FRAMES to 2 (from 1):
Why: Requiring a blink to last at least 2 frames confirms a full eye closure and reopening cycle, preventing false positives from brief EAR dips and ensuring only genuine blinks are counted.
Impact: Enhances blink detection accuracy, reducing the chance of missing real blinks.
Improved Blink Detection Logic:
Why: The previous logic wasn’t consistently registering blinks, as Blink Detected stayed False even during obvious blinks (e.g., EAR 0.16). The new logic sets blink_detected to True during the blink and resets it only after confirmation, ensuring blinks are tracked properly.
Change: blink_detected = (blink_counter >= BLINK_CONSEC_FRAMES) is set within the if ear < EAR_THRESHOLD block, and the full blink cycle is recorded when EAR rises again.
Impact: Ensures blinks like EAR: 0.16 are detected and contribute to maintaining "Real" status.
Increased Spoof Detection Threshold to 300 (from 200):
Why: The condition np.std(intervals) < 200 was too strict, flagging my natural blinking as a spoof due to perceived regularity. Increasing it to 300 allows more variation (e.g., intervals of 30-90 frames) typical of real users, while still catching highly regular video patterns (e.g., std_dev < 50).
Impact: Reduces false "Spoof" triggers for real users, maintaining accuracy for video detection.
Enhanced Liveness Persistence:
Why: The liveness_counter was depleting too quickly, allowing "Spoof" status to take over after initial blinks or motion. The new logic refreshes it more aggressively.
Change:
Resets liveness_counter to 180 frames (~6 seconds) on confirmed blinks.
Refreshes to at least 120 frames (~4 seconds) on motion detection alone.
Maintains "Real" status if liveness_counter > 0, even without immediate blinks.
Impact: Keeps you classified as "Real" longer, preventing premature switches to "Spoof."
Cleared Old Blink Intervals:
Why: The intervals list was accumulating outdated data, skewing the spoof check with low standard deviations. Clearing it after a 5-second gap ensures only recent blink intervals are considered.
Change: Added if len(blink_events) > 0 and (frame_counter - blink_events[-1]) > 150: intervals.clear().
Impact: Prevents false positives from stale data, improving spoof detection accuracy.'''

'''import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist
from collections import deque

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def create_liveness_model(input_shape=(64, 64, 3)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (64, 64))
    face_img = face_img / 255.0
    return np.expand_dims(face_img, axis=0)

# Initialize model and detectors
model = create_liveness_model()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Variables
prev_frame = None
motion_threshold = 200000  # Detects subtle live motion
EAR_THRESHOLD = 0.20  # Lowered from 0.22 for better sensitivity
BLINK_CONSEC_FRAMES = 2  # Increased to ensure full blinks
blink_counter = 0
blink_detected = False
liveness_buffer = 120  # ~4 seconds at 30 fps
liveness_counter = 0
blink_timeout = 150  # ~5 seconds at 30 fps
blink_timer = 0
frame_counter = 0
prev_ear_below_threshold = False
blink_events = []
intervals = deque(maxlen=5)
(left_start, left_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_start, right_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
motion_history = deque(maxlen=5)
motion_variance_history = deque(maxlen=20)
avg_variance = 0

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    frame_counter += 1
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    status = "Liveness: Spoof"

    # Motion detection
    motion_detected = False
    if prev_frame is not None:
        diff = cv2.absdiff(gray, prev_frame)
        motion_score = np.sum(diff)
        motion_history.append(motion_score)
        avg_motion = np.mean(motion_history)
        motion_detected = avg_motion > motion_threshold
        if len(motion_history) == motion_history.maxlen:
            motion_variance = np.var(motion_history)
            motion_variance_history.append(motion_variance)
        avg_variance = np.mean(motion_variance_history) if motion_variance_history else 0
        print(f"Avg Motion: {avg_motion:.0f}, Detected: {motion_detected}, Variance: {avg_variance:.0f}")

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        left_eye = shape[left_start:left_end]
        right_eye = shape[right_start:right_end]

        for (x, y) in left_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        for (x, y) in right_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Blink detection
        if ear < EAR_THRESHOLD:
            if not prev_ear_below_threshold:
                blink_counter = 1
            else:
                blink_counter += 1
            blink_detected = (blink_counter >= BLINK_CONSEC_FRAMES)
        else:
            if prev_ear_below_threshold and blink_counter >= BLINK_CONSEC_FRAMES:
                blink_detected = False
                blink_timer = 0
                liveness_counter = 180  # Reset to 6 seconds on confirmed blink
                if len(blink_events) > 0 and (frame_counter - blink_events[-1]) > 150:
                    intervals.clear()  # Clear old intervals after a 5-second gap
                interval = frame_counter - blink_events[-1] if blink_events else 0
                intervals.append(interval)
                blink_events.append(frame_counter)
            else:
                blink_detected = False
            blink_counter = 0
        prev_ear_below_threshold = ear < EAR_THRESHOLD
        print(f"EAR: {ear:.2f}, Blink Counter: {blink_counter}, Blink Detected: {blink_detected}")

        (x, y, w, h) = face_utils.rect_to_bb(rect)
        face_roi = frame[y:y+h, x:x+w]
        if face_roi.size == 0:
            continue
        face_input = preprocess_face(face_roi)
        liveness_score = model.predict(face_input, verbose=0)[0][0]
        print(f"AI Score: {liveness_score:.2f}")

        # Spoof detection
        is_spoof = False
        blink_timer += 1

        if blink_timer > blink_timeout:
            is_spoof = True
            print("No blink detected in 5 seconds - possible spoof")
        elif len(intervals) >= 5 and np.std(intervals) < 300:  # Increased from 200 to 300
            is_spoof = True
            print("Regular blink pattern detected - possible video spoof")
        elif avg_variance < 5e9 and avg_motion > 100000:
            is_spoof = True
            print("Low motion variance with motion - possible video spoof")
        elif avg_motion < 50000 and avg_variance < 1e9:
            is_spoof = True
            print("Minimal motion and variance - possible picture spoof")

        # Liveness decision
        if motion_detected and not is_spoof:
            status = "Liveness: Real"
            color = (0, 255, 0)
            liveness_counter = max(liveness_counter, 120)  # Refresh to 4 seconds on motion
        elif liveness_counter > 0:
            status = "Liveness: Real"
            color = (0, 255, 0)
        else:
            status = "Liveness: Spoof"
            color = (0, 0, 255)

        if liveness_counter > 0:
            liveness_counter -= 1

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"EAR: {ear:.2f}", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Blink: {'Yes' if blink_detected else 'No'}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Motion: {'Yes' if motion_detected else 'No'}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Blink Timer: {blink_timer}", (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow('Advanced Liveness Detection', frame)
    prev_frame = gray.copy()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()'''

'''Motion Smoothness Analysis:
Change: Replaced motion_variance_history (20 frames) with motion_history (30 frames) and calculate motion_variance directly over these 30 frames. Added a spoof condition: motion_detected and motion_variance < 1e9,Helps flag videos with consistent motion patterns as spoofs, improving differentiation from real 3D movement.
Change: Renamed intervals to blink_intervals for clarity and tightened the spoof threshold from np.std(blink_intervals) < 300 to < 50. Only applies when 5+ intervals are recorded,Increases sensitivity to unnatural blink regularity, helping identify 2D video spoofs.
Change: Updated liveness decision logic:
liveness_score > 0.6 confirms "Real" directly with motion and no spoof flags.
Scores between 0.4–0.6 require motion_variance > 1e11 to pass as "Real."
Scores <0.4 or failing other conditions default to "Spoof.Prevents videos with middling AI scores from being classified as "Real," enhancing accuracy.
Blink Detection Refinement:
Change: Kept EAR_THRESHOLD = 0.20 and BLINK_CONSEC_FRAMES = 2, but ensured blink_detected triggers during the blink and resets only after confirmation. Logs intervals when blinks complete.Improves reliability of blink detection, supporting both "Real" status persistence and spoof checks.
Change: No change to liveness_buffer (120) or blink_timeout (150), but liveness_counter now resets to 180 on blinks and refreshes to at least 120 on motion alone.Maintains "Real" status for you while allowing spoofs to be flagged when conditions lapse.'''

'''import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist
from collections import deque

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def create_liveness_model(input_shape=(64, 64, 3)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (64, 64))
    face_img = face_img / 255.0
    return np.expand_dims(face_img, axis=0)

# Initialize model and detectors
model = create_liveness_model()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Variables
prev_frame = None
motion_threshold = 200000  # Detects subtle live motion
EAR_THRESHOLD = 0.20  # Blink detection threshold
BLINK_CONSEC_FRAMES = 2  # Ensure full blinks
blink_counter = 0
blink_detected = False
liveness_buffer = 120  # ~4 seconds at 30 fps
liveness_counter = 0
blink_timeout = 150  # ~5 seconds at 30 fps
blink_timer = 0
frame_counter = 0
prev_ear_below_threshold = False
blink_events = []
blink_intervals = deque(maxlen=5)  # Track last 5 blink intervals
motion_history = deque(maxlen=30)  # Track motion over 30 frames for smoothness
(left_start, left_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_start, right_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    frame_counter += 1
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    status = "Liveness: Spoof"

    # Motion detection
    motion_detected = False
    motion_variance = 0
    if prev_frame is not None:
        diff = cv2.absdiff(gray, prev_frame)
        motion_score = np.sum(diff)
        motion_history.append(motion_score)
        avg_motion = np.mean(motion_history)
        motion_detected = avg_motion > motion_threshold
        if len(motion_history) == motion_history.maxlen:
            motion_variance = np.var(motion_history)
        print(f"Avg Motion: {avg_motion:.0f}, Detected: {motion_detected}, Variance: {motion_variance:.0f}")

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        left_eye = shape[left_start:left_end]
        right_eye = shape[right_start:right_end]

        for (x, y) in left_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        for (x, y) in right_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Blink detection
        if ear < EAR_THRESHOLD:
            if not prev_ear_below_threshold:
                blink_counter = 1
            else:
                blink_counter += 1
            blink_detected = blink_counter >= BLINK_CONSEC_FRAMES
        else:
            if prev_ear_below_threshold and blink_counter >= BLINK_CONSEC_FRAMES:
                blink_detected = False
                blink_timer = 0
                liveness_counter = 180  # Reset to 6 seconds on confirmed blink
                if len(blink_events) > 0 and (frame_counter - blink_events[-1]) > 150:
                    blink_intervals.clear()  # Clear old intervals after a 5-second gap
                interval = frame_counter - blink_events[-1] if blink_events else 0
                blink_intervals.append(interval)
                blink_events.append(frame_counter)
            else:
                blink_detected = False
            blink_counter = 0
        prev_ear_below_threshold = ear < EAR_THRESHOLD
        print(f"EAR: {ear:.2f}, Blink Counter: {blink_counter}, Blink Detected: {blink_detected}")

        (x, y, w, h) = face_utils.rect_to_bb(rect)
        face_roi = frame[y:y+h, x:x+w]
        if face_roi.size == 0:
            continue
        face_input = preprocess_face(face_roi)
        liveness_score = model.predict(face_input, verbose=0)[0][0]
        print(f"AI Score: {liveness_score:.2f}")

        # Spoof detection
        is_spoof = False
        blink_timer += 1

        if blink_timer > blink_timeout:
            is_spoof = True
            print("No blink detected in 5 seconds - possible spoof")
        elif len(blink_intervals) >= 5 and np.std(blink_intervals) < 50:  # Stricter regularity check
            is_spoof = True
            print("Regular blink pattern detected - possible video spoof")
        elif motion_detected and motion_variance < 1e9:  # Check for smooth motion
            is_spoof = True
            print("Low motion variance with motion - possible video spoof")

        # Liveness decision
        if motion_detected and not is_spoof and liveness_score > 0.6:  # Stricter AI threshold
            status = "Liveness: Real"
            color = (0, 255, 0)
            liveness_counter = max(liveness_counter, 120)  # Refresh to 4 seconds on motion
        elif liveness_score > 0.4 and motion_variance > 1e11:  # High variance backup
            status = "Liveness: Real"
            color = (0, 255, 0)
            liveness_counter = max(liveness_counter, 120)
        elif liveness_counter > 0:
            status = "Liveness: Real"
            color = (0, 255, 0)
        else:
            status = "Liveness: Spoof"
            color = (0, 0, 255)

        if liveness_counter > 0:
            liveness_counter -= 1

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"EAR: {ear:.2f}", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Blink: {'Yes' if blink_detected else 'No'}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Motion: {'Yes' if motion_detected else 'No'}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Blink Timer: {blink_timer}", (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow('Advanced Liveness Detection', frame)
    prev_frame = gray.copy()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()'''

'''Current Issue: Both you and the video score 0.54–0.57, and high motion variance (e.g., 1e12) pushes the video over the "Real" threshold.
With Pretrained Model:
A better model might output a wider score range (e.g., 0.8–0.9 for real, 0.2–0.4 for spoofs), making classification clearer.
It could detect video-specific artifacts (e.g., screen glare, frame rate consistency) that my CNN misses, lowering the video’s score.
Combined with my existing motion and blink checks, this would reduce false positives.
NOW INTEGRATING PRE-TRAINED MODEL MOBILENETV2'''

'''import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist
from collections import deque

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def create_liveness_model(input_shape=(64, 64, 3)):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (64, 64))
    face_img = tf.keras.applications.mobilenet_v2.preprocess_input(face_img)
    return np.expand_dims(face_img, axis=0)

# Initialize model and detectors
model = create_liveness_model()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Variables
prev_frame = None
motion_threshold = 200000
EAR_THRESHOLD = 0.20
BLINK_CONSEC_FRAMES = 2
blink_counter = 0
blink_detected = False
liveness_buffer = 120
liveness_counter = 0
blink_timeout = 150
blink_timer = 0
frame_counter = 0
prev_ear_below_threshold = False
blink_events = []
blink_intervals = deque(maxlen=10)  # Increased to 10 for better regularity check
motion_history = deque(maxlen=30)
variance_history = deque(maxlen=30)  # Track variance over 30 frames
(left_start, left_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_start, right_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    frame_counter += 1
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    status = "Liveness: Spoof"

    # Motion detection
    motion_detected = False
    motion_variance = 0
    if prev_frame is not None:
        diff = cv2.absdiff(gray, prev_frame)
        motion_score = np.sum(diff)
        motion_history.append(motion_score)
        avg_motion = np.mean(motion_history)
        motion_detected = avg_motion > motion_threshold
        if len(motion_history) == motion_history.maxlen:
            motion_variance = np.var(motion_history)
            variance_history.append(motion_variance)
            if len(variance_history) > variance_history.maxlen:
                variance_history.pop(0)
        variance_of_variance = np.var(variance_history) if len(variance_history) == variance_history.maxlen else 0
        print(f"Avg Motion: {avg_motion:.0f}, Detected: {motion_detected}, Variance: {motion_variance:.0f}")

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        left_eye = shape[left_start:left_end]
        right_eye = shape[right_start:right_end]

        for (x, y) in left_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        for (x, y) in right_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Blink detection
        if ear < EAR_THRESHOLD:
            if not prev_ear_below_threshold:
                blink_counter = 1
            else:
                blink_counter += 1
            blink_detected = blink_counter >= BLINK_CONSEC_FRAMES
        else:
            if prev_ear_below_threshold and blink_counter >= BLINK_CONSEC_FRAMES:
                blink_detected = False
                blink_timer = 0
                liveness_counter = 180
                if len(blink_events) > 0:
                    interval = frame_counter - blink_events[-1]
                    blink_intervals.append(interval)
                blink_events.append(frame_counter)
            else:
                blink_detected = False
            blink_counter = 0
        prev_ear_below_threshold = ear < EAR_THRESHOLD
        print(f"EAR: {ear:.2f}, Blink Counter: {blink_counter}, Blink Detected: {blink_detected}")

        (x, y, w, h) = face_utils.rect_to_bb(rect)
        face_roi = frame[y:y+h, x:x+w]
        if face_roi.size == 0:
            continue
        face_input = preprocess_face(face_roi)
        liveness_score = model.predict(face_input, verbose=0)[0][0]
        print(f"AI Score: {liveness_score:.2f}")

        # Spoof detection
        is_spoof = False
        blink_timer += 1

        if blink_timer > blink_timeout:
            is_spoof = True
            print("No blink detected in 5 seconds - possible spoof")
        elif len(blink_intervals) >= 10 and np.std(blink_intervals) < 100:
            is_spoof = True
            print("Regular blink pattern detected - possible video spoof")
        elif motion_detected and variance_of_variance < 1e10 and motion_variance > 0:
            is_spoof = True
            print("Consistent motion variance - possible video spoof")

        # Liveness decision
        if motion_detected and not is_spoof and liveness_score > 0.7:
            status = "Liveness: Real"
            color = (0, 255, 0)
            liveness_counter = max(liveness_counter, 120)
        elif liveness_score > 0.5 and motion_variance > 1e12 and (len(blink_intervals) < 10 or np.std(blink_intervals) > 50):
            status = "Liveness: Real"
            color = (0, 255, 0)
            liveness_counter = max(liveness_counter, 120)
        elif liveness_counter > 0:
            status = "Liveness: Real"
            color = (0, 255, 0)
        else:
            status = "Liveness: Spoof"
            color = (0, 0, 255)

        if liveness_counter > 0:
            liveness_counter -= 1

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"EAR: {ear:.2f}", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Blink: {'Yes' if blink_detected else 'No'}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Motion: {'Yes' if motion_detected else 'No'}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Blink Timer: {blink_timer}", (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow('Advanced Liveness Detection', frame)
    prev_frame = gray.copy()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()'''

'''Key Changes Explained
Fine-Tunable MobileNetV2:
Change: Kept base_model.trainable = True in create_liveness_model and included a commented fine-tuning section.
Why: logs show high AI scores (e.g., 0.98) for the video, indicating the pretrained model isn’t spoof-aware. Fine-tuning with a dataset (real vs. spoof) would lower spoof scores, but this runs without it for now.
Impact: Prepares for future accuracy boosts; currently relies on spoof logic enhancements.
Texture Analysis with LBP:
Change: Added compute_texture_score function and a spoof condition texture_score < 0.02. Logs now include Texture Score.
Why: Videos have smoother textures (lower variance) due to compression or screen rendering, unlike real skin. my logs lack this data, but it should differentiate the video.
Impact: Flags videos with uniform texture, catching spoofs missed by AI scores (e.g., 0.94).
Stricter Spoof Logic Priority:
Change: Reordered liveness decision to check not is_spoof before liveness_score > 0.7, ensuring spoof flags (e.g., "No blink detected") aren’t overridden.
Why: logs show "No blink detected" triggering is_spoof = True, but high AI scores (e.g., 0.92) still classified "Real." This fix enforces spoof precedence.
Impact: Videos with sparse blinks (e.g., last 30-35 seconds) should now be "Spoof," regardless of AI score.
Tighter Blink Regularity Check:
Change: Reduced np.std(blink_intervals) < 100 to < 75 and lowered the minimum blinks to 5 (from 10) for earlier detection.
Why: video has irregular blinks (e.g., EAR: 0.19 late), but tightening this catches subtle patterns. Fewer blinks required speeds up spoof detection.
Impact: Increases sensitivity to video-like blink consistency, flagging spoofs faster.
Adjusted Liveness Decision:
Change: Simplified conditions:
Primary: liveness_score > 0.7 and not is_spoof.
Fallback: liveness_score >= 0.5 with motion_variance > 1e12, irregular blinks, high texture, and not is_spoof.
Why: High AI scores (e.g., 0.98) bypassed spoof flags in my logs. This ensures spoof conditions dominate, balancing model confidence with spoof checks.
Impact: Reduces false positives, correctly classifying my video as "Spoof" when spoof flags trigger.'''

'''import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist
from collections import deque
from skimage.feature import local_binary_pattern

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def create_liveness_model(input_shape=(64, 64, 3)):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = True  # Allow fine-tuning
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (64, 64))
    face_img = tf.keras.applications.mobilenet_v2.preprocess_input(face_img)
    return np.expand_dims(face_img, axis=0)

def compute_texture_score(gray_img):
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray_img, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return np.var(hist)  # Variance of LBP histogram as texture score

# Initialize model and detectors
model = create_liveness_model()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Fine-tuning placeholder (uncomment and adjust if you have a dataset)
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# train_datagen = ImageDataGenerator(rescale=1./255)
# train_generator = train_datagen.flow_from_directory(
#     'path_to_dataset', target_size=(64, 64), batch_size=32, class_mode='binary'
# )
# model.fit(train_generator, epochs=10)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Variables
prev_frame = None
motion_threshold = 200000
EAR_THRESHOLD = 0.20
BLINK_CONSEC_FRAMES = 2
blink_counter = 0
blink_detected = False
liveness_buffer = 120
liveness_counter = 0
blink_timeout = 150  # ~5 seconds at 30 fps
blink_timer = 0
frame_counter = 0
prev_ear_below_threshold = False
blink_events = []
blink_intervals = deque(maxlen=10)
motion_history = deque(maxlen=30)
variance_history = deque(maxlen=30)
(left_start, left_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_start, right_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    frame_counter += 1
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    status = "Liveness: Spoof"

    # Motion detection
    motion_detected = False
    motion_variance = 0
    if prev_frame is not None:
        diff = cv2.absdiff(gray, prev_frame)
        motion_score = np.sum(diff)
        motion_history.append(motion_score)
        avg_motion = np.mean(motion_history)
        motion_detected = avg_motion > motion_threshold
        if len(motion_history) == motion_history.maxlen:
            motion_variance = np.var(motion_history)
            variance_history.append(motion_variance)
            if len(variance_history) > variance_history.maxlen:
                variance_history.pop(0)
        variance_of_variance = np.var(variance_history) if len(variance_history) == variance_history.maxlen else 0
        print(f"Avg Motion: {avg_motion:.0f}, Detected: {motion_detected}, Variance: {motion_variance:.0f}")

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        left_eye = shape[left_start:left_end]
        right_eye = shape[right_start:right_end]

        for (x, y) in left_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        for (x, y) in right_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Blink detection
        if ear < EAR_THRESHOLD:
            if not prev_ear_below_threshold:
                blink_counter = 1
            else:
                blink_counter += 1
            blink_detected = blink_counter >= BLINK_CONSEC_FRAMES
        else:
            if prev_ear_below_threshold and blink_counter >= BLINK_CONSEC_FRAMES:
                blink_detected = False
                blink_timer = 0
                liveness_counter = 180
                if len(blink_events) > 0:
                    interval = frame_counter - blink_events[-1]
                    blink_intervals.append(interval)
                blink_events.append(frame_counter)
            else:
                blink_detected = False
            blink_counter = 0
        prev_ear_below_threshold = ear < EAR_THRESHOLD
        print(f"EAR: {ear:.2f}, Blink Counter: {blink_counter}, Blink Detected: {blink_detected}")

        (x, y, w, h) = face_utils.rect_to_bb(rect)
        face_roi = frame[y:y+h, x:x+w]
        face_roi_gray = gray[y:y+h, x:x+w]
        if face_roi.size == 0 or face_roi_gray.size == 0:
            continue
        face_input = preprocess_face(face_roi)
        liveness_score = model.predict(face_input, verbose=0)[0][0]
        texture_score = compute_texture_score(face_roi_gray)
        print(f"AI Score: {liveness_score:.2f}, Texture Score: {texture_score:.4f}")

        # Spoof detection
        is_spoof = False
        blink_timer += 1

        if blink_timer > blink_timeout:
            is_spoof = True
            print("No blink detected in 5 seconds - possible spoof")
        elif len(blink_intervals) >= 5 and np.std(blink_intervals) < 75:  # Tightened from 100 to 75
            is_spoof = True
            print("Regular blink pattern detected - possible video spoof")
        elif motion_detected and variance_of_variance < 1e10 and motion_variance > 0:
            is_spoof = True
            print("Consistent motion variance - possible video spoof")
        elif texture_score < 0.02:  # Adjusted threshold for video texture
            is_spoof = True
            print("Low texture variance - possible video spoof")

        # Liveness decision
        if motion_detected and liveness_score > 0.7 and not is_spoof:  # Reordered to prioritize spoof check
            status = "Liveness: Real"
            color = (0, 255, 0)
            liveness_counter = max(liveness_counter, 120)
        elif liveness_score >= 0.5 and motion_variance > 1e12 and (len(blink_intervals) < 5 or np.std(blink_intervals) > 50) and texture_score > 0.02 and not is_spoof:
            status = "Liveness: Real"
            color = (0, 255, 0)
            liveness_counter = max(liveness_counter, 120)
        elif liveness_counter > 0 and not is_spoof:
            status = "Liveness: Real"
            color = (0, 255, 0)
        else:
            status = "Liveness: Spoof"
            color = (0, 0, 255)

        if liveness_counter > 0:
            liveness_counter -= 1

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"EAR: {ear:.2f}", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Blink: {'Yes' if blink_detected else 'No'}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Motion: {'Yes' if motion_detected else 'No'}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Blink Timer: {blink_timer}", (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow('Advanced Liveness Detection', frame)
    prev_frame = gray.copy()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#trying to adjust the thresholds'''

'''import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist
from collections import deque
from skimage.feature import local_binary_pattern

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def create_liveness_model(input_shape=(64, 64, 3)):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = True  # Allow fine-tuning
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (64, 64))
    face_img = tf.keras.applications.mobilenet_v2.preprocess_input(face_img)
    return np.expand_dims(face_img, axis=0)

def compute_texture_score(gray_img):
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray_img, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return np.var(hist)  # Variance of LBP histogram as texture score

# Initialize model and detectors
model = create_liveness_model()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Fine-tuning placeholder (uncomment and adjust if you have a dataset)
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# train_datagen = ImageDataGenerator(rescale=1./255)
# train_generator = train_datagen.flow_from_directory(
#     'path_to_dataset', target_size=(64, 64), batch_size=32, class_mode='binary'
# )
# model.fit(train_generator, epochs=10)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Variables
prev_frame = None
motion_threshold = 200000
EAR_THRESHOLD = 0.20
BLINK_CONSEC_FRAMES = 2
blink_counter = 0
blink_detected = False
liveness_buffer = 120
liveness_counter = 0
blink_timeout = 150  # ~5 seconds at 30 fps
blink_timer = 0
frame_counter = 0
prev_ear_below_threshold = False
blink_events = []
blink_intervals = deque(maxlen=10)
motion_history = deque(maxlen=30)
variance_history = deque(maxlen=30)
(left_start, left_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_start, right_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    frame_counter += 1
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    status = "Liveness: Spoof"

    # Motion detection
    motion_detected = False
    motion_variance = 0
    if prev_frame is not None:
        diff = cv2.absdiff(gray, prev_frame)
        motion_score = np.sum(diff)
        motion_history.append(motion_score)
        avg_motion = np.mean(motion_history)
        motion_detected = avg_motion > motion_threshold
        if len(motion_history) == motion_history.maxlen:
            motion_variance = np.var(motion_history)
            variance_history.append(motion_variance)
            if len(variance_history) > variance_history.maxlen:
                variance_history.pop(0)
        variance_of_variance = np.var(variance_history) if len(variance_history) == variance_history.maxlen else 0
        print(f"Avg Motion: {avg_motion:.0f}, Detected: {motion_detected}, Variance: {motion_variance:.0f}")

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        left_eye = shape[left_start:left_end]
        right_eye = shape[right_start:right_end]

        for (x, y) in left_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        for (x, y) in right_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Blink detection
        if ear < EAR_THRESHOLD:
            if not prev_ear_below_threshold:
                blink_counter = 1
            else:
                blink_counter += 1
            blink_detected = blink_counter >= BLINK_CONSEC_FRAMES
        else:
            if prev_ear_below_threshold and blink_counter >= BLINK_CONSEC_FRAMES:
                blink_detected = False
                blink_timer = 0
                liveness_counter = 180
                if len(blink_events) > 0:
                    interval = frame_counter - blink_events[-1]
                    blink_intervals.append(interval)
                blink_events.append(frame_counter)
            else:
                blink_detected = False
            blink_counter = 0
        prev_ear_below_threshold = ear < EAR_THRESHOLD
        print(f"EAR: {ear:.2f}, Blink Counter: {blink_counter}, Blink Detected: {blink_detected}")

        (x, y, w, h) = face_utils.rect_to_bb(rect)
        face_roi = frame[y:y+h, x:x+w]
        face_roi_gray = gray[y:y+h, x:x+w]
        if face_roi.size == 0 or face_roi_gray.size == 0:
            continue
        face_input = preprocess_face(face_roi)
        liveness_score = model.predict(face_input, verbose=0)[0][0]
        texture_score = compute_texture_score(face_roi_gray)
        print(f"AI Score: {liveness_score:.2f}, Texture Score: {texture_score:.4f}")

        # Spoof detection
        is_spoof = False
        blink_timer += 1

        if blink_timer > blink_timeout:
            is_spoof = True
            print("No blink detected in 5 seconds - possible spoof")
        elif texture_score < 0.0015:  # Adjusted from 0.02 to 0.0015
            is_spoof = True
            print("Low texture variance - possible video spoof")
        elif motion_detected and variance_of_variance < 1e10 and motion_variance > 0:
            is_spoof = True
            print("Consistent motion variance - possible video spoof")
        elif len(blink_intervals) >= 5 and np.std(blink_intervals) < 75:
            is_spoof = True
            print("Regular blink pattern detected - possible video spoof")

        # Override spoof if a blink was detected recently
        if blink_detected:
            is_spoof = False
            print("Recent blink detected - likely real")
            blink_timer = 0  # Reset timer on blink

        # Liveness decision with relaxed AI threshold
        if motion_detected and liveness_score > 0.6 and not is_spoof:  # Lowered from 0.7 to 0.6
            status = "Liveness: Real"
            color = (0, 255, 0)
            liveness_counter = max(liveness_counter, 120)
        elif liveness_score >= 0.4 and motion_variance > 1e12 and not is_spoof:  # New fallback threshold
            status = "Liveness: Real"
            color = (0, 255, 0)
            liveness_counter = max(liveness_counter, 120)
        elif liveness_counter > 0 and not is_spoof:
            status = "Liveness: Real"
            color = (0, 255, 0)
        else:
            status = "Liveness: Spoof"
            color = (0, 0, 255)

        if liveness_counter > 0:
            liveness_counter -= 1

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"EAR: {ear:.2f}", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Blink: {'Yes' if blink_detected else 'No'}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Motion: {'Yes' if motion_detected else 'No'}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Blink Timer: {blink_timer}", (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow('Advanced Liveness Detection', frame)
    prev_frame = gray.copy()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()'''

'''Tracking Recent Blinks Over a Time Window:
Change: Added recent_blinks = sum(1 for event in blink_events if frame_counter - event < 300) to count blinks in the last 10 seconds (~300 frames at 30 fps).
Why: my logs showed sporadic blinks in the video (e.g., EAR 0.13-0.18) causing it to flip to "real." Real people blink 15-20 times per minute (2-3 times in 10 seconds), so requiring at least 2 blinks ensures consistent liveness, while videos with fewer blinks are flagged.
Impact: Prevents the video from being classified as "real" unless it mimics a natural blink rate, which is unlikely.
Strengthened Spoof Detection:
Change: Updated spoof condition to if recent_blinks < 2 and blink_timer > 150, replacing the simpler blink_timer > 150 check.
Why: The video’s initial lack of blinks triggered "spoof," but later blinks reset this. This new condition ensures insufficient blinking over time (fewer than 2 blinks in 10 seconds) keeps it as "spoof," even with occasional blinks.
Impact: Stabilizes "spoof" classification for the video, avoiding flips back to "real" after sparse blinks.
Adjusted Liveness Decision Criteria:
Change: Modified primary condition to liveness_score > 0.6 and recent_blinks >= 2 and not is_spoof, and fallback to liveness_score >= 0.4 and motion_variance > 1e12 and recent_blinks >= 1 and not is_spoof.
Why: my face had consistent blinks early on, ensuring recent_blinks >= 2, while the video’s sporadic blinks (e.g., 10 late blinks then none) drop recent_blinks below 2 after 10 seconds, preventing "real" classification even with high AI scores (e.g., 0.78).
Impact: Ensures "real" status requires sustained blinking, keeping my live detection accurate and video detection as "spoof."
Reset Liveness Counter on Spoof:
Change: Added if is_spoof: liveness_counter = 0 before the counter decrement logic.
Why: The video’s initial "real" classification might have been due to a lingering liveness_counter from my prior detection. Resetting it on spoof detection prevents carryover effects.
Impact: Eliminates the initial "real" misclassification when switching to the video.
Clean Up Old Blink Events:
Change: Added if frame_counter % 100 == 0: blink_events = [event for event in blink_events if frame_counter - event < 600] to remove events older than 20 seconds.
Why: Keeps the blink_events list manageable and relevant, avoiding memory buildup and ensuring recent_blinks reflects current behavior.
Impact: Maintains system efficiency without affecting accuracy.
Debugging Aid:
Change: Added print(f"Recent Blinks: {recent_blinks}, Variance of Variance: {variance_of_variance:.0f}") to logs.
Why: Helps verify that recent_blinks and variance_of_variance are behaving as expected, especially during video playback.
Impact: Facilitates troubleshooting if further tuning is needed.'''

'''import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist
from collections import deque
from skimage.feature import local_binary_pattern

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def create_liveness_model(input_shape=(64, 64, 3)):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = True  # Allow fine-tuning
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (64, 64))
    face_img = tf.keras.applications.mobilenet_v2.preprocess_input(face_img)
    return np.expand_dims(face_img, axis=0)

def compute_texture_score(gray_img):
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray_img, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return np.var(hist)  # Variance of LBP histogram as texture score

# Initialize model and detectors
model = create_liveness_model()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Fine-tuning placeholder (uncomment and adjust if you have a dataset)
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# train_datagen = ImageDataGenerator(rescale=1./255)
# train_generator = train_datagen.flow_from_directory(
#     'path_to_dataset', target_size=(64, 64), batch_size=32, class_mode='binary'
# )
# model.fit(train_generator, epochs=10)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Variables
prev_frame = None
motion_threshold = 200000
EAR_THRESHOLD = 0.20
BLINK_CONSEC_FRAMES = 2
blink_counter = 0
blink_detected = False
liveness_buffer = 120
liveness_counter = 0
blink_timeout = 150  # ~5 seconds at 30 fps
blink_timer = 0
frame_counter = 0
prev_ear_below_threshold = False
blink_events = []
blink_intervals = deque(maxlen=10)
motion_history = deque(maxlen=30)
variance_history = deque(maxlen=30)
(left_start, left_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_start, right_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    frame_counter += 1
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    status = "Liveness: Spoof"

    # Motion detection
    motion_detected = False
    motion_variance = 0
    if prev_frame is not None:
        diff = cv2.absdiff(gray, prev_frame)
        motion_score = np.sum(diff)
        motion_history.append(motion_score)
        avg_motion = np.mean(motion_history)
        motion_detected = avg_motion > motion_threshold
        if len(motion_history) == motion_history.maxlen:
            motion_variance = np.var(motion_history)
            variance_history.append(motion_variance)
            if len(variance_history) > variance_history.maxlen:
                variance_history.pop(0)
        variance_of_variance = np.var(variance_history) if len(variance_history) == variance_history.maxlen else 0
        print(f"Avg Motion: {avg_motion:.0f}, Detected: {motion_detected}, Variance: {motion_variance:.0f}")

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        left_eye = shape[left_start:left_end]
        right_eye = shape[right_start:right_end]

        for (x, y) in left_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        for (x, y) in right_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Blink detection
        if ear < EAR_THRESHOLD:
            if not prev_ear_below_threshold:
                blink_counter = 1
            else:
                blink_counter += 1
            blink_detected = blink_counter >= BLINK_CONSEC_FRAMES
        else:
            if prev_ear_below_threshold and blink_counter >= BLINK_CONSEC_FRAMES:
                blink_detected = False
                blink_timer = 0
                liveness_counter = 180
                if len(blink_events) > 0:
                    interval = frame_counter - blink_events[-1]
                    blink_intervals.append(interval)
                blink_events.append(frame_counter)
            else:
                blink_detected = False
            blink_counter = 0
        prev_ear_below_threshold = ear < EAR_THRESHOLD
        print(f"EAR: {ear:.2f}, Blink Counter: {blink_counter}, Blink Detected: {blink_detected}")

        (x, y, w, h) = face_utils.rect_to_bb(rect)
        face_roi = frame[y:y+h, x:x+w]
        face_roi_gray = gray[y:y+h, x:x+w]
        if face_roi.size == 0 or face_roi_gray.size == 0:
            continue
        face_input = preprocess_face(face_roi)
        liveness_score = model.predict(face_input, verbose=0)[0][0]
        texture_score = compute_texture_score(face_roi_gray)

        # Calculate recent blinks (last 10 seconds, ~300 frames at 30 fps)
        recent_blinks = sum(1 for event in blink_events if frame_counter - event < 300)

        print(f"AI Score: {liveness_score:.2f}, Texture Score: {texture_score:.4f}")
        print(f"Recent Blinks: {recent_blinks}, Variance of Variance: {variance_of_variance:.0f}")

        # Spoof detection
        is_spoof = False
        blink_timer += 1

        if recent_blinks < 2 and blink_timer > 150:  # Require at least 2 blinks in 10 seconds
            is_spoof = True
            print("Insufficient blinks - possible spoof")
        elif texture_score < 0.0015:
            is_spoof = True
            print("Low texture variance - possible video spoof")
        elif motion_detected and variance_of_variance < 1e10 and motion_variance > 0:
            is_spoof = True
            print("Consistent motion variance - possible video spoof")
        elif len(blink_intervals) >= 5 and np.std(blink_intervals) < 75:
            is_spoof = True
            print("Regular blink pattern detected - possible video spoof")

        if blink_detected:
            is_spoof = False
            print("Recent blink detected - likely real")
            blink_timer = 0

        # Liveness decision
        if motion_detected and liveness_score > 0.6 and recent_blinks >= 2 and not is_spoof:
            status = "Liveness: Real"
            color = (0, 255, 0)
            liveness_counter = max(liveness_counter, 120)
        elif liveness_score >= 0.4 and motion_variance > 1e12 and recent_blinks >= 1 and not is_spoof:
            status = "Liveness: Real"
            color = (0, 255, 0)
            liveness_counter = max(liveness_counter, 120)
        elif liveness_counter > 0 and recent_blinks >= 1 and not is_spoof:
            status = "Liveness: Real"
            color = (0, 255, 0)
        else:
            status = "Liveness: Spoof"
            color = (0, 0, 255)

        # Reset liveness counter if spoof is detected
        if is_spoof:
            liveness_counter = 0
        elif liveness_counter > 0:
            liveness_counter -= 1

        # Clean up old blink events every 100 frames
        if frame_counter % 100 == 0:
            blink_events = [event for event in blink_events if frame_counter - event < 600]

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"EAR: {ear:.2f}", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Blink: {'Yes' if blink_detected else 'No'}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Motion: {'Yes' if motion_detected else 'No'}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Blink Timer: {blink_timer}", (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow('Advanced Liveness Detection', frame)
    prev_frame = gray.copy()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()'''

'''Relaxed Blink Requirement:
Change: Changed spoof condition to recent_blinks < 2 and blink_timer > 150 (from < 3 and > 90), and primary liveness condition to recent_blinks >= 2.
Why: my logs show sparse blinks (e.g., recent_blinks at 2), typical for some people (15-20 blinks/min ≈ 1-2 in 10 sec). 5 seconds aligns better with natural rates, avoiding false spoofs.
Impact: You’ll pass with 2 blinks in 10 seconds, reducing spoof flags unless you stop blinking for over 5 seconds.
Adjusted EAR Threshold:
Change: Increased EAR_THRESHOLD from 0.20 to 0.25.
Why: my EAR drops to 0.19–0.26 during blinks, often missing the 0.20 threshold (e.g., 0.21–0.24 not detected). 0.25 catches these, increasing recent_blinks.
Impact: More blinks detected (e.g., EAR 0.24–0.25), ensuring recent_blinks >= 2 more often.
Prioritized AI Score:
Change: Added override: liveness_score > 0.9 and not is_spoof before the primary condition.
Why: my AI scores are frequently >0.9 (e.g., 0.95, 0.99), indicating high confidence in a live face, yet spoof flags override this. This prioritizes strong AI evidence.
Impact: You’re classified as "real" when AI scores are very high, even with sparse blinks, while videos with lower scores or regular patterns remain "spoof."
Kept Reset Sensitivity:
Change: Retained motion reset logic unchanged.
Why: No video transition here, but it ensures robustness for future tests. my motion stays consistent (2M–0.5M), avoiding unnecessary resets.
Impact: Maintains state unless a significant change occurs (e.g., video swap).'''

'''import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist
from collections import deque
from skimage.feature import local_binary_pattern

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def create_liveness_model(input_shape=(64, 64, 3)):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = True
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (64, 64))
    face_img = tf.keras.applications.mobilenet_v2.preprocess_input(face_img)
    return np.expand_dims(face_img, axis=0)

def compute_texture_score(gray_img):
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray_img, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return np.var(hist)

# Initialize model and detectors
model = create_liveness_model()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Variables
prev_frame = None
motion_threshold = 200000
EAR_THRESHOLD = 0.25  # Increased from 0.20 to catch more blinks
BLINK_CONSEC_FRAMES = 2
blink_counter = 0
blink_detected = False
liveness_buffer = 120
liveness_counter = 0
blink_timeout = 150  # Extended to 5 seconds
blink_timer = 0
frame_counter = 0
prev_ear_below_threshold = False
blink_events = []
blink_intervals = deque(maxlen=10)
motion_history = deque(maxlen=30)
variance_history = deque(maxlen=30)
prev_avg_motion = 0
(left_start, left_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_start, right_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    frame_counter += 1
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    status = "Liveness: Spoof"

    # Motion detection
    motion_detected = False
    motion_variance = 0
    if prev_frame is not None:
        diff = cv2.absdiff(gray, prev_frame)
        motion_score = np.sum(diff)
        motion_history.append(motion_score)
        avg_motion = np.mean(motion_history)
        motion_detected = avg_motion > motion_threshold
        if len(motion_history) == motion_history.maxlen:
            motion_variance = np.var(motion_history)
            variance_history.append(motion_variance)
            if len(variance_history) > variance_history.maxlen:
                variance_history.pop(0)
        variance_of_variance = np.var(variance_history) if len(variance_history) == variance_history.maxlen else 0

        # Reset state on significant motion change
        if abs(avg_motion - prev_avg_motion) > 500000:
            blink_events = []
            liveness_counter = 0
            print("Significant motion change detected - resetting state")

        prev_avg_motion = avg_motion
        print(f"Avg Motion: {avg_motion:.0f}, Detected: {motion_detected}, Variance: {motion_variance:.0f}")

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        left_eye = shape[left_start:left_end]
        right_eye = shape[right_start:right_end]

        for (x, y) in left_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        for (x, y) in right_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Blink detection
        if ear < EAR_THRESHOLD:
            if not prev_ear_below_threshold:
                blink_counter = 1
            else:
                blink_counter += 1
            blink_detected = blink_counter >= BLINK_CONSEC_FRAMES
        else:
            if prev_ear_below_threshold and blink_counter >= BLINK_CONSEC_FRAMES:
                blink_detected = False
                blink_timer = 0
                liveness_counter = 180
                if len(blink_events) > 0:
                    interval = frame_counter - blink_events[-1]
                    blink_intervals.append(interval)
                blink_events.append(frame_counter)
            else:
                blink_detected = False
            blink_counter = 0
        prev_ear_below_threshold = ear < EAR_THRESHOLD
        print(f"EAR: {ear:.2f}, Blink Counter: {blink_counter}, Blink Detected: {blink_detected}")

        (x, y, w, h) = face_utils.rect_to_bb(rect)
        face_roi = frame[y:y+h, x:x+w]
        face_roi_gray = gray[y:y+h, x:x+w]
        if face_roi.size == 0 or face_roi_gray.size == 0:
            continue
        face_input = preprocess_face(face_roi)
        liveness_score = model.predict(face_input, verbose=0)[0][0]
        texture_score = compute_texture_score(face_roi_gray)

        # Calculate recent blinks (last 10 seconds, ~300 frames at 30 fps)
        recent_blinks = sum(1 for event in blink_events if frame_counter - event < 300)

        print(f"AI Score: {liveness_score:.2f}, Texture Score: {texture_score:.4f}")
        print(f"Recent Blinks: {recent_blinks}, Variance of Variance: {variance_of_variance:.0f}")

        # Spoof detection
        is_spoof = False
        blink_timer += 1

        if recent_blinks < 2 and blink_timer > 150:  # Relaxed to 2 blinks, 5 seconds
            is_spoof = True
            print("Insufficient blinks - possible spoof")
        elif texture_score < 0.0015:
            is_spoof = True
            print("Low texture variance - possible video spoof")
        elif motion_detected and variance_of_variance < 1e10 and motion_variance > 0:
            is_spoof = True
            print("Consistent motion variance - possible video spoof")
        elif len(blink_intervals) >= 5 and np.std(blink_intervals) < 75:
            is_spoof = True
            print("Regular blink pattern detected - possible video spoof")

        if blink_detected:
            is_spoof = False
            print("Recent blink detected - likely real")
            blink_timer = 0

        # Liveness decision
        if motion_detected and liveness_score > 0.9 and not is_spoof:  # High AI score override
            status = "Liveness: Real"
            color = (0, 255, 0)
            liveness_counter = max(liveness_counter, 120)
        elif motion_detected and liveness_score > 0.7 and recent_blinks >= 2 and not is_spoof:
            status = "Liveness: Real"
            color = (0, 255, 0)
            liveness_counter = max(liveness_counter, 120)
        elif liveness_score >= 0.4 and motion_variance > 1e12 and recent_blinks >= 1 and not is_spoof:
            status = "Liveness: Real"
            color = (0, 255, 0)
            liveness_counter = max(liveness_counter, 120)
        elif liveness_counter > 0 and recent_blinks >= 1 and not is_spoof:
            status = "Liveness: Real"
            color = (0, 255, 0)
        else:
            status = "Liveness: Spoof"
            color = (0, 0, 255)

        # Reset liveness counter if spoof is detected
        if is_spoof:
            liveness_counter = 0
        elif liveness_counter > 0:
            liveness_counter -= 1

        # Clean up old blink events every 100 frames
        if frame_counter % 100 == 0:
            blink_events = [event for event in blink_events if frame_counter - event < 600]

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"EAR: {ear:.2f}", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Blink: {'Yes' if blink_detected else 'No'}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Motion: {'Yes' if motion_detected else 'No'}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Blink Timer: {blink_timer}", (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow('Advanced Liveness Detection', frame)
    prev_frame = gray.copy()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()'''

'''Strengthened Blink Validation:
Change: Updated to if blink_detected and recent_blinks >= 2 to override is_spoof.
Why: Single blinks (e.g., EAR 0.23–0.24, 0.15–0.22) reset is_spoof too easily. Requiring 2+ blinks ensures frequent blinking, rare in videos.
Impact: Video stays "spoof" unless blinks occur rapidly, which this video doesn’t sustain.
Adjusted AI Threshold:
Change: Removed liveness_score > 0.9 override, keeping liveness_score > 0.7 and recent_blinks >= 2.
Why: High scores (e.g., 0.90) allowed "real" despite video-like motion. This ties "real" to frequent blinks.
Impact: Prevents false "real" from sporadic high AI scores (e.g., 0.94).
Fixed Motion Variance Logic:
Change: Raised threshold to variance_of_variance < 1e12, default to float('inf') until 5 samples.
Why: Early 0 values weakened spoof detection; large later values (e.g., 3.1e22) were inconsistent. 1e12 better captures video stability.
Impact: Consistently flags video-like motion (e.g., 1.2e13 variance).
Increased Spoof Sensitivity:
Change: Added recent_blinks < 2 and blink_timer > 90 to spoof conditions.
Why: Video has sparse blinks (long gaps, e.g., 0.29–0.40), but wasn’t penalized enough. 3 seconds catches this.
Impact: Reinforces "spoof" during non-blinking periods.'''

'''import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist
from collections import deque
from skimage.feature import local_binary_pattern

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def create_liveness_model(input_shape=(64, 64, 3)):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = True
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (64, 64))
    face_img = tf.keras.applications.mobilenet_v2.preprocess_input(face_img)
    return np.expand_dims(face_img, axis=0)

def compute_texture_score(gray_img):
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray_img, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return np.var(hist)

# Initialize model and detectors
model = create_liveness_model()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Variables
prev_frame = None
motion_threshold = 200000
EAR_THRESHOLD = 0.25
BLINK_CONSEC_FRAMES = 2
blink_counter = 0
blink_detected = False
liveness_buffer = 120
liveness_counter = 0
blink_timeout = 90
blink_timer = 0
frame_counter = 0
prev_ear_below_threshold = False
blink_events = []
blink_intervals = deque(maxlen=10)
motion_history = deque(maxlen=30)
variance_history = deque(maxlen=30)
prev_avg_motion = 0
variance_of_variance = float('inf')  # Initialize outside the loop
(left_start, left_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_start, right_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    frame_counter += 1
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    status = "Liveness: Spoof"

    # Motion detection
    motion_detected = False
    motion_variance = 0
    if prev_frame is not None:
        diff = cv2.absdiff(gray, prev_frame)
        motion_score = np.sum(diff)
        motion_history.append(motion_score)
        avg_motion = np.mean(motion_history)
        motion_detected = avg_motion > motion_threshold
        if len(motion_history) == motion_history.maxlen:
            motion_variance = np.var(motion_history)
            variance_history.append(motion_variance)
            if len(variance_history) > variance_history.maxlen:
                variance_history.pop(0)
            variance_of_variance = np.var(variance_history) if len(variance_history) >= 5 else float('inf')

        if abs(avg_motion - prev_avg_motion) > 500000:
            blink_events = []
            liveness_counter = 0
            print("Significant motion change detected - resetting state")

        prev_avg_motion = avg_motion
        print(f"Avg Motion: {avg_motion:.0f}, Detected: {motion_detected}, Variance: {motion_variance:.0f}")

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        left_eye = shape[left_start:left_end]
        right_eye = shape[right_start:right_end]

        for (x, y) in left_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        for (x, y) in right_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Blink detection
        if ear < EAR_THRESHOLD:
            if not prev_ear_below_threshold:
                blink_counter = 1
            else:
                blink_counter += 1
            blink_detected = blink_counter >= BLINK_CONSEC_FRAMES
        else:
            if prev_ear_below_threshold and blink_counter >= BLINK_CONSEC_FRAMES:
                blink_detected = False
                blink_timer = 0
                if len(blink_events) > 0:
                    interval = frame_counter - blink_events[-1]
                    blink_intervals.append(interval)
                blink_events.append(frame_counter)
            else:
                blink_detected = False
            blink_counter = 0
        prev_ear_below_threshold = ear < EAR_THRESHOLD
        print(f"EAR: {ear:.2f}, Blink Counter: {blink_counter}, Blink Detected: {blink_detected}")

        (x, y, w, h) = face_utils.rect_to_bb(rect)
        face_roi = frame[y:y+h, x:x+w]
        face_roi_gray = gray[y:y+h, x:x+w]
        if face_roi.size == 0 or face_roi_gray.size == 0:
            continue
        face_input = preprocess_face(face_roi)
        liveness_score = model.predict(face_input, verbose=0)[0][0]
        texture_score = compute_texture_score(face_roi_gray)

        # Calculate recent blinks (last 10 seconds, ~300 frames at 30 fps)
        recent_blinks = sum(1 for event in blink_events if frame_counter - event < 300)

        print(f"AI Score: {liveness_score:.2f}, Texture Score: {texture_score:.4f}")
        print(f"Recent Blinks: {recent_blinks}, Variance of Variance: {variance_of_variance:.0f}")

        # Spoof detection
        is_spoof = False
        blink_timer += 1

        if recent_blinks < 2 and blink_timer > 90:
            is_spoof = True
            print("Insufficient blinks - possible spoof")
        elif texture_score < 0.0015:
            is_spoof = True
            print("Low texture variance - possible video spoof")
        elif motion_detected and variance_of_variance < 1e12 and motion_variance > 0:
            is_spoof = True
            print("Consistent motion variance - possible video spoof")
        regular_pattern = len(blink_intervals) >= 5 and np.std(blink_intervals) < 150
        if regular_pattern and (texture_score < 0.0020 or variance_of_variance < 1e12):
            is_spoof = True
            print("Regular blink pattern detected with additional spoof indicators - possible video spoof")

        # Override spoof with sufficient blinks and high AI score
        if recent_blinks >= 2 and liveness_score > 0.7:
            is_spoof = False
            print("Sufficient blink frequency and high AI score - likely real")
            blink_timer = 0

        # Liveness decision
        if motion_detected and liveness_score > 0.7 and recent_blinks >= 2 and not is_spoof:
            status = "Liveness: Real"
            color = (0, 255, 0)
            liveness_counter = max(liveness_counter, 120)
        elif liveness_score >= 0.4 and motion_variance > 1e12 and recent_blinks >= 1 and not is_spoof:
            status = "Liveness: Real"
            color = (0, 255, 0)
            liveness_counter = max(liveness_counter, 120)
        elif liveness_counter > 0 and recent_blinks >= 1 and not is_spoof:
            status = "Liveness: Real"
            color = (0, 255, 0)
        else:
            status = "Liveness: Spoof"
            color = (0, 0, 255)

        # Reset liveness counter if spoof is detected
        if is_spoof:
            liveness_counter = 0
        elif liveness_counter > 0:
            liveness_counter -= 1

        # Clean up old blink events every 100 frames
        if frame_counter % 100 == 0:
            blink_events = [event for event in blink_events if frame_counter - event < 600]

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"EAR: {ear:.2f}", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Blink: {'Yes' if blink_detected else 'No'}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Motion: {'Yes' if motion_detected else 'No'}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Blink Timer: {blink_timer}", (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow('Advanced Liveness Detection', frame)
    prev_frame = gray.copy()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()'''

'''Removed Untrained MobileNet Model:

The original model wasn't pretrained on liveness data

Focused on more reliable heuristic features instead

Enhanced Feature Extraction:

Added Fourier transform analysis for screen pattern detection

Improved texture analysis with multi-scale LBP

Added HSV color analysis for screen reflection detection

Enhanced motion analysis with face position tracking

Improved Blink Detection:

Added blink interval variability analysis

Implemented natural blink pattern recognition

Advanced Spoof Detection:

Added screen reflection detection using HSV

Implemented frequency domain analysis

Added multiple consistency checks

Optimized Thresholds:

Adjusted thresholds based on empirical testing

Added dynamic threshold adjustments

Added Logs too'''
'''
import cv2
import numpy as np
import dlib
import csv
import time
from imutils import face_utils
from scipy.spatial import distance as dist
from collections import deque
from skimage.feature import local_binary_pattern

# Configuration
LOG_FILE = "liveness_logs.csv"
EAR_SMOOTHING_WINDOW = 5
MIN_FACE_SIZE = 100  # Minimum face size in pixels (w * h)

# Eye Aspect Ratio (EAR) calculation
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Initialize logger
def init_logger():
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "face_detected", "texture_score", "avg_ear",
            "motion_score", "face_size", "decision", "confidence"
        ])

def log_data(data):
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(data)

# Feature extractors
def get_face_quality(face_img):
    if face_img.size == 0:
        return 0
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def adaptive_texture_score(gray_img):
    try:
        lbp = local_binary_pattern(gray_img, 24, 3, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 27))
        hist = hist.astype("float") / (hist.sum() + 1e-7)
        return np.std(hist)
    except:
        return 0

# Main detection logic
def main():
    init_logger()
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # State variables
    ear_history = deque(maxlen=EAR_SMOOTHING_WINDOW)
    position_history = deque(maxlen=30)
    prev_face_position = None
    frame_count = 0
    baseline_texture = None
    baseline_motion = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame_time = time.strftime("%Y-%m-%d %H:%M:%S")
            log_entry = [frame_time, 0, 0, 0, 0, 0, "Unknown", 0]
            
            # Preprocess
            gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (640, 480))
            rects = detector(gray, 0)
            
            if len(rects) > 0:
                rect = rects[0]
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                
                # Get face ROI
                (x, y, w, h) = face_utils.rect_to_bb(rect)
                face_size = w * h
                log_entry[5] = face_size
                
                if face_size < MIN_FACE_SIZE:
                    log_entry[7] = "Face too small"
                    continue
                
                face_roi = frame[y:y+h, x:x+w]
                quality = get_face_quality(face_roi)
                
                if quality < 50:  # Image is too blurry
                    log_entry[7] = f"Low quality ({quality:.1f})"
                    continue
                
                # Eye detection
                left_eye = shape[42:48]
                right_eye = shape[36:42]
                ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2
                ear_history.append(ear)
                avg_ear = np.mean(ear_history)
                log_entry[3] = avg_ear
                
                # Texture analysis
                texture = adaptive_texture_score(gray[y:y+h, x:x+w])
                log_entry[2] = texture
                
                # Motion analysis
                current_pos = (x + w/2, y + h/2)
                if prev_face_position:
                    motion = dist.euclidean(current_pos, prev_face_position)
                    position_history.append(motion)
                prev_face_position = current_pos
                avg_motion = np.mean(position_history) if position_history else 0
                log_entry[4] = avg_motion
                
                # Dynamic baselines
                if frame_count < 30:  # First second of footage
                    if baseline_texture is None:
                        baseline_texture = texture
                    else:
                        baseline_texture = 0.9 * baseline_texture + 0.1 * texture
                    
                    if baseline_motion is None:
                        baseline_motion = avg_motion
                    else:
                        baseline_motion = 0.9 * baseline_motion + 0.1 * avg_motion
                    continue  # Skip detection during warmup
                
                # Decision logic
                texture_diff = texture - baseline_texture
                motion_diff = avg_motion - baseline_motion
                
                live_score = 0
                if avg_ear > 0.2:
                    live_score += 0.3
                if texture_diff > -0.0002:
                    live_score += 0.3
                if motion_diff > 0.5:
                    live_score += 0.4
                
                decision = "Real" if live_score > 0.6 else "Suspicious" if live_score > 0.3 else "Spoof"
                log_entry[6] = decision
                log_entry[7] = live_score
                
                # Update display
                color = (0, 255, 0) if "Real" in decision else (0, 255, 255) if "Suspicious" in decision else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"{decision} ({live_score:.2f})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                log_entry[1] = 1

            # Log and display
            log_data(log_entry)
            cv2.imshow("Liveness Detection", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() '''

'''challenge Prompt Generation detection added'''
'''
import cv2
import dlib
import numpy as np
import pytesseract
import random
import os
from imutils import face_utils
from scipy.fftpack import fft2, fftshift

# Set path to Tesseract-OCR
pytesseract.pytesseract.tesseract_cmd = r"C:\Tesseract-OCR\tesseract.exe"

# Load Dlib's face detector and shape predictor.
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download this separately.

# Create a folder for debugging OCR images
DEBUG_FOLDER = "debug_ocr"
os.makedirs(DEBUG_FOLDER, exist_ok=True)

# --- Utility Functions ---

def generate_challenge_text(length=5):
    """Generate a random alphanumeric challenge text."""
    characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return ''.join(random.choices(characters, k=length))

def preprocess_frame(frame):
    """Convert frame to grayscale and apply CLAHE for enhanced contrast."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray_blurred)
    return enhanced

def apply_fft(image):
    """Compute FFT and return the log-magnitude spectrum (optional for liveness cues)."""
    f = fft2(image)
    fshift = fftshift(f)
    magnitude = np.log1p(np.abs(fshift))
    return magnitude

def detect_face(frame):
    """Detect faces in the frame."""
    gray = preprocess_frame(frame)
    faces = detector(gray)
    return faces

def detect_challenge(frame, challenge_text):
    """
    Run OCR on the frame to extract text and check if the challenge text is present.
    Returns (passed, detected_text).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Improve OCR readability using adaptive thresholding and noise reduction
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    sharpened = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    thresh = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Save preprocessed image for debugging
    cv2.imwrite(os.path.join(DEBUG_FOLDER, "ocr_input.jpg"), thresh)

    # Try different PSM modes to improve detection
    ocr_result = pytesseract.image_to_string(thresh, config='--psm 6')

    # Clean OCR output
    ocr_clean = ''.join(ocr_result.split()).upper()

    # Debugging output
    print(f"OCR Detected: '{ocr_clean}' | Expected: '{challenge_text}'")

    return challenge_text in ocr_clean, ocr_clean

# --- Main Application Function ---

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    challenge_text = generate_challenge_text()
    challenge_passed = False
    challenge_result_display = ""
    challenge_display_timer = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        overlay = frame.copy()

        # Display challenge text
        instruction = f"Write '{challenge_text}' on paper and hold it up"
        cv2.putText(overlay, instruction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(overlay, "Press 'c' to capture challenge", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Detect and highlight faces
        faces = detect_face(frame)
        for rect in faces:
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the challenge result
        if challenge_display_timer > 0:
            cv2.putText(overlay, challenge_result_display, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            challenge_display_timer -= 1

        cv2.imshow("Liveness Detection Challenge", overlay)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            capture_frame = frame.copy()
            faces = detect_face(capture_frame)

            if len(faces) == 0:
                challenge_result_display = "No face detected. Try again."
                challenge_display_timer = 60
                continue

            # Run OCR to verify challenge
            passed, ocr_text = detect_challenge(capture_frame, challenge_text)
            if passed:
                challenge_result_display = f"✅ Challenge passed: {ocr_text}"
                challenge_passed = True
            else:
                challenge_result_display = f"❌ Challenge failed: '{ocr_text}'"
                challenge_passed = False
            challenge_display_timer = 120

            # Generate a new challenge for the next attempt
            challenge_text = generate_challenge_text()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
'''

''' Auto-detects the text box – Finds a rectangular region, crops it, and runs OCR only on that part.
✅ Highlights the detected text box – Draws a blue box around the challenge text for debugging.
✅ Tests multiple OCR modes (--psm 6, 7, 8, 11) – Tries different OCR settings to maximize accuracy.
✅ Saves the extracted text image – Check debug_ocr/ocr_input.jpg to see what OCR is reading.'''

'''import cv2
import dlib
import numpy as np
import pytesseract
import random
from imutils import face_utils

# Set the path to Tesseract-OCR (adjust if needed based on my installation)
pytesseract.pytesseract.tesseract_cmd = r"C:\Tesseract-OCR\tesseract.exe"

# Initialize dlib's face detector
detector = dlib.get_frontal_face_detector()

# Generate Random Challenge
def generate_challenge_text(length=5):
    """Generate a random challenge text."""
    characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return ''.join(random.choices(characters, k=length))

# Preprocessing Function
def preprocess_frame(frame):
    """Apply grayscale, thresholding, sharpening, and noise removal for OCR."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply adaptive thresholding for black text on white background
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    # Sharpen the image to enhance text edges
    sharpening_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(thresh, -1, sharpening_kernel)
    # Remove noise with morphological operations
    kernel = np.ones((3, 3), np.uint8)
    processed = cv2.morphologyEx(sharpened, cv2.MORPH_CLOSE, kernel, iterations=2)
    return processed

# Detect Text Region
def detect_text_region(frame):
    """Detect the text region using filtered contours."""
    processed = preprocess_frame(frame)
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, processed
    
    # Filter contours by area and aspect ratio to target text
    valid_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect_ratio = w / h if h != 0 else 0
        # Adjust these thresholds based on my text size and camera setup
        if 1000 < area < 10000 and 0.5 < aspect_ratio < 2.0:
            valid_contours.append(cnt)
    
    if not valid_contours:
        return None, processed
    
    # Select the largest valid contour
    largest_valid_contour = max(valid_contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_valid_contour)
    
    # Add padding for better OCR detection
    padding = 10
    x, y = max(0, x - padding), max(0, y - padding)
    w, h = w + 2 * padding, h + 2 * padding
    
    return (x, y, w, h), processed

# OCR Detection Function
def detect_challenge(frame, challenge_text):
    """Perform OCR and compare with expected text."""
    bbox, processed = detect_text_region(frame)
    
    if bbox is None:
        return False, "", None, processed
    
    x, y, w, h = bbox
    roi = processed[y:y + h, x:x + w]
    
    # Resize ROI to make text larger (target height = 50 pixels)
    if h > 0:
        scale_factor = 50 / h
        roi = cv2.resize(roi, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    
    # Save ROI for debugging
    cv2.imwrite("roi.png", roi)
    
    # Tesseract configuration: single line, LSTM engine, restrict to challenge characters
    config = '--psm 7 --oem 1 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    ocr_result = pytesseract.image_to_string(roi, config=config)
    detected_text = ''.join(ocr_result.split()).upper()
    
    # Exact match for accuracy
    passed = challenge_text == detected_text
    
    return passed, detected_text, bbox, processed

# Main Function
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    challenge_text = generate_challenge_text()
    challenge_result_display = ""
    challenge_display_timer = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame = cv2.resize(frame, (640, 480))
        overlay = frame.copy()

        # Display instructions
        instruction = f"Write '{challenge_text}' and hold it up"
        cv2.putText(overlay, instruction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(overlay, "Press 'c' to capture", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Detect faces and draw bounding boxes
        faces = detector(frame)
        for rect in faces:
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display challenge result
        if challenge_display_timer > 0:
            cv2.putText(overlay, challenge_result_display, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            challenge_display_timer -= 1

        cv2.imshow("Liveness Detection Challenge", overlay)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord('c'):  # Capture
            capture_frame = frame.copy()
            passed, detected_text, bbox, processed = detect_challenge(capture_frame, challenge_text)
            
            if bbox:
                x, y, w, h = bbox
                cv2.rectangle(capture_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                # Show processed frame for additional debugging
                cv2.imshow("Processed Frame", processed)
                cv2.imshow("Detected Text Region", capture_frame)
            
            # Update result display
            if passed:
                challenge_result_display = f"✅ Challenge Passed! (Detected: {detected_text})"
            else:
                challenge_result_display = f"❌ Challenge Failed! (Detected: {detected_text})"
            
            challenge_display_timer = 120  # ~4 seconds at 30 fps
            challenge_text = generate_challenge_text()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()'''

'''INCORPORATING HAND GESTURES INSTEAD


IMPROVING THE FINGER DETECTION'''

'''
import cv2
import dlib
import mediapipe as mp
import random
import math
from imutils import face_utils

# Initialize dlib face detector
detector = dlib.get_frontal_face_detector()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to count raised fingers
def count_raised_fingers(hand_landmarks):
    """Count only fully extended fingers pointing upward, with refined thumb logic."""
    landmarks = hand_landmarks.landmark
    # Define finger indices: (tip, DIP, PIP, MCP)
    fingers = [
        (4, 3, 2, 1),   # Thumb: tip (4), DIP (3), PIP (2), MCP (1)
        (8, 7, 6, 5),   # Index: tip (8), DIP (7), PIP (6), MCP (5)
        (12, 11, 10, 9), # Middle: tip (12), DIP (11), PIP (10), MCP (9)
        (16, 15, 14, 13), # Ring: tip (16), DIP (15), PIP (14), MCP (13)
        (20, 19, 18, 17)  # Pinky: tip (20), DIP (19), PIP (18), MCP (17)
    ]
    count = 0
    
    # Wrist landmark (0) as reference for hand orientation
    wrist_y = landmarks[0].y
    
    # Thumb logic
    thumb_tip_idx, thumb_dip_idx, thumb_pip_idx, thumb_mcp_idx = fingers[0]
    thumb_tip_x, thumb_tip_y = landmarks[thumb_tip_idx].x, landmarks[thumb_tip_idx].y
    thumb_dip_y = landmarks[thumb_dip_idx].y
    thumb_pip_y = landmarks[thumb_pip_idx].y
    thumb_mcp_x, thumb_mcp_y = landmarks[thumb_mcp_idx].x, landmarks[thumb_mcp_idx].y
    
    # Thumb conditions:
    # 1. Vertical alignment with significant y-difference
    y_diff_pip = thumb_pip_y - thumb_tip_y
    y_diff_mcp = thumb_mcp_y - thumb_tip_y
    thumb_extended_vertically = thumb_tip_y < thumb_dip_y < thumb_pip_y < thumb_mcp_y and y_diff_pip > 0.03 and y_diff_mcp > 0.05
    
    # 2. Angle check: within 60° of vertical (up or down)
    dx = thumb_tip_x - thumb_mcp_x
    dy = thumb_tip_y - thumb_mcp_y  # Negative if tip is above MCP
    angle = math.degrees(math.atan2(-dy, dx))
    thumb_angle_vertical = abs(angle) < 60 or abs(angle - 180) < 60  # Relaxed from 45° to 60°
    
    # 3. Tip above wrist
    thumb_above_wrist = thumb_tip_y < wrist_y
    
    thumb_is_raised = thumb_extended_vertically and thumb_angle_vertical and thumb_above_wrist
    
    # Debugging output for thumb
    if thumb_is_raised:
        print(f"Thumb detected as raised: y_diff_pip={y_diff_pip:.3f}, y_diff_mcp={y_diff_mcp:.3f}, angle={angle:.1f}°")
    else:
        print(f"Thumb not raised: y_diff_pip={y_diff_pip:.3f}, y_diff_mcp={y_diff_mcp:.3f}, angle={angle:.1f}°")
    
    if thumb_is_raised:
        count += 1
    
    # Logic for other fingers (index to pinky)
    for tip_idx, dip_idx, pip_idx, mcp_idx in fingers[1:]:
        tip_y = landmarks[tip_idx].y
        dip_y = landmarks[dip_idx].y
        pip_y = landmarks[pip_idx].y
        mcp_y = landmarks[mcp_idx].y
        
        # Check if finger is extended upward
        is_extended = (tip_y < dip_y < pip_y < mcp_y) and (tip_y < wrist_y)
        
        if is_extended:
            count += 1
    
    return count

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize variables
challenge_fingers = random.randint(1, 5)
challenge_result_display = ""
challenge_display_timer = 0
detected_fingers = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Resize frame for consistency
    frame = cv2.resize(frame, (640, 480))
    overlay = frame.copy()

    # Detect faces using dlib
    faces = detector(frame)
    for rect in faces:
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Detect hands using MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    if results.multi_hand_landmarks:
        # Process the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        # Draw hand landmarks for visualization
        mp_drawing.draw_landmarks(overlay, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        # Count raised fingers
        detected_fingers = count_raised_fingers(hand_landmarks)
    else:
        detected_fingers = None

    # Display instructions and detected finger count
    instruction = f"Hold up {challenge_fingers} fingers with my hand upright"
    cv2.putText(overlay, instruction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(overlay, "Press 'c' to capture", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    if detected_fingers is not None:
        cv2.putText(overlay, f"Detected Fingers: {detected_fingers}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(overlay, "No hand detected", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display challenge result for a short duration
    if challenge_display_timer > 0:
        cv2.putText(overlay, challenge_result_display, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        challenge_display_timer -= 1

    # Show the frame
    cv2.imshow("Liveness Detection Challenge", overlay)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == ord('c'):  # Capture and verify
        if len(faces) > 0 and detected_fingers is not None and detected_fingers == challenge_fingers:
            challenge_result_display = f"✅ Challenge Passed! (Detected: {detected_fingers} fingers)"
            print("Person is likely real.")
        else:
            challenge_result_display = f"❌ Challenge Failed! (Detected: {detected_fingers if detected_fingers is not None else 'no hand'})"
            print("Person may not be real.")
        challenge_display_timer = 120  # Display result for ~2 seconds at 60 FPS
        challenge_fingers = random.randint(1, 5)  # Generate new challenge

# Clean up
cap.release()
cv2.destroyAllWindows()
hands.close()'''

'''


import cv2
import dlib
import mediapipe as mp
import random
import math
from imutils import face_utils

# Initialize dlib face detector
detector = dlib.get_frontal_face_detector()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to count raised fingers
def count_raised_fingers(hand_landmarks):
    """Count only fully extended fingers pointing upward, with refined thumb logic."""
    landmarks = hand_landmarks.landmark
    # Define finger indices: (tip, DIP, PIP, MCP)
    fingers = [
        (4, 3, 2, 1),   # Thumb: tip (4), DIP (3), PIP (2), MCP (1)
        (8, 7, 6, 5),   # Index: tip (8), DIP (7), PIP (6), MCP (5)
        (12, 11, 10, 9), # Middle: tip (12), DIP (11), PIP (10), MCP (9)
        (16, 15, 14, 13), # Ring: tip (16), DIP (15), PIP (14), MCP (13)
        (20, 19, 18, 17)  # Pinky: tip (20), DIP (19), PIP (18), MCP (17)
    ]
    count = 0
    
    # Wrist landmark (0) as reference for hand orientation
    wrist_y = landmarks[0].y
    
    # Thumb logic
    thumb_tip_idx, thumb_dip_idx, thumb_pip_idx, thumb_mcp_idx = fingers[0]
    thumb_tip_x, thumb_tip_y = landmarks[thumb_tip_idx].x, landmarks[thumb_tip_idx].y
    thumb_dip_y = landmarks[thumb_dip_idx].y
    thumb_pip_y = landmarks[thumb_pip_idx].y
    thumb_mcp_x, thumb_mcp_y = landmarks[thumb_mcp_idx].x, landmarks[thumb_mcp_idx].y
    
    # Thumb conditions:
    # 1. Vertical alignment with significant y-differences
    y_diff_pip = thumb_pip_y - thumb_tip_y
    y_diff_mcp = thumb_mcp_y - thumb_tip_y
    y_diff_dip_pip = thumb_pip_y - thumb_dip_y  # New: DIP to PIP difference
    thumb_extended_vertically = (thumb_tip_y < thumb_dip_y < thumb_pip_y < thumb_mcp_y and 
                                 y_diff_pip > 0.03 and 
                                 y_diff_mcp > 0.05 and 
                                 y_diff_dip_pip > 0.02)  # Ensure straight alignment
    
    # 2. Angle check: within 75° of vertical (up or down)
    dx = thumb_tip_x - thumb_mcp_x
    dy = thumb_tip_y - thumb_mcp_y  # Negative if tip is above MCP
    angle = math.degrees(math.atan2(-dy, dx))
    thumb_angle_vertical = abs(angle) < 75 or abs(angle - 180) < 75  # Relaxed to 75°
    
    # 3. Tip above wrist
    thumb_above_wrist = thumb_tip_y < wrist_y
    
    thumb_is_raised = thumb_extended_vertically and thumb_angle_vertical and thumb_above_wrist
    
    # Debugging output for thumb
    if thumb_is_raised:
        print(f"Thumb detected as raised: y_diff_pip={y_diff_pip:.3f}, y_diff_mcp={y_diff_mcp:.3f}, y_diff_dip_pip={y_diff_dip_pip:.3f}, angle={angle:.1f}°")
    else:
        print(f"Thumb not raised: y_diff_pip={y_diff_pip:.3f}, y_diff_mcp={y_diff_mcp:.3f}, y_diff_dip_pip={y_diff_dip_pip:.3f}, angle={angle:.1f}°")
    
    if thumb_is_raised:
        count += 1
    
    # Logic for other fingers (index to pinky)
    for tip_idx, dip_idx, pip_idx, mcp_idx in fingers[1:]:
        tip_y = landmarks[tip_idx].y
        dip_y = landmarks[dip_idx].y
        pip_y = landmarks[pip_idx].y
        mcp_y = landmarks[mcp_idx].y
        
        # Check if finger is extended upward
        is_extended = (tip_y < dip_y < pip_y < mcp_y) and (tip_y < wrist_y)
        
        if is_extended:
            count += 1
    
    return count

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize variables
challenge_fingers = random.randint(1, 5)
challenge_result_display = ""
challenge_display_timer = 0
detected_fingers = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Resize frame for consistency
    frame = cv2.resize(frame, (640, 480))
    overlay = frame.copy()

    # Detect faces using dlib
    faces = detector(frame)
    for rect in faces:
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Detect hands using MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    if results.multi_hand_landmarks:
        # Process the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        # Draw hand landmarks for visualization
        mp_drawing.draw_landmarks(overlay, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        # Count raised fingers
        detected_fingers = count_raised_fingers(hand_landmarks)
    else:
        detected_fingers = None

    # Display instructions and detected finger count
    instruction = f"Hold up {challenge_fingers} fingers with my hand upright"
    cv2.putText(overlay, instruction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(overlay, "Press 'c' to capture", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    if detected_fingers is not None:
        cv2.putText(overlay, f"Detected Fingers: {detected_fingers}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(overlay, "No hand detected", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display challenge result for a short duration
    if challenge_display_timer > 0:
        cv2.putText(overlay, challenge_result_display, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        challenge_display_timer -= 1

    # Show the frame
    cv2.imshow("Liveness Detection Challenge", overlay)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == ord('c'):  # Capture and verify
        if len(faces) > 0 and detected_fingers is not None and detected_fingers == challenge_fingers:
            challenge_result_display = f"✅ Challenge Passed! (Detected: {detected_fingers} fingers)"
            print("Person is likely real.")
        else:
            challenge_result_display = f"❌ Challenge Failed! (Detected: {detected_fingers if detected_fingers is not None else 'no hand'})"
            print("Person may not be real.")
        challenge_display_timer = 120  # Display result for ~2 seconds at 60 FPS
        challenge_fingers = random.randint(1, 5)  # Generate new challenge

# Clean up
cap.release()
cv2.destroyAllWindows()
hands.close()

Improving the text
'''
'''
import cv2
import dlib
import mediapipe as mp
import random
import math
from imutils import face_utils

# Initialize dlib face detector
detector = dlib.get_frontal_face_detector()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to count raised fingers
def count_raised_fingers(hand_landmarks):
    """Count only fully extended fingers pointing upward, with refined thumb logic."""
    landmarks = hand_landmarks.landmark
    # Define finger indices: (tip, DIP, PIP, MCP)
    fingers = [
        (4, 3, 2, 1),   # Thumb: tip (4), DIP (3), PIP (2), MCP (1)
        (8, 7, 6, 5),   # Index: tip (8), DIP (7), PIP (6), MCP (5)
        (12, 11, 10, 9), # Middle: tip (12), DIP (11), PIP (10), MCP (9)
        (16, 15, 14, 13), # Ring: tip (16), DIP (15), PIP (14), MCP (13)
        (20, 19, 18, 17)  # Pinky: tip (20), DIP (19), PIP (18), MCP (17)
    ]
    count = 0
    
    # Wrist landmark (0) as reference for hand orientation
    wrist_y = landmarks[0].y
    
    # Thumb logic
    thumb_tip_idx, thumb_dip_idx, thumb_pip_idx, thumb_mcp_idx = fingers[0]
    thumb_tip_x, thumb_tip_y = landmarks[thumb_tip_idx].x, landmarks[thumb_tip_idx].y
    thumb_dip_y = landmarks[thumb_dip_idx].y
    thumb_pip_y = landmarks[thumb_pip_idx].y
    thumb_mcp_x, thumb_mcp_y = landmarks[thumb_mcp_idx].x, landmarks[thumb_mcp_idx].y
    
    # Thumb conditions:
    # 1. Vertical alignment with significant y-differences
    y_diff_pip = thumb_pip_y - thumb_tip_y
    y_diff_mcp = thumb_mcp_y - thumb_tip_y
    y_diff_dip_pip = thumb_pip_y - thumb_dip_y  # New: DIP to PIP difference
    thumb_extended_vertically = (thumb_tip_y < thumb_dip_y < thumb_pip_y < thumb_mcp_y and 
                                 y_diff_pip > 0.03 and 
                                 y_diff_mcp > 0.05 and 
                                 y_diff_dip_pip > 0.02)  # Ensure straight alignment
    
    # 2. Angle check: within 75° of vertical (up or down)
    dx = thumb_tip_x - thumb_mcp_x
    dy = thumb_tip_y - thumb_mcp_y  # Negative if tip is above MCP
    angle = math.degrees(math.atan2(-dy, dx))
    thumb_angle_vertical = abs(angle) < 75 or abs(angle - 180) < 75  # Relaxed to 75°
    
    # 3. Tip above wrist
    thumb_above_wrist = thumb_tip_y < wrist_y
    
    thumb_is_raised = thumb_extended_vertically and thumb_angle_vertical and thumb_above_wrist
    
    # Debugging output for thumb
    if thumb_is_raised:
        print(f"Thumb detected as raised: y_diff_pip={y_diff_pip:.3f}, y_diff_mcp={y_diff_mcp:.3f}, y_diff_dip_pip={y_diff_dip_pip:.3f}, angle={angle:.1f}°")
    else:
        print(f"Thumb not raised: y_diff_pip={y_diff_pip:.3f}, y_diff_mcp={y_diff_mcp:.3f}, y_diff_dip_pip={y_diff_dip_pip:.3f}, angle={angle:.1f}°")
    
    if thumb_is_raised:
        count += 1
    
    # Logic for other fingers (index to pinky)
    for tip_idx, dip_idx, pip_idx, mcp_idx in fingers[1:]:
        tip_y = landmarks[tip_idx].y
        dip_y = landmarks[dip_idx].y
        pip_y = landmarks[pip_idx].y
        mcp_y = landmarks[mcp_idx].y
        
        # Check if finger is extended upward
        is_extended = (tip_y < dip_y < pip_y < mcp_y) and (tip_y < wrist_y)
        
        if is_extended:
            count += 1
    
    return count

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize variables
challenge_fingers = random.randint(1, 5)
challenge_result_display = ""
challenge_display_timer = 0
detected_fingers = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Resize frame for consistency
    frame = cv2.resize(frame, (640, 480))
    overlay = frame.copy()

    # Detect faces using dlib
    faces = detector(frame)
    for rect in faces:
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Detect hands using MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    if results.multi_hand_landmarks:
        # Process the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        # Draw hand landmarks for visualization
        mp_drawing.draw_landmarks(overlay, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        # Count raised fingers
        detected_fingers = count_raised_fingers(hand_landmarks)
    else:
        detected_fingers = None

    # Display instructions and detected finger count
    instruction = f"Hold up {challenge_fingers} fingers with my hand upright. If thumb is not used please try to bend it downwards to pass the challenge easily"
    cv2.putText(overlay, instruction, (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(overlay, "Press 'c' to capture", (10, 60), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 2)
    
    if detected_fingers is not None:
        cv2.putText(overlay, f"Detected Fingers: {detected_fingers}", (10, 120), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(overlay, "No hand detected", (10, 120), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255), 2)

    # Display challenge result for a short duration
    if challenge_display_timer > 0:
        cv2.putText(overlay, challenge_result_display, (10, 90), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255), 2)
        challenge_display_timer -= 1

    # Show the frame
    cv2.imshow("Liveness Detection Challenge", overlay)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == ord('c'):  # Capture and verify
        if len(faces) > 0 and detected_fingers is not None and detected_fingers == challenge_fingers:
            challenge_result_display = f"✅ Challenge Passed! (Detected: {detected_fingers} fingers)"
            print("Person is likely real.")
        else:
            challenge_result_display = f"❌ Challenge Failed! (Detected: {detected_fingers if detected_fingers is not None else 'no hand'})"
            print("Person may not be real.")
        challenge_display_timer = 120  # Display result for ~2 seconds at 60 FPS
        challenge_fingers = random.randint(1, 5)  # Generate new challenge

# Clean up
cap.release()
cv2.destroyAllWindows()
hands.close()'''

#Improving face detection using media pipe instead of dlib
'''
import cv2
import mediapipe as mp
import random
import math

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to count raised fingers
def count_raised_fingers(hand_landmarks):
    """Count only fully extended fingers pointing upward, with refined thumb logic."""
    landmarks = hand_landmarks.landmark
    # Define finger indices: (tip, DIP, PIP, MCP)
    fingers = [
        (4, 3, 2, 1),   # Thumb: tip (4), DIP (3), PIP (2), MCP (1)
        (8, 7, 6, 5),   # Index: tip (8), DIP (7), PIP (6), MCP (5)
        (12, 11, 10, 9), # Middle: tip (12), DIP (11), PIP (10), MCP (9)
        (16, 15, 14, 13), # Ring: tip (16), DIP (15), PIP (14), MCP (13)
        (20, 19, 18, 17)  # Pinky: tip (20), DIP (19), PIP (18), MCP (17)
    ]
    count = 0
    
    # Wrist landmark (0) as reference for hand orientation
    wrist_y = landmarks[0].y
    
    # Thumb logic
    thumb_tip_idx, thumb_dip_idx, thumb_pip_idx, thumb_mcp_idx = fingers[0]
    thumb_tip_x, thumb_tip_y = landmarks[thumb_tip_idx].x, landmarks[thumb_tip_idx].y
    thumb_dip_y = landmarks[thumb_dip_idx].y
    thumb_pip_y = landmarks[thumb_pip_idx].y
    thumb_mcp_x, thumb_mcp_y = landmarks[thumb_mcp_idx].x, landmarks[thumb_mcp_idx].y
    
    # Thumb conditions:
    # 1. Vertical alignment with significant y-differences
    y_diff_pip = thumb_pip_y - thumb_tip_y
    y_diff_mcp = thumb_mcp_y - thumb_tip_y
    y_diff_dip_pip = thumb_pip_y - thumb_dip_y  # New: DIP to PIP difference
    thumb_extended_vertically = (thumb_tip_y < thumb_dip_y < thumb_pip_y < thumb_mcp_y and 
                                 y_diff_pip > 0.03 and 
                                 y_diff_mcp > 0.05 and 
                                 y_diff_dip_pip > 0.02)  # Ensure straight alignment
    
    # 2. Angle check: within 75° of vertical (up or down)
    dx = thumb_tip_x - thumb_mcp_x
    dy = thumb_tip_y - thumb_mcp_y  # Negative if tip is above MCP
    angle = math.degrees(math.atan2(-dy, dx))
    thumb_angle_vertical = abs(angle) < 75 or abs(angle - 180) < 75  # Relaxed to 75°
    
    # 3. Tip above wrist
    thumb_above_wrist = thumb_tip_y < wrist_y
    
    thumb_is_raised = thumb_extended_vertically and thumb_angle_vertical and thumb_above_wrist
    
    # Debugging output for thumb
    if thumb_is_raised:
        print(f"Thumb detected as raised: y_diff_pip={y_diff_pip:.3f}, y_diff_mcp={y_diff_mcp:.3f}, y_diff_dip_pip={y_diff_dip_pip:.3f}, angle={angle:.1f}°")
    else:
        print(f"Thumb not raised: y_diff_pip={y_diff_pip:.3f}, y_diff_mcp={y_diff_mcp:.3f}, y_diff_dip_pip={y_diff_dip_pip:.3f}, angle={angle:.1f}°")
    
    if thumb_is_raised:
        count += 1
    
    # Logic for other fingers (index to pinky)
    for tip_idx, dip_idx, pip_idx, mcp_idx in fingers[1:]:
        tip_y = landmarks[tip_idx].y
        dip_y = landmarks[dip_idx].y
        pip_y = landmarks[pip_idx].y
        mcp_y = landmarks[mcp_idx].y
        
        # Check if finger is extended upward
        is_extended = (tip_y < dip_y < pip_y < mcp_y) and (tip_y < wrist_y)
        
        if is_extended:
            count += 1
    
    return count

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize variables
challenge_fingers = random.randint(1, 5)
challenge_result_display = ""
challenge_display_timer = 0
detected_fingers = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Resize frame for consistency
    frame = cv2.resize(frame, (640, 480))
    overlay = frame.copy()

    # Detect faces using MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    faces = []
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)
            # Adjust for negative coordinates
            x, y = max(0, x), max(0, y)
            faces.append((x, y, w, h))
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Detect hands using MediaPipe
    results = hands.process(rgb_frame)
    if results.multi_hand_landmarks:
        # Process the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        # Draw hand landmarks for visualization
        mp_drawing.draw_landmarks(overlay, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        # Count raised fingers
        detected_fingers = count_raised_fingers(hand_landmarks)
    else:
        detected_fingers = None

    # Display instructions and detected finger count
    instruction = f"Hold up {challenge_fingers} fingers with my hand upright. If a finger is not used make sure it is below my knuckles"
    cv2.putText(overlay, instruction, (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(overlay, "Press 'c' to capture", (10, 60), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 2)
    
    if detected_fingers is not None:
        cv2.putText(overlay, f"Detected Fingers: {detected_fingers}", (10, 120), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(overlay, "No hand detected", (10, 120), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255), 2)

    # Display challenge result for a short duration
    if challenge_display_timer > 0:
        cv2.putText(overlay, challenge_result_display, (10, 90), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255), 2)
        challenge_display_timer -= 1

    # Show the frame
    cv2.imshow("Liveness Detection Challenge", overlay)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == ord('c'):  # Capture and verify
        if len(faces) > 0 and detected_fingers is not None and detected_fingers == challenge_fingers:
            challenge_result_display = f"✅ Challenge Passed! (Detected: {detected_fingers} fingers)"
            print("Person is likely real.")
        else:
            challenge_result_display = f"❌ Challenge Failed! (Detected: {detected_fingers if detected_fingers is not None else 'no hand'})"
            print("Person may not be real.")
        challenge_display_timer = 120  # Display result for ~2 seconds at 60 FPS
        challenge_fingers = random.randint(1, 5)  # Generate new challenge

# Clean up
cap.release()
cv2.destroyAllWindows()
hands.close()
face_detection.close()'''

#Added resizable functionality to camera window

'''import cv2
import mediapipe as mp
import random
import math

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to count raised fingers
def count_raised_fingers(hand_landmarks):
    """Count only fully extended fingers pointing upward, with refined thumb logic."""
    landmarks = hand_landmarks.landmark
    # Define finger indices: (tip, DIP, PIP, MCP)
    fingers = [
        (4, 3, 2, 1),   # Thumb: tip (4), DIP (3), PIP (2), MCP (1)
        (8, 7, 6, 5),   # Index: tip (8), DIP (7), PIP (6), MCP (5)
        (12, 11, 10, 9), # Middle: tip (12), DIP (11), PIP (10), MCP (9)
        (16, 15, 14, 13), # Ring: tip (16), DIP (15), PIP (14), MCP (13)
        (20, 19, 18, 17)  # Pinky: tip (20), DIP (19), PIP (18), MCP (17)
    ]
    count = 0
    
    # Wrist landmark (0) as reference for hand orientation
    wrist_y = landmarks[0].y
    
    # Thumb logic
    thumb_tip_idx, thumb_dip_idx, thumb_pip_idx, thumb_mcp_idx = fingers[0]
    thumb_tip_x, thumb_tip_y = landmarks[thumb_tip_idx].x, landmarks[thumb_tip_idx].y
    thumb_dip_y = landmarks[thumb_dip_idx].y
    thumb_pip_y = landmarks[thumb_pip_idx].y
    thumb_mcp_x, thumb_mcp_y = landmarks[thumb_mcp_idx].x, landmarks[thumb_mcp_idx].y
    
    # Thumb conditions:
    # 1. Vertical alignment with significant y-differences
    y_diff_pip = thumb_pip_y - thumb_tip_y
    y_diff_mcp = thumb_mcp_y - thumb_tip_y
    y_diff_dip_pip = thumb_pip_y - thumb_dip_y  # New: DIP to PIP difference
    thumb_extended_vertically = (thumb_tip_y < thumb_dip_y < thumb_pip_y < thumb_mcp_y and 
                                 y_diff_pip > 0.03 and 
                                 y_diff_mcp > 0.05 and 
                                 y_diff_dip_pip > 0.02)  # Ensure straight alignment
    
    # 2. Angle check: within 75° of vertical (up or down)
    dx = thumb_tip_x - thumb_mcp_x
    dy = thumb_tip_y - thumb_mcp_y  # Negative if tip is above MCP
    angle = math.degrees(math.atan2(-dy, dx))
    thumb_angle_vertical = abs(angle) < 75 or abs(angle - 180) < 75  # Relaxed to 75°
    
    # 3. Tip above wrist
    thumb_above_wrist = thumb_tip_y < wrist_y
    
    thumb_is_raised = thumb_extended_vertically and thumb_angle_vertical and thumb_above_wrist
    
    # Debugging output for thumb
    if thumb_is_raised:
        print(f"Thumb detected as raised: y_diff_pip={y_diff_pip:.3f}, y_diff_mcp={y_diff_mcp:.3f}, y_diff_dip_pip={y_diff_dip_pip:.3f}, angle={angle:.1f}°")
    else:
        print(f"Thumb not raised: y_diff_pip={y_diff_pip:.3f}, y_diff_mcp={y_diff_mcp:.3f}, y_diff_dip_pip={y_diff_dip_pip:.3f}, angle={angle:.1f}°")
    
    if thumb_is_raised:
        count += 1
    
    # Logic for other fingers (index to pinky)
    for tip_idx, dip_idx, pip_idx, mcp_idx in fingers[1:]:
        tip_y = landmarks[tip_idx].y
        dip_y = landmarks[dip_idx].y
        pip_y = landmarks[pip_idx].y
        mcp_y = landmarks[mcp_idx].y
        
        # Check if finger is extended upward
        is_extended = (tip_y < dip_y < pip_y < mcp_y) and (tip_y < wrist_y)
        
        if is_extended:
            count += 1
    
    return count

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Create a named window
window_name = "Liveness Detection Challenge"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Window state variables
is_fullscreen = False
custom_width = 800
custom_height = 600

# Set initial window size
cv2.resizeWindow(window_name, custom_width, custom_height)

# Initialize variables
challenge_fingers = random.randint(1, 5)
challenge_result_display = ""
challenge_display_timer = 0
detected_fingers = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Get the actual frame dimensions
    frame_height, frame_width = frame.shape[:2]

    overlay = frame.copy()

    # Detect faces using MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    faces = []
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)
            # Adjust for negative coordinates
            x, y = max(0, x), max(0, y)
            faces.append((x, y, w, h))
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Detect hands using MediaPipe
    results = hands.process(rgb_frame)
    if results.multi_hand_landmarks:
        # Process the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        # Draw hand landmarks for visualization
        mp_drawing.draw_landmarks(overlay, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        # Count raised fingers
        detected_fingers = count_raised_fingers(hand_landmarks)
    else:
        detected_fingers = None

    # Calculate text positions based on frame dimensions
    # Scale font size based on frame width
    font_scale = frame_width / 640 * 0.7
    text_thickness = max(1, int(frame_width / 640 * 2))
    line_spacing = int(30 * (frame_height / 480))
    
    # Display instructions and detected finger count
    instruction = f"Hold up {challenge_fingers} fingers with my hand upright. If a finger is not used make sure it is below my knuckles"
    cv2.putText(overlay, instruction, (20, line_spacing), cv2.FONT_HERSHEY_TRIPLEX, font_scale, (0, 0, 0), text_thickness)
    cv2.putText(overlay, "Press 'c' to capture, 'f' to toggle fullscreen, '+/-' to resize, 'q' to quit", 
                (20, 2*line_spacing), cv2.FONT_HERSHEY_TRIPLEX, font_scale, (0, 0, 0), text_thickness)
    
    if detected_fingers is not None:
        cv2.putText(overlay, f"Detected Fingers: {detected_fingers}", (20, 4*line_spacing), cv2.FONT_HERSHEY_TRIPLEX, font_scale, (0, 255, 0), text_thickness)
    else:
        cv2.putText(overlay, "No hand detected", (20, 4*line_spacing), cv2.FONT_HERSHEY_TRIPLEX, font_scale, (0, 0, 255), text_thickness)

    # Display challenge result for a short duration
    if challenge_display_timer > 0:
        cv2.putText(overlay, challenge_result_display, (20, 3*line_spacing), cv2.FONT_HERSHEY_TRIPLEX, font_scale, (0, 0, 255), text_thickness)
        challenge_display_timer -= 1

    # Display current window size
    size_text = f"Window Size: {custom_width}x{custom_height}"
    cv2.putText(overlay, size_text, (20, 5*line_spacing), cv2.FONT_HERSHEY_TRIPLEX, font_scale, (0, 0, 0), text_thickness)

    # Show the frame
    cv2.imshow(window_name, overlay)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == ord('c'):  # Capture and verify
        if len(faces) > 0 and detected_fingers is not None and detected_fingers == challenge_fingers:
            challenge_result_display = f"✅ Challenge Passed! (Detected: {detected_fingers} fingers)"
            print("Person is likely real.")
        else:
            challenge_result_display = f"❌ Challenge Failed! (Detected: {detected_fingers if detected_fingers is not None else 'no hand'})"
            print("Person may not be real.")
        challenge_display_timer = 120  # Display result for ~2 seconds at 60 FPS
        challenge_fingers = random.randint(1, 5)  # Generate new challenge
    elif key == ord('f'):  # Toggle fullscreen
        is_fullscreen = not is_fullscreen
        if is_fullscreen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, custom_width, custom_height)
    elif key == ord('+') or key == ord('='):  # Increase window size
        if not is_fullscreen:
            custom_width = min(custom_width + 100, 1920)  # Max width 1920
            custom_height = min(custom_height + 75, 1080)  # Max height 1080
            cv2.resizeWindow(window_name, custom_width, custom_height)
    elif key == ord('-') or key == ord('_'):  # Decrease window size
        if not is_fullscreen:
            custom_width = max(custom_width - 100, 400)  # Min width 400
            custom_height = max(custom_height - 75, 300)  # Min height 300
            cv2.resizeWindow(window_name, custom_width, custom_height)

# Clean up
cap.release()
cv2.destroyAllWindows()
hands.close()
face_detection.close()'''
# improved instructions and text
'''
import cv2
import mediapipe as mp
import random
import math

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to count raised fingers
def count_raised_fingers(hand_landmarks):
    """Count only fully extended fingers pointing upward, with refined thumb logic."""
    landmarks = hand_landmarks.landmark
    # Define finger indices: (tip, DIP, PIP, MCP)
    fingers = [
        (4, 3, 2, 1),   # Thumb: tip (4), DIP (3), PIP (2), MCP (1)
        (8, 7, 6, 5),   # Index: tip (8), DIP (7), PIP (6), MCP (5)
        (12, 11, 10, 9), # Middle: tip (12), DIP (11), PIP (10), MCP (9)
        (16, 15, 14, 13), # Ring: tip (16), DIP (15), PIP (14), MCP (13)
        (20, 19, 18, 17)  # Pinky: tip (20), DIP (19), PIP (18), MCP (17)
    ]
    count = 0
    
    # Wrist landmark (0) as reference for hand orientation
    wrist_y = landmarks[0].y
    
    # Thumb logic
    thumb_tip_idx, thumb_dip_idx, thumb_pip_idx, thumb_mcp_idx = fingers[0]
    thumb_tip_x, thumb_tip_y = landmarks[thumb_tip_idx].x, landmarks[thumb_tip_idx].y
    thumb_dip_y = landmarks[thumb_dip_idx].y
    thumb_pip_y = landmarks[thumb_pip_idx].y
    thumb_mcp_x, thumb_mcp_y = landmarks[thumb_mcp_idx].x, landmarks[thumb_mcp_idx].y
    
    # Thumb conditions:
    # 1. Vertical alignment with significant y-differences
    y_diff_pip = thumb_pip_y - thumb_tip_y
    y_diff_mcp = thumb_mcp_y - thumb_tip_y
    y_diff_dip_pip = thumb_pip_y - thumb_dip_y  # New: DIP to PIP difference
    thumb_extended_vertically = (thumb_tip_y < thumb_dip_y < thumb_pip_y < thumb_mcp_y and 
                                 y_diff_pip > 0.03 and 
                                 y_diff_mcp > 0.05 and 
                                 y_diff_dip_pip > 0.02)  # Ensure straight alignment
    
    # 2. Angle check: within 75° of vertical (up or down)
    dx = thumb_tip_x - thumb_mcp_x
    dy = thumb_tip_y - thumb_mcp_y  # Negative if tip is above MCP
    angle = math.degrees(math.atan2(-dy, dx))
    thumb_angle_vertical = abs(angle) < 75 or abs(angle - 180) < 75  # Relaxed to 75°
    
    # 3. Tip above wrist
    thumb_above_wrist = thumb_tip_y < wrist_y
    
    thumb_is_raised = thumb_extended_vertically and thumb_angle_vertical and thumb_above_wrist
    
    # Debugging output for thumb
    if thumb_is_raised:
        print(f"Thumb detected as raised: y_diff_pip={y_diff_pip:.3f}, y_diff_mcp={y_diff_mcp:.3f}, y_diff_dip_pip={y_diff_dip_pip:.3f}, angle={angle:.1f}°")
    else:
        print(f"Thumb not raised: y_diff_pip={y_diff_pip:.3f}, y_diff_mcp={y_diff_mcp:.3f}, y_diff_dip_pip={y_diff_dip_pip:.3f}, angle={angle:.1f}°")
    
    if thumb_is_raised:
        count += 1
    
    # Logic for other fingers (index to pinky)
    for tip_idx, dip_idx, pip_idx, mcp_idx in fingers[1:]:
        tip_y = landmarks[tip_idx].y
        dip_y = landmarks[dip_idx].y
        pip_y = landmarks[pip_idx].y
        mcp_y = landmarks[mcp_idx].y
        
        # Check if finger is extended upward
        is_extended = (tip_y < dip_y < pip_y < mcp_y) and (tip_y < wrist_y)
        
        if is_extended:
            count += 1
    
    return count

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Create a named window
window_name = "Liveness Detection Challenge"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Window state variables
is_fullscreen = False
custom_width = 800
custom_height = 600

# Set initial window size
cv2.resizeWindow(window_name, custom_width, custom_height)

# Initialize variables
challenge_fingers = random.randint(1, 5)
challenge_result_display = ""
challenge_result_color = (0, 0, 0)  # Default black
challenge_display_timer = 0
detected_fingers = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Get the actual frame dimensions
    frame_height, frame_width = frame.shape[:2]

    overlay = frame.copy()

    # Detect faces using MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    faces = []
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)
            # Adjust for negative coordinates
            x, y = max(0, x), max(0, y)
            faces.append((x, y, w, h))
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Detect hands using MediaPipe
    results = hands.process(rgb_frame)
    if results.multi_hand_landmarks:
        # Process the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        # Draw hand landmarks for visualization
        mp_drawing.draw_landmarks(overlay, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        # Count raised fingers
        detected_fingers = count_raised_fingers(hand_landmarks)
    else:
        detected_fingers = None

    # Calculate text positions based on frame dimensions
    # Use smaller font size
    font_scale = frame_width / 1280 * 0.7  # Reduced font scale
    text_thickness = max(1, int(frame_width / 1280 * 2))
    line_spacing = int(25 * (frame_height / 480))  # Reduced line spacing
    
    # Use a cleaner font
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Create a semi-transparent overlay for text background
    text_bg = overlay.copy()
    cv2.rectangle(text_bg, (10, 10), (frame_width - 10, 6*line_spacing), (240, 240, 240), -1)
    cv2.addWeighted(text_bg, 0.7, overlay, 0.3, 0, overlay)
    
    # Display instructions and detected finger count
    instruction = f"Hold up {challenge_fingers} fingers with my hand upright."
    cv2.putText(overlay, instruction, (20, line_spacing), font, font_scale, (0, 0, 0), text_thickness)
    cv2.putText(overlay, "Press 'c' to capture, 'f' to toggle fullscreen, '+/-' to resize, 'q' to quit", 
                (20, 2*line_spacing), font, font_scale*0.8, (0, 0, 0), text_thickness-1)  # Smaller font for controls
    
    if detected_fingers is not None:
        cv2.putText(overlay, f"Detected Fingers: {detected_fingers}", (20, 3*line_spacing), font, font_scale, (0, 0, 0), text_thickness)
    else:
        cv2.putText(overlay, "No hand detected", (20, 3*line_spacing), font, font_scale, (0, 0, 0), text_thickness)

    # Display challenge result for a short duration
    if challenge_display_timer > 0:
        cv2.putText(overlay, challenge_result_display, (20, 4*line_spacing), font, font_scale, challenge_result_color, text_thickness)
        challenge_display_timer -= 1

    # Display current window size
    size_text = f"Window Size: {custom_width}x{custom_height}"
    cv2.putText(overlay, size_text, (20, 5*line_spacing), font, font_scale*0.8, (0, 0, 0), text_thickness-1)  # Smaller font for window size

    # Show the frame
    cv2.imshow(window_name, overlay)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == ord('c'):  # Capture and verify
        if len(faces) > 0 and detected_fingers is not None and detected_fingers == challenge_fingers:
            challenge_result_display = "Live person detected (Challenge passed)"
            challenge_result_color = (0, 255, 0)  # Green for pass
            print("Person is likely real.")
        else:
            challenge_result_display = "Challenge failed. Please try again"
            challenge_result_color = (0, 0, 255)  # Red for fail
            print("Person may not be real.")
        challenge_display_timer = 120  # Display result for ~2 seconds at 60 FPS
        challenge_fingers = random.randint(1, 5)  # Generate new challenge
    elif key == ord('f'):  # Toggle fullscreen
        is_fullscreen = not is_fullscreen
        if is_fullscreen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, custom_width, custom_height)
    elif key == ord('+') or key == ord('='):  # Increase window size
        if not is_fullscreen:
            custom_width = min(custom_width + 100, 1920)  # Max width 1920
            custom_height = min(custom_height + 75, 1080)  # Max height 1080
            cv2.resizeWindow(window_name, custom_width, custom_height)
    elif key == ord('-') or key == ord('_'):  # Decrease window size
        if not is_fullscreen:
            custom_width = max(custom_width - 100, 400)  # Min width 400
            custom_height = max(custom_height - 75, 300)  # Min height 300
            cv2.resizeWindow(window_name, custom_width, custom_height)

# Clean up
cap.release()
cv2.destroyAllWindows()
hands.close()
face_detection.close()'''
#Latest Working model
'''import cv2
import mediapipe as mp
import random
import math

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to count raised fingers
def count_raised_fingers(hand_landmarks):
    """Count only fully extended fingers pointing upward, with refined thumb logic."""
    landmarks = hand_landmarks.landmark
    # Define finger indices: (tip, DIP, PIP, MCP)
    fingers = [
        (4, 3, 2, 1),   # Thumb: tip (4), DIP (3), PIP (2), MCP (1)
        (8, 7, 6, 5),   # Index: tip (8), DIP (7), PIP (6), MCP (5)
        (12, 11, 10, 9), # Middle: tip (12), DIP (11), PIP (10), MCP (9)
        (16, 15, 14, 13), # Ring: tip (16), DIP (15), PIP (14), MCP (13)
        (20, 19, 18, 17)  # Pinky: tip (20), DIP (19), PIP (18), MCP (17)
    ] #the finger structure based on MediaPipe’s hand tracking index.
    count = 0
    
    # Wrist landmark (0) as reference for hand orientation
    wrist_y = landmarks[0].y
    
    # Thumb logic
    thumb_tip_idx, thumb_dip_idx, thumb_pip_idx, thumb_mcp_idx = fingers[0]
    thumb_tip_x, thumb_tip_y = landmarks[thumb_tip_idx].x, landmarks[thumb_tip_idx].y
    thumb_dip_y = landmarks[thumb_dip_idx].y
    thumb_pip_y = landmarks[thumb_pip_idx].y
    thumb_mcp_x, thumb_mcp_y = landmarks[thumb_mcp_idx].x, landmarks[thumb_mcp_idx].y
    
    # Thumb conditions:
    # 1. Vertical alignment with significant y-differences
    y_diff_pip = thumb_pip_y - thumb_tip_y
    y_diff_mcp = thumb_mcp_y - thumb_tip_y
    y_diff_dip_pip = thumb_pip_y - thumb_dip_y  # New: DIP to PIP difference
    thumb_extended_vertically = (thumb_tip_y < thumb_dip_y < thumb_pip_y < thumb_mcp_y and 
                                 y_diff_pip > 0.03 and 
                                 y_diff_mcp > 0.05 and 
                                 y_diff_dip_pip > 0.02)  # Ensure straight alignment
    
    # 2. Angle check: within 75° of vertical (up or down)
    dx = thumb_tip_x - thumb_mcp_x
    dy = thumb_tip_y - thumb_mcp_y  # Negative if tip is above MCP
    angle = math.degrees(math.atan2(-dy, dx))
    thumb_angle_vertical = abs(angle) < 75 or abs(angle - 180) < 75  # Relaxed to 75°
    
    # 3. Tip above wrist
    thumb_above_wrist = thumb_tip_y < wrist_y
    
    thumb_is_raised = thumb_extended_vertically and thumb_angle_vertical and thumb_above_wrist
    
    # Debugging output for thumb
    if thumb_is_raised:
        print(f"Thumb detected as raised: y_diff_pip={y_diff_pip:.3f}, y_diff_mcp={y_diff_mcp:.3f}, y_diff_dip_pip={y_diff_dip_pip:.3f}, angle={angle:.1f}°")
    else:
        print(f"Thumb not raised: y_diff_pip={y_diff_pip:.3f}, y_diff_mcp={y_diff_mcp:.3f}, y_diff_dip_pip={y_diff_dip_pip:.3f}, angle={angle:.1f}°")
    
    if thumb_is_raised:
        count += 1
    
    # Logic for other fingers (index to pinky)
    for tip_idx, dip_idx, pip_idx, mcp_idx in fingers[1:]:
        tip_y = landmarks[tip_idx].y
        dip_y = landmarks[dip_idx].y
        pip_y = landmarks[pip_idx].y
        mcp_y = landmarks[mcp_idx].y
        
        # Check if finger is extended upward
        is_extended = (tip_y < dip_y < pip_y < mcp_y) and (tip_y < wrist_y)
        
        if is_extended:
            count += 1
    
    return count

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Create a named window
window_name = "Liveness Detection Challenge"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Window state variables
is_fullscreen = False
custom_width = 800
custom_height = 600

# Set initial window size
cv2.resizeWindow(window_name, custom_width, custom_height)

# Initialize variables
challenge_fingers = random.randint(1, 5)
challenge_result_display = ""
challenge_display_timer = 0
detected_fingers = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Get the actual frame dimensions
    frame_height, frame_width = frame.shape[:2]

    overlay = frame.copy()

    # Detect faces using MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    faces = []
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)
            # Adjust for negative coordinates
            x, y = max(0, x), max(0, y)
            faces.append((x, y, w, h))
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Detect hands using MediaPipe
    results = hands.process(rgb_frame)
    if results.multi_hand_landmarks:
        # Process the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        # Draw hand landmarks for visualization
        mp_drawing.draw_landmarks(overlay, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        # Count raised fingers
        detected_fingers = count_raised_fingers(hand_landmarks)
    else:
        detected_fingers = None

    # Calculate text positions based on frame dimensions
    # Scale font size based on frame width - INCREASED BUT PROPORTIONAL TO SCREEN
    font_scale = min(frame_width / 1000, frame_height / 750) * 1.2  # More adaptive scaling
    text_thickness = max(1, int(min(frame_width, frame_height) / 500))
    line_spacing = int(35 * (frame_height / 480))
    
    # Break down instructions into smaller chunks that fit on screen
    instruction1 = f"Hold up {challenge_fingers} fingers with my hand upright."
    instruction2 = "Unused fingers should be below my knuckles."
    controls = "Press 'c': capture  'f': fullscreen  '+/-': resize  'q': quit"
    
    # Display instructions and controls with background
    # Create dark semi-transparent rectangle for better text visibility
    overlay_h = 3 * line_spacing + 10
    cv2.rectangle(overlay, (10, 10), (frame_width - 10, 10 + overlay_h), (0, 0, 0), -1)
    overlay_alpha = 0.6
    frame_part = frame[10:10 + overlay_h, 10:frame_width - 10].copy()
    cv2.rectangle(overlay, (10, 10), (frame_width - 10, 10 + overlay_h), (0, 0, 0), -1)
    overlay[10:10 + overlay_h, 10:frame_width - 10] = cv2.addWeighted(
        overlay[10:10 + overlay_h, 10:frame_width - 10], overlay_alpha, 
        frame_part, 1 - overlay_alpha, 0)
    
    # Display text on semi-transparent background
    cv2.putText(overlay, instruction1, (20, line_spacing), 
               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), text_thickness)
    cv2.putText(overlay, instruction2, (20, 2*line_spacing), 
               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), text_thickness)
    cv2.putText(overlay, controls, (20, 3*line_spacing), 
               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), text_thickness)
    
    # Display finger detection status
    status_y = 4 * line_spacing + 20
    if detected_fingers is not None:
        finger_text = f"Detected Fingers: {detected_fingers}"
        # Create background for detection status
        text_size = cv2.getTextSize(finger_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)[0]
        cv2.rectangle(overlay, (10, status_y - 30), (10 + text_size[0] + 20, status_y + 10), (0, 0, 0), -1)
        frame_part = frame[status_y - 30:status_y + 10, 10:10 + text_size[0] + 20].copy()
        overlay[status_y - 30:status_y + 10, 10:10 + text_size[0] + 20] = cv2.addWeighted(
            overlay[status_y - 30:status_y + 10, 10:10 + text_size[0] + 20], overlay_alpha, 
            frame_part, 1 - overlay_alpha, 0)
        cv2.putText(overlay, finger_text, (20, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), text_thickness)
    else:
        finger_text = "No hand detected"
        # Create background for detection status
        text_size = cv2.getTextSize(finger_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)[0]
        cv2.rectangle(overlay, (10, status_y - 30), (10 + text_size[0] + 20, status_y + 10), (0, 0, 0), -1)
        frame_part = frame[status_y - 30:status_y + 10, 10:10 + text_size[0] + 20].copy()
        overlay[status_y - 30:status_y + 10, 10:10 + text_size[0] + 20] = cv2.addWeighted(
            overlay[status_y - 30:status_y + 10, 10:10 + text_size[0] + 20], overlay_alpha, 
            frame_part, 1 - overlay_alpha, 0)
        cv2.putText(overlay, finger_text, (20, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), text_thickness)

    # Display challenge result
    if challenge_display_timer > 0:
        result_y = 5 * line_spacing + 20
        text_size = cv2.getTextSize(challenge_result_display, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)[0]
        cv2.rectangle(overlay, (10, result_y - 30), (10 + text_size[0] + 20, result_y + 10), (0, 0, 0), -1)
        frame_part = frame[result_y - 30:result_y + 10, 10:10 + text_size[0] + 20].copy()
        overlay[result_y - 30:result_y + 10, 10:10 + text_size[0] + 20] = cv2.addWeighted(
            overlay[result_y - 30:result_y + 10, 10:10 + text_size[0] + 20], overlay_alpha, 
            frame_part, 1 - overlay_alpha, 0)
        cv2.putText(overlay, challenge_result_display, (20, result_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), text_thickness)
        challenge_display_timer -= 1

    # Show the frame
    cv2.imshow(window_name, overlay)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == ord('c'):  # Capture and verify
        if len(faces) > 0 and detected_fingers is not None and detected_fingers == challenge_fingers:
            challenge_result_display = f"✅ Challenge Passed! (Detected: {detected_fingers} fingers)"
            print("Person is likely real.")
        else:
            challenge_result_display = f"❌ Challenge Failed! (Detected: {detected_fingers if detected_fingers is not None else 'no hand'})"
            print("Person may not be real.")
        challenge_display_timer = 120  # Display result for ~2 seconds at 60 FPS
        challenge_fingers = random.randint(1, 5)  # Generate new challenge
    elif key == ord('f'):  # Toggle fullscreen
        is_fullscreen = not is_fullscreen
        if is_fullscreen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, custom_width, custom_height)
    elif key == ord('+') or key == ord('='):  # Increase window size
        if not is_fullscreen:
            custom_width = min(custom_width + 100, 1920)  # Max width 1920
            custom_height = min(custom_height + 75, 1080)  # Max height 1080
            cv2.resizeWindow(window_name, custom_width, custom_height)
    elif key == ord('-') or key == ord('_'):  # Decrease window size
        if not is_fullscreen:
            custom_width = max(custom_width - 100, 400)  # Min width 400
            custom_height = max(custom_height - 75, 300)  # Min height 300
            cv2.resizeWindow(window_name, custom_width, custom_height)

# Clean up
cap.release()
cv2.destroyAllWindows()
hands.close()
face_detection.close()'''
#INTRODUCING EYE TRACKING AND BLIINK DETECTION
'''import cv2
import mediapipe as mp
import random
import math

# Initialize MediaPipe Face Mesh (includes iris tracking)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to calculate Eye Aspect Ratio (EAR) for blink detection
def calculate_ear(eye_landmarks):
    """Calculate EAR from eye landmarks."""
    # Vertical distances
    v1 = math.sqrt((eye_landmarks[1].x - eye_landmarks[5].x) ** 2 + (eye_landmarks[1].y - eye_landmarks[5].y) ** 2)
    v2 = math.sqrt((eye_landmarks[2].x - eye_landmarks[4].x) ** 2 + (eye_landmarks[2].y - eye_landmarks[4].y) ** 2)
    # Horizontal distance
    h = math.sqrt((eye_landmarks[0].x - eye_landmarks[3].x) ** 2 + (eye_landmarks[0].y - eye_landmarks[3].y) ** 2)
    ear = (v1 + v2) / (2.0 * h)
    return ear

# Function to count raised fingers
def count_raised_fingers(hand_landmarks):
    """Count only fully extended fingers pointing upward, with refined thumb logic."""
    landmarks = hand_landmarks.landmark
    fingers = [
        (4, 3, 2, 1),   # Thumb: tip (4), DIP (3), PIP (2), MCP (1)
        (8, 7, 6, 5),   # Index: tip (8), DIP (7), PIP (6), MCP (5)
        (12, 11, 10, 9), # Middle: tip (12), DIP (11), PIP (10), MCP (9)
        (16, 15, 14, 13), # Ring: tip (16), DIP (15), PIP (14), MCP (13)
        (20, 19, 18, 17)  # Pinky: tip (20), DIP (19), PIP (18), MCP (17)
    ]
    count = 0
    
    wrist_y = landmarks[0].y
    
    thumb_tip_idx, thumb_dip_idx, thumb_pip_idx, thumb_mcp_idx = fingers[0]
    thumb_tip_x, thumb_tip_y = landmarks[thumb_tip_idx].x, landmarks[thumb_tip_idx].y
    thumb_dip_y = landmarks[thumb_dip_idx].y
    thumb_pip_y = landmarks[thumb_pip_idx].y
    thumb_mcp_x, thumb_mcp_y = landmarks[thumb_mcp_idx].x, landmarks[thumb_mcp_idx].y
    
    y_diff_pip = thumb_pip_y - thumb_tip_y
    y_diff_mcp = thumb_mcp_y - thumb_tip_y
    y_diff_dip_pip = thumb_pip_y - thumb_dip_y
    thumb_extended_vertically = (thumb_tip_y < thumb_dip_y < thumb_pip_y < thumb_mcp_y and 
                                 y_diff_pip > 0.03 and 
                                 y_diff_mcp > 0.05 and 
                                 y_diff_dip_pip > 0.02)
    
    dx = thumb_tip_x - thumb_mcp_x
    dy = thumb_tip_y - thumb_mcp_y
    angle = math.degrees(math.atan2(-dy, dx))
    thumb_angle_vertical = abs(angle) < 75 or abs(angle - 180) < 75
    
    thumb_above_wrist = thumb_tip_y < wrist_y
    
    thumb_is_raised = thumb_extended_vertically and thumb_angle_vertical and thumb_above_wrist
    
    if thumb_is_raised:
        print(f"Thumb detected as raised: y_diff_pip={y_diff_pip:.3f}, y_diff_mcp={y_diff_mcp:.3f}, y_diff_dip_pip={y_diff_dip_pip:.3f}, angle={angle:.1f}°")
    else:
        print(f"Thumb not raised: y_diff_pip={y_diff_pip:.3f}, y_diff_mcp={y_diff_mcp:.3f}, y_diff_dip_pip={y_diff_dip_pip:.3f}, angle={angle:.1f}°")
    
    if thumb_is_raised:
        count += 1
    
    for tip_idx, dip_idx, pip_idx, mcp_idx in fingers[1:]:
        tip_y = landmarks[tip_idx].y
        dip_y = landmarks[dip_idx].y
        pip_y = landmarks[pip_idx].y
        mcp_y = landmarks[mcp_idx].y
        
        is_extended = (tip_y < dip_y < pip_y < mcp_y) and (tip_y < wrist_y)
        
        if is_extended:
            count += 1
    
    return count

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize variables
challenge_type = random.choice(["fingers", "blinks"])  # Randomly choose challenge
challenge_fingers = random.randint(1, 5) if challenge_type == "fingers" else None
challenge_blinks = random.randint(1, 3) if challenge_type == "blinks" else None
challenge_result_display = ""
challenge_display_timer = 0
detected_fingers = None
blink_count = 0
blink_detected = False
blink_timer = 0
EAR_THRESHOLD = 0.25  # Adjust as needed for sensitivity
BLINK_CONSEC_FRAMES = 2
blink_counter = 0
prev_ear = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    frame = cv2.resize(frame, (640, 480))
    overlay = frame.copy()

    # Process frame with MediaPipe Face Mesh (for face and iris)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_mesh.process(rgb_frame)
    faces = []
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            # Approximate face bounding box from landmarks
            ih, iw, _ = frame.shape
            x_min = min([lm.x for lm in face_landmarks.landmark]) * iw
            y_min = min([lm.y for lm in face_landmarks.landmark]) * ih
            x_max = max([lm.x for lm in face_landmarks.landmark]) * iw
            y_max = max([lm.y for lm in face_landmarks.landmark]) * ih
            x, y, w, h = int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)
            x, y = max(0, x), max(0, y)
            faces.append((x, y, w, h))
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Blink detection using iris landmarks (left eye example)
            left_eye_indices = [33, 160, 158, 133, 153, 144]  # Left eye landmarks
            left_eye_landmarks = [face_landmarks.landmark[i] for i in left_eye_indices]
            ear = calculate_ear(left_eye_landmarks)
            
            if ear < EAR_THRESHOLD:
                blink_counter += 1
            else:
                if blink_counter >= BLINK_CONSEC_FRAMES:
                    blink_count += 1
                    blink_detected = True
                    blink_timer = 0
                blink_counter = 0
            
            prev_ear = ear
            blink_timer += 1
            if blink_timer > 30:  # Reset after ~1 second if no new blinks
                blink_detected = False

    # Detect hands using MediaPipe
    hand_results = hands.process(rgb_frame)
    if hand_results.multi_hand_landmarks:
        hand_landmarks = hand_results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(overlay, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        detected_fingers = count_raised_fingers(hand_landmarks)
    else:
        detected_fingers = None

    # Display instructions based on challenge type
    if challenge_type == "fingers":
        instruction = f"Hold up {challenge_fingers} fingers with my hand upright"
    else:
        instruction = f"Blink {challenge_blinks} times"
    cv2.putText(overlay, instruction, (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(overlay, "Press 'c' to capture", (10, 60), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 2)
    
    if challenge_type == "fingers" and detected_fingers is not None:
        cv2.putText(overlay, f"Detected Fingers: {detected_fingers}", (10, 120), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 2)
    elif challenge_type == "blinks":
        cv2.putText(overlay, f"Detected Blinks: {blink_count}", (10, 120), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(overlay, "No hand detected", (10, 120), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255), 2)

    # Display challenge result
    if challenge_display_timer > 0:
        cv2.putText(overlay, challenge_result_display, (10, 90), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255), 2)
        challenge_display_timer -= 1

    cv2.imshow("Liveness Detection Challenge", overlay)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == ord('c'):  # Capture and verify
        if len(faces) > 0:
            if challenge_type == "fingers" and detected_fingers == challenge_fingers:
                challenge_result_display = f"✅ Challenge Passed! (Detected: {detected_fingers} fingers)"
                print("Person is likely real.")
            elif challenge_type == "blinks" and blink_count >= challenge_blinks:
                challenge_result_display = f"✅ Challenge Passed! (Detected: {blink_count} blinks)"
                print("Person is likely real.")
            else:
                challenge_result_display = f"❌ Challenge Failed! (Detected: {detected_fingers if challenge_type == 'fingers' else blink_count})"
                print("Person may not be real.")
        else:
            challenge_result_display = "❌ Challenge Failed! (No face detected)"
            print("Person may not be real.")
        challenge_display_timer = 120
        challenge_type = random.choice(["fingers", "blinks"])
        challenge_fingers = random.randint(1, 5) if challenge_type == "fingers" else None
        challenge_blinks = random.randint(1, 3) if challenge_type == "blinks" else None
        blink_count = 0  # Reset blink count for new challenge

# Clean up
cap.release()
cv2.destroyAllWindows()
hands.close()
face_mesh.close()'''

#adding options for closing and choosing the test according to the users 
'''
import cv2
import mediapipe as mp
import random
import math

# Initialize MediaPipe Face Mesh (includes iris tracking)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to calculate Eye Aspect Ratio (EAR) for blink detection
def calculate_ear(eye_landmarks):
    """Calculate EAR from eye landmarks."""
    v1 = math.sqrt((eye_landmarks[1].x - eye_landmarks[5].x) ** 2 + (eye_landmarks[1].y - eye_landmarks[5].y) ** 2)
    v2 = math.sqrt((eye_landmarks[2].x - eye_landmarks[4].x) ** 2 + (eye_landmarks[2].y - eye_landmarks[4].y) ** 2)
    h = math.sqrt((eye_landmarks[0].x - eye_landmarks[3].x) ** 2 + (eye_landmarks[0].y - eye_landmarks[3].y) ** 2)
    ear = (v1 + v2) / (2.0 * h)
    return ear

# Function to draw eye outline
def draw_eye_outline(frame, eye_landmarks):
    """Draw outline around the eye using landmarks."""
    points = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in eye_landmarks]
    for i in range(len(points) - 1):
        cv2.line(frame, points[i], points[i + 1], (0, 255, 0), 1)
    cv2.line(frame, points[-1], points[0], (0, 255, 0), 1)  # Close the loop

# Function to count raised fingers
def count_raised_fingers(hand_landmarks):
    """Count only fully extended fingers pointing upward, with refined thumb logic."""
    landmarks = hand_landmarks.landmark
    fingers = [
        (4, 3, 2, 1),   # Thumb: tip (4), DIP (3), PIP (2), MCP (1)
        (8, 7, 6, 5),   # Index: tip (8), DIP (7), PIP (6), MCP (5)
        (12, 11, 10, 9), # Middle: tip (12), DIP (11), PIP (10), MCP (9)
        (16, 15, 14, 13), # Ring: tip (16), DIP (15), PIP (14), MCP (13)
        (20, 19, 18, 17)  # Pinky: tip (20), DIP (19), PIP (18), MCP (17)
    ]
    count = 0
    
    wrist_y = landmarks[0].y
    
    thumb_tip_idx, thumb_dip_idx, thumb_pip_idx, thumb_mcp_idx = fingers[0]
    thumb_tip_x, thumb_tip_y = landmarks[thumb_tip_idx].x, landmarks[thumb_tip_idx].y
    thumb_dip_y = landmarks[thumb_dip_idx].y
    thumb_pip_y = landmarks[thumb_pip_idx].y
    thumb_mcp_x, thumb_mcp_y = landmarks[thumb_mcp_idx].x, landmarks[thumb_mcp_idx].y
    
    y_diff_pip = thumb_pip_y - thumb_tip_y
    y_diff_mcp = thumb_mcp_y - thumb_tip_y
    y_diff_dip_pip = thumb_pip_y - thumb_dip_y
    thumb_extended_vertically = (thumb_tip_y < thumb_dip_y < thumb_pip_y < thumb_mcp_y and 
                                 y_diff_pip > 0.03 and 
                                 y_diff_mcp > 0.05 and 
                                 y_diff_dip_pip > 0.02)
    
    dx = thumb_tip_x - thumb_mcp_x
    dy = thumb_tip_y - thumb_mcp_y
    angle = math.degrees(math.atan2(-dy, dx))
    thumb_angle_vertical = abs(angle) < 75 or abs(angle - 180) < 75
    
    thumb_above_wrist = thumb_tip_y < wrist_y
    
    thumb_is_raised = thumb_extended_vertically and thumb_angle_vertical and thumb_above_wrist
    
    if thumb_is_raised:
        print(f"Thumb detected as raised: y_diff_pip={y_diff_pip:.3f}, y_diff_mcp={y_diff_mcp:.3f}, y_diff_dip_pip={y_diff_dip_pip:.3f}, angle={angle:.1f}°")
    else:
        print(f"Thumb not raised: y_diff_pip={y_diff_pip:.3f}, y_diff_mcp={y_diff_mcp:.3f}, y_diff_dip_pip={y_diff_dip_pip:.3f}, angle={angle:.1f}°")
    
    if thumb_is_raised:
        count += 1
    
    for tip_idx, dip_idx, pip_idx, mcp_idx in fingers[1:]:
        tip_y = landmarks[tip_idx].y
        dip_y = landmarks[dip_idx].y
        pip_y = landmarks[pip_idx].y
        mcp_y = landmarks[mcp_idx].y
        
        is_extended = (tip_y < dip_y < pip_y < mcp_y) and (tip_y < wrist_y)
        
        if is_extended:
            count += 1
    
    return count

# Function to prompt user for challenge choice
def get_user_choice():
    print("\nChoose my liveness detection challenge:")
    print("1. Blink Detection Test")
    print("2. Gesture Recognition Test (Finger Counting)")
    print("3. End Test")
    while True:
        choice = input("Enter 1, 2, or 3: ")
        if choice in ["1", "2", "3"]:
            return "blinks" if choice == "1" else "fingers" if choice == "2" else "end"
        print("Invalid choice. Please enter 1, 2, or 3.")

# Main loop with camera management
while True:
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        break

    # Get initial challenge choice
    challenge_type = get_user_choice()
    if challenge_type == "end":
        print("Ending test.")
        cap.release()
        break

    challenge_fingers = random.randint(1, 5) if challenge_type == "fingers" else None
    challenge_blinks = random.randint(1, 3) if challenge_type == "blinks" else None
    challenge_result_display = ""
    challenge_display_timer = 0
    detected_fingers = None
    blink_count = 0
    blink_detected = False
    blink_timer = 0
    EAR_THRESHOLD = 0.25
    BLINK_CONSEC_FRAMES = 2
    blink_counter = 0
    prev_ear = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame = cv2.resize(frame, (640, 480))
        overlay = frame.copy()

        # Process frame with MediaPipe Face Mesh (for face and iris)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(rgb_frame)
        faces = []
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                ih, iw, _ = frame.shape
                x_min = min([lm.x for lm in face_landmarks.landmark]) * iw
                y_min = min([lm.y for lm in face_landmarks.landmark]) * ih
                x_max = max([lm.x for lm in face_landmarks.landmark]) * iw
                y_max = max([lm.y for lm in face_landmarks.landmark]) * ih
                x, y, w, h = int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)
                x, y = max(0, x), max(0, y)
                faces.append((x, y, w, h))
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Draw eye outlines
                left_eye_indices = [33, 160, 158, 133, 153, 144]
                right_eye_indices = [362, 385, 387, 263, 373, 380]
                left_eye_landmarks = [face_landmarks.landmark[i] for i in left_eye_indices]
                right_eye_landmarks = [face_landmarks.landmark[i] for i in right_eye_indices]
                draw_eye_outline(overlay, left_eye_landmarks)
                draw_eye_outline(overlay, right_eye_landmarks)

                # Blink detection
                ear = calculate_ear(left_eye_landmarks)
                if ear < EAR_THRESHOLD:
                    blink_counter += 1
                else:
                    if blink_counter >= BLINK_CONSEC_FRAMES:
                        blink_count += 1
                        blink_detected = True
                        blink_timer = 0
                    blink_counter = 0
                
                prev_ear = ear
                blink_timer += 1
                if blink_timer > 30:
                    blink_detected = False

        # Detect hands using MediaPipe
        hand_results = hands.process(rgb_frame)
        if hand_results.multi_hand_landmarks:
            hand_landmarks = hand_results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(overlay, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            detected_fingers = count_raised_fingers(hand_landmarks)
        else:
            detected_fingers = None

        # Display instructions based on challenge type
        if challenge_type == "fingers":
            instruction = f"Hold up {challenge_fingers} fingers with my hand upright"
        else:
            instruction = f"Blink {challenge_blinks} times"
        cv2.putText(overlay, instruction, (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(overlay, "Press 'c' to capture", (10, 60), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 2)
        
        if challenge_type == "fingers" and detected_fingers is not None:
            cv2.putText(overlay, f"Detected Fingers: {detected_fingers}", (10, 120), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 2)
        elif challenge_type == "blinks":
            cv2.putText(overlay, f"Detected Blinks: {blink_count}", (10, 120), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(overlay, "No hand detected", (10, 120), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255), 2)

        # Display challenge result
        if challenge_display_timer > 0:
            cv2.putText(overlay, challenge_result_display, (10, 90), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255), 2)
            challenge_display_timer -= 1
            if challenge_display_timer == 0:
                cap.release()  # Close camera when result display ends
                cv2.destroyAllWindows()  # Close window
                break  # Exit inner loop to prompt for new choice

        cv2.imshow("Liveness Detection Challenge", overlay)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit immediately
            cap.release()
            break
        elif key == ord('c'):  # Capture and verify
            if len(faces) > 0:
                if challenge_type == "fingers" and detected_fingers == challenge_fingers:
                    challenge_result_display = f"✅ Challenge Passed! (Detected: {detected_fingers} fingers)"
                    print("Person is likely real.")
                elif challenge_type == "blinks" and blink_count >= challenge_blinks:
                    challenge_result_display = f"✅ Challenge Passed! (Detected: {blink_count} blinks)"
                    print("Person is likely real.")
                else:
                    challenge_result_display = f"❌ Challenge Failed! (Detected: {detected_fingers if challenge_type == 'fingers' else blink_count})"
                    print("Person may not be real.")
            else:
                challenge_result_display = "❌ Challenge Failed! (No face detected)"
                print("Person may not be real.")
            challenge_display_timer = 360  # Display for 6 seconds at 60 FPS

    if key == ord('q'):  # Break outer loop if 'q' was pressed
        break

# Final cleanup
hands.close()
face_mesh.close()
print("Test terminated.")'''

#Fine tuning the code further(BEST)

'''import cv2
import mediapipe as mp
import random
import math

# Initialize MediaPipe Face Mesh (includes iris tracking)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to calculate Eye Aspect Ratio (EAR) for blink detection
def calculate_ear(eye_landmarks):
    """Calculate EAR from eye landmarks."""
    v1 = math.sqrt((eye_landmarks[1].x - eye_landmarks[5].x) ** 2 + (eye_landmarks[1].y - eye_landmarks[5].y) ** 2)
    v2 = math.sqrt((eye_landmarks[2].x - eye_landmarks[4].x) ** 2 + (eye_landmarks[2].y - eye_landmarks[4].y) ** 2)
    h = math.sqrt((eye_landmarks[0].x - eye_landmarks[3].x) ** 2 + (eye_landmarks[0].y - eye_landmarks[3].y) ** 2)
    ear = (v1 + v2) / (2.0 * h)
    return ear

# Function to draw eye outline
def draw_eye_outline(frame, eye_landmarks):
    """Draw outline around the eye using landmarks."""
    points = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in eye_landmarks]
    for i in range(len(points) - 1):
        cv2.line(frame, points[i], points[i + 1], (0, 255, 0), 1)
    cv2.line(frame, points[-1], points[0], (0, 255, 0), 1)  # Close the loop

# Function to count raised fingers
def count_raised_fingers(hand_landmarks):
    """Count only fully extended fingers pointing upward, with refined thumb logic."""
    landmarks = hand_landmarks.landmark
    fingers = [
        (4, 3, 2, 1),   # Thumb: tip (4), DIP (3), PIP (2), MCP (1)
        (8, 7, 6, 5),   # Index: tip (8), DIP (7), PIP (6), MCP (5)
        (12, 11, 10, 9), # Middle: tip (12), DIP (11), PIP (10), MCP (9)
        (16, 15, 14, 13), # Ring: tip (16), DIP (15), PIP (14), MCP (13)
        (20, 19, 18, 17)  # Pinky: tip (20), DIP (19), PIP (18), MCP (17)
    ]
    count = 0
    
    wrist_y = landmarks[0].y
    
    thumb_tip_idx, thumb_dip_idx, thumb_pip_idx, thumb_mcp_idx = fingers[0]
    thumb_tip_x, thumb_tip_y = landmarks[thumb_tip_idx].x, landmarks[thumb_tip_idx].y
    thumb_dip_y = landmarks[thumb_dip_idx].y
    thumb_pip_y = landmarks[thumb_pip_idx].y
    thumb_mcp_x, thumb_mcp_y = landmarks[thumb_mcp_idx].x, landmarks[thumb_mcp_idx].y
    
    y_diff_pip = thumb_pip_y - thumb_tip_y
    y_diff_mcp = thumb_mcp_y - thumb_tip_y
    y_diff_dip_pip = thumb_pip_y - thumb_dip_y
    thumb_extended_vertically = (thumb_tip_y < thumb_dip_y < thumb_pip_y < thumb_mcp_y and 
                                 y_diff_pip > 0.03 and 
                                 y_diff_mcp > 0.05 and 
                                 y_diff_dip_pip > 0.02)
    
    dx = thumb_tip_x - thumb_mcp_x
    dy = thumb_tip_y - thumb_mcp_y
    angle = math.degrees(math.atan2(-dy, dx))
    thumb_angle_vertical = abs(angle) < 75 or abs(angle - 180) < 75
    
    thumb_above_wrist = thumb_tip_y < wrist_y
    
    thumb_is_raised = thumb_extended_vertically and thumb_angle_vertical and thumb_above_wrist
    
    if thumb_is_raised:
        print(f"Thumb detected as raised: y_diff_pip={y_diff_pip:.3f}, y_diff_mcp={y_diff_mcp:.3f}, y_diff_dip_pip={y_diff_dip_pip:.3f}, angle={angle:.1f}°")
    else:
        print(f"Thumb not raised: y_diff_pip={y_diff_pip:.3f}, y_diff_mcp={y_diff_mcp:.3f}, y_diff_dip_pip={y_diff_dip_pip:.3f}, angle={angle:.1f}°")
    
    if thumb_is_raised:
        count += 1
    
    for tip_idx, dip_idx, pip_idx, mcp_idx in fingers[1:]:
        tip_y = landmarks[tip_idx].y
        dip_y = landmarks[dip_idx].y
        pip_y = landmarks[pip_idx].y
        mcp_y = landmarks[mcp_idx].y
        
        is_extended = (tip_y < dip_y < pip_y < mcp_y) and (tip_y < wrist_y)
        
        if is_extended:
            count += 1
    
    return count

# Function to prompt user for challenge choice
def get_user_choice():
    print("\nChoose my liveness detection challenge:")
    print("1. Blink Detection Test")
    print("2. Gesture Recognition Test (Finger Counting)")
    print("3. End Test")
    while True:
        choice = input("Enter 1, 2, or 3: ")
        if choice in ["1", "2", "3"]:
            return "blinks" if choice == "1" else "fingers" if choice == "2" else "end"
        print("Invalid choice. Please enter 1, 2, or 3.")

# Main loop with camera management
while True:
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        break

    # Get initial challenge choice
    challenge_type = get_user_choice()
    if challenge_type == "end":
        print("Ending test.")
        cap.release()
        break

    challenge_fingers = random.randint(1, 5) if challenge_type == "fingers" else None
    challenge_blinks = random.randint(1, 3) if challenge_type == "blinks" else None
    challenge_result_display = ""
    challenge_display_timer = 0
    detected_fingers = None
    blink_count = 0
    blink_detected = False
    blink_timer = 0
    EAR_THRESHOLD = 0.25
    BLINK_CONSEC_FRAMES = 2
    blink_counter = 0
    prev_ear = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame = cv2.resize(frame, (640, 480))
        overlay = frame.copy()

        # Process frame with MediaPipe Face Mesh (for face and iris)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(rgb_frame)
        faces = []
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                ih, iw, _ = frame.shape
                x_min = min([lm.x for lm in face_landmarks.landmark]) * iw
                y_min = min([lm.y for lm in face_landmarks.landmark]) * ih
                x_max = max([lm.x for lm in face_landmarks.landmark]) * iw
                y_max = max([lm.y for lm in face_landmarks.landmark]) * ih
                x, y, w, h = int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)
                x, y = max(0, x), max(0, y)
                faces.append((x, y, w, h))
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Draw eye outlines
                left_eye_indices = [33, 160, 158, 133, 153, 144]
                right_eye_indices = [362, 385, 387, 263, 373, 380]
                left_eye_landmarks = [face_landmarks.landmark[i] for i in left_eye_indices]
                right_eye_landmarks = [face_landmarks.landmark[i] for i in right_eye_indices]
                draw_eye_outline(overlay, left_eye_landmarks)
                draw_eye_outline(overlay, right_eye_landmarks)

                # Blink detection
                ear = calculate_ear(left_eye_landmarks)
                if ear < EAR_THRESHOLD:
                    blink_counter += 1
                else:
                    if blink_counter >= BLINK_CONSEC_FRAMES:
                        blink_count += 1
                        blink_detected = True
                        blink_timer = 0
                    blink_counter = 0
                
                prev_ear = ear
                blink_timer += 1
                if blink_timer > 30:
                    blink_detected = False

        # Detect hands using MediaPipe
        hand_results = hands.process(rgb_frame)
        if hand_results.multi_hand_landmarks:
            hand_landmarks = hand_results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(overlay, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            detected_fingers = count_raised_fingers(hand_landmarks)
        else:
            detected_fingers = None

        # Display instructions based on challenge type
        if challenge_type == "fingers":
            instruction = f"Hold up {challenge_fingers} fingers with my hand upright"
        else:
            instruction = f"Blink {challenge_blinks} times"
        cv2.putText(overlay, instruction, (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(overlay, "Press 'c' to capture", (10, 60), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 2)
        
        if challenge_type == "fingers" and detected_fingers is not None:
            cv2.putText(overlay, f"Detected Fingers: {detected_fingers}", (10, 120), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 2)
        elif challenge_type == "blinks":
            cv2.putText(overlay, f"Detected Blinks: {blink_count}", (10, 120), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(overlay, "No hand detected", (10, 120), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255), 2)

        # Display challenge result
        if challenge_display_timer > 0:
            cv2.putText(overlay, challenge_result_display, (10, 90), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255), 2)
            challenge_display_timer -= 1
            if challenge_display_timer == 0:
                cap.release()  # Close camera
                cv2.destroyAllWindows()  # Close window
                break  # Exit inner loop to prompt for new choice

        cv2.imshow("Liveness Detection Challenge", overlay)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit immediately
            cap.release()
            break
        elif key == ord('c'):  # Capture and verify
            if len(faces) > 0:
                if challenge_type == "fingers" and detected_fingers == challenge_fingers:
                    challenge_result_display = f"✅ Challenge Passed! (Detected: {detected_fingers} fingers)"
                    print("Person is likely real.")
                elif challenge_type == "blinks" and blink_count >= challenge_blinks:
                    challenge_result_display = f"✅ Challenge Passed! (Detected: {blink_count} blinks)"
                    print("Person is likely real.")
                else:
                    challenge_result_display = f"❌ Challenge Failed! (Detected: {detected_fingers if challenge_type == 'fingers' else blink_count})"
                    print("Person may not be real.")
            else:
                challenge_result_display = "❌ Challenge Failed! (No face detected)"
                print("Person may not be real.")
            challenge_display_timer = 120  # Display for 2 seconds at 60 FPS

    if key == ord('q'):  # Break outer loop if 'q' was pressed
        break

# Final cleanup
hands.close()
face_mesh.close()
print("Test terminated.")'''

# Adding Head Pose Estimation:

'''import cv2
import mediapipe as mp
import random
import math
import numpy as np  # New prerequisite

# Initialize MediaPipe Face Mesh (includes iris tracking)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to calculate Eye Aspect Ratio (EAR) for blink detection
def calculate_ear(eye_landmarks):
    v1 = math.sqrt((eye_landmarks[1].x - eye_landmarks[5].x) ** 2 + (eye_landmarks[1].y - eye_landmarks[5].y) ** 2)
    v2 = math.sqrt((eye_landmarks[2].x - eye_landmarks[4].x) ** 2 + (eye_landmarks[2].y - eye_landmarks[4].y) ** 2)
    h = math.sqrt((eye_landmarks[0].x - eye_landmarks[3].x) ** 2 + (eye_landmarks[0].y - eye_landmarks[3].y) ** 2)
    ear = (v1 + v2) / (2.0 * h)
    return ear

# Function to draw eye outline
def draw_eye_outline(frame, eye_landmarks):
    points = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in eye_landmarks]
    for i in range(len(points) - 1):
        cv2.line(frame, points[i], points[i + 1], (0, 255, 0), 1)
    cv2.line(frame, points[-1], points[0], (0, 255, 0), 1)  # Close the loop

# Function to count raised fingers
def count_raised_fingers(hand_landmarks):
    landmarks = hand_landmarks.landmark
    fingers = [
        (4, 3, 2, 1),   # Thumb: tip, DIP, PIP, MCP
        (8, 7, 6, 5),   # Index
        (12, 11, 10, 9), # Middle
        (16, 15, 14, 13), # Ring
        (20, 19, 18, 17)  # Pinky
    ]
    count = 0
    wrist_y = landmarks[0].y
    
    # Thumb logic (more refined)
    thumb_tip_idx, thumb_dip_idx, thumb_pip_idx, thumb_mcp_idx = fingers[0]
    thumb_tip_x, thumb_tip_y = landmarks[thumb_tip_idx].x, landmarks[thumb_tip_idx].y
    thumb_dip_y = landmarks[thumb_dip_idx].y
    thumb_pip_y = landmarks[thumb_pip_idx].y
    thumb_mcp_x, thumb_mcp_y = landmarks[thumb_mcp_idx].x, landmarks[thumb_mcp_idx].y
    
    y_diff_pip = thumb_pip_y - thumb_tip_y
    y_diff_mcp = thumb_mcp_y - thumb_tip_y
    y_diff_dip_pip = thumb_pip_y - thumb_dip_y
    thumb_extended_vertically = (thumb_tip_y < thumb_dip_y < thumb_pip_y < thumb_mcp_y and 
                                 y_diff_pip > 0.03 and 
                                 y_diff_mcp > 0.05 and 
                                 y_diff_dip_pip > 0.02)
    
    dx = thumb_tip_x - thumb_mcp_x
    dy = thumb_tip_y - thumb_mcp_y
    angle = math.degrees(math.atan2(-dy, dx))
    thumb_angle_vertical = abs(angle) < 75 or abs(angle - 180) < 75
    thumb_above_wrist = thumb_tip_y < wrist_y
    thumb_is_raised = thumb_extended_vertically and thumb_angle_vertical and thumb_above_wrist
    
    if thumb_is_raised:
        print(f"Thumb detected as raised: y_diff_pip={y_diff_pip:.3f}, y_diff_mcp={y_diff_mcp:.3f}, y_diff_dip_pip={y_diff_dip_pip:.3f}, angle={angle:.1f}°")
    else:
        print(f"Thumb not raised: y_diff_pip={y_diff_pip:.3f}, y_diff_mcp={y_diff_mcp:.3f}, y_diff_dip_pip={y_diff_dip_pip:.3f}, angle={angle:.1f}°")
    
    if thumb_is_raised:
        count += 1
    
    for tip_idx, dip_idx, pip_idx, mcp_idx in fingers[1:]:
        tip_y = landmarks[tip_idx].y
        dip_y = landmarks[dip_idx].y
        pip_y = landmarks[pip_idx].y
        mcp_y = landmarks[mcp_idx].y
        
        is_extended = (tip_y < dip_y < pip_y < mcp_y) and (tip_y < wrist_y)
        if is_extended:
            count += 1
    return count

# New function: Compute head pose (Euler angles) using face landmarks
def get_head_pose(frame, face_landmarks):
    h, w, _ = frame.shape
    # Select 2D image points from specific landmarks:
    image_points = np.array([
        (face_landmarks.landmark[1].x * w, face_landmarks.landmark[1].y * h),     # Nose tip
        (face_landmarks.landmark[152].x * w, face_landmarks.landmark[152].y * h),   # Chin
        (face_landmarks.landmark[33].x * w, face_landmarks.landmark[33].y * h),     # Left eye corner
        (face_landmarks.landmark[263].x * w, face_landmarks.landmark[263].y * h),   # Right eye corner
        (face_landmarks.landmark[61].x * w, face_landmarks.landmark[61].y * h),     # Left mouth corner
        (face_landmarks.landmark[291].x * w, face_landmarks.landmark[291].y * h)    # Right mouth corner
    ], dtype="double")
    
    # 3D model points of a generic face model (approximate values in mm)
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -63.6, -12.5),         # Chin
        (-43.3, 32.7, -26.0),        # Left eye corner
        (43.3, 32.7, -26.0),         # Right eye corner
        (-28.9, -28.9, -24.1),       # Left Mouth corner
        (28.9, -28.9, -24.1)         # Right mouth corner
    ])
    
    # Camera internals
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    dist_coeffs = np.zeros((4,1))  # Assuming no lens distortion

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    
    if not success:
        return None, None
    
    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    
    # Use projection matrix decomposition to get Euler angles
    proj_matrix = np.hstack((rotation_matrix, translation_vector))
    _, _, _, _, _, _, eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)
    # eulerAngles are in degrees: [pitch, yaw, roll]
    return eulerAngles, translation_vector

# Helper function to check head movement challenge
def check_head_movement(eulerAngles, direction="left", threshold=15):
    # eulerAngles is a 3x1 matrix: [pitch, yaw, roll]
    # For a left turn, yaw should be negative beyond the threshold; for right, positive.
    yaw = float(eulerAngles[1])
    if direction == "left":
        return yaw < -threshold
    elif direction == "right":
        return yaw > threshold
    return False

# Modified function to include head movement test challenge
def get_user_choice():
    print("\nChoose my liveness detection challenge:")
    print("1. Blink Detection Test")
    print("2. Gesture Recognition Test (Finger Counting)")
    print("3. Head Movement Test")
    print("4. End Test")
    while True:
        choice = input("Enter 1, 2, 3, or 4: ")
        if choice in ["1", "2", "3", "4"]:
            if choice == "1":
                return "blinks"
            elif choice == "2":
                return "fingers"
            elif choice == "3":
                return "head"
            else:
                return "end"
        print("Invalid choice. Please enter 1, 2, 3, or 4.")

# Main loop with camera management
while True:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        break

    challenge_type = get_user_choice()
    if challenge_type == "end":
        print("Ending test.")
        cap.release()
        break

    # For head movement, choose a random direction
    if challenge_type == "head":
        challenge_direction = random.choice(["left", "right"])
    else:
        challenge_direction = None

    challenge_fingers = random.randint(1, 5) if challenge_type == "fingers" else None
    challenge_blinks = random.randint(1, 3) if challenge_type == "blinks" else None
    challenge_result_display = ""
    challenge_display_timer = 0
    detected_fingers = None
    blink_count = 0
    blink_detected = False
    blink_timer = 0
    EAR_THRESHOLD = 0.25
    BLINK_CONSEC_FRAMES = 2
    blink_counter = 0
    prev_ear = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame = cv2.resize(frame, (640, 480))
        overlay = frame.copy()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame for face detection
        face_results = face_mesh.process(rgb_frame)
        faces = []
        head_pose_angles = None
        face_landmark_used = None
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                ih, iw, _ = frame.shape
                x_min = min([lm.x for lm in face_landmarks.landmark]) * iw
                y_min = min([lm.y for lm in face_landmarks.landmark]) * ih
                x_max = max([lm.x for lm in face_landmarks.landmark]) * iw
                y_max = max([lm.y for lm in face_landmarks.landmark]) * ih
                x, y, w, h = int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)
                x, y = max(0, x), max(0, y)
                faces.append((x, y, w, h))
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw eye outlines
                left_eye_indices = [33, 160, 158, 133, 153, 144]
                right_eye_indices = [362, 385, 387, 263, 373, 380]
                left_eye_landmarks = [face_landmarks.landmark[i] for i in left_eye_indices]
                right_eye_landmarks = [face_landmarks.landmark[i] for i in right_eye_indices]
                draw_eye_outline(overlay, left_eye_landmarks)
                draw_eye_outline(overlay, right_eye_landmarks)
                
                # Blink detection
                ear = calculate_ear(left_eye_landmarks)
                if ear < EAR_THRESHOLD:
                    blink_counter += 1
                else:
                    if blink_counter >= BLINK_CONSEC_FRAMES:
                        blink_count += 1
                        blink_detected = True
                        blink_timer = 0
                    blink_counter = 0
                prev_ear = ear
                blink_timer += 1
                if blink_timer > 30:
                    blink_detected = False

                # For head movement challenge, compute head pose from the first detected face
                if challenge_type == "head" and head_pose_angles is None:
                    head_pose_angles, _ = get_head_pose(frame, face_landmarks)
                    face_landmark_used = face_landmarks

        # Process hand landmarks for finger counting challenge
        hand_results = hands.process(rgb_frame)
        if hand_results.multi_hand_landmarks:
            hand_landmarks = hand_results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(overlay, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            detected_fingers = count_raised_fingers(hand_landmarks)
        else:
            detected_fingers = None

        # Display instructions based on challenge type
        if challenge_type == "fingers":
            instruction = f"Hold up {challenge_fingers} fingers with my hand upright"
        elif challenge_type == "blinks":
            instruction = f"Blink {challenge_blinks} times"
        elif challenge_type == "head":
            instruction = f"Turn my head to the {challenge_direction}"
        else:
            instruction = ""
        cv2.putText(overlay, instruction, (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(overlay, "Press 'c' to capture", (10, 60), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 2)
        
        if challenge_type == "fingers" and detected_fingers is not None:
            cv2.putText(overlay, f"Detected Fingers: {detected_fingers}", (10, 120), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 2)
        elif challenge_type == "blinks":
            cv2.putText(overlay, f"Detected Blinks: {blink_count}", (10, 120), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 2)
        elif challenge_type == "head" and head_pose_angles is not None:
            # Optionally display the yaw angle for debugging
            yaw = float(head_pose_angles[1])
            cv2.putText(overlay, f"Yaw: {yaw:.1f}°", (10, 120), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 0, 0), 2)
        else:
            cv2.putText(overlay, "No hand/face detected", (10, 120), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255), 2)

        # Display challenge result
        if challenge_display_timer > 0:
            cv2.putText(overlay, challenge_result_display, (10, 90), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255), 2)
            challenge_display_timer -= 1
            if challenge_display_timer == 0:
                cap.release()
                cv2.destroyAllWindows()
                break

        cv2.imshow("Liveness Detection Challenge", overlay)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cap.release()
            break
        elif key == ord('c'):  # Capture and verify challenge response
            if challenge_type == "fingers":
                if len(faces) > 0 and detected_fingers == challenge_fingers:
                    challenge_result_display = f"✅ Challenge Passed! (Detected: {detected_fingers} fingers)"
                    print("Person is likely real.")
                else:
                    challenge_result_display = f"❌ Challenge Failed! (Detected: {detected_fingers})"
                    print("Person may not be real.")
            elif challenge_type == "blinks":
                if len(faces) > 0 and blink_count >= challenge_blinks:
                    challenge_result_display = f"✅ Challenge Passed! (Detected: {blink_count} blinks)"
                    print("Person is likely real.")
                else:
                    challenge_result_display = f"❌ Challenge Failed! (Detected: {blink_count} blinks)"
                    print("Person may not be real.")
            elif challenge_type == "head":
                if len(faces) > 0 and head_pose_angles is not None:
                    if check_head_movement(head_pose_angles, challenge_direction):
                        challenge_result_display = f"✅ Challenge Passed! (Head turned {challenge_direction})"
                        print("Person is likely real.")
                    else:
                        challenge_result_display = f"❌ Challenge Failed! (Head not turned {challenge_direction})"
                        print("Person may not be real.")
                else:
                    challenge_result_display = "❌ Challenge Failed! (No face detected)"
                    print("Person may not be real.")
            else:
                challenge_result_display = "❌ Challenge Failed!"
                print("Person may not be real.")
            challenge_display_timer = 120  # Display result for a short duration

    if key == ord('q'):
        break

# Cleanup
hands.close()
face_mesh.close()
cv2.destroyAllWindows()
print("Test terminated.")'''

#Fine tuning code with headpose detection test (Current Best)
'''
import cv2
import mediapipe as mp
import random
import math
import numpy as np

# Initialize MediaPipe Face Mesh (includes iris tracking)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to calculate Eye Aspect Ratio (EAR) for blink detection
def calculate_ear(eye_landmarks):
    v1 = math.sqrt((eye_landmarks[1].x - eye_landmarks[5].x) ** 2 + (eye_landmarks[1].y - eye_landmarks[5].y) ** 2)
    v2 = math.sqrt((eye_landmarks[2].x - eye_landmarks[4].x) ** 2 + (eye_landmarks[2].y - eye_landmarks[4].y) ** 2)
    h = math.sqrt((eye_landmarks[0].x - eye_landmarks[3].x) ** 2 + (eye_landmarks[0].y - eye_landmarks[3].y) ** 2)
    ear = (v1 + v2) / (2.0 * h)
    return ear

# Function to draw eye outline
def draw_eye_outline(frame, eye_landmarks):
    points = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in eye_landmarks]
    for i in range(len(points) - 1):
        cv2.line(frame, points[i], points[i + 1], (0, 255, 0), 1)
    cv2.line(frame, points[-1], points[0], (0, 255, 0), 1)

# Function to count raised fingers
def count_raised_fingers(hand_landmarks):
    landmarks = hand_landmarks.landmark
    fingers = [
        (4, 3, 2, 1),   # Thumb: tip, DIP, PIP, MCP
        (8, 7, 6, 5),   # Index
        (12, 11, 10, 9), # Middle
        (16, 15, 14, 13), # Ring
        (20, 19, 18, 17)  # Pinky
    ]
    count = 0
    wrist_y = landmarks[0].y
    
    thumb_tip_idx, thumb_dip_idx, thumb_pip_idx, thumb_mcp_idx = fingers[0]
    thumb_tip_x, thumb_tip_y = landmarks[thumb_tip_idx].x, landmarks[thumb_tip_idx].y
    thumb_dip_y = landmarks[thumb_dip_idx].y
    thumb_pip_y = landmarks[thumb_pip_idx].y
    thumb_mcp_x, thumb_mcp_y = landmarks[thumb_mcp_idx].x, landmarks[thumb_mcp_idx].y
    
    y_diff_pip = thumb_pip_y - thumb_tip_y
    y_diff_mcp = thumb_mcp_y - thumb_tip_y
    y_diff_dip_pip = thumb_pip_y - thumb_dip_y
    thumb_extended_vertically = (thumb_tip_y < thumb_dip_y < thumb_pip_y < thumb_mcp_y and 
                                 y_diff_pip > 0.03 and 
                                 y_diff_mcp > 0.05 and 
                                 y_diff_dip_pip > 0.02)
    
    dx = thumb_tip_x - thumb_mcp_x
    dy = thumb_tip_y - thumb_mcp_y
    angle = math.degrees(math.atan2(-dy, dx))
    thumb_angle_vertical = abs(angle) < 75 or abs(angle - 180) < 75
    thumb_above_wrist = thumb_tip_y < wrist_y
    thumb_is_raised = thumb_extended_vertically and thumb_angle_vertical and thumb_above_wrist
    
    if thumb_is_raised:
        print(f"Thumb detected as raised: y_diff_pip={y_diff_pip:.3f}, y_diff_mcp={y_diff_mcp:.3f}, y_diff_dip_pip={y_diff_dip_pip:.3f}, angle={angle:.1f}°")
    else:
        print(f"Thumb not raised: y_diff_pip={y_diff_pip:.3f}, y_diff_mcp={y_diff_mcp:.3f}, y_diff_dip_pip={y_diff_dip_pip:.3f}, angle={angle:.1f}°")
    
    if thumb_is_raised:
        count += 1
    
    for tip_idx, dip_idx, pip_idx, mcp_idx in fingers[1:]:
        tip_y = landmarks[tip_idx].y
        dip_y = landmarks[dip_idx].y
        pip_y = landmarks[pip_idx].y
        mcp_y = landmarks[mcp_idx].y
        
        is_extended = (tip_y < dip_y < pip_y < mcp_y) and (tip_y < wrist_y)
        if is_extended:
            count += 1
    return count

# Function to compute head pose (Euler angles) using face landmarks
def get_head_pose(frame, face_landmarks):
    h, w, _ = frame.shape
    image_points = np.array([
        (face_landmarks.landmark[1].x * w, face_landmarks.landmark[1].y * h),     # Nose tip
        (face_landmarks.landmark[152].x * w, face_landmarks.landmark[152].y * h), # Chin
        (face_landmarks.landmark[33].x * w, face_landmarks.landmark[33].y * h),   # Left eye corner
        (face_landmarks.landmark[263].x * w, face_landmarks.landmark[263].y * h), # Right eye corner
        (face_landmarks.landmark[61].x * w, face_landmarks.landmark[61].y * h),   # Left mouth corner
        (face_landmarks.landmark[291].x * w, face_landmarks.landmark[291].y * h)  # Right mouth corner
    ], dtype="double")
    
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -63.6, -12.5),         # Chin
        (-43.3, 32.7, -26.0),        # Left eye corner
        (43.3, 32.7, -26.0),         # Right eye corner
        (-28.9, -28.9, -24.1),       # Left Mouth corner
        (28.9, -28.9, -24.1)         # Right mouth corner
    ])
    
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    dist_coeffs = np.zeros((4,1))
    
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    
    if not success:
        return None, None
    
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    proj_matrix = np.hstack((rotation_matrix, translation_vector))
    _, _, _, _, _, _, eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)
    return eulerAngles, translation_vector

# Helper function to check head movement
def check_head_movement(eulerAngles, direction="left", threshold=15):
    yaw = float(eulerAngles[1])
    if direction == "left":
        return yaw < -threshold
    elif direction == "right":
        return yaw > threshold
    return False

# Modified function to include head movement with blink and gesture challenges
def get_user_choice():
    print("\nChoose my liveness detection challenge:")
    print("1. Blink Detection with Head Movement")
    print("2. Gesture Recognition with Head Movement")
    print("3. End Test")
    while True:
        choice = input("Enter 1, 2, or 3: ")
        if choice in ["1", "2", "3"]:
            return "blinks" if choice == "1" else "fingers" if choice == "2" else "end"
        print("Invalid choice. Please enter 1, 2, or 3.")

# Main loop with camera management
while True:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        break

    challenge_type = get_user_choice()
    if challenge_type == "end":
        print("Ending test.")
        cap.release()
        break

    # Random head movement direction for both challenges
    challenge_direction = random.choice(["left", "right"])
    challenge_fingers = random.randint(1, 5) if challenge_type == "fingers" else None
    challenge_blinks = random.randint(1, 3) if challenge_type == "blinks" else None
    challenge_result_display = ""
    challenge_display_timer = 0
    detected_fingers = None
    blink_count = 0
    blink_detected = False
    blink_timer = 0
    EAR_THRESHOLD = 0.25
    BLINK_CONSEC_FRAMES = 2
    blink_counter = 0
    prev_ear = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame = cv2.resize(frame, (640, 480))
        overlay = frame.copy()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame for face detection
        face_results = face_mesh.process(rgb_frame)
        faces = []
        head_pose_angles = None
        face_landmark_used = None
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                ih, iw, _ = frame.shape
                x_min = min([lm.x for lm in face_landmarks.landmark]) * iw
                y_min = min([lm.y for lm in face_landmarks.landmark]) * ih
                x_max = max([lm.x for lm in face_landmarks.landmark]) * iw
                y_max = max([lm.y for lm in face_landmarks.landmark]) * ih
                x, y, w, h = int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)
                x, y = max(0, x), max(0, y)
                faces.append((x, y, w, h))
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw eye outlines
                left_eye_indices = [33, 160, 158, 133, 153, 144]
                right_eye_indices = [362, 385, 387, 263, 373, 380]
                left_eye_landmarks = [face_landmarks.landmark[i] for i in left_eye_indices]
                right_eye_landmarks = [face_landmarks.landmark[i] for i in right_eye_indices]
                draw_eye_outline(overlay, left_eye_landmarks)
                draw_eye_outline(overlay, right_eye_landmarks)
                
                # Blink detection
                ear = calculate_ear(left_eye_landmarks)
                if ear < EAR_THRESHOLD:
                    blink_counter += 1
                else:
                    if blink_counter >= BLINK_CONSEC_FRAMES:
                        blink_count += 1
                        blink_detected = True
                        blink_timer = 0
                    blink_counter = 0
                prev_ear = ear
                blink_timer += 1
                if blink_timer > 30:
                    blink_detected = False

                # Compute head pose for both challenges
                if head_pose_angles is None:
                    head_pose_angles, _ = get_head_pose(frame, face_landmarks)
                    face_landmark_used = face_landmarks

        # Process hand landmarks for finger counting challenge
        hand_results = hands.process(rgb_frame)
        if hand_results.multi_hand_landmarks:
            hand_landmarks = hand_results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(overlay, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            detected_fingers = count_raised_fingers(hand_landmarks)
        else:
            detected_fingers = None

        # Display instructions based on challenge type with head movement
        if challenge_type == "fingers":
            instruction = f"Hold up {challenge_fingers} fingers and turn head {challenge_direction}"
        elif challenge_type == "blinks":
            instruction = f"Blink {challenge_blinks} times and turn head {challenge_direction}"
        else:
            instruction = ""
        cv2.putText(overlay, instruction, (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(overlay, "Press 'c' to capture", (10, 60), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 2)
        
        if challenge_type == "fingers" and detected_fingers is not None:
            cv2.putText(overlay, f"Detected Fingers: {detected_fingers}", (10, 120), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 2)
        elif challenge_type == "blinks":
            cv2.putText(overlay, f"Detected Blinks: {blink_count}", (10, 120), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 2)
        if head_pose_angles is not None:
            yaw = float(head_pose_angles[1])
            cv2.putText(overlay, f"Yaw: {yaw:.1f}°", (10, 150), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 0, 0), 2)
        else:
            cv2.putText(overlay, "No head pose detected", (10, 150), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255), 2)

        # Display challenge result
        if challenge_display_timer > 0:
            cv2.putText(overlay, challenge_result_display, (10, 90), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255), 2)
            challenge_display_timer -= 1
            if challenge_display_timer == 0:
                cap.release()
                cv2.destroyAllWindows()
                break

        cv2.imshow("Liveness Detection Challenge", overlay)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cap.release()
            break
        elif key == ord('c'):
            if len(faces) > 0 and head_pose_angles is not None:
                head_moved_correctly = check_head_movement(head_pose_angles, challenge_direction)
                if challenge_type == "fingers":
                    if detected_fingers == challenge_fingers and head_moved_correctly:
                        challenge_result_display = f"✅ Challenge Passed! (Detected: {detected_fingers} fingers, head {challenge_direction})"
                        print("Person is likely real.")
                    else:
                        challenge_result_display = f"❌ Challenge Failed! (Fingers: {detected_fingers}, Head moved: {head_moved_correctly})"
                        print("Person may not be real.")
                elif challenge_type == "blinks":
                    if blink_count >= challenge_blinks and head_moved_correctly:
                        challenge_result_display = f"✅ Challenge Passed! (Detected: {blink_count} blinks, head {challenge_direction})"
                        print("Person is likely real.")
                    else:
                        challenge_result_display = f"❌ Challenge Failed! (Blinks: {blink_count}, Head moved: {head_moved_correctly})"
                        print("Person may not be real.")
            else:
                challenge_result_display = "❌ Challenge Failed! (No face or head pose detected)"
                print("Person may not be real.")
            challenge_display_timer = 120

    if key == ord('q'):
        break

# Cleanup
hands.close()
face_mesh.close()
cv2.destroyAllWindows()
print("Test terminated.")'''

#further fine tuning to UI with headmovment challenge integrated (Current BEST 1)
'''
import cv2
import mediapipe as mp
import random
import math
import numpy as np

# Initialize MediaPipe Face Mesh (includes iris tracking)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Global variable to store the last challenge result
last_result = "No previous challenge completed yet."

# Function to calculate Eye Aspect Ratio (EAR) for blink detection
def calculate_ear(eye_landmarks):
    v1 = math.sqrt((eye_landmarks[1].x - eye_landmarks[5].x) ** 2 + (eye_landmarks[1].y - eye_landmarks[5].y) ** 2)
    v2 = math.sqrt((eye_landmarks[2].x - eye_landmarks[4].x) ** 2 + (eye_landmarks[2].y - eye_landmarks[4].y) ** 2)
    h = math.sqrt((eye_landmarks[0].x - eye_landmarks[3].x) ** 2 + (eye_landmarks[0].y - eye_landmarks[3].y) ** 2)
    ear = (v1 + v2) / (2.0 * h)
    return ear

# Function to draw eye outline
def draw_eye_outline(frame, eye_landmarks):
    points = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in eye_landmarks]
    for i in range(len(points) - 1):
        cv2.line(frame, points[i], points[i + 1], (0, 255, 0), 1)
    cv2.line(frame, points[-1], points[0], (0, 255, 0), 1)

# Function to count raised fingers
def count_raised_fingers(hand_landmarks):
    landmarks = hand_landmarks.landmark
    fingers = [
        (4, 3, 2, 1),   # Thumb: tip, DIP, PIP, MCP
        (8, 7, 6, 5),   # Index
        (12, 11, 10, 9), # Middle
        (16, 15, 14, 13), # Ring
        (20, 19, 18, 17)  # Pinky
    ]
    count = 0
    wrist_y = landmarks[0].y
    
    thumb_tip_idx, thumb_dip_idx, thumb_pip_idx, thumb_mcp_idx = fingers[0]
    thumb_tip_x, thumb_tip_y = landmarks[thumb_tip_idx].x, landmarks[thumb_tip_idx].y
    thumb_dip_y = landmarks[thumb_dip_idx].y
    thumb_pip_y = landmarks[thumb_pip_idx].y
    thumb_mcp_x, thumb_mcp_y = landmarks[thumb_mcp_idx].x, landmarks[thumb_mcp_idx].y
    
    y_diff_pip = thumb_pip_y - thumb_tip_y
    y_diff_mcp = thumb_mcp_y - thumb_tip_y
    y_diff_dip_pip = thumb_pip_y - thumb_dip_y
    thumb_extended_vertically = (thumb_tip_y < thumb_dip_y < thumb_pip_y < thumb_mcp_y and 
                                 y_diff_pip > 0.03 and 
                                 y_diff_mcp > 0.05 and 
                                 y_diff_dip_pip > 0.02)
    
    dx = thumb_tip_x - thumb_mcp_x
    dy = thumb_tip_y - thumb_mcp_y
    angle = math.degrees(math.atan2(-dy, dx))
    thumb_angle_vertical = abs(angle) < 75 or abs(angle - 180) < 75
    thumb_above_wrist = thumb_tip_y < wrist_y
    thumb_is_raised = thumb_extended_vertically and thumb_angle_vertical and thumb_above_wrist
    
    if thumb_is_raised:
        print(f"Thumb detected as raised: y_diff_pip={y_diff_pip:.3f}, y_diff_mcp={y_diff_mcp:.3f}, y_diff_dip_pip={y_diff_dip_pip:.3f}, angle={angle:.1f}°")
    else:
        print(f"Thumb not raised: y_diff_pip={y_diff_pip:.3f}, y_diff_mcp={y_diff_mcp:.3f}, y_diff_dip_pip={y_diff_dip_pip:.3f}, angle={angle:.1f}°")
    
    if thumb_is_raised:
        count += 1
    
    for tip_idx, dip_idx, pip_idx, mcp_idx in fingers[1:]:
        tip_y = landmarks[tip_idx].y
        dip_y = landmarks[dip_idx].y
        pip_y = landmarks[pip_idx].y
        mcp_y = landmarks[mcp_idx].y
        
        is_extended = (tip_y < dip_y < pip_y < mcp_y) and (tip_y < wrist_y)
        if is_extended:
            count += 1
    return count

# Function to compute head pose (Euler angles) using face landmarks
def get_head_pose(frame, face_landmarks):
    h, w, _ = frame.shape
    image_points = np.array([
        (face_landmarks.landmark[1].x * w, face_landmarks.landmark[1].y * h),     # Nose tip
        (face_landmarks.landmark[152].x * w, face_landmarks.landmark[152].y * h), # Chin
        (face_landmarks.landmark[33].x * w, face_landmarks.landmark[33].y * h),   # Left eye corner
        (face_landmarks.landmark[263].x * w, face_landmarks.landmark[263].y * h), # Right eye corner
        (face_landmarks.landmark[61].x * w, face_landmarks.landmark[61].y * h),   # Left mouth corner
        (face_landmarks.landmark[291].x * w, face_landmarks.landmark[291].y * h)  # Right mouth corner
    ], dtype="double")
    
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -63.6, -12.5),         # Chin
        (-43.3, 32.7, -26.0),        # Left eye corner
        (43.3, 32.7, -26.0),         # Right eye corner
        (-28.9, -28.9, -24.1),       # Left Mouth corner
        (28.9, -28.9, -24.1)         # Right mouth corner
    ])
    
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    dist_coeffs = np.zeros((4,1))
    
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    
    if not success:
        return None, None
    
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    proj_matrix = np.hstack((rotation_matrix, translation_vector))
    _, _, _, _, _, _, eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)
    return eulerAngles, translation_vector

# Helper function to check head movement
def check_head_movement(eulerAngles, direction="left", threshold=15):
    yaw = float(eulerAngles[1])
    if direction == "left":
        return yaw < -threshold
    elif direction == "right":
        return yaw > threshold
    return False

# Modified function to show last result and include head movement with challenges
def get_user_choice():
    print(f"\nLast Challenge Result: {last_result}")
    print("Choose my liveness detection challenge:")
    print("1. Blink Detection with Head Movement")
    print("2. Gesture Recognition with Head Movement")
    print("3. End Test")
    while True:
        choice = input("Enter 1, 2, or 3: ")
        if choice in ["1", "2", "3"]:
            return "blinks" if choice == "1" else "fingers" if choice == "2" else "end"
        print("Invalid choice. Please enter 1, 2, or 3.")

# Main loop with camera management
while True:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        break

    challenge_type = get_user_choice()
    if challenge_type == "end":
        print("Ending test.")
        cap.release()
        break

    # Random head movement direction for both challenges
    challenge_direction = random.choice(["left", "right"])
    challenge_fingers = random.randint(1, 5) if challenge_type == "fingers" else None
    challenge_blinks = random.randint(1, 3) if challenge_type == "blinks" else None
    challenge_result_display = ""
    challenge_display_timer = 0
    detected_fingers = None
    blink_count = 0
    blink_detected = False
    blink_timer = 0
    EAR_THRESHOLD = 0.25
    BLINK_CONSEC_FRAMES = 2
    blink_counter = 0
    prev_ear = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame = cv2.resize(frame, (640, 480))
        overlay = frame.copy()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame for face detection
        face_results = face_mesh.process(rgb_frame)
        faces = []
        head_pose_angles = None
        face_landmark_used = None
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                ih, iw, _ = frame.shape
                x_min = min([lm.x for lm in face_landmarks.landmark]) * iw
                y_min = min([lm.y for lm in face_landmarks.landmark]) * ih
                x_max = max([lm.x for lm in face_landmarks.landmark]) * iw
                y_max = max([lm.y for lm in face_landmarks.landmark]) * ih
                x, y, w, h = int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)
                x, y = max(0, x), max(0, y)
                faces.append((x, y, w, h))
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw eye outlines
                left_eye_indices = [33, 160, 158, 133, 153, 144]
                right_eye_indices = [362, 385, 387, 263, 373, 380]
                left_eye_landmarks = [face_landmarks.landmark[i] for i in left_eye_indices]
                right_eye_landmarks = [face_landmarks.landmark[i] for i in right_eye_indices]
                draw_eye_outline(overlay, left_eye_landmarks)
                draw_eye_outline(overlay, right_eye_landmarks)
                
                # Blink detection
                ear = calculate_ear(left_eye_landmarks)
                if ear < EAR_THRESHOLD:
                    blink_counter += 1
                else:
                    if blink_counter >= BLINK_CONSEC_FRAMES:
                        blink_count += 1
                        blink_detected = True
                        blink_timer = 0
                    blink_counter = 0
                prev_ear = ear
                blink_timer += 1
                if blink_timer > 30:
                    blink_detected = False

                # Compute head pose for both challenges
                if head_pose_angles is None:
                    head_pose_angles, _ = get_head_pose(frame, face_landmarks)
                    face_landmark_used = face_landmarks

        # Process hand landmarks for finger counting challenge
        hand_results = hands.process(rgb_frame)
        if hand_results.multi_hand_landmarks:
            hand_landmarks = hand_results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(overlay, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            detected_fingers = count_raised_fingers(hand_landmarks)
        else:
            detected_fingers = None

        # Display instructions based on challenge type with head movement
        if challenge_type == "fingers":
            instruction = f"Hold up {challenge_fingers} fingers and turn head {challenge_direction}"
        elif challenge_type == "blinks":
            instruction = f"Blink {challenge_blinks} times and turn head {challenge_direction}"
        else:
            instruction = ""
        cv2.putText(overlay, instruction, (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(overlay, "Press 'c' to capture", (10, 60), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 2)
        
        if challenge_type == "fingers" and detected_fingers is not None:
            cv2.putText(overlay, f"Detected Fingers: {detected_fingers}", (10, 120), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 2)
        elif challenge_type == "blinks":
            cv2.putText(overlay, f"Detected Blinks: {blink_count}", (10, 120), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 2)
        if head_pose_angles is not None:
            yaw = float(head_pose_angles[1])
            cv2.putText(overlay, f"Yaw: {yaw:.1f}°", (10, 150), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 0, 0), 2)
        else:
            cv2.putText(overlay, "No head pose detected", (10, 150), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255), 2)

        # Display challenge result
        if challenge_display_timer > 0:
            cv2.putText(overlay, challenge_result_display, (10, 90), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255), 2)
            challenge_display_timer -= 1
            if challenge_display_timer == 0:
                cap.release()
                cv2.destroyAllWindows()
                break

        cv2.imshow("Liveness Detection Challenge", overlay)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cap.release()
            break
        elif key == ord('c'):
            if len(faces) > 0 and head_pose_angles is not None:
                head_moved_correctly = check_head_movement(head_pose_angles, challenge_direction)
                if challenge_type == "fingers":
                    if detected_fingers == challenge_fingers and head_moved_correctly:
                        challenge_result_display = f"✅ Challenge Passed! (Detected: {detected_fingers} fingers, head {challenge_direction})"
                        print("Person is likely real.")
                    else:
                        challenge_result_display = f"❌ Challenge Failed! (Fingers: {detected_fingers}, Head moved: {head_moved_correctly})"
                        print("Person may not be real.")
                elif challenge_type == "blinks":
                    if blink_count >= challenge_blinks and head_moved_correctly:
                        challenge_result_display = f"✅ Challenge Passed! (Detected: {blink_count} blinks, head {challenge_direction})"
                        print("Person is likely real.")
                    else:
                        challenge_result_display = f"❌ Challenge Failed! (Blinks: {blink_count}, Head moved: {head_moved_correctly})"
                        print("Person may not be real.")
                # Update last_result with the current challenge outcome
                last_result = challenge_result_display
            else:
                challenge_result_display = "❌ Challenge Failed! (No face or head pose detected)"
                print("Person may not be real.")
                last_result = challenge_result_display
            challenge_display_timer = 120

    if key == ord('q'):
        break

# Cleanup
hands.close()
face_mesh.close()
cv2.destroyAllWindows()
print("Test terminated.")'''







# Updating the UI making the code more readable


import cv2
import mediapipe as mp
import random
import math
import numpy as np

# Initialize MediaPipe Face Mesh (includes iris tracking)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Global variable to store the last challenge result
last_result = "No previous challenge completed yet."

# Function to calculate Eye Aspect Ratio (EAR) for blink detection
def calculate_ear(eye_landmarks):
    v1 = math.sqrt((eye_landmarks[1].x - eye_landmarks[5].x) ** 2 + (eye_landmarks[1].y - eye_landmarks[5].y) ** 2)
    v2 = math.sqrt((eye_landmarks[2].x - eye_landmarks[4].x) ** 2 + (eye_landmarks[2].y - eye_landmarks[4].y) ** 2)
    h = math.sqrt((eye_landmarks[0].x - eye_landmarks[3].x) ** 2 + (eye_landmarks[0].y - eye_landmarks[3].y) ** 2)
    ear = (v1 + v2) / (2.0 * h)
    return ear

# Function to draw eye outline
def draw_eye_outline(frame, eye_landmarks):
    points = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in eye_landmarks]
    for i in range(len(points) - 1):
        cv2.line(frame, points[i], points[i + 1], (0, 255, 0), 1)
    cv2.line(frame, points[-1], points[0], (0, 255, 0), 1)

# Function to count raised fingers
def count_raised_fingers(hand_landmarks):
    landmarks = hand_landmarks.landmark
    fingers = [
        (4, 3, 2, 1),   # Thumb: tip, DIP, PIP, MCP
        (8, 7, 6, 5),   # Index
        (12, 11, 10, 9), # Middle
        (16, 15, 14, 13), # Ring
        (20, 19, 18, 17)  # Pinky
    ]
    count = 0
    wrist_y = landmarks[0].y
    
    thumb_tip_idx, thumb_dip_idx, thumb_pip_idx, thumb_mcp_idx = fingers[0]
    thumb_tip_x, thumb_tip_y = landmarks[thumb_tip_idx].x, landmarks[thumb_tip_idx].y
    thumb_dip_y = landmarks[thumb_dip_idx].y
    thumb_pip_y = landmarks[thumb_pip_idx].y
    thumb_mcp_x, thumb_mcp_y = landmarks[thumb_mcp_idx].x, landmarks[thumb_mcp_idx].y
    
    y_diff_pip = thumb_pip_y - thumb_tip_y
    y_diff_mcp = thumb_mcp_y - thumb_tip_y
    y_diff_dip_pip = thumb_pip_y - thumb_dip_y
    thumb_extended_vertically = (thumb_tip_y < thumb_dip_y < thumb_pip_y < thumb_mcp_y and 
                                 y_diff_pip > 0.03 and 
                                 y_diff_mcp > 0.05 and 
                                 y_diff_dip_pip > 0.02)
    
    dx = thumb_tip_x - thumb_mcp_x
    dy = thumb_tip_y - thumb_mcp_y
    angle = math.degrees(math.atan2(-dy, dx))
    thumb_angle_vertical = abs(angle) < 75 or abs(angle - 180) < 75
    thumb_above_wrist = thumb_tip_y < wrist_y
    thumb_is_raised = thumb_extended_vertically and thumb_angle_vertical and thumb_above_wrist
    
    if thumb_is_raised:
        print(f"Thumb detected as raised: y_diff_pip={y_diff_pip:.3f}, y_diff_mcp={y_diff_mcp:.3f}, y_diff_dip_pip={y_diff_dip_pip:.3f}, angle={angle:.1f}°")
    else:
        print(f"Thumb not raised: y_diff_pip={y_diff_pip:.3f}, y_diff_mcp={y_diff_mcp:.3f}, y_diff_dip_pip={y_diff_dip_pip:.3f}, angle={angle:.1f}°")
    
    if thumb_is_raised:
        count += 1
    
    for tip_idx, dip_idx, pip_idx, mcp_idx in fingers[1:]:
        tip_y = landmarks[tip_idx].y
        dip_y = landmarks[dip_idx].y
        pip_y = landmarks[pip_idx].y
        mcp_y = landmarks[mcp_idx].y
        
        is_extended = (tip_y < dip_y < pip_y < mcp_y) and (tip_y < wrist_y)
        if is_extended:
            count += 1
    return count

# Function to compute head pose (Euler angles) using face landmarks
def get_head_pose(frame, face_landmarks):
    h, w, _ = frame.shape
    image_points = np.array([
        (face_landmarks.landmark[1].x * w, face_landmarks.landmark[1].y * h),     # Nose tip
        (face_landmarks.landmark[152].x * w, face_landmarks.landmark[152].y * h), # Chin
        (face_landmarks.landmark[33].x * w, face_landmarks.landmark[33].y * h),   # Left eye corner
        (face_landmarks.landmark[263].x * w, face_landmarks.landmark[263].y * h), # Right eye corner
        (face_landmarks.landmark[61].x * w, face_landmarks.landmark[61].y * h),   # Left mouth corner
        (face_landmarks.landmark[291].x * w, face_landmarks.landmark[291].y * h)  # Right mouth corner
    ], dtype="double")
    
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -63.6, -12.5),         # Chin
        (-43.3, 32.7, -26.0),        # Left eye corner
        (43.3, 32.7, -26.0),         # Right eye corner
        (-28.9, -28.9, -24.1),       # Left Mouth corner
        (28.9, -28.9, -24.1)         # Right mouth corner
    ])
    
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    dist_coeffs = np.zeros((4,1))
    
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    
    if not success:
        return None, None
    
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    proj_matrix = np.hstack((rotation_matrix, translation_vector))
    _, _, _, _, _, _, eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)
    return eulerAngles, translation_vector

# Helper function to check head movement
def check_head_movement(eulerAngles, direction="left", threshold=15):
    yaw = float(eulerAngles[1])
    if direction == "left":
        return yaw < -threshold
    elif direction == "right":
        return yaw > threshold
    return False

# Modified function to show last result and include head movement with challenges
def get_user_choice():
    print(f"\nLast Challenge Result: {last_result}")
    print("Choose my liveness detection challenge:")
    print("1. Blink Detection with Head Movement")
    print("2. Gesture Recognition with Head Movement")
    print("3. End Test")
    while True:
        choice = input("Enter 1, 2, or 3: ")
        if choice in ["1", "2", "3"]:
            return "blinks" if choice == "1" else "fingers" if choice == "2" else "end"
        print("Invalid choice. Please enter 1, 2, or 3.")

# Main loop with camera management
while True:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        break

    challenge_type = get_user_choice()
    if challenge_type == "end":
        print("Ending test.")
        cap.release()
        break

    # Random head movement direction for both challenges
    challenge_direction = random.choice(["left", "right"])
    challenge_fingers = random.randint(1, 5) if challenge_type == "fingers" else None
    challenge_blinks = random.randint(1, 3) if challenge_type == "blinks" else None
    challenge_result_display = ""
    challenge_display_timer = 0
    detected_fingers = None
    blink_count = 0
    blink_detected = False
    blink_timer = 0
    EAR_THRESHOLD = 0.25
    BLINK_CONSEC_FRAMES = 2
    blink_counter = 0
    prev_ear = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame = cv2.resize(frame, (640, 480))
        overlay = frame.copy()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame for face detection
        face_results = face_mesh.process(rgb_frame)
        faces = []
        head_pose_angles = None
        face_landmark_used = None
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                ih, iw, _ = frame.shape
                x_min = min([lm.x for lm in face_landmarks.landmark]) * iw
                y_min = min([lm.y for lm in face_landmarks.landmark]) * ih
                x_max = max([lm.x for lm in face_landmarks.landmark]) * iw
                y_max = max([lm.y for lm in face_landmarks.landmark]) * ih
                x, y, w, h = int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)
                x, y = max(0, x), max(0, y)
                faces.append((x, y, w, h))
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw eye outlines
                left_eye_indices = [33, 160, 158, 133, 153, 144]
                right_eye_indices = [362, 385, 387, 263, 373, 380]
                left_eye_landmarks = [face_landmarks.landmark[i] for i in left_eye_indices]
                right_eye_landmarks = [face_landmarks.landmark[i] for i in right_eye_indices]
                draw_eye_outline(overlay, left_eye_landmarks)
                draw_eye_outline(overlay, right_eye_landmarks)
                
                # Blink detection
                ear = calculate_ear(left_eye_landmarks)
                if ear < EAR_THRESHOLD:
                    blink_counter += 1
                else:
                    if blink_counter >= BLINK_CONSEC_FRAMES:
                        blink_count += 1
                        blink_detected = True
                        blink_timer = 0
                    blink_counter = 0
                prev_ear = ear
                blink_timer += 1
                if blink_timer > 30:
                    blink_detected = False

                # Compute head pose for both challenges
                if head_pose_angles is None:
                    head_pose_angles, _ = get_head_pose(frame, face_landmarks)
                    face_landmark_used = face_landmarks

        # Process hand landmarks for finger counting challenge
        hand_results = hands.process(rgb_frame)
        if hand_results.multi_hand_landmarks:
            hand_landmarks = hand_results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(overlay, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            detected_fingers = count_raised_fingers(hand_landmarks)
        else:
            detected_fingers = None

        # Display instructions based on challenge type with head movement
        if challenge_type == "fingers":
            instruction = f"Hold up {challenge_fingers} fingers and turn head {challenge_direction}"
        elif challenge_type == "blinks":
            instruction = f"Blink {challenge_blinks} times and turn head {challenge_direction}"
        else:
            instruction = ""
        # Draw instructions with black outline and green text
        cv2.putText(overlay, instruction, (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(overlay, instruction, (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 1)
        cv2.putText(overlay, "Press 'c' to capture", (10, 60), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(overlay, "Press 'c' to capture", (10, 60), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 1)
        
        if challenge_type == "fingers" and detected_fingers is not None:
            cv2.putText(overlay, f"Detected Fingers: {detected_fingers}", (10, 120), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(overlay, f"Detected Fingers: {detected_fingers}", (10, 120), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 1)
        elif challenge_type == "blinks":
            cv2.putText(overlay, f"Detected Blinks: {blink_count}", (10, 120), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(overlay, f"Detected Blinks: {blink_count}", (10, 120), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 1)
        if head_pose_angles is not None:
            yaw = float(head_pose_angles[1])
            cv2.putText(overlay, f"Yaw: {yaw:.1f}°", (10, 150), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(overlay, f"Yaw: {yaw:.1f}°", (10, 150), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 1)
        else:
            cv2.putText(overlay, "No head pose detected", (10, 150), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(overlay, "No head pose detected", (10, 150), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 1)

        # Display challenge result
        if challenge_display_timer > 0:
            cv2.putText(overlay, challenge_result_display, (10, 90), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(overlay, challenge_result_display, (10, 90), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 1)
            challenge_display_timer -= 1
            if challenge_display_timer == 0:
                cap.release()
                cv2.destroyAllWindows()
                break

        cv2.imshow("Liveness Detection Challenge", overlay)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cap.release()
            break
        elif key == ord('c'):
            if len(faces) > 0 and head_pose_angles is not None:
                head_moved_correctly = check_head_movement(head_pose_angles, challenge_direction)
                if challenge_type == "fingers":
                    if detected_fingers == challenge_fingers and head_moved_correctly:
                        challenge_result_display = f"✅ Challenge Passed! (Detected: {detected_fingers} fingers, head {challenge_direction})"
                        print("Person is likely real.")
                    else:
                        challenge_result_display = f"❌ Challenge Failed! (Fingers: {detected_fingers}, Head moved: {head_moved_correctly})"
                        print("Person may not be real.")
                elif challenge_type == "blinks":
                    if blink_count >= challenge_blinks and head_moved_correctly:
                        challenge_result_display = f"✅ Challenge Passed! (Detected: {blink_count} blinks, head {challenge_direction})"
                        print("Person is likely real.")
                    else:
                        challenge_result_display = f"❌ Challenge Failed! (Blinks: {blink_count}, Head moved: {head_moved_correctly})"
                        print("Person may not be real.")
                # Update last_result with the current challenge outcome
                last_result = challenge_result_display
            else:
                challenge_result_display = "❌ Challenge Failed! (No face or head pose detected)"
                print("Person may not be real.")
                last_result = challenge_result_display
            challenge_display_timer = 120

    if key == ord('q'):
        break

# Cleanup
hands.close()
face_mesh.close()
cv2.destroyAllWindows()
print("Test terminated.")

#Adding the pre trained anti spoofing model, download and clone from github and integrated into code(https://github.com/paulovpcotta/antispoofing/tree/master)
#Note it only works with a good quality camera please ensure the same 
'''
import cv2
import mediapipe as mp
import random
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense

# --------------------------
# 1. Create the Pretrained Anti-Spoofing Model
# --------------------------
def create_model():
    model = Sequential()
    # Explicit input layer: (24, 24, 1)
    model.add(Input(shape=(24, 24, 1)))
    # First convolutional layer: 6 filters, 3x3 kernel, ReLU activation
    model.add(Conv2D(6, kernel_size=(3, 3), activation="relu"))
    # First pooling layer
    model.add(AveragePooling2D(pool_size=(2, 2)))
    # Second convolutional layer: 16 filters, 3x3 kernel, ReLU activation
    model.add(Conv2D(16, kernel_size=(3, 3), activation="relu"))
    # Second pooling layer
    model.add(AveragePooling2D(pool_size=(2, 2)))
    # Flatten the output
    model.add(Flatten())
    # Dense layer with 120 units
    model.add(Dense(120, activation="relu"))
    # Dense layer with 84 units
    model.add(Dense(84, activation="relu"))
    # Output layer with 1 unit and sigmoid activation (for binary classification)
    model.add(Dense(1, activation="sigmoid"))
    return model

# Create the model
anti_spoofing_model = create_model()

# Load weights from my HDF5 file (update the path if needed)
anti_spoofing_model.load_weights("D:/LivenessDetection/antispoofing-master/model.h5")

# Compile the model
anti_spoofing_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# (Optional) Print model summary to verify input shape and layers
anti_spoofing_model.summary()

# --------------------------
# 2. Initialize MediaPipe Modules
# --------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Global variable to store the last challenge result
last_result = "No previous challenge completed yet."

# --------------------------
# 3. Define Helper Functions
# --------------------------
def calculate_ear(eye_landmarks):
    # Calculate Eye Aspect Ratio for blink detection.
    v1 = math.sqrt((eye_landmarks[1].x - eye_landmarks[5].x) ** 2 +
                   (eye_landmarks[1].y - eye_landmarks[5].y) ** 2)
    v2 = math.sqrt((eye_landmarks[2].x - eye_landmarks[4].x) ** 2 +
                   (eye_landmarks[2].y - eye_landmarks[4].y) ** 2)
    h = math.sqrt((eye_landmarks[0].x - eye_landmarks[3].x) ** 2 +
                  (eye_landmarks[0].y - eye_landmarks[3].y) ** 2)
    ear = (v1 + v2) / (2.0 * h)
    return ear

def draw_eye_outline(frame, eye_landmarks):
    # Draw lines between eye landmarks.
    points = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0]))
              for lm in eye_landmarks]
    for i in range(len(points) - 1):
        cv2.line(frame, points[i], points[i+1], (0, 255, 0), 1)
    cv2.line(frame, points[-1], points[0], (0, 255, 0), 1)

def count_raised_fingers(hand_landmarks):
    # Count raised fingers based on hand landmarks.
    landmarks = hand_landmarks.landmark
    fingers = [
        (4, 3, 2, 1),   # Thumb
        (8, 7, 6, 5),   # Index
        (12, 11, 10, 9),# Middle
        (16, 15, 14, 13),# Ring
        (20, 19, 18, 17) # Pinky
    ]
    count = 0
    wrist_y = landmarks[0].y
    
    # Process thumb separately
    thumb_tip_idx, thumb_dip_idx, thumb_pip_idx, thumb_mcp_idx = fingers[0]
    thumb_tip_y = landmarks[thumb_tip_idx].y
    thumb_dip_y = landmarks[thumb_dip_idx].y
    thumb_pip_y = landmarks[thumb_pip_idx].y
    thumb_mcp_y = landmarks[thumb_mcp_idx].y

    y_diff_pip = thumb_pip_y - thumb_tip_y
    y_diff_mcp = thumb_mcp_y - thumb_tip_y
    y_diff_dip_pip = thumb_pip_y - thumb_dip_y
    thumb_extended = (thumb_tip_y < thumb_dip_y < thumb_pip_y < thumb_mcp_y and
                      y_diff_pip > 0.03 and y_diff_mcp > 0.05 and y_diff_dip_pip > 0.02)
    if thumb_extended:
        count += 1

    # Process other fingers
    for tip_idx, dip_idx, pip_idx, mcp_idx in fingers[1:]:
        tip_y = landmarks[tip_idx].y
        dip_y = landmarks[dip_idx].y
        pip_y = landmarks[pip_idx].y
        mcp_y = landmarks[mcp_idx].y
        if tip_y < dip_y < pip_y < mcp_y and tip_y < wrist_y:
            count += 1
    return count

def get_head_pose(frame, face_landmarks):
    # Compute head pose using 6 facial landmarks.
    h, w, _ = frame.shape
    image_points = np.array([
        (face_landmarks.landmark[1].x * w, face_landmarks.landmark[1].y * h),   # Nose tip
        (face_landmarks.landmark[152].x * w, face_landmarks.landmark[152].y * h), # Chin
        (face_landmarks.landmark[33].x * w, face_landmarks.landmark[33].y * h),   # Left eye corner
        (face_landmarks.landmark[263].x * w, face_landmarks.landmark[263].y * h), # Right eye corner
        (face_landmarks.landmark[61].x * w, face_landmarks.landmark[61].y * h),   # Left mouth corner
        (face_landmarks.landmark[291].x * w, face_landmarks.landmark[291].y * h)  # Right mouth corner
    ], dtype="double")
    
    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1)
    ])
    
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4,1))
    
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    
    if not success:
        return None, None
    
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    proj_matrix = np.hstack((rotation_matrix, translation_vector))
    _, _, _, _, _, _, eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)
    return eulerAngles, translation_vector

def check_head_movement(eulerAngles, direction="left", threshold=15):
    # Check if the head has turned sufficiently in the specified direction.
    yaw = float(eulerAngles[1])
    if direction == "left":
        return yaw < -threshold
    elif direction == "right":
        return yaw > threshold
    return False

def get_user_choice():
    # Print the last result and prompt the user for a challenge choice.
    print(f"\nLast Challenge Result: {last_result}")
    print("Choose my liveness detection challenge:")
    print("1. Blink Detection with Head Movement")
    print("2. Gesture Recognition with Head Movement")
    print("3. End Test")
    while True:
        choice = input("Enter 1, 2, or 3: ")
        if choice in ["1", "2", "3"]:
            return "blinks" if choice == "1" else "fingers" if choice == "2" else "end"
        print("Invalid choice. Please enter 1, 2, or 3.")

'''
'''def predict_live_face(frame, x, y, w, h):
    face_roi = frame[y:y+h, x:x+w]
    try:
        face_roi = cv2.resize(face_roi, (24, 24))
    except Exception as e:
        return "N/A"
    face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    face_roi = face_roi.astype("float32") / 255.0
    face_roi = np.expand_dims(face_roi, axis=-1)   # shape: (24, 24, 1)
    face_roi = np.expand_dims(face_roi, axis=0)      # shape: (1, 24, 24, 1)
    
    pred = anti_spoofing_model.predict(face_roi)
    prob = pred[0][0]
    print(f"Raw prediction probability: {prob:.3f}")  # Debug print

    # Adjust threshold here if needed (e.g., change 0.5 to 0.4)
    threshold = 0.5
    return "Real Person" if prob > threshold else "Fake"'''
'''
def predict_live_face(frame, x, y, w, h):
    face_roi = frame[y:y+h, x:x+w]
    try:
        face_roi = cv2.resize(face_roi, (24, 24))
    except Exception as e:
        return "N/A"
    face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    face_roi = face_roi.astype("float32") / 255.0
    face_roi = np.expand_dims(face_roi, axis=-1)   # shape: (24, 24, 1)
    face_roi = np.expand_dims(face_roi, axis=0)      # shape: (1, 24, 24, 1)

    # Debug: save the ROI image to check if it's correct (optional)
    cv2.imwrite("debug_roi.jpg", (face_roi[0]*255).astype(np.uint8))
    
    pred = anti_spoofing_model.predict(face_roi)
    prob = pred[0][0]
    print(f"Raw prediction probability: {prob:.3f}")
    
    # Try adjusting the threshold if needed:
    threshold = 0.5  # You might experiment with values like 0.3 or 0.4
    return "Real Person" if prob > threshold else "Fake"


# --------------------------
# 4. Main Loop with Camera Management
# --------------------------
while True:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        break

    challenge_type = get_user_choice()
    if challenge_type == "end":
        print("Ending test.")
        cap.release()
        break

    # Set random challenge parameters
    challenge_direction = random.choice(["left", "right"])
    challenge_fingers = random.randint(1, 5) if challenge_type == "fingers" else None
    challenge_blinks = random.randint(1, 3) if challenge_type == "blinks" else None
    challenge_result_display = ""
    challenge_display_timer = 0
    detected_fingers = None
    blink_count = 0
    blink_detected = False
    blink_timer = 0
    EAR_THRESHOLD = 0.25
    BLINK_CONSEC_FRAMES = 2
    blink_counter = 0
    prev_ear = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame = cv2.resize(frame, (640, 480))
        overlay = frame.copy()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process face detection using MediaPipe Face Mesh
        face_results = face_mesh.process(rgb_frame)
        faces = []
        head_pose_angles = None
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                ih, iw, _ = frame.shape
                x_min = min([lm.x for lm in face_landmarks.landmark]) * iw
                y_min = min([lm.y for lm in face_landmarks.landmark]) * ih
                x_max = max([lm.x for lm in face_landmarks.landmark]) * iw
                y_max = max([lm.y for lm in face_landmarks.landmark]) * ih
                x, y, w, h = int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)
                x, y = max(0, x), max(0, y)
                faces.append((x, y, w, h))
                
                # Draw bounding box around face
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Use the anti-spoofing model to predict if the face is real
                spoof_label = predict_live_face(frame, x, y, w, h)
                cv2.putText(overlay, spoof_label, (x, y - 30),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(overlay, spoof_label, (x, y - 30),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 1)
                
                # Draw eye outlines
                left_eye_indices = [33, 160, 158, 133, 153, 144]
                right_eye_indices = [362, 385, 387, 263, 373, 380]
                left_eye_landmarks = [face_landmarks.landmark[i] for i in left_eye_indices]
                right_eye_landmarks = [face_landmarks.landmark[i] for i in right_eye_indices]
                draw_eye_outline(overlay, left_eye_landmarks)
                draw_eye_outline(overlay, right_eye_landmarks)
                
                # Blink detection using left eye landmarks
                ear = calculate_ear(left_eye_landmarks)
                if ear < EAR_THRESHOLD:
                    blink_counter += 1
                else:
                    if blink_counter >= BLINK_CONSEC_FRAMES:
                        blink_count += 1
                        blink_detected = True
                        blink_timer = 0
                    blink_counter = 0
                prev_ear = ear
                blink_timer += 1
                if blink_timer > 30:
                    blink_detected = False

                # Compute head pose (once per face)
                if head_pose_angles is None:
                    head_pose_angles, _ = get_head_pose(frame, face_landmarks)

        # Process hand landmarks for finger counting challenge
        hand_results = hands.process(rgb_frame)
        if hand_results.multi_hand_landmarks:
            hand_landmarks = hand_results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(overlay, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            detected_fingers = count_raised_fingers(hand_landmarks)
        else:
            detected_fingers = None

        # Display instructions based on challenge type
        if challenge_type == "fingers":
            instruction = f"Hold up {challenge_fingers} fingers and turn head {challenge_direction}"
        elif challenge_type == "blinks":
            instruction = f"Blink {challenge_blinks} times and turn head {challenge_direction}"
        else:
            instruction = ""
        cv2.putText(overlay, instruction, (10, 30),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(overlay, instruction, (10, 30),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 1)
        cv2.putText(overlay, "Press 'c' to capture", (10, 60),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(overlay, "Press 'c' to capture", (10, 60),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 1)
        
        if challenge_type == "fingers" and detected_fingers is not None:
            cv2.putText(overlay, f"Detected Fingers: {detected_fingers}", (10, 120),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(overlay, f"Detected Fingers: {detected_fingers}", (10, 120),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 1)
        elif challenge_type == "blinks":
            cv2.putText(overlay, f"Detected Blinks: {blink_count}", (10, 120),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(overlay, f"Detected Blinks: {blink_count}", (10, 120),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 1)
        if head_pose_angles is not None:
            yaw = float(head_pose_angles[1])
            cv2.putText(overlay, f"Yaw: {yaw:.1f}°", (10, 150),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(overlay, f"Yaw: {yaw:.1f}°", (10, 150),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 1)
        else:
            cv2.putText(overlay, "No head pose detected", (10, 150),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(overlay, "No head pose detected", (10, 150),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 1)

        if challenge_display_timer > 0:
            cv2.putText(overlay, challenge_result_display, (10, 90),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(overlay, challenge_result_display, (10, 90),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 1)
            challenge_display_timer -= 1
            if challenge_display_timer == 0:
                cap.release()
                cv2.destroyAllWindows()
                break

        cv2.imshow("Liveness Detection Challenge", overlay)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cap.release()
            break
        elif key == ord('c'):
            if len(faces) > 0 and head_pose_angles is not None:
                head_moved_correctly = check_head_movement(head_pose_angles, challenge_direction)
                if challenge_type == "fingers":
                    if detected_fingers == challenge_fingers and head_moved_correctly:
                        challenge_result_display = f"✅ Challenge Passed! (Detected: {detected_fingers} fingers, head {challenge_direction})"
                        print("Person is likely real.")
                    else:
                        challenge_result_display = f"❌ Challenge Failed! (Fingers: {detected_fingers}, Head moved: {head_moved_correctly})"
                        print("Person may not be real.")
                elif challenge_type == "blinks":
                    if blink_count >= challenge_blinks and head_moved_correctly:
                        challenge_result_display = f"✅ Challenge Passed! (Detected: {blink_count} blinks, head {challenge_direction})"
                        print("Person is likely real.")
                    else:
                        challenge_result_display = f"❌ Challenge Failed! (Blinks: {blink_count}, Head moved: {head_moved_correctly})"
                        print("Person may not be real.")
                last_result = challenge_result_display
            else:
                challenge_result_display = "❌ Challenge Failed! (No face or head pose detected)"
                print("Person may not be real.")
                last_result = challenge_result_display
            challenge_display_timer = 120

    if key == ord('q'):
        break

hands.close()
face_mesh.close()
cv2.destroyAllWindows()
print("Test terminated.")'''
