_**Liveness Detection System**_


Real-time liveness detection to verify human presence using MediaPipe and OpenCV. 
Offers two challenges: blink detection (via Eye Aspect Ratio) and gesture recognition (finger counting).
Users choose via console, with results shown for 2 seconds before refreshing.

_UPDATE_: Added head movement detection challenege integrated with blink detection challenge and gesture recognition challenge       

_Update_: Added a commented out code adding and integrating a pre trained Anti spoofing model grom Git, link--> https://github.com/paulovpcotta/antispoofing/tree/master
        Note: It only works with a good quality camera.(with depth detection)



        
**Features-**
-Blink Detection: Tracks eye contours and counts blinks using iris landmarks.

-Gesture Recognition: Counts raised fingers with refined thumb logic.

-User Choice: Console selection (1: Blink, 2: Gesture, 3: End).

-Visual Feedback: Displays eye outlines and detection results.

-Camera Management: Closes/reopens webcam between challenges.



**Prerequisites**

-Python 3.6+

-Webcam



**Installation**
-Clone the repository: ->git clone https://github.com/SNEAKO07/liveness-detection-system.git

                      ->cd liveness-detection-system

-Install dependencies: ->pip install mediapipe opencv-python

-Run the script: ->python LivenessDetection.py



**Follow console prompts:**

1: Blink Detection Test
    (blink the specified number of times and move your head in the specified direction , i.e, Left or Right).

2: Gesture Recognition Test
    (raise the specified fingers and move your head in the specified direction , i.e, Left or Right).

3: End Test.

Press c to submit your challenge response;



_**How It Works**_

_Blink Detection_: Uses MediaPipe Face Mesh to track iris landmarks, calculates EAR to count blinks.

_Gesture Recognition_: Uses MediaPipe Hands to detect finger positions.

_Head Pose Detection_: Compute head pose (Euler angles) using face landmarks

_Flow_: Camera opens, user selects challenge, result displays for 2s, camera closes, then re-prompts.
