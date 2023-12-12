import cv2
import csv
import mediapipe as mp
import pandas as pd
import keyboard
import os
import uuid
import time
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler #Standardizes the data so that no one feature overshadows one feature
from sklearn.linear_model import LogisticRegression, RidgeClassifier # Classifciation Algorithms
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score # Calculates Accuracy
import pickle # Save their models down to disks

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# TRAINING SET UP
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

#os.mkdir('Output Images')


df = pd.read_csv('hand_vals.csv')
print(df)

ButtonPosX = 525
ButtonPosY = 450
ButtonWidth = 100
ButtonHeight = 50
canClick = False
count = 0
classname = "J" # Change This per Gesture

#Setting Up Features and Classes
x = df.drop("Class", axis=1) # Every Value Except Our class column
y = df['Class'] # Only our class column

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=1234) # Train and Test Split of the information
# Training the model/ Separate Machine learning algorithms
# Dicstionary of Pipelines
pipelines = {
  'lr':make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
  'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
  'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
  'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
}

fit_models = {}
for algo, pipeline in pipelines.items():
  model = pipeline.fit(X_train, Y_train)
  fit_models[algo] = model


#Evaluate Models
for algo, model in fit_models.items():
  yhat = model.predict(X_test)
  print(algo, accuracy_score(Y_test, yhat))

# Exports the model into pickel, uncomment if needed
with open('sign_language.pkl', 'wb') as f:
  pickle.dump(fit_models['lr'],f)


#Test Tree
clf = DecisionTreeClassifier()
clf.fit(X_train, Y_train)
#print(tree.export_text(clf))
fig  = plt.figure(figsize=(100,60))
tree.plot_tree(clf, feature_names=X_train.columns, class_names=clf.classes_, filled=True)
fig.savefig("decision_tree.png")

# TRAINING SET UP END

#Load our model
with open('sign_language.pkl', 'rb') as f:
    model = pickle.load(f)


class Button:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.can_click = False

button = Button(ButtonPosX, ButtonPosY - ButtonHeight, ButtonWidth, ButtonHeight)

count = 0

def mouse_callback(event, x, y, flags, param):
    global count
    if event == cv2.EVENT_LBUTTONDOWN:
        if button.can_click == True and button.x < x < button.x + button.width and button.y < y < button.y + button.height:
            #print(f"Mouse clicked at position ({x}, {y})")

            lm_row = [classname]
            for index, lm in enumerate(mp_hands.HandLandmark):
              lm_point = hand_landmarks.landmark[index]
              lm_row.extend([lm_point.x, lm_point.y, lm_point.z])

            csv_writer.writerow(lm_row)
            cv2.imwrite(os.path.join('Output Images', '{}.jpg'.format(uuid.uuid1())), image) #Create an image with a unique identifer

# For static images:
IMAGE_FILES = []
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:
  for idx, file in enumerate(IMAGE_FILES):
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    image = cv2.flip(cv2.imread(file), 1)

    # added
    # cv2.rectangle(image.copy(), (int(100), int(100)), (int(100), int(100)), (255, 12, 145), 2)
    # ends here


    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print handedness and draw hand landmarks on the image.
    print('Handedness:', results.multi_handedness)
    if not results.multi_hand_landmarks:
      continue
    image_height, image_width, _ = image.shape
    annotated_image = image.copy()

    for hand_landmarks in results.multi_hand_landmarks:
      print('hand_landmarks:', hand_landmarks)
      print(
          f'Index finger tip coordinates: (',
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
      )
      mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
    cv2.imwrite(
        '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
    # Draw hand world landmarks.
    if not results.multi_hand_world_landmarks:
      continue
    for hand_world_landmarks in results.multi_hand_world_landmarks:
      mp_drawing.plot_landmarks(
        hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=10,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands, open('hand_vals.csv', 'a', newline='') as csv_file:
  
  # WRITE HEADER
  csv_writer = csv.writer(csv_file)
  csv_header = ["Class","WRIST_X","WRIST_Y","WRIST_Z","THUMB_CMC_X","THUMB_CMC_Y","THUMB_CMC_Z","THUMB_MCP_X","THUMB_MCP_Y","THUMB_MCP_Z","THUMB_IP_X","THUMB_IP_Y","THUMB_IP_Z","THUMB_TIP_X","THUMB_TIP_Y","THUMB_TIP_Z","INDEX_FINGER_MCP_X","INDEX_FINGER_MCP_Y","INDEX_FINGER_MCP_Z","INDEX_FINGER_PIP_X","INDEX_FINGER_PIP_Y","INDEX_FINGER_PIP_Z","INDEX_FINGER_DIP_X","INDEX_FINGER_DIP_Y","INDEX_FINGER_DIP_Z","INDEX_FINGER_TIP_X","INDEX_FINGER_TIP_Y","INDEX_FINGER_TIP_Z","MIDDLE_FINGER_MCP_X","MIDDLE_FINGER_MCP_Y","MIDDLE_FINGER_MCP_Z","MIDDLE_FINGER_PIP_X","MIDDLE_FINGER_PIP_Y","MIDDLE_FINGER_PIP_Z","MIDDLE_FINGER_DIP_X","MIDDLE_FINGER_DIP_Y","MIDDLE_FINGER_DIP_Z","MIDDLE_FINGER_TIP_X","MIDDLE_FINGER_TIP_Y","MIDDLE_FINGER_TIP_Z","RING_FINGER_MCP_X","RING_FINGER_MCP_Y","RING_FINGER_MCP_Z","RING_FINGER_PIP_X","RING_FINGER_PIP_Y","RING_FINGER_PIP_Z","RING_FINGER_DIP_X","RING_FINGER_DIP_Y","RING_FINGER_DIP_Z","RING_FINGER_TIP_X","RING_FINGER_TIP_Y","RING_FINGER_TIP_Z","PINKY_MCP_X","PINKY_MCP_Y","PINKY_MCP_Z","PINKY_PIP_X","PINKY_PIP_Y","PINKY_PIP_Z","PINKY_DIP_X","PINKY_DIP_Y","PINKY_DIP_Z","PINKY_TIP_X","PINKY_TIP_Y","PINKY_TIP_Z"]
  #csv_writer.writerow(csv_header)


  while cap.isOpened():
    success, image = cap.read()
  
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
  
    # Set Flag
    image.flags.writeable = False

    # BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    #Detections
    results = hands.process(image) 
    
    # Set Flag to True
    image.flags.writeable = True

    #RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #cv2.rectangle(image, (80, 150), (280, 350),(0, 255, 0), 2)



    

    if results.multi_hand_landmarks:
      button.can_click = True


      for hand_landmarks in results.multi_hand_landmarks:
        max_x = 0
        min_x = 1
        max_y = 0
        min_y = 1


        for index, lm in enumerate(mp_hands.HandLandmark):

          for point in hand_landmarks.landmark:
            max_x = max(max_x,point.x)
            min_x = min(min_x,point.x)
            max_y = max(max_y,point.y)
            min_y = min(min_y,point.y)


          # Add Button
          cv2.rectangle(image, (ButtonPosX, ButtonPosY - 50), (ButtonPosX + 100, ButtonPosY), (0, 255, 0), -1)
          cv2.putText(image, 'Capture', (ButtonPosX + 20, ButtonPosY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
          """
          if max_x < 0.50 and min_x > 0.10 and max_y < 0.75 and min_y > 0.25:
            button.can_click = True
            # Add Button
            cv2.rectangle(image, (ButtonPosX, ButtonPosY - 50), (ButtonPosX + 100, ButtonPosY), (0, 255, 0), -1)
            cv2.putText(image, 'Capture', (ButtonPosX + 20, ButtonPosY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            # Add Bounding Box 
            cv2.rectangle(image, (80, 150), (280, 350), (0, 0, 255), 2) #change to button
          else:
            button.can_click = False
          """

        # Draws Hand Land Marks
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    
        try: 
          
          lm_row = []
          for index, lm in enumerate(mp_hands.HandLandmark):
              lm_point = hand_landmarks.landmark[index]
              lm_row.extend([lm_point.x, lm_point.y, lm_point.z])

          X = pd.DataFrame([lm_row], columns=csv_header[1:])
          # Make Detections
          body_language_class = model.predict(X)[0]
          body_language_prob = model.predict_proba(X)[0]
          #print(body_language_class, body_language_prob)


          #Grab Coord
          image_height, image_width, _ = image.shape
          max_x, max_y, min_x, min_y = float('-inf'), float('-inf'), float('inf'), float('inf')

          for index, lm in enumerate(mp_hands.HandLandmark):
              lm_point = hand_landmarks.landmark[index]
              x, y = int(lm_point.x * image_width), int(lm_point.y * image_height)

              # Update maximum and minimum coordinates
              max_x = max(max_x, x)
              max_y = max(max_y, y)
              min_x = min(min_x, x)
              min_y = min(min_y, y)

          # Draw a rectangle around the hand based on max and min coordinates
          cv2.rectangle(image, (min_x, min_y - 50), (max_x, max_y), (255,255,255), 2)
          
          #Add Text
          cv2.rectangle(image, (min_x, min_y - 70), (max_x, min_y - 50), (255,255,255), -1)
          cv2.putText(image, body_language_class, (min_x + 10, min_y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)


        except Exception as e:
          # By this way we can know about the type of error occurring
          print("The error is: ",e)
          pass
    else:
      button.can_click = True
   
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', image)

    cv2.setMouseCallback("MediaPipe Hands", mouse_callback)

    

    
    
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()


