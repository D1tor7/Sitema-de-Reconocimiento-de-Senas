import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe import ImageFormat
import numpy as np
import matplotlib.pyplot as plt
import csv
from dtw import *

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

#file = open('D:\\UPAO\\SistemasGestuales&Conversacionales\\Semana15\\U.txt', 'w')


x_indice = []
y_indice = [] 
x_meñique = []
y_meñique = [] 
n = []
n_i = 0
#indice
with open('E:\\User\\Escritorio\\Sistemas_Conversacionales\\Semana16\\LETRAS_MOVIMIENTO\\Ñ_indice.txt') as Ñi:
    readerÑi = csv.reader(Ñi, delimiter=' ')
    referenceÑi = list(readerÑi)
with open('E:\\User\\Escritorio\\Sistemas_Conversacionales\\Semana16\\LETRAS_MOVIMIENTO\\S_indice.txt') as Si:
    readerSi = csv.reader(Si, delimiter=' ')
    referenceSi = list(readerSi)
with open('E:\\User\\Escritorio\\Sistemas_Conversacionales\\Semana16\\LETRAS_MOVIMIENTO\\Z_indice.txt') as Zi:
    readerZi = csv.reader(Zi, delimiter=' ')
    referenceZi = list(readerZi)
with open('E:\\User\\Escritorio\\Sistemas_Conversacionales\\Semana16\\LETRAS_MOVIMIENTO\\J_indice.txt') as Ji:
    readerJi = csv.reader(Ji, delimiter=' ')
    referenceJi = list(readerJi)
#meñique

with open('E:\\User\\Escritorio\\Sistemas_Conversacionales\\Semana16\\LETRAS_MOVIMIENTO\\Ñ_meñique.txt') as Ñm:
    readerÑm = csv.reader(Ñm, delimiter=' ')
    referenceÑm = list(readerÑm)
with open('E:\\User\\Escritorio\\Sistemas_Conversacionales\\Semana16\\LETRAS_MOVIMIENTO\\S_meñique.txt') as Sm:
    readerSm = csv.reader(Sm, delimiter=' ')
    referenceSm = list(readerSm)
with open('E:\\User\\Escritorio\\Sistemas_Conversacionales\\Semana16\\LETRAS_MOVIMIENTO\\Z_meñique.txt') as Zm:
    readerZm = csv.reader(Zm, delimiter=' ')
    referenceZm = list(readerZm)
with open('E:\\User\\Escritorio\\Sistemas_Conversacionales\\Semana16\\LETRAS_MOVIMIENTO\\J_meñique.txt') as Jm:
    readerJm = csv.reader(Jm, delimiter=' ')
    referenceJm = list(readerJm)



def draw_landmarks_on_image_indice(rgb_image, detection_result):
  global xindice, yindice, n, n_i
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)  

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
                        
    x_indice.append(-x_coordinates[8])
    y_indice.append(-y_coordinates[8])
    n.append(n_i)
    n_i = n_i+1      
    text_x = int(min(x_coordinates) * width)
    text_y = int(max(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, str(hand_landmarks[0].x) + ',' + str(hand_landmarks[0].y),
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image

def draw_landmarks_on_image_meñique(rgb_image, detection_result):
  global x_meñique, y_meñique, n, n_i
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image_m = np.copy(rgb_image)  

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image_m,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image_m.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
                        
    x_meñique.append(-x_coordinates[20])
    y_meñique.append(-y_coordinates[20])
    n.append(n_i)
    n_i = n_i+1      
    text_x = int(min(x_coordinates) * width)
    text_y = int(max(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image_m, str(hand_landmarks[0].x) + ',' + str(hand_landmarks[0].y),
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image_m

# STEP 2: Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path='E:\\User\\Escritorio\\Sistemas_Conversacionales\\Semana16\\hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)


# For webcam input:
cap = cv2.VideoCapture(0)
while cap.isOpened():
  success, image = cap.read()
  if not success:
    print("Ignoring empty camera frame.")
    # If loading a video, use 'break' instead of 'continue'.
    continue

  # To improve performance, optionally mark the image as not writeable to
  # pass by reference.
  image.flags.writeable = False
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  #image = image[:, :, ::-1]
  #print(image.shape)
  rgb_frame = mp.Image(image_format=ImageFormat.SRGB, data=image)

  # Draw the hand annotations on the image.
  #image.flags.writeable = True
  #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  #image_height, image_width, _ = image.shape
  # STEP 4: Detect hand landmarks from the input image.
  detection_result = detector.detect(rgb_frame)

  # STEP 5: Process the classification result. In this case, visualize it.
  annotated_image = draw_landmarks_on_image_indice(rgb_frame.numpy_view(), detection_result)
  annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

  annotated_image_m = draw_landmarks_on_image_meñique(rgb_frame.numpy_view(), detection_result)
  annotated_image_m = cv2.cvtColor(annotated_image_m, cv2.COLOR_RGB2BGR)
  #cv2.imshow('MediaPipe Hands', cv2.flip(annotated_image, 1))
  cv2.imshow('MediaPipe Hands', annotated_image)
  if cv2.waitKey(5) & 0xFF == 27:
    break
  if cv2.waitKey(17) & 0xFF == 27:
    break
cap.release()

query_i = [x_indice, y_indice]
query_i = np.array(query_i)
query_i = query_i.T

referenceÑi = np.array(referenceÑi)
referenceÑi = referenceÑi.astype(float)

referenceSi = np.array(referenceSi)
referenceSi = referenceSi.astype(float)

referenceZi = np.array(referenceZi)
referenceZi = referenceZi.astype(float)

referenceJi = np.array(referenceJi)
referenceJi = referenceJi.astype(float)


query_m = [x_meñique, y_meñique]
query_m = np.array(query_m)
query_m = query_m.T

referenceÑm = np.array(referenceÑm)
referenceÑm = referenceÑm.astype(float)

referenceSm = np.array(referenceSm)
referenceSm = referenceSm.astype(float)

referenceZm = np.array(referenceZm)
referenceZm = referenceZm.astype(float)

referenceJm = np.array(referenceJm)
referenceJm = referenceJm.astype(float)

#dx = dtw(query[:,0],reference[:,0],keep_internals=True,step_pattern=rabinerJuangStepPattern(6, "c"))
#dx.plot(type='twoway')

#dy = dtw(query[:,1],reference[:,1],keep_internals=True,step_pattern=rabinerJuangStepPattern(6, "c"))
#dy.plot(type='twoway')

Ñi = dtw(query_i,referenceÑi,keep_internals=True)
Ñm=dtw(query_m,referenceÑm,keep_internals=True)
print(Ñi.distance)
print(Ñm.distance)

if(Ñi.distance<15 and Ñm.distance<15):
    print('Hiciste una Ñ')
else:
    print('No es Ñ')

Si = dtw(query_i,referenceSi,keep_internals=True)
Sm = dtw(query_m,referenceSm,keep_internals=True)
print(Si.distance)
print(Sm.distance)
if(Si.distance<15 and Sm.distance<15):
    print('Hiciste una S')
else:
    print('No es S')
    
Zi = dtw(query_i,referenceZi,keep_internals=True)
Zm = dtw(query_m,referenceZm,keep_internals=True)
print(Zm.distance)
print(Zi.distance)
if(Zi.distance<15 and Zm.distance<15):
    print('Hiciste una Z')
else:
    print('No es Z')

Ji = dtw(query_i,referenceJi,keep_internals=True)
Jm = dtw(query_m,referenceJm,keep_internals=True)
print(Jm.distance)
print(Ji.distance)
if(Ji.distance<15 and Jm.distance<15):
    print('Hiciste una J')
else:
    print('No es J')