import os
path = r'C:\user\ai-lab\exp-3'
os.environ['PATH']+=':'+path
from deepface import DeepFace
from PIL import Image
import cv2
import matplotlib.pyplot as plt
img_path = 'image4.jpg'
img = cv2.imread(img_path)
plt.imshow(img[:, :, ::-1])
image = Image.open(img_path)
image.show()
#print(img)
demography = DeepFace.analyze(img_path) #passing nothing as 2nd argument will find everything
#demography = DeepFace.analyze("img4.jpg",['age', 'gender', 'emotion']) #identical to the line above
#demography = DeepFace.analyze(["img1.jpg", "img4.jpg"]) #analyzing multiple faces same time
print("Age: ", demography["age"])
print("Gender: ", demography["gender"])
print("Emotion: ", demography["dominant_emotion"])
print("Race: ", demography["dominant_race"])
