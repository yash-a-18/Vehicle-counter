#pip install opencv-python
#pip install opencv-python-headless
#pip install matplotlib
#pip install cvlib
#pip install tensorflow
import cv2
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox

im = cv2.imread('third.jpg')

bbox, label, conf = cv.detect_common_objects(im)
output_image = draw_bbox(im, bbox, label, conf)

print('Number of cars in image is:', label.count('car'))
print('Number of persons in image is:', label.count('person'))
print('Number of trucks in image is:', label.count('truck'))
print('Number of buses in image is:', label.count('bus'))
print('Number of motorcycles in image is:', label.count('motorcycle'))

plt.imshow(output_image)
plt.show()
