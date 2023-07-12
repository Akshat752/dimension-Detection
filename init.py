"""
Filename: init.py
Usage: This script will measure different objects in the frame using a reference object of known dimension. 
The object with known dimension must be the leftmost object.
Author: Shashank Sharma
"""

import random
from scipy.spatial.distance import euclidean
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2

# Function to show array of images (intermediate results)
def show_images(images):
	for i, img in enumerate(images):
		cv2.imshow("image_" + str(i), img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def show_all_images():
	cv2.imshow("final", image)
	cv2.imshow("erode", erode)
        
	cv2.imshow("dilated", dialated)
	cv2.imshow("Canny", Canny)
	cv2.imshow("blurred", blur)
	cv2.imshow("gray", gray)
	cv2.imshow("original", cv2.imread(img_path))
	
	cv2.waitKey(0)
	cv2.destroyAllWindows()

img_path = "test/text6.jpg"

# Read image and preprocess
image = cv2.imread(img_path)




gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



clahe = cv2.createCLAHE(clipLimit=random.uniform(0, 100.0))
cl1 = clahe.apply(gray)



blur = cv2.GaussianBlur(gray, (9, 9), 0)

Canny = cv2.Canny(blur, 21, 42)



show_images([cl1])

# for x in range(50):
#     c2 = cv2.Canny(blur, x, x*2)
#     c2 = cv2.dilate(c2, None, iterations=1)
#     cv2.imshow(str(x),  c2)
#     cv2.waitKey(1000)
	
	
    


    
# for x in range(100):
# 	cv2.imshow(str(x), cl1)
# 	cv2.waitKey(1000)
# 	clahe = cv2.createCLAHE(clipLimit=0.0001)
# 	cl1 = clahe.apply(gray)
    
# 	cv2.setWindowTitle(str(x-1), str(x))

dialated = cv2.dilate(Canny, None, iterations=1)

# show_images([edged])
erode = cv2.erode(dialated, None, iterations=1)
erode = dialated
# show_images([edged])

#show_images([blur, edged])

# Find contours
cnts = cv2.findContours(erode.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# Sort contours from left to right as leftmost contour is reference object
(cnts, _) = contours.sort_contours(cnts)

# Remove contours which are not large enough
cnts = [x for x in cnts if cv2.contourArea(x) > 100]

#cv2.drawContours(image, cnts, -1, (0,255,0), 3)

#show_images([image, edged])
#print(len(cnts))

# Reference object dimensions
# Here for reference I have used a 2cm x 2cm square
ref_object = cnts[0]
box = cv2.minAreaRect(ref_object)
box = cv2.boxPoints(box)
box = np.array(box, dtype="int")
box = perspective.order_points(box)
(tl, tr, br, bl) = box
dist_in_pixel = euclidean(tl, tr)
dist_in_cm = 2
pixel_per_cm = dist_in_pixel/dist_in_cm

# Draw remaining contours
# for cnt in cnts:
	
# 	box = cv2.minAreaRect(cnt)
# 	box = cv2.boxPoints(box)
# 	box = np.array(box, dtype="int")
# 	box = perspective.order_points(box)
# 	(tl, tr, br, bl) = box
# 	cv2.drawContours(image, [box.astype("int")], -1, (0, 0, 255), 2)
# 	mid_pt_horizontal = (tl[0] + int(abs(tr[0] - tl[0])/2), tl[1] + int(abs(tr[1] - tl[1])/2))
# 	mid_pt_verticle = (tr[0] + int(abs(tr[0] - br[0])/2), tr[1] + int(abs(tr[1] - br[1])/2))
# 	wid = euclidean(tl, tr)/pixel_per_cm
# 	ht = euclidean(tr, br)/pixel_per_cm
# 	cv2.putText(image, "{:.1f}cm".format(wid), (int(mid_pt_horizontal[0] - 15), int(mid_pt_horizontal[1] - 10)), 
# 		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
# 	cv2.putText(image, "{:.1f}cm".format(ht), (int(mid_pt_verticle[0] + 10), int(mid_pt_verticle[1])), 
# 		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
	# show_images([image])
for cnt in cnts:
   x1,y1 = cnt[0][0]
   approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True) , True) 
   if len(approx) == 4:
      x, y, w, h = cv2.boundingRect(cnt)
      ratio = float(w)/h
      if ratio >= 0.9 and ratio <= 1.1:
         image = cv2.drawContours(image, [cnt], -1, (0,255,255), 3)
         cv2.putText(image, 'Square', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
      else:
         cv2.putText(image, 'Rectangle', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
         image = cv2.drawContours(image, [cnt], -1, (0,255,0), 3)

show_all_images()
