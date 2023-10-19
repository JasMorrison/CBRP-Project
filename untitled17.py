# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 10::18 2023

@author: Morrison
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt 
import pandas as pd 
import csv


#Applying the circles to the Green Channel 

image = cv2.imread("")
blur = cv2.GaussianBlur(image, (5,5), 0)
RGBImage = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
gray_image = cv2.cvtColor(RGBImage, cv2.COLOR_RGB2GRAY)

intensities = gray_image[gray_image > 32]
circles = []
for intensity in np.unique(intensities):
    _, binary_mask = cv2.threshold(gray_image, intensity, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circle_area = 20
    radius = int(np.sqrt(circle_area/np.pi))
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area > 0:
            M = cv2.moments(contour)
            center_X = int(M["m10"]/M["m00"])
            center_Y = int(M["m01"]/M["m00"])
            new_circle = ((center_X, center_Y), radius)
            overlap = False
            for circle in circles:
                center, prev_radius = circle
                distance = np.sqrt((center_X - center[0]) ** 2 + (center_Y - center[1])**2)
                if distance < radius + prev_radius: 
                    overlap = True
                    break
            if not overlap:
                circles.append(new_circle)
        
image_with_circles = image.copy()
for circle in circles:
    center, radius = circle 
    cv2.circle(image_with_circles, center, radius, (0, 0, 255), thickness = cv2.FILLED)
    

#cv2.imshow("Points of highest intensity", image_with_circles)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('', image_with_circles)

#Creating the contour from the blue channel

image1 = cv2.imread("")
npimage = np.array(image1)

blur = cv2.GaussianBlur(image1, (5,5),0)
gray_image = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY) 
ret1, th1 = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

npimage2 = np.array(th1)

_, threshold = cv2.threshold(npimage2, 0, 25/5, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


#Thresholding for selection of contours 


contours, hierarchy1 = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
min_area_threshold = 0
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area_threshold]

#For all contours 
#filtered_contours = contours 


image2 = cv2.imread("")
contour_image = image2.copy()
cv2.drawContours(contour_image, filtered_contours, -1, (0, 255, 0), thickness=2)


cv2.imshow("Contoured green channel w highest intensity points", contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('', contour_image)

#Isolate the Contours

total_area = 0 
individual_areas = []
for contour in filtered_contours:
    area = cv2.contourArea(contour)
    total_area += area
    individual_areas.append(area)

print("Individual areas inside contours", individual_areas)


#Count the number of circles within each filtered contour
image_with_circles = image.copy()
for contour in filtered_contours:
    contour_circles = 0
    for circle in circles:
        center, radius = circle
        if cv2.pointPolygonTest(contour, center, False) > 0:
            cv2.circle(image_with_circles, center, radius, (0, 0, 255), thickness=cv2.FILLED)
            contour_circles += 1
    print(contour_circles)
    
#cv2.imshow("Points of highest intensity", image_with_circles)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('', image_with_circles)

#Plotting
"""
x = individual_areas
y = contour_circles

figure = plt.figure(figsize = (10, 5))
plt.bar(x, y, color = 'blue', width = 0.4)
plt.xlabel("Areas of nuclei")
plt.ylabel("No.of present foci")
plt.title("Number of gamma illuminated foci present in irradiated epithiliod nuclei")
plt.show()
"""

output_path = ''
with open(output_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Contour_Area', 'Number_of_Circles'])
    for contour in filtered_contours:
        contour_area = cv2.contourArea(contour)
        num_circles = sum(1 for circle in circles if cv2.pointPolygonTest(contour, circle[0], False) >= 0)
        csv_writer.writerow([contour_area, num_circles])

print("Circles information saved to:", output_path)

#Labelling the contours with their respective areas (useful for comparing to manual count)

contour_image_with_text = contour_image.copy()
for i, contour in enumerate(filtered_contours):
    if cv2.contourArea(contour) > 0:
        cv2.drawContours(contour_image_with_text, [contour], -1, (0, 255, 0), thickness=2)
        M = cv2.moments(contour)
        center_X = int(M["m10"] / M["m00"])
        center_Y = int(M["m01"] / M["m00"])
        contour_area = cv2.contourArea(contour)
        cv2.putText(
            contour_image_with_text,
            f"Area: {contour_area:.2f}",
            (center_X, center_Y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

# Display the image with contour areas
cv2.imshow("", contour_image_with_text)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(
    "",
    contour_image_with_text,
)

"""
#Test for number of circles not counted

detected_circles = []
circle_image = contour_image.copy()
for contour in filtered_contours:
    mask = np.zeros_like(circle_image)
    cv2.drawContours(mask, contour, 0, 255, thickness = cv2.FILLED)
    roi = cv2.bitwise_and(circle_image, circle_image, mask)
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    circles2 = cv2.HoughCircles(roi_gray, cv2.HOUGH_GRADIENT, dp = 1, minDist = 20, param1 = 50, param2 = 30, minRadius = radius, maxRadius = radius)
    if circles2 is not None:
        circles2 = np.unit16(np.around(circles2))
        for pt in circles2[0, :]:
            a, b, r = pt[0], pt[1], pt[2]


cv2.imshow("Detected Circle", circle_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

if detected_circles:
    for (x, y, r) in detected_circles:
        cv2.circle(contour_image, (x, y), r, (0, 0, 255), thickness=2)

cv2.imshow("Recognised Circles", contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
num_circles = len(detected_circles)
print(f"Total number of circles detected: {num_circles}")
"""
