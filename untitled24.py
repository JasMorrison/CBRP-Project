# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 13:48:58 2023

@author: Morrison
"""

import numpy as np
import cv2
import csv
import glob 
import os

CentersOfAllCircles = [] 
LimitedAreasContours = []
CirclePositions = {}
CirclesbyLayer = {}

image_path1 = glob.glob('blue channel')
image_path2 = glob.glob('green channel')


for path1, path2 in zip(image_path1, image_path2):
    image1 = cv2.imread(path1) 
    blur1 = cv2.GaussianBlur(image1, (5, 5), 0)
    RGBImage1 = cv2.cvtColor(blur1, cv2.COLOR_BGR2RGB)
    gray_image1 = cv2.cvtColor(RGBImage1, cv2.COLOR_RGB2GRAY)
    _, threshold1 = cv2.threshold(gray_image1, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, threshold2 = cv2.threshold(threshold1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours1, _ = cv2.findContours(threshold2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    minimum_area = 0
    LimitedAreasContours = [cnt for cnt in contours1 if cv2.contourArea(cnt) >= minimum_area]
    
    image2 = cv2.imread(path2)
    blur2 = cv2.GaussianBlur(image2, (5, 5), 0)
    RGBImage2 = cv2.cvtColor(blur2, cv2.COLOR_BGR2RGB)
    gray_image2 = cv2.cvtColor(RGBImage2, cv2.COLOR_RGB2GRAY)
    intensities = gray_image2[gray_image2 > 32]
    circles = []
    for intensity in np.unique(intensities):
        _, binary_mask = cv2.threshold(gray_image2, intensity, 255, cv2.THRESH_BINARY)
        contours2, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        AreaofCircle = 20
        radius = int(np.sqrt(AreaofCircle/np.pi))
        for contour in contours2:
            AreaofContour = cv2.contourArea(contour)
            if AreaofContour > 0:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    center_1 = int(M["m10"]/M["m00"])
                    center_2 = int(M["m01"]/M["m00"])
                    CircleN = ((center_1, center_2), radius)
                    overlap = False
                    for CenterofCircle in CentersOfAllCircles:
                        distance = np.sqrt((center_1 - CenterofCircle[0]) ** 2 + (center_2 - CenterofCircle[1]) ** 2)
                        if distance < 2 * radius:
                            overlap = True
                            break
                    if not overlap:
                        circles.append(CircleN)
                        CentersOfAllCircles.append((center_1, center_2))
    
    
    GreenChannelwCircles = image2.copy()
    for circle in circles:
        center, radius = circle 
        cv2.circle(GreenChannelwCircles, center, radius, (0, 0, 255), thickness = cv2.FILLED)
        
DestinationFolder = ''
os.makedirs(DestinationFolder, exist_ok=True)
PathforCircleCount = os.path.join(DestinationFolder, f"output_all_images 300823.csv")

with open(PathforCircleCount, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Contour_Area', 'Number_of_Circles'])
    
    for contour, circles in zip(LimitedAreasContours, CirclesbyLayer.values()):
        AreaofContour = cv2.contourArea(contour)
        NumberofCircles = sum(1 for circle in circles if cv2.pointPolygonTest(contour, circle[0], False) >= 0)
        csv_writer.writerow([AreaofContour, NumberofCircles])


def imageprocess(image):
    BlurredImage2 = cv2.GaussianBlur(image, (5, 5), 0)
    RGBImage = cv2.cvtColor(BlurredImage2, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(RGBImage, cv2.COLOR_RGB2GRAY)
    restricted2, threshold3 = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return threshold3

processed_images = [imageprocess(cv2.imread(path)) for path in image_path1]


contours_list = [cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] for img in processed_images]
area_list = [] 

for contours in contours_list:
    areas = [cv2.contourArea(contour) for contour in contours]
    area_list.append(areas)

proximity_threshold = 50
centroid_list_per_image = []
for contours in contours_list:
    centroids = []
    for contour in contours:
       
        moments = cv2.moments(contour)
        
        
        if moments['m00'] != 0:
            centroid_x = moments['m10'] / moments['m00']
            centroid_y = moments['m01'] / moments['m00']
            centroids.append([centroid_x, centroid_y])
    centroid_list_per_image.append(centroids)

grouped_contours = {}


for i, centroids in enumerate(centroid_list_per_image):
    for idx, centroid in enumerate(centroids):
        min_distance = proximity_threshold
        closest_idx = -1
        for j, other_centroids in enumerate(centroid_list_per_image):
            if i != j:
                for other_idx, other_centroid in enumerate(other_centroids):
                    distance = np.linalg.norm(np.array(centroid) - np.array(other_centroid))
                    if distance < min_distance:
                        min_distance = distance
                        closest_idx = other_idx

        if closest_idx != -1:  
            if closest_idx != i:  
                num_circles = sum(1 for circle in circles if cv2.pointPolygonTest(contours[idx], circle[0], False) >= 0)

                
                if closest_idx not in grouped_contours:
                    grouped_contours[closest_idx] = {"areas": [], "circles": 0}
                grouped_contours[closest_idx]["areas"].append(area_list[i][idx])
                grouped_contours[closest_idx]["circles"] += num_circles

for i, centroids in enumerate(centroid_list_per_image):
    for idx, centroid in enumerate(centroids):
        if idx not in grouped_contours and idx != i:
            grouped_contours[idx] = [area_list[i][idx]]


output_directory = ""
csv_filename = os.path.join(output_directory, "")
with open(csv_filename, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Group", "Contour Area", "Number of Circles"])
    
    for group_idx, data in grouped_contours.items():
        for area, num_circles in zip(data["areas"], data["circles"]):
            csv_writer.writerow([group_idx, area, num_circles])

print(f"Linked contours saved to {csv_filename}")


