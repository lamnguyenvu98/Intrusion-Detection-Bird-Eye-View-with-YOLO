import cv2
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

def isInside(points, centroid):
    polygon = Polygon(points)
    centroid = Point(centroid)
    return polygon.contains(centroid)

def point_to_new_perspective(matrix, list_points):
    list_points_to_detect = np.float32(list_points).reshape(-1, 1, 2)
    transformed_points = cv2.perspectiveTransform(list_points_to_detect, matrix)
    transformed_points_list = list()
    for i in range(0, transformed_points.shape[0]):
        transformed_points_list.append([transformed_points[i][0][0], transformed_points[i][0][1]])
    return np.array(transformed_points_list).astype('int')

def compute_point_perspective_transformation(matrix, boxes):
    list_downoids = [[box[4], box[5] + box[3] // 2] for box in boxes]
    list_points_to_detect = np.float32(list_downoids).reshape(-1, 1, 2)
    transformed_points = cv2.perspectiveTransform(list_points_to_detect, matrix)
    transformed_points_list = list()
    for i in range(0, transformed_points.shape[0]):
        transformed_points_list.append([transformed_points[i][0][0], transformed_points[i][0][1]])
    return np.array(transformed_points_list).astype('int')

def get_red_green_boxes(intrusion_points, birds_eye_points, boxes):
    green_boxes = []
    red_boxes = []
    
    for i in range(birds_eye_points.shape[0]):
        if isInside(intrusion_points, birds_eye_points[i]):
            red_boxes.append(boxes[i] + birds_eye_points[i].tolist())
        else:
            green_boxes.append(boxes[i] + birds_eye_points[i].tolist())
    
    return red_boxes, green_boxes

def get_birds_eye_view_image(green_box, red_box, intrusion_eye_view_area, eye_view_height, eye_view_width, background_image):
    # blank_image = image
    blank_image = background_image.copy()
    cv2.putText(blank_image, "Instrusion: " + str(len(red_box)), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
    # cv2.putText(blank_image, "No-Instrusion: " + str(len(green_box)), (400, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
    for point in green_box:
        cv2.circle(blank_image, tuple([point[6], point[7]]), 20, (0, 255, 0), -5)
    for point in red_box:
        cv2.circle(blank_image, tuple([point[6], point[7]]), 20, (0, 0, 255), -5)
    # blank_image = cv2.polylines(blank_image, [intrusion_eye_view_area], True, (255, 0, 0), 3)
    # blank_image = cv2.resize(blank_image, (eye_view_width, eye_view_height))
    return blank_image

def get_red_green_box_image(new_box_image, green_box, red_box):
    for point in green_box:
        cv2.rectangle(new_box_image, (point[0], point[1]), (point[0] + point[2], point[1] + point[3]), (0, 255, 0), 2)
    for point in red_box:
        cv2.rectangle(new_box_image, (point[0], point[1]), (point[0] + point[2], point[1] + point[3]), (0, 0, 255), 2)
    return new_box_image