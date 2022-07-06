import cv2
import numpy as np
import json
from utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--video", "--v", type=str, default='videos/virat.mp4', help="Video Input")
parser.add_argument("--config", "--c", type=str, default='configs/data.json', help="Json file that contains calibration data")
args = vars(parser.parse_args())

video = cv2.VideoCapture(args['video'])

# load json
f = open(args['config'], 'r')
metadata = json.load(f)

src = np.float32(metadata['calib_points'])
intrusion_points = np.float32(metadata['intrusion_points'])
dst = np.float32(metadata['dst'])
dst_size = metadata['dst_size']

intrusion_points_new = intrusion_points.reshape((-1,1,2)).astype(np.int32)

dst = dst * np.float32(dst_size)

H_matrix = cv2.getPerspectiveTransform(src, dst)

# Model Init
class_names = []
with open("models/coco.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

net = cv2.dnn.readNet("models/yolov3.weights", "models/yolov3.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1 / 255)

confidence_threshold = 0.5
nms_threshold = 0.5

# Init video writer
frame_width = int(video.get(3))
frame_height = int(video.get(4))
fps = int(video.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("results/res.mp4", fourcc, fps, (1440, 540))

while True:
    ret, frame = video.read()
    if not ret:
        break
    image_height, image_width = frame.shape[:2]
    warped = cv2.warpPerspective(frame, H_matrix, dst_size)
    classes, _ , boxes = model.detect(frame, confidence_threshold, nms_threshold)
    list_boxes = []

    for (classid, box) in zip(classes, boxes):
        if class_names[classid] != 'person': continue
        x, y, w, h = box
        center_x, center_y = int(x+w/2), int(y+h/2)
        list_boxes.append([x, y, w, h, center_x, center_y])
        
    birds_eye_points = compute_point_perspective_transformation(H_matrix, list_boxes)
    
    instrusion_eye_points = point_to_new_perspective(H_matrix, intrusion_points.tolist())
    
    red_boxes, green_boxes = get_red_green_boxes(instrusion_eye_points, birds_eye_points, list_boxes)
    
    instrusion_eye_points = instrusion_eye_points.reshape((-1, 1, 2)).astype(np.int32)
    
    birds_eye_view_image = get_birds_eye_view_image(green_boxes, red_boxes, 
                                                    intrusion_eye_view_area=instrusion_eye_points,
                                                    eye_view_height=image_height,eye_view_width=image_width//2, background_image="background.png")
    
    # instrusion_eye_points = instrusion_eye_points.reshape((-1, 1, 2)).astype(np.int32)
    # cv2.polylines(birds_eye_view_image, [instrusion_eye_points], True, (0, 255, 0), thickness=4)
    
    cv2.polylines(frame, [intrusion_points_new], True, (255, 0, 0), thickness=2)
    
    box_red_green_image = get_red_green_box_image(frame.copy(), green_boxes, red_boxes)
    
    combined_image = np.concatenate((birds_eye_view_image, box_red_green_image), axis=1)
    resize_combined_image = cv2.resize(combined_image, (1440, 540))
    out.write(resize_combined_image)
    cv2.imshow("Social Distance", resize_combined_image)
    if cv2.waitKey(1) == ord('q'): break

video.release()
out.release()
cv2.destroyAllWindows()