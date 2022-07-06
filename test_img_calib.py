import cv2
import numpy as np
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", "--i", type=str, default='videos/virat.mp4', help="Video Input")
parser.add_argument("--json", "--f", type=str, default='points.json', help="Json file contains all points")
args = parser.parse_args()

cap = cv2.VideoCapture(args['input'])

_, img = cap.read()

h, w  = img.shape[:2]

# load json
f = open(args['json'], 'r')
metadata = json.load(f)

src = np.float32(metadata['calib_points'])
intrusion_points = np.array(metadata['intrusion_points'])

points = src.reshape((-1,1,2)).astype(np.int32)
cv2.polylines(img, [points], True, (0,255,0), thickness=4)

dst = np.float32([(0.2, 0.79), 
                  (0.6, 0.79), 
                  (0.6, 0.87), 
                  (0.2, 0.87)])

dst_size = (h, w)
dst = dst * np.float32(dst_size)

print(dst)

H_matrix = cv2.getPerspectiveTransform(src, dst)
points = intrusion_points.reshape((-1,1,2)).astype(np.int32) 
img = cv2.polylines(img, [points], True, (0,0,255), thickness=2)
warped = cv2.warpPerspective(img, H_matrix, dst_size)

warped = cv2.resize(warped, (800, 600))

cv2.imshow('Og', img)
cv2.imshow('warped', warped)
cv2.waitKey(0)

cv2.destroyAllWindows()
