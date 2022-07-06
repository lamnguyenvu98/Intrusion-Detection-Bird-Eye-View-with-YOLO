import cv2
import numpy as np
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--video", "--v", type=str, default='videos/virat.mp4', help="Video Input")
parser.add_argument("--points", "--p", type=str, default='configs/points.json', help="Json file that contains calibration points and intrusion points")
parser.add_argument('--config', "--c", type=str, default='configs/data.json', help="Json file that will store calibration data")
args = vars(parser.parse_args())

cap = cv2.VideoCapture(args['video'])

_, img = cap.read()

height, width = img.shape[:2]

def on_change(value):
    pass

cv2.namedWindow('trackbar')
cv2.resizeWindow('trackbar', 640, 240)
cv2.createTrackbar("Xtl", "trackbar", 20, 100, on_change)
cv2.createTrackbar("Xtr", "trackbar", 60, 100, on_change)
cv2.createTrackbar("Ytl", "trackbar", 79, 100, on_change)
cv2.createTrackbar("Ybr", "trackbar", 87, 100, on_change)

# load json
f = open(args['points'], 'r')
metadata = json.load(f)

src = np.float32(metadata['calib_points'])
intrusion_points = np.array(metadata['intrusion_points'])

dst_size = (800, 1080)

og_img = img.copy()

while True:
    points = src.reshape((-1,1,2)).astype(np.int32)
    img = cv2.polylines(og_img, [points], True, (0,255,0), thickness=4)
    
    xtl = cv2.getTrackbarPos("Xtl", "trackbar") / 100
    xtr = cv2.getTrackbarPos("Xtr", "trackbar") / 100
    ytl = cv2.getTrackbarPos("Ytl", "trackbar") / 100
    ybr = cv2.getTrackbarPos("Ybr", "trackbar") / 100
    
    dst_offset = np.float32([
        [xtl, ytl],
        [xtr, ytl],
        [xtr, ybr],
        [xtl, ybr]
    ])
         
    dst = dst_offset * np.float32(dst_size)
    
    H_matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(og_img, H_matrix, dst_size)
    
    warped = cv2.resize(warped, (300, 600))
    
    cv2.imshow('warped', warped)
    key = cv2.waitKey(1)
    
    if key == ord('s'):
        metadata.update({'dst': dst_offset.tolist(), 'dst_size': dst_size})
        out_file = open(args['config'], "w")
        json.dump(metadata, out_file, indent = 4)
        out_file.close()
        break
    
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()