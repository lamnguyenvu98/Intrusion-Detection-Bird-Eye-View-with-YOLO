import cv2
import numpy as np
import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--video", "--v", type=str, default='videos/virat.mp4', help="Video Input")
parser.add_argument("--points", "--p", type=str, default='configs/points.json', help="Json file that will store calibration points and intrusion points")
args = vars(parser.parse_args())

ix, iy = 0, 0


def draw_circle(event, x, y, flags, param):
    global ix, iy
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 4, (255, 0, 0), -1)
        ix, iy = x, y


def get_points(image, numOfPoints=None, image_size=(800, 600)):
    global img
    img = image.copy()
    saved = True
    # img = cv2.resize(img, image_size)
    width, height = image.shape[:2]
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", draw_circle)
    points = []
    queue_images = []
    while True:
        img = cv2.polylines(img, [np.int32(points)], False, (255,0, 0), thickness=2)
        cv2.imshow("image", img)
        k = cv2.waitKey(1)
        
        # Append point to list
        if k == ord('a'):
            points.append([int(ix), int(iy)])
            img = cv2.circle(img, (ix, iy), 3, (0, 0, 255), -1)
            queue_images.append(img)
                
        # Quit and don't save result
        if k == ord('q'):
            saved = False
            break
        
        # Reset
        if k == ord('r'):
            img = image.copy()
            points = []
        
        if numOfPoints is not None:
            text = "Enter {} points for calib area: ".format(numOfPoints)
            if len(points) == numOfPoints:
                break
        else:
            text = "Enter points for intrusion area".format(len(points))
            # Complete
            if k == ord('c'):
                break
            # Quit and don't save result
            if k == ord('q'):
                saved = False
                break
        
                 
        cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "Press a to save points", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 1, cv2.LINE_AA)
    
    cv2.destroyAllWindows()
    
    return points, saved


def save_json(filename, save_file, is_video=True):
    if is_video:
        cap = cv2.VideoCapture(filename)
        _, img = cap.read()
    else:
        img = cv2.imread(filename)
              
    metadata = {}
    metadata['calib_points'], saved = get_points(img, numOfPoints=4)
    if saved:
        metadata['intrusion_points'], saved = get_points(img, numOfPoints=None)
        if saved:
            out_file = open(save_file, "w")
            json.dump(metadata, out_file, indent = 4)
            out_file.close()
    

if __name__ == '__main__':
    save_json(filename=args['video'], save_file=args['points'])