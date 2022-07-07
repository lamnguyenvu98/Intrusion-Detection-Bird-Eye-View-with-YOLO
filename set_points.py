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
preview = None

def draw_polygon(img, points):
    arr_points = np.int32(points).reshape((-1, 1, 2))
    img = cv2.polylines(img, [arr_points], True, (255,0, 0), thickness=2)
    return img

def draw_circle(event, x, y, flags, param):
    global ix, iy
    global preview
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 4, (255, 0, 0), -1)
        preview = img.copy()
        ix, iy = x, y
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if preview is not None:
            preview = img.copy()
            if len(param) > 0:
                curr_x, curr_y = param[-1]
                cv2.line(preview, (curr_x, curr_y), (x, y), (0, 0, 255), 2)

def get_points(image, numOfPoints=None, image_size=(800, 600)):
    global img
    colors = (255, 0, 0) if numOfPoints is None else (0, 255, 0)
    points = []
    img = image.copy()
    saved = True
    # img = cv2.resize(img, image_size)
    width, height = image.shape[:2]
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", draw_circle, points)
    while True:            
        img = cv2.polylines(img, [np.int32(points)], False, colors, thickness=2)
        if preview is None:
            cv2.imshow("image", img)
        else:
            cv2.imshow("image", preview)
        k = cv2.waitKey(1)
        
        # Append point to list
        if k == ord('a'):
            points.append([int(ix), int(iy)])
            img = cv2.circle(img, (ix, iy), 3, (0, 0, 255), -1)
                
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
    metadata['intrusion_points'], saved = get_points(img, numOfPoints=None)
    img = draw_polygon(img.copy(), metadata['intrusion_points'])
    if saved:
        metadata['calib_points'], saved = get_points(img, numOfPoints=4)
        if saved:
            out_file = open(save_file, "w")
            json.dump(metadata, out_file, indent = 4)
            out_file.close()
    

if __name__ == '__main__':
    save_json(filename=args['video'], save_file=args['points'])