# Intrusion Warning with Bird's Eye View Transformation using OpenCV and YOLO
Demo of Intrusion Warning with OpenCV & Yolo

![Alt Text](results/res.gif)

## Installing package
```
pip install opencv-python shapely numpy
```
or
```
pip install -r requirements.txt
```
## How to run
### 1. Set 4 points for calibration bird's eye view and instrusion area
```
python set_points.py --video videos/virat.mp4 --points configs/points.json
```

- Use left-click to select 4 calibration points on the a sample image and press "a" to append selected point to a list.
- All points are stored in a json file ("points.json" by default) 
- If you want to escape the process and don't want to save current seleted points, press "q".
- After setting 4 points for calibration, appilication automatically switch to next part where you have to provide region of instrusion area. When you've finished selecting points for intrusion area, press "c" to complete the process and save calibration result.

### 2. Calibration bird's eye view transformation
```
python calibration.py --video videos/virat.mp4 --points configs/points.json --config configs/data.json
```
- Use trackbars to calibrate bird's eye view.
- Values, that are selected by trackbar, are off-set value for destination image (image are transformed to bird's eye view)
- After finishing calibration, press "s" to save value to "data.json" file
- If you want to escape the process and don't want to save current seleted points, press "q".

### 3. Run program
```
 python intrusion.py --video videos/virat.mp4 --config configs/data.json
```




