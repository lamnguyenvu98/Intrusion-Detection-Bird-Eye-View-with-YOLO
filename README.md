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

- Use left click to mark points on the image and press "a" to append selected point to a list.
- After setting 4 points for calibration, appilication automatically switch to next part where you have to provide region of instrusion area. When you've finished selecting points for intrusion area, press "c" to complete
- All points, that were set, are stored in a json file ("points.json" by default) 
- Press 'q' to quit and no result is saved

### 2. Calibration bird's eye view transformation
```
python calibration.py --video videos/virat.mp4 --points configs/points.json --config configs/data.json
```
- Use trackbar window to calibrate bird's eye view
- Values, that are selected by trackbar, are off-set value for destination image (image are transformed to bird's eye view)
- After finishing calibration, press "s" to save value to "data.json" file
- Press 'q' to quit and no result is saved

### 3. Run program
```
 python intrusion.py --video videos/virat.mp4 --config configs/data.json
```

This project was inspired by MiAI

*InstrusionWarning.pdf* was taken from MiAI's repo: https://github.com/thangnch/MiAI_Intrusion_Warning<br>

Facebook: http://facebook.com/pep.oversesa<br>




