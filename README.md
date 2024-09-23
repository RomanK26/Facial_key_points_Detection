# Facial_key_points_Detection 
Identifying keypoints on an Image

### To run first it is recommended to initialize and activate a virtual enviroinment

To run locally execute  below snippet in terminal
```
git clone https://github.com/RomanK26/Facial_key_points_Detection

```
```
python3 -m venv <env_name>
source <path to env name/bin/activate>
```
and then run

``` 
pip install -r requirements.txt
```

## Project structure
### |--- Dataset structure
Here, we have Images in **data/Training** and keypoints in **training_frames_keypoint.csv**  
The structure of csv file is 
```
Luis_Fonsi_21.jpg,45.0,98.0,47.0,106.0,49.0,110.0,53.0,119.0...
``` 
where first column denotes image name, and remaining as keypoints i.e
```
X1, Y1, X2, Y2, X3, Y3, ..., X68, Y68
```

## 

## Training and Test loss curve
![Alt text](https://github.com/RomanK26/Facial_key_points_Detection/blob/main/Saved/version2/train_curve.png)


## Inference
![Alt text](https://github.com/RomanK26/Facial_key_points_Detection/blob/main/Saved/version2/inference.png)




## TO DOs
- [ 1 ] update markdown file
- [ 2 ] add code to select best model(to prevent overfitting)  
- [ 3 ] add scheduler







