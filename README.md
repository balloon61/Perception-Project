# Perception-Project
### Description:
This repository includes several basic perception project
### Requirement:
numpy, cv2, matplotlib

# AR Tag Detection:
### Description:
Detect AR tag in any direction.

### Result:
![AR_detect_AdobeExpress (1)](https://user-images.githubusercontent.com/55338365/198097203-46b5d08d-dc20-454f-ac1c-fef83a75893d.gif)

Video 1. AR Tag detection


## Coin Seperation
### Description:

Calculate how many coins in this image.
### Result:
Image processing is shown as follows, and the number of coins will be published in your terminal when you executed the code.
![Q1image](https://user-images.githubusercontent.com/55338365/198095332-982d5242-2c4b-4faf-b03b-ec831095093b.png)![Screenshot from 2022-10-26 13-23-55](https://user-images.githubusercontent.com/55338365/198095670-84c265ac-78ff-4614-a638-a62f74dee052.png)

Fig 1. (Left) Preprocessing image (Right) After processing image



## Historgram Equalization
### Description:

Given a sequence of image and do Historgram Equalization and Adaptive Historgram Equalization to each images.

### Result:

![0000000000](https://user-images.githubusercontent.com/55338365/198097957-109ed8ca-8915-4f94-afdf-823aedb983f9.png)
Fig 2. Original image
![output00](https://user-images.githubusercontent.com/55338365/198097987-0e01a798-9b18-4807-878f-6b7f360a0270.png)
Fig 3. Historgram Equalization (top) original (mid) Historgram Equalization (bot) Adaptive Historgram Equalization

## Stereo Vision 
### Description:
implement the concept of Stereo Vision. given several different datasets, each of them contains 2 images of the same scenario but
taken from two different camera angles. By comparing the information about a scene from 2 vantage points, we can obtain the 3D information by examining the relative positions of objects.

### Pipeline for creating a Stereo Vision System:
1. Calibration: Get the F (fundamental matrix), K (intrinsic matrix), and E (essential matrix)
2. Rectification: Apply the perspective transformation to make sure that the epipolar lines are horizontal for both images
3. Correspondence: Get the disparity image which is the image that given where each pixels gives the disparity of the 3D points.
4. Compute Depth Image: Directly use the disparity matrix to calculate the depth image.

### Results:
### Rectification:


![Screenshot from 2022-10-26 13-51-10](https://user-images.githubusercontent.com/55338365/198101309-44edcc94-1f52-437a-becf-470e98b8dace.png)
Fig 4. Corresponding Eplliplolar line

![Screenshot from 2022-10-26 13-50-59](https://user-images.githubusercontent.com/55338365/198101316-db664991-d101-4463-90cf-0cd7ae5a8274.png)

Fig 5. Rectified
### Correspondence:
![Screenshot from 2022-10-26 13-50-27](https://user-images.githubusercontent.com/55338365/198100835-40d103f6-0735-42e4-849f-31c55b843493.png)

Fig 6. Disparity


### Depth Image:
![Screenshot from 2022-10-26 13-52-40](https://user-images.githubusercontent.com/55338365/198101273-1ce7a01d-bfa6-479d-ae8b-9e7716c25ccc.png)

Fig 7. Depth image


## Line Detection
### Description: 
Detect Lane in the highway.

![Lane_detection_AdobeExpress](https://user-images.githubusercontent.com/55338365/198093735-5015928b-d5dd-47e8-b88a-1b95e473738b.gif)

Video 1. Line Detection


# Turning Prediction
### Description:
Calculate the curvature of the lane in the highway and predict the steering direction.
### Results:
![turning_prediction_AdobeExpress (1)](https://user-images.githubusercontent.com/55338365/198093383-3f7910b5-3a58-44ab-a66c-7409b0dbcc66.gif)

Video 2. Turning Prediction







