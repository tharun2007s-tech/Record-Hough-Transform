# Record-Hough-Transform
##  Lane Detection

##  Aim

To implement a basic lane detection pipeline using OpenCV by completing missing code segments at specified locations.

## Learning Objective

* Understand each stage of image processing
* Learn how to build a complete computer vision pipeline
* Practice writing code in guided sections

##  Software Used

* Anaconda – Python 3.7
* Jupyter Notebook / VS Code
* OpenCV (cv2)
* NumPy
* Matplotlib


##  Algorithm & Explanation

1.Import OpenCV, NumPy and Matplotlib libraries and read the input image.

2.Convert the image into grayscale and apply thresholding.

3.Define the Region of Interest (ROI) and apply Canny edge detection.

4.Apply Gaussian Blur and Hough Transform to detect lane lines.

5.Draw the detected lane lines on the original image and display the final output.

## Program

~~~python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
img = cv2.imread('lane.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Display images
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(img)
plt.title('Original Image')

plt.subplot(1,2,2)
plt.imshow(gray, cmap='gray')
plt.title('Grayscale Image')
plt.show()

# Apply thresholding
threshold = 150
_, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

# Region of Interest
height, width = gray.shape
mask = np.zeros_like(gray)
polygon = np.array([[
    (0, height),
    (width, height),
    (width, height//2),
    (0, height//2)
]])
cv2.fillPoly(mask, polygon, 255)
roi = cv2.bitwise_and(thresh, mask)

# Edge detection
edges = cv2.Canny(roi, 50, 150)

# Gaussian Blur
blur = cv2.GaussianBlur(edges, (5,5), 0)

# Hough Transform
lines = cv2.HoughLinesP(
    blur, 1, np.pi/180, 50,
    minLineLength=50, maxLineGap=100
)

# Lane detection logic
line_img = np.zeros_like(img)

if lines is not None:
    lines = np.array(lines).reshape(-1, 4)

    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(line_img, (x1,y1), (x2,y2), (255,0,0), 5)

output = cv2.addWeighted(img, 0.8, line_img, 1, 0)

# Display all outputs
plt.figure(figsize=(12,8))

images = [img, gray, thresh, roi, edges, blur, line_img, output]
titles = ['Original', 'Grayscale', 'Threshold',
          'ROI', 'Edges', 'Blur', 'Detected Lines', 'Final Output']

for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(images[i], cmap='gray' if i in [1,2,3,4,5] else None)
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
~~~

##  Expected Output

* Original image
* Grayscale image
* Thresholded image
* ROI masked image
* Edge detected image
* Smoothed image
* Detected lines
* Final lane detection output

## Output

<img width="1642" height="345" alt="image" src="https://github.com/user-attachments/assets/098568d3-2c06-4baf-a9a4-47b6a1a10345" />
<img width="1688" height="360" alt="image" src="https://github.com/user-attachments/assets/bd3866c1-5131-4a12-992a-f8f1667eed16" />


## Result

Thus, the lane detection pipeline is successfully implemented by completing the missing code sections. The system detects and highlights lane lines effectively.

---

##  Developed By

* **Name:** Tharun S
* **Register No:** 212225240174
