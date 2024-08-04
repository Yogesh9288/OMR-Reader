import cv2  # type: ignore
import numpy as np # type: ignore
import util

# Constants
imgWidth = 700
imgHeight = 700
questions = 5
choices = 5
ans = [1, 2, 3, 1, 3]  # Correct answers

# Load and Preprocess the Image
img = cv2.imread("opencv//assets//omr3.jpg")
img = cv2.resize(img, (imgWidth, imgHeight)) 

# Create copies of the image for various steps
imgContours = img.copy()
imgBigContours = img.copy()
imgFinal = img.copy()

# Convert to grayscale and apply Gaussian Blur
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)

# Detect edges using Canny edge detector
imgCanny = cv2.Canny(imgBlur, 10, 50)

# Find all contours
countors, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContours, countors, -1, (0, 255, 0), 10)

# Find rectangular contours
rectCont = util.rectCounter(countors)
biggestContour = util.getCornerPoints(rectCont[0])
gradePoints = util.getCornerPoints(rectCont[1])

if biggestContour.size != 0 and gradePoints.size != 0:
    # Draw the contours
    cv2.drawContours(imgBigContours, biggestContour, -1, (0, 255, 0), 20)
    cv2.drawContours(imgBigContours, gradePoints, -1, (255, 0, 0), 20)

    # Reorder the corner points
    biggestContour = util.reorder(biggestContour)
    gradePoints = util.reorder(gradePoints)

    # Transform perspective to obtain a top-down view of the OMR sheet
    pt1 = np.float32(biggestContour)
    pt2 = np.float32([[0, 0], [imgWidth, 0], [0, imgHeight], [imgWidth, imgHeight]])
    matrix = cv2.getPerspectiveTransform(pt1, pt2)
    imageColored = cv2.warpPerspective(img, matrix, (imgWidth, imgHeight))

    # Transform perspective for the grading area
    ptg1 = np.float32(gradePoints)
    ptg2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
    matrixg = cv2.getPerspectiveTransform(ptg1, ptg2)
    imageGrade = cv2.warpPerspective(img, matrixg, (325, 150))

# Apply thresholding
imgWarpGray = cv2.cvtColor(imageColored, cv2.COLOR_BGR2GRAY)
imgThresh = cv2.threshold(imgWarpGray, 150, 255, cv2.THRESH_BINARY_INV)[1]

# Split the thresholded image into individual boxes
boxes = util.splitBoxes(imgThresh)

# Count non-zero pixels in each box
myPixelVal = np.zeros((questions, choices))
countC = 0
countR = 0

for imgp in boxes:
    totalPixels = cv2.countNonZero(imgp)
    myPixelVal[countR][countC] = totalPixels
    countC += 1
    if countC == choices:
        countR += 1
        countC = 0

# Find index of marked bubbles
myIndex = []

for x in range(questions):
    arr = myPixelVal[x]
    myIndexVal = np.where(arr == np.amax(arr))
    myIndex.append(myIndexVal[0][0])

# Grade the responses
grading = []

for x in range(questions):
    if ans[x] == myIndex[x]:
        grading.append(1)
    else:
        grading.append(0)

# Calculate the score
score = (sum(grading) / questions) * 100
print(score)

# Display the results
imgResult = imageColored.copy()
imgResult = util.showAnswers(imgResult, myIndex, grading, ans, questions, choices)

# Draw the answers and grading on a blank image
imgDrawing = np.zeros_like(imageColored)
imgDrawing = util.showAnswers(imgDrawing, myIndex, grading, ans, questions, choices)

# Transform perspective back to original
inversematrix = cv2.getPerspectiveTransform(pt2, pt1)
imageInverse = cv2.warpPerspective(imgDrawing, inversematrix, (imgWidth, imgHeight))
imgFinal = cv2.addWeighted(imgFinal, 1, imageInverse, 1, 0)

# Add the score to the grading area
imgRawGrade = np.zeros_like(imageGrade)
cv2.putText(imgRawGrade, str(int(score)) + "%", (60, 100), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 255), 3)
inversematrixg = cv2.getPerspectiveTransform(ptg2, ptg1)
imageInverseG = cv2.warpPerspective(imgRawGrade, inversematrixg, (imgWidth, imgHeight))
imgFinal = cv2.addWeighted(imgFinal, 1, imageInverseG, 1, 0)

# Stack images for display
imgBlank = np.zeros_like(img)
imgArray = ([img, imgGray, imgBlur, imgCanny],
            [imgContours, imgBigContours, imageColored, imgThresh],
            [imgResult, imgDrawing, imageInverse, imgFinal])
labels = [['original', 'Gray', 'Blur', 'canny'],
          ['contors', 'biggest con', 'warp', 'thresh'],
          ['result', 'raw drawing', 'inv warp', 'img final']]
imageStack = util.stackImages(imgArray, 0.3, labels)

cv2.imwrite("opencv//assets//final.jpg",imgFinal)
# Display the final images
cv2.imshow('final', imgFinal)
cv2.imshow('omr', imageStack)
cv2.waitKey(0)
