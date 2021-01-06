from pyimagesearch.shapedetector import ShapeDetector
import argparse
import imutils
import cv2

'''
!!IMPORTANT!!

In order to activate this thing on an image, input the terminal with
python detect_shapes.py --image Mill_10666.tif and press SPACE (or any key)
in order to advance the script
'''

light_orange = (0, 0, 20)
dark_orange = (200, 255, 255)

# CREATING A SCATTER PLOT

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to the input image")
args = vars(ap.parse_args())

# load the image and resize it to a smaller factor so that
# the shapes can be approximated better
image = cv2.imread(args["image"])

hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)  # converting the color of the image
mask = cv2.inRange(hsv_image, light_orange, dark_orange)
result = cv2.bitwise_and(image, image, mask=mask)

result = imutils.resize(result, width=1000)
ratio = result.shape[0] / float(result.shape[0])

# convert the resized image to grayscale, blur it slightly,
# and threshold it
gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

# find contours in the threshold image and initialize the
# shape detector
contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
sd = ShapeDetector()

# loop over the contours
for c in contours:
	# compute the center of the contour, then detect the name of the
	# shape using only the contour
	M = cv2.moments(c)
	cX = int((M["m10"] / (M["m00"] + 1e-7)) * ratio)  # added 1e-7 to the calculation so to not multiply by 0
	cY = int((M["m01"] / (M["m00"] + 1e-7)) * ratio)
	shape = sd.detect(c)

	# multiply the contour (x, y)-coordinates by the resize ratio,
	# then draw the contours and the name of the shape on the image
	c = c.astype("float")
	c *= ratio
	c = c.astype("int")
	cv2.drawContours(result, [c], -1, (0, 255, 0), 2)
	cv2.putText(result, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
	            0.5, (255, 255, 255), 2)

	# show the output image
	cv2.imshow("Image", result)
	cv2.waitKey(0)
