from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
from git import Repo
import time
from yeelight import Bulb
import os

bulb = Bulb("172.20.10.3", auto_on = True)
bulb.set_rgb(0,255,0)
bulb.start_music(port=0)
NUM_TABLES = 0
TABLES = []
PTS = []
SELECTING = False
calibrated = False


class Table:
    def __init__(self):
		self.x = 0
		self.y = 0
		self.w = 0
		self.h = 0
		self.occupied = False

    def setLocation(self, newX, newY, newW, newH):
		self.x = newX
		self.y = newY
		self.w = newW
		self.h = newH

class Person:
    def __init__(self):
        self.x = 0
        self.y = 0

    def setLocation(self, newX, newY):
        self.x = newX
        self.y = newY


def pushToGit():
	print('Pushing to GitHub')
	repo_dir = '/home/pi/Project/IrishBerryPi.github.io'
	repo = Repo(repo_dir)
	f = ["results.csv"]
	repo.index.add(f)
	repo.index.commit('Uploading Table CSV')
	origin = repo.remote('origin')
	origin.push()

def configure():
	global calibrated
	print "calibrated"
	calibrated = True
	startTime = time.time()

def setTable(event, x, y, flags, param):
	global PTS, TABLES, SELECTING, NUM_TABLES

	if event == cv2.EVENT_LBUTTONDOWN and calibrated == False:
		PTS = [(x, y)]
		SELECTING = True

	elif event == cv2.EVENT_LBUTTONUP and calibrated == False:
		newTable = Table()
		newTable.setLocation(PTS[0][0], PTS[0][1], abs(x - PTS[0][0]), abs(PTS[0][1] - y))
		TABLES.append(newTable)
		SELECTING = False
		PTS.append((x, y))
		NUM_TABLES += 1

websiteTables = np.ones(25,dtype = np.int8)
startTime = 0

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())


# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
 
# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
#vs = VideoStream(src=0).start()
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
fps = FPS().start()

# set callback for locating tables
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", setTable)

# loop over the frames from the video stream
while True:
       
	#time.sleep(.5)
	if calibrated == False: # reset the tables until it is calibrated. Once Calibrated, the list of tables wont change
		tables = []
	else:
		people = [] # once it's calibrated, start looking for people

	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	if calibrated == True: # if calibrated, start object detection
		# grab the frame dimensions and convert it to a blob
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
			0.007843, (300, 300), 127.5)
	 
		# pass the blob through the network and obtain the detections and
		# predictions
		net.setInput(blob)
		detections = net.forward()

		# loop over the detections
		for i in np.arange(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with
			# the prediction
			confidence = detections[0, 0, i, 2]
	 
			# filter out weak detections by ensuring the `confidence` is
			# greater than the minimum confidence
			if confidence > args["confidence"]:
				# extract the index of the class label from the
				# `detections`, then compute the (x, y)-coordinates of
				# the bounding box for the object
				idx = int(detections[0, 0, i, 1])
							
				# Until it is calibrated, we only want to look for tables
				if calibrated == False and idx != 11:
					continue

				# once it is calibrated we only care about people (can add bottles)
				if calibrated == True and idx != 15:
					continue

				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# add tables to list
				if idx == 11:
					newTable = Table()
					newTable.setLocation((startX + endX) / 2, (startY + endY) / 2, startX - endX, startY - endY)
					tables.append(newTable)

				# add people to list
				if idx == 15:
					newPerson = Person()
					newPerson.setLocation((startX + endX) / 2, (startY + endY) / 2)
					people.append(newPerson)
	 
				# draw the prediction on the frame
				label = "{}: {:.2f}%".format(CLASSES[idx],
					confidence * 100)
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					COLORS[idx], 2)
				y = startY - 15 if startY - 15 > 15 else startY + 15
				cv2.putText(frame, label, (startX, y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	# show the output frame
	for table in TABLES:
		cv2.rectangle(frame, (table.x, table.y), (table.x + table.w, table.y + table.h), (0, 255, 0), 2)
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
    # Determine if tables are occupied
	if calibrated:
		for i, table in enumerate(TABLES):
			if len(people) == 0:
				websiteTables[i] = 1
				table.occupied = False
			for person in people:
				if abs(person.x - (table.x + table.w / 2)) < 500 and abs(person.y - (table.y + table.h / 2)) < 500:
					table.occupied = True
					websiteTables[i] = 0
					break
				else:
					table.occupted = False
					websiteTables[i] = 1

	# change bulb color
	if websiteTables[0] == 0:
		bulb.set_rgb(255, 0, 0)
	else:
		bulb.set_rgb(0, 255, 0)
       

	print "tables:", str(websiteTables[0])
	#print "numTables:", NUM_TABLES

	if calibrated  and time.time() - startTime > 30: # run every 5 minutes
		# write to file
		np.savetxt('../results.csv', websiteTables, delimiter=",")

		# push to git
		pushToGit()
		# reset time
		startTime = time.time()

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
	elif key == ord("c"):
		configure()

	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
 
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
bulb.turn_off()
