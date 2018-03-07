__author__ = 'Vishal'
from Tkinter import *
from PIL import ImageTk, Image
import cv2
import pickle
import numpy as np
import pytesseract
import serial
import time

ser = serial.Serial('COM3', 9600, timeout=0) # Establish the connection on a specific port

refPt = []
cropping = False
img = cv2.imread('C:\chi5.jpg') #path for template image for training

camera = cv2.VideoCapture(0)


def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping,img

	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		cropping = True

	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))
		cropping = False

		# draw a rectangle around the region of interest
		cv2.rectangle(img, refPt[0], refPt[1], (0, 255, 0), 2)
		cv2.imshow("image", img)

def get_image():
 # read is the easiest way to get a full image out of a VideoCapture object.
 retval, im = camera.read()
 return im

class Application(Frame):

    def say_hi(self):


        camera_port = 0

        #Number of frames to throw away while the camera adjusts to light levels
        ramp_frames = 30

        # Now we can initialize the camera capture object with the cv2.VideoCapture class.
        # All it needs is the index to a camera port.

        for i in xrange(ramp_frames):
            temp = get_image()

            count = 0

        while True:
            global camera
            ret_val, img = camera.read()

            imgColor = img
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            retval, threshold = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY) #img


            kernel = np.ones((5,5),np.uint8)
            imgEroded = cv2.erode(threshold,kernel,1)


            erodedBlur = cv2.medianBlur(imgEroded,5)

            edges = cv2.Canny(erodedBlur,10,255)


            (contours,_)=cv2.findContours(edges.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours,key=cv2.contourArea,reverse=True)[:10]

            chipContour = None
            print 'No of contours:'
            print len(contours)

            # loop over our contours
            for c in contours:
                # approximate he contour
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c,0.02 * peri, True)

                # if our approximated contour has four points, then
                # we can assume that we have found our screen
                if len(approx) == 4:
                    chipContour = approx
                    break

            print 'Chip contour:'
            print chipContour
            cv2.drawContours(imgColor,[chipContour],-1,(0,255,0),2)

            cv2.imshow('my webcam', imgColor)
            if cv2.waitKey(1) == 27:
                cv2.imwrite('cam.png',img)
                break  # esc to quit
            if count > 100:
                cv2.imwrite('cam.png',img)
                break
            else:
                print 'count:', count
                count+=1

        cv2.destroyAllWindows()

        print("Taking image...")
# Take the actual image we want to keep
        del(camera)



        fileName = "config"
        fileObject = open(fileName,'rb')
        pts = pickle.load(fileObject)

        newImg = cv2.imread('cam.png',0)
        newImg = cv2.resize(newImg,(700,600))

        imgColor = cv2.cvtColor(newImg, cv2.COLOR_GRAY2BGR)  # img
        retval, threshold = cv2.threshold(newImg, 200, 255, cv2.THRESH_BINARY)  # img

        kernel = np.ones((5, 5), np.uint8)
        imgEroded = cv2.erode(threshold, kernel, 1)

        erodedBlur = cv2.medianBlur(imgEroded, 5)

        edges = cv2.Canny(erodedBlur, 10, 255)

        (contours, _) = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        chipContour = None


        # loop over our contours
        for c in contours:
            # approximate he contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            # if our approximated contour has four points, then
            # we can assume that we have found our screen
            if len(approx) == 4:
                chipContour = approx
                break

        print 'Chip contour:'
        print chipContour
        cv2.drawContours(imgColor, [chipContour], -1, (0, 255, 0), 2)

        cv2.imshow('contour',imgColor)

        contourList = map(list, chipContour)

        xCoord = [None] * 4
        yCoord = [None] * 4

        i = 0
        while i < 4:
            xCoord[i] = contourList[i][0][0]
            yCoord[i] = contourList[i][0][1]
            i += 1

        cropped = newImg[min(yCoord):max(yCoord), min(xCoord):max(xCoord)]

        cv2.imshow('cropped',cropped)

        line1 = []
        for i in range(20, cropped.shape[0]-40):
            line1.append(cropped[i, 10])

        print line1

        line2 = []
        for i in range(20, cropped.shape[0]-40):
            line2.append(cropped[i, cropped.shape[1] - 20])

        print line2

        pixels = []
        pixels.append(min(line1))
        pixels.append(max(line1))

        pixels.append(min(line2))
        pixels.append(max(line2))

        minPixel = min(pixels)
        maxPixel = max(pixels)

        thresh = cropped.copy()

        for i in range(0, thresh.shape[1]):
            for j in range(0, thresh.shape[0]):
                k = thresh.item(j, i)
                if k >= minPixel + 1 and k <= maxPixel + 1:
                    thresh.itemset(j, i, 0)
                else:
                    thresh.itemset(j, i, 255)

        thresh = cv2.resize(thresh,(700,600))

        thresh= cv2.medianBlur(thresh,5)

        cv2.imshow('gen thresh', thresh)

        kernel = np.ones((5,5),np.uint8)
        img_erosion = cv2.erode(thresh,kernel,1)

        cv2.imwrite('eroded.jpg',img_erosion)

        roi = thresh[pts[0][1]:pts[1][1], pts[0][0]:pts[1][0]]

        roi = cv2.medianBlur(roi,5)

        cv2.imwrite('roi.jpg',roi)

        imgString = pytesseract.image_to_string(Image.open('C:\\roi.jpg'))

        _,threshInv = cv2.threshold(thresh,0,255,cv2.THRESH_BINARY_INV)

        cv2.imshow('inv thresh', threshInv)

        xCrop, yCrop = cropped.shape[0],cropped.shape[1]

        cropUL = cropped[0:cropped.shape[1]/2, 0:cropped.shape[1]/2]
        cropUR = cropped[0:cropped.shape[1]/2, cropped.shape[0]/2:xCrop+100]
        cropDL = cropped[cropped.shape[0]/2:yCrop+100, 0:cropped.shape[0]/2]
        cropDR = cropped[cropped.shape[0]/2:yCrop+1000, cropped.shape[0]/2:xCrop+100]

        print 'difference:'
        print max(xCoord) - xCrop

        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = 10;
        params.maxThreshold = 255;


        params.filterByCircularity = True
        params.minCircularity = 0.4

        params.filterByArea = True;
        params.minArea = 1000;

        detector = cv2.SimpleBlobDetector(params)

        quadDict = {'Upper Right':cropUR, 'Upper Left':cropUL, 'Down Right':cropDR, 'Down Left':cropDL}

        i=0
        blobquad =None
        for key,value in quadDict.iteritems():
            imgQuad = value
           # _,thresh = cv2.threshold(imgQuad,200,255,cv2.THRESH_BINARY_INV)
            quadBlur = cv2.medianBlur(imgQuad,5)
            imKey = detector.detect(quadBlur)
            if len(imKey) == 0:
                i+=1
                continue
            else:
                blobquad = key
                #imKey = cv2.drawKeypoints(imgQuad, imKey, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                break

        if blobquad is not None:
            print 'blob Found in:'
            print blobquad
            if blobquad == "Down Right":
                ser.write('2')
                time.sleep(2)
            elif blobquad == 'Upper Right':
                ser.write('1')
                time.sleep(2)
            else:
                ser.write('0')
                time.sleep(0)

        else:
            print 'blob not found'

            cv2.waitKey(0)


        print imgString

        w = Label(root, text="Logo")
        w.configure(justify="center")
        w.pack(pady=10)

        x = Label(root, text=imgString)
        x.configure(justify="center")
        x.pack(pady=10)

        y = Label(root, text=blobquad)
        y.configure(justify="center")
        y.pack(pady=10)

        z = Label(root, text="PIN MARK EXISTS")
        z.configure(justify="center")
        z.pack(pady=10)

        global same
        same = TRUE
        n=0.25

        path = 'C:\chi4.jpg'
        global img  # The problem was that the image had to be displayed outide and it was created inside a function. I made the img variable global  now its accesible osutside the function
        global image
        image = Image.open(path)
        [imageSizeWidth, imageSizeHeight] = image.size

        newImageSizeWidth = int(imageSizeWidth * n)
        if same:
            newImageSizeHeight = int(imageSizeHeight * n)
        else:
            newImageSizeHeight = int(imageSizeHeight / n)

        image = image.resize((newImageSizeWidth, newImageSizeHeight), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(image)
        panel = Label(root, image=img)
        panel.pack(fill="both", expand="yes", side=RIGHT, padx=20)


        cv2.imshow('ROI',roi)
        cv2.imshow('quad1',cropUL)
        cv2.imshow('quad2', cropUR)
        cv2.imshow('quad3',cropDL)
        cv2.imshow('quad4',cropDR)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        fileObject.close()



    def setRoi(self):
        global img
        clone = img.copy()
        newX,newY = 700,600
        img= cv2.resize(img,(int(newX),int(newY)))
        cv2.namedWindow("image")
        cv2.setMouseCallback("image",click_and_crop)

# keep looping until the 'q' key is pressed
        while True:
            # display the image and wait for a keypress

            cv2.imshow("image", img)
            key = cv2.waitKey(1) & 0xFF

            # if the 'r' key is pressed, reset the cropping region
            if key == ord("r"):
             img = clone.copy()

            # if the 'c' key is pressed, break from the loop
            elif key == ord("c"):
                break

        if len(refPt) == 2:
            fileName = "config"
            fileObject = open(fileName,'wb')

            pickle.dump(refPt,fileObject)

            fileObject.close()
            print 'Config file saved'
            cv2.destroyWindow("image")





    def createWidgets(self):
        self.QUIT = Button(self , text="QUIT", fg='red', command=self.quit).pack(side=RIGHT, padx=10, pady=20)
        self.hi_there = Button(self, text="Generate", command= self.say_hi).pack(side=RIGHT, padx=20,pady=30)
        self.add = Button(self,text="Add template", command=self.setRoi).pack(side=RIGHT, padx=30,pady=40)



    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.pack()
        group = LabelFrame(master, text="Group", padx=5, pady=5)
        group.pack(padx=10, pady=10)
        w = Entry(group)

        self.createWidgets()


root = Tk()
app = Application(master=root)
app.mainloop()
