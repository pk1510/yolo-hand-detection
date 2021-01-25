import argparse
import cv2
import numpy as np
import vlc
import time
from yolo import YOLO
from operator import itemgetter
ap = argparse.ArgumentParser()
ap.add_argument('-n', '--network', default="normal", help='Network Type: normal / tiny / prn / v4-tiny')
ap.add_argument('-d', '--device', default=0, help='Device to use')
ap.add_argument('-s', '--size', default=416, help='Size for yolo')
ap.add_argument('-c', '--confidence', default=0.2, help='Confidence for yolo')
args = ap.parse_args()

if args.network == "normal":
    print("loading yolo...")
    yolo = YOLO("models/cross-hands.cfg", "models/cross-hands.weights", ["hand"])
elif args.network == "prn":
    print("loading yolo-tiny-prn...")
    yolo = YOLO("models/cross-hands-tiny-prn.cfg", "models/cross-hands-tiny-prn.weights", ["hand"])
elif args.network == "v4-tiny":
    print("loading yolov4-tiny-prn...")
    yolo = YOLO("models/cross-hands-yolov4-tiny.cfg", "models/cross-hands-yolov4-tiny.weights", ["hand"])
else:
    print("loading yolo-tiny...")
    yolo = YOLO("models/cross-hands-tiny.cfg", "models/cross-hands-tiny.weights", ["hand"])

yolo.size = int(args.size)
yolo.confidence = float(args.confidence)


img = cv2.imread("timeStone.jpg")
lower_bgr = np.uint8([[[30, 30, 30]]])
upper_bgr = np.uint8([[[255, 255, 255]]])
mask = cv2.inRange(img, lower_bgr, upper_bgr)
img = cv2.bitwise_and(img, img, mask=mask)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

vdo = ("test_vdo.mp4", "Faded.mp3")                              ##add ur vdo list here and access via the index
# create video capture object
data = cv2.VideoCapture(vdo[0])

# count the number of frames
frames = data.get(cv2.CAP_PROP_FRAME_COUNT)
fps = int(data.get(cv2.CAP_PROP_FPS))

# calculate dusration of the video
seconds = int(frames / fps)
milliseconds = seconds*1000
data.release()

a = 500
r = 1.0001
n = 0
direction = "rest"                                                 #direction of rotation. Use this variable with vlc library
media_player = vlc.MediaPlayer()
media = vlc.Media(vdo[0])
media_player.set_media(media)

first = False

print("starting webcam...")
cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)
if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
    flip = cv2.flip(frame, 1)
else:
    rval = False

while rval:
    width, height, inference_time, results = yolo.inference(flip)
    if len(results)!=0:
        res = max(results, key=itemgetter(2))
    # for detection in results:
    #     id, name, confidence, x, y, w, h = detection
    #     cx = x + (w / 2)
    #     cy = y + (h / 2)
    #
    #     # draw a bounding box rectangle and label on the image
    #     color = (0, 255, 255)
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    #     text = "%s (%s)" % (name, round(confidence, 2))
    #     cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
    #                 0.5, color, 2)
        id, name, confidence, x, y, w, h = res
        img = cv2.resize(img, (w,h))
        gray = cv2.resize(gray, (w, h))
        thresh = cv2.threshold(gray, 29, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.merge((thresh, thresh, thresh))
        cx = x + (w // 2)
        cy = y + (h // 2)

        # draw a bounding box rectangle and label on the image
        color = (0, 255, 255)
        cv2.rectangle(flip, (x, y), (x + w, y + h), color, 2)
        text = "%s (%s)" % (name, round(confidence, 2))
        cv2.putText(flip, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)
        cv2.circle(flip, (cx, cy), 5, (0,0,0), -1)
        #flip[y:y+h, x:x+w, :] = np.where(thresh != 0, img, flip[y:y+h, x:x+w, :])
        if cx >=420 and cx < 510 and w>=120 and w<=220:
            direction = "rest"
        elif cx>=505 and cx<=600 and w>=190 and w<=290:
            direction = "clockwise"
        elif cx>=430 and cx<=500 and w>=210 and w<=310:
            direction = "anticlockwise"
        else:
            direction = "rest"
        print(direction, cx, cy, w, h)
    else:
        direction="rest"
        #print(cx, cy, w, h)
    media_player.play()
    prev = direction
    if direction == "anticlockwise":
        curr_time = media_player.get_time()
        if curr_time < 0:
            media_player.set_time(0)
        else:
            if not first:
                first = True
                n=0
                #r = 1 - a/curr_time
            n+=1
            media_player.set_time(curr_time - int(a*pow(r, n)))
            media_player.play()
            time.sleep(0.1)
    if direction == "clockwise":
        curr_time = media_player.get_time()
        if curr_time > milliseconds:
            media_player.set_time(0)
        else:
            if not first:
                n = 0
                first = True
                #r = 1 - curr_time/milliseconds if curr_time!=0 else 1 - a/milliseconds
            n+=1
            media_player.set_time(curr_time + int(a*pow(r, n)))
            time.sleep(0.1)
            media_player.play()
    if prev != direction:
        first = False
        prev = direction
        n = 0
    cv2.imshow("preview", flip)

    rval, frame = vc.read()
    flip = cv2.flip(frame , 1)
    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break

cv2.destroyWindow("preview")
vc.release()
