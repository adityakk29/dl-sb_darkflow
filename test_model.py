import cv2
from darkflow.net.build import TFNet
import numpy as np
import time

options ={
	"model": "cfg/tiny-yolo-voc-2c.cfg",
	"load" : 9375,
	"threshold" : 0.05,
	"gpu" : 0.2
}

tfnet = TFNet(options)
# VID_20190916_112725, video_20190913_172119
capture = cv2.VideoCapture("demo_5_2.mp4")
colors = [tuple(255 * np.random.rand(3)) for i in range(2)]

while (capture.isOpened()):
    cv2.waitKey(20)
    stime = time.time()
    ret, frame = capture.read()
    # frame.set(3, 640)
    # frame.set(4, 480)
    if ret:
        # frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = tfnet.return_predict(frame)
        for color, result in zip(colors, results):
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label'] if result["label"] in ["seat_belt"] else "cell_phone"
            confidence = result['confidence']
            text = '{}: {:.0f}%'.format(label, confidence * 100)
            txt_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN , 4, 4)
            frame = cv2.rectangle(img = frame, pt1 = tl, pt2 = (tl[0]+txt_size[0][0], tl[1] - txt_size[0][1]), color = color, thickness = cv2.FILLED)
            frame = cv2.rectangle(frame, tl, br, color, 2)
            frame = cv2.putText(frame, text, tl, cv2.FONT_HERSHEY_PLAIN , 1, (255, 255, 255), 1)
        cv2.imshow('frame', frame)
        # print('FPS {:.1f}'.format(1 / (time.time() - stime)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        capture.release()
        cv2.destroyAllWindows()
        break