import cv2
import numpy as np
import time
vid = cv2.VideoCapture("test1.mp4")

label = 'coco.names'
classes = []
with open(label, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

weight_height_target = 416
model_config = 'yolov3.cfg'
model_weights = 'yolov3.weights'
confThreshold = 0.4
nmsThreshold = 0.2
inccount1 = 0
inccount4 = 0
inccount5 = 0
inccount_reset = 0
start_time = time.time()

network = cv2.dnn.readNetFromDarknet(model_config, model_weights)
network.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
network.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
 
def findobject(outputs, img):
    heightTar, weightTar, channelTars = img.shape
    bbox = []
    classid = []
    confidance = []
    count1 = 0
    count4 = 0

    for output in outputs:
        for det in output:
            score = det[5:]
            classids = np.argmax(score)
            confids = score[classids]
            if classids == 2 or classids == 7:
                if confids > confThreshold:
                    w, h = int(det[2]*weightTar), int(det[3]*heightTar)
                    x, y = int((det[0]*weightTar)-w/2), int((det[1]*heightTar)-h/2)
                    bbox.append([x, y, w, h])
                    classid.append(classids) 
                    confidance.append(float(confids))
                    if (int(img.shape[0]/2)-3) < y < (int(img.shape[0]/2)+3):
                        if classids == 2:
                            count1 += 1
                        if classids == 7:
                            count4 += 1
            else:
                continue
    draw_box = cv2.dnn.NMSBoxes(bbox, confidance, confThreshold, nmsThreshold)
    for i in draw_box:
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)
        cv2.putText(img, f'{classes[classid[i]].upper()} {int(confidance[i]*100)}%',
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
        cv2.line(img, (0, int(img.shape[0]/2)+3), (int(img.shape[1]), int(img.shape[0]/2)+3), (0, 0, 100), 1)
        cv2.line(img, (0, int(img.shape[0]/2)-3), (int(img.shape[1]), int(img.shape[0]/2)-3), (0, 0, 100), 1)

    return count1, count4

while True:
    success, img = vid.read()
    if not success:
        break

    img = cv2.resize(img, (1280, 720))
    cv2.imshow('video', img)
    blob = cv2.dnn.blobFromImage(img, 1/255, (weight_height_target, weight_height_target), [0, 0, 0, 0], 1, crop=False)
    network.setInput(blob)
    LayerNames = network.getLayerNames()
    outputNames = [LayerNames[i - 1] for i in network.getUnconnectedOutLayers()]
    outputs = network.forward(outputNames)
    counter1, counter4 = findobject(outputs, img)
    inccount1 += counter1
    inccount4 += counter4
    inccount5 = inccount1 + inccount4
    run_time = time.time()
    inccount_reset = int(time.time() - start_time)
    if inccount_reset == 3600:
        inccount1 = 0
        inccount4 = 0
        inccount5 = 0
        inccount_reset = 0
        start_time = run_time

    cv2.putText(img, f'counting car : {inccount1}', (25, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(img, f'counting TRUCK : {inccount4}', (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(img, f'counting TOTAL : {inccount5}', (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.imshow('video', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
