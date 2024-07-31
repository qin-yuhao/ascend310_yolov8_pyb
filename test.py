import sampleYOLOV7
import cv2
import time 
# yolov8 n
coco_classes = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
#py::arg("model_path"),py::arg("model_size"),py::arg("confidenceThreshold"),py::arg("NMSThreshold"),py::arg("classNum"))
runner = sampleYOLOV7.sampleYOLOV7("/home/HwHiAiUser/Desktop/sampleYOLOV7/yolov8n_normal.om",640,0.5,0.5,80) 
img = cv2.imread("/home/HwHiAiUser/Desktop/sampleYOLOV7/people.jpg")
#img  = cv2.resize(img,(640,640))
out = runner.inference(img)
print(out)
#"sampleYOLV7,return classIndex,x_center,y_center,width,height,score"
for i in out:
    classIndex,x_center,y_center,width,height,score = int(i[0]),i[1],i[2],i[3],i[4],i[5]
    cv2.rectangle(img, (int(x_center - width / 2), int(y_center - height / 2)), (int(x_center + width / 2), int(y_center + height / 2)), (0, 255, 0), 2)
    cv2.putText(img, coco_classes[classIndex] + ' ' + str(round(score, 2)), (int(x_center), int(y_center)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
cv2.imwrite("output.jpg", img)

#test time cost
start = time.time()
for i in range(1000):
    out = runner.inference(img)
end = time.time()
print("1000 inference cost time avg ms:",(end-start)/1000*1000)
