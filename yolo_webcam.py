import argparse
import os

import cv2
import numpy as np

whT = 320
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.3


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-m",
        "--model",
        help="base path to YOLO model weight directory",
        default="models/yolov3/",
    )
    args = vars(ap.parse_args())

    return args


def findObjects(outputs, img, LABELS):
    hT, wT, cT = img.shape
    bbox = []
    class_ids = []
    confs = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > CONF_THRESHOLD:
                w, h = int(detection[2] * wT), int(detection[3] * hT)
                x, y = int((detection[0] * wT) - w / 2), int(
                    (detection[1] * hT) - h / 2
                )
                bbox.append([x, y, w, h])
                class_ids.append(class_id)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox, confs, CONF_THRESHOLD, NMS_THRESHOLD)

    for i in indices:
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(
            img,
            f"{LABELS[class_ids[i]].upper()} {int(confs[i]*100)}%",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2,
        )


def main(args):
    cap = cv2.VideoCapture(0)
    labels_path = os.path.sep.join([args["model"], "coco.names"])
    LABELS = open(labels_path).read().strip().split("\n")

    model_weight = next(
        (file for file in os.listdir(args["model"]) if file.endswith(".weights")), None
    )
    model_config = next(
        (file for file in os.listdir(args["model"]) if file.endswith(".cfg")), None
    )
    weights_path = os.path.sep.join([args["model"], model_weight])
    config_path = os.path.sep.join([args["model"], model_config])

    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    while True:
        success, img = cap.read()

        blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], crop=False)
        net.setInput(blob)

        layerNames = net.getLayerNames()
        outputNames = [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]

        outputs = net.forward(outputNames)
        findObjects(outputs, img, LABELS)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    args = get_args()
    main(args)
