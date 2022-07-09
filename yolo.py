import argparse
import os
import time

import cv2
import numpy as np


def get_args():

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to input image")
    ap.add_argument(
        "-m", "--model", required=True, help="base path to YOLO model weight directory"
    )
    ap.add_argument(
        "-c",
        "--confidence",
        type=float,
        default=0.5,
        help="minimum probability to filter weak detections",
    )
    ap.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.3,
        help="threshold when applying non-maxima suppression",
    )
    args = vars(ap.parse_args())

    return args


def main(args):

    labels_path = os.path.sep.join([args["model"], "coco.names"])
    LABELS = open(labels_path).read().strip().split("\n")

    # define constant color list for each classes
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

    model_weight = next(
        (file for file in os.listdir(args["model"]) if file.endswith(".weights")), None
    )
    model_config = next(
        (file for file in os.listdir(args["model"]) if file.endswith(".cfg")), None
    )
    weights_path = os.path.sep.join([args["model"], model_weight])
    config_path = os.path.sep.join([args["model"], model_config])

    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    image = cv2.imread(args["image"])
    (H, W) = image.shape[:2]

    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    start = time.time()
    layer_outputs = net.forward(ln)
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # discard bbox with low confidence
            if confidence > args["confidence"]:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x1 = int(centerX - (width / 2))
                y1 = int(centerY - (height / 2))
                x2 = int(centerX + (width / 2))
                y2 = int(centerY + (height / 2))

                boxes.append([x1, y1, x2, y2])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x1, y1) = (boxes[i][0], boxes[i][1])
            (x2, y2) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in COLORS[class_ids[i]]]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            text = "{}: {:.4f}".format(LABELS[class_ids[i]], confidences[i])
            cv2.putText(
                image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

    cv2.imshow("Image", image)
    cv2.waitKey(0)


if __name__ == "__main__":
    args = get_args()

    main(args)
