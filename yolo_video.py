import argparse
import os
import time

import cv2
import imutils
import numpy as np


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="path to input video")
    ap.add_argument("-o", "--output", required=True, help="path to output video")
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
        help="threshold when applyong non-maxima suppression",
    )
    args = vars(ap.parse_args())

    return args


def main(args):
    labels_path = os.path.sep.join([args["model"], "coco.names"])
    LABELS = open(labels_path).read().strip().split("\n")

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

    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    vs = cv2.VideoCapture(args["input"])
    writer = None
    (W, H) = (None, None)

    try:
        prop = (
            cv2.cv.CV_CAP_PROP_FRAME_COUNT
            if imutils.is_cv2()
            else cv2.CAP_PROP_FRAME_COUNT
        )
        total = int(vs.get(prop))
        print("[INFO] {} total frames in video".format(total))
    except Exception as ex:
        print("[INFO] could not determine # of frames in video")
        print("[INFO] no approx. completion time can be provided")
        print(f"[ERROR] {ex}")
        total = -1

    while True:
        (grabbed, frame) = vs.read()

        if not grabbed:
            break

        if W is None or H is None:
            (H, W) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (416, 416), swapRB=True, crop=False
        )
        net.setInput(blob)

        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        boxes = []
        confidences = []
        class_ids = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

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

        idxs = cv2.dnn.NMSBoxes(
            boxes, confidences, args["confidence"], args["threshold"]
        )

        if len(idxs) > 0:
            for i in idxs.flatten():
                (x1, y1) = (boxes[i][0], boxes[i][1])
                (x2, y2) = (boxes[i][2], boxes[i][3])
                color = [int(c) for c in COLORS[class_ids[i]]]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text = "{}: {:.4f}".format(LABELS[class_ids[i]], confidences[i])
                cv2.putText(
                    frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(
                args["output"], fourcc, 30, (frame.shape[1], frame.shape[0]), True
            )
            if total > 0:
                elap = end - start
                print("[INFO] single frame took {:.4f} seconds".format(elap))
                print(
                    "[INFO] estimated total time to finish: {:.4f}".format(elap * total)
                )

        writer.write(frame)

    print("[INFO] cleaning up...")
    writer.release()
    vs.release()


if __name__ == "__main__":
    args = get_args()

    main(args)
