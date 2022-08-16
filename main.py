from keras.models import model_from_json
import torch.nn.functional as F
import numpy as np
import cv2 as cv
from time import process_time
from mss import mss
import YOLO.config as config
import torch
import YOLO.utils as utils
import torch.nn as nn
from YOLO.MyModel_model import MyModel

WEIGHTS_DIST = 'dist/model_keras@last.h5'
MODEL_DIST = 'dist/model_keras@last.json'
font = cv.FONT_HERSHEY_SIMPLEX
fontScale = 0.4

"""LOAD OBJECT DETECTION MODEL"""
model = MyModel(num_classes=20).to(config.DEVICE)
try:
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
except Exception as e:
    print('No weight_file:', e)
WIDTH, HEIGHT = 416, 416
window = {'left': 0, 'top': 325, 'width': WIDTH, 'height': HEIGHT}


"""LOAD DISTANCE DETECTION MODEL"""
json_file = open(MODEL_DIST)
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(WEIGHTS_DIST)

def plot_image(img, lines, nms_boxes, loaded_model):
    left_line = []
    right_line = []
    
    """Try plotting the road path"""
    try:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            parms = np.polyfit((x1, x2), (y1, y2), 1)

            """Grouping lanes with positive slope together """
            if parms[0] < 1 and parms[0] > 0.3:
                left_line.append((parms[0], parms[1]))

            """Grouping lanes with negative slope together """
            if parms[0] < -0.3 and parms[0] > -1:
                right_line.append((parms[0], parms[1]))

        left_line = np.average(left_line, axis=0)
        right_line = np.average(right_line, axis=0)

        y1_1 = int(416*(4.1/5))
        y2_1 = int(416*(3.4/5))
        x1_1 = int((y1_1-right_line[1]) / right_line[0])
        x2_1 = int((y2_1-right_line[1]) / right_line[0])

        y1_2 = int(416*(4.1/5))
        y2_2 = int(416*(3.4/5))
        x1_2 = int((y1_2-left_line[1]) / left_line[0])
        x2_2 = int((y2_2-left_line[1]) / left_line[0])
        alpha = (1.0 - 0.6)
        overlay  = img.copy()

        contours = np.array(
            [[x1_1, y1_1], [x2_1, y2_1], [x2_2, y2_2], [x1_2, y1_2]])
        cv.fillPoly(overlay, pts=[contours], color=(0, 0, 255))
        alpha = 0.2 
        img = cv.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        cv.line(img, (x1_1, y1_1), (x2_1, y2_1), (0, 255, 0), 2)
        cv.line(img, (x1_2, y1_2), (x2_2, y2_2), (0, 255, 0), 2)

    except Exception as e:
        pass
        print('No lane:', e)


    """Try making distancee for only car classes prediction and plotting the distance together with all bounding boxes"""
    try:
        for box in nms_boxes:
            

            class_pred = box[0]
            box = box[2:]
            upper_left_x = box[0] - box[2] / 2
            upper_left_y = box[1] - box[3] / 2
            lower_left_x = box[0] + box[2] / 2
            lower_left_y = box[1] + box[3] / 2

            x1, y1 = upper_left_x * 416, upper_left_y * 416
            x2, y2 = lower_left_x * 416, lower_left_y * 416

            if config.CLASSES[int(class_pred)] == 'car':
                square = (x2-x1)*(y2-y1)
                bbox = np.array([x1, y1, x2, y2, square])
                bbox = np.expand_dims(bbox, axis=0)
                dist = loaded_model.predict(bbox)
                dist = dist 

                print('dist:', int(dist))
            else:
                dist = None

            rect = cv.rectangle(img, (int(x1), int(y1)),
                                (int(x2), int(y2)), (255, 0, 0), 2)
            img = cv.putText(img, f'{config.CLASSES[int(class_pred)]}: {int(dist)}', (int(x1), int(y1)), font,
                                      fontScale, (255, 255, 0), 1, cv.LINE_AA)
    except Exception as e:
        print('No box:', e)

    return img


def line_det(img_np):
    lines = None
    scc = cv.cvtColor(img_np, cv.COLOR_BGR2RGB)
    height = scc.shape[0]
    width = scc.shape[1]
    channel_count = scc.shape[2]

    match_mask_color = (255,) * channel_count
    vertices = np.array([
        (-10, height-85),
        (width/2-35, height/2+50),
        (width/2+27, height/2+60),
        (width-65, height-85)
    ], np.int32)

    vertices = np.expand_dims(vertices, axis=0)
    blur = cv.GaussianBlur(img_np, (5, 5), 0)

    value = 50
    edges = cv.Canny(blur, value, value*3)
    mask = np.zeros_like(edges)
    cv.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv.bitwise_and(edges, mask)
    lines = cv.HoughLinesP(masked_image, 2, np.pi /
                            180, 0, minLineLength=20, maxLineGap=3)

    return lines


def object_det(sct_img):
    confThreshold = 0.8
    nmsThreshold = 0.05
    sct_img = sct_img.to(config.DEVICE)
    out = model(sct_img)
    bboxes = [[] for _ in range(sct_img.shape[0])]
    for i in range(3):
        batch_size, A, S, a, b = out[i].shape
        anchor = config.SCALED_ANCHORS[i]
        boxes_scale_i = utils.cells_to_bboxes(
            out[i], anchor, S=S, is_preds=True
        )
        for idx, (box) in enumerate(boxes_scale_i):
            bboxes[idx] += box

    nms_boxes = utils.non_max_suppression(
        bboxes, iou_threshold=0.2, threshold=0.8)
    return nms_boxes


def capture():
    lines = []
    nms_boxes = []
    with torch.no_grad():
        with mss() as sct:
            while(1):
                t1_start = process_time()

                'Frame preparation'
                sct_img = sct.grab(window)
                img_np = np.array(sct_img)
                blob = cv.dnn.blobFromImage(
                    img_np[:, :, :3], 1/255.0, (416, 416), swapRB=False, crop=True)
                blob = torch.tensor(blob)

                'Create bounding boxes after nms'
                nms_boxes = object_det(blob)

                'Detect lines'
                lanes = line_det(img_np)

                try:
                    plot_img = plot_image(
                        img_np, lanes, nms_boxes, loaded_model)
                except Exception as e:
                    print('Error', e)
                    plot_img = img_np

                'Plot image detections + make distance estimation'
                cv.imshow('Lane_detection', plot_img)
                t1_stop = process_time()
                print("Elapsed time during the whole program in seconds:",
                      t1_stop-t1_start)
                if cv.waitKey(1) == ord('q'):
                    cv.destroyAllWindows()
                    break
    return 


if __name__ == "__main__":
    capture()
