import torch.nn as nn
import torch
import numpy as np

def IoU(prediction, targ_labels, anchors=None):
    if anchors:
        box1_w, box1_h = torch.tensor(prediction)
        box2_w, box2_h = targ_labels[:, 0], targ_labels[:, 1]
        x1 = torch.min(box1_w, box2_w)
        x2 = torch.min(box1_h, box2_h)
        intersection = abs(x1) * abs(x2)
        box1_area = abs(box1_w * box1_h)
        box2_area = abs(box2_w * box2_h)
        union = box1_area + box2_area - intersection

    else:
        box1_x1, box1_y1, box1_x2, box1_y2 = prediction[0]
        box2_x1, box2_y1, box2_x2, box2_y2 = targ_labels[:, 1:]
        x1 = torch.max(box1_x1, box2_x1)
        y1 = torch.max(box1_y1, box2_y1)
        x2 = torch.min(box1_x2, box2_x2)
        y2 = torch.min(box1_y2, box2_y2)
        box1_area = (box1_x1 - box1_x2) * (box1_y1 - box1_y2)
        box2_area = (box2_x1 - box2_x2) * (box2_y1 - box2_y2)
        union = box1_area + box2_area - intersection

    return intersection / union


def cells_to_bboxes(predictions, anchors, S, is_preds=True):
    BATCH_SIZE = predictions.shape[0]
    num_anchors = len(anchors)
    box_predictions = predictions[..., 1:5]
    if is_preds:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:] = torch.exp(
            box_predictions[..., 2:]) * anchors
        scores = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]

    cell_indices = (
        torch.arange(S)
        .repeat(predictions.shape[0], 3, S, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )
    x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / S * (box_predictions[..., 1:2] +
                 cell_indices.permute(0, 1, 3, 2, 4))
    w_h = 1 / S * box_predictions[..., 2:4]
    converted_bboxes = torch.cat(
        (best_class, scores, x, y, w_h), dim=-1).reshape(BATCH_SIZE, num_anchors * S * S, 6)
    return converted_bboxes.tolist()


def intersection_over_union(boxes_preds, boxes_labels):

    box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
    box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
    box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
    box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
    box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
    box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
    box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
    box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_suppression(bboxes, iou_threshold, threshold):

    boxes = np.asarray(bboxes[0])

    #1, 5, 6, 13, 14 classes
    boxes1 = [box for box in boxes if box[1] > threshold and (
        box[0] == 1 or box[0] == 5 or box[0] == 6 or box[0] == 13 or box[0] == 14)]
    s_boxes = sorted(boxes1, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while s_boxes:
        chosen_box = s_boxes.pop(0)

        s_boxes = [
            box
            for box in s_boxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)
    print(
        f'all_bboxes:{len(boxes)}, filtered_boxes:{len(boxes1)}, best_box:{len(bboxes_after_nms)}')

    return bboxes_after_nms



