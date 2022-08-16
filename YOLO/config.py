import torch
CLASSES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]


S = [416 // 32, 416 // 16, 416 // 8]

SCALED_ANCHORS = (
    torch.tensor(ANCHORS)
    * torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
).to(DEVICE)

MODEL_PATH = ''
