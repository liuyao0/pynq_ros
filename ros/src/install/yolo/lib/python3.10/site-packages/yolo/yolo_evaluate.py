# from models.yolo import DetectMultiBackend
import cv2
import torch
import pkg_resources as pkg
import torchvision
import time

def check_version(current='0.0.0', minimum='0.0.0', name='version ', pinned=False, hard=False, verbose=False):
    # Check version vs. required version
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    s = f'WARNING ⚠️ {name}{minimum} is required by YOLOv5, but {name}{current} is currently installed'  # string
    if hard:
        assert result, print(s)  # assert min requirements met
    if verbose and not result:
        print(s)
    return result

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

def post_process(output):
    if isinstance(output,tuple):
        output = list(output)
    dcfg = DetectConfiguration()
    z = []
    for i in range(3):
        bs, _, ny, nx = output[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
        output[i] = output[i].view(bs, dcfg.na, dcfg.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

        if dcfg.grid[i].shape[2:4] != output[i].shape[2:4] :
            dcfg.grid[i], dcfg.anchor_grid[i] = dcfg._make_grid(nx, ny, i)
            # print("[%d] grid:" % i,dcfg.grid[i])
            # print("[%d] anchor_grid:" % i,dcfg.anchor_grid[i])
        
        xy, wh, conf = output[i].sigmoid().split((2, 2, dcfg.nc + 1), 4)
        xy = (xy * 2 + dcfg.grid[i]) * dcfg.stride[i]
        wh = (wh * 2) ** 2 * dcfg.anchor_grid[i]  # wh
        y = torch.cat((xy, wh, conf), 4)
        z.append(y.view(bs, dcfg.na * nx * ny, dcfg.no))

    pred = torch.cat(z,1)
    pred = non_max_suppression(pred)
    return pred
    

def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        multi_label=False,
        max_det=1000,
        nm=0,  # number of masks
):
    
    bs = 1  # batch size
    nc = 76  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    
    # print("classes: ", classes) # None
    # print("agnostic: ", agnostic) # False
    # print("multi_label: ", multi_label) # False 
    # print("labels: ", labels) # ()
    
    
    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    # print('prediction size: ', prediction.shape)
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)


        conf, j = x[:, 5:mi].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]


        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections

        output[xi] = x[i]
    return output

class DetectConfiguration():
    def __init__(self):  # detection layer
        super().__init__()
        self.anchors = [[1.25, 1.625,2.0, 3.75,4.125, 2.875], 
                        [1.875, 3.8125,3.875, 2.8125,3.6875, 7.4375],
                        [3.625, 2.8125,4.875, 6.1875,11.65625, 10.1875]]
        self.nc = 76
        self.no = self.nc + 5
        self.na = len(self.anchors[0]) // 2
        self.nl = len(self.anchors)
        self.grid = [torch.empty(0) for _ in range(self.nl)]
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]
        self.anchors = torch.tensor(self.anchors).float().view(self.nl, -1, 2)
        self.stride = torch.tensor([8., 16., 32.])
    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, '1.10.0')):
        d = torch.device("cpu")
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing='ij') if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


def evaluate(model, img_path):
    image = cv2.imread(img_path)
    image = image.transpose((2, 0, 1))[::-1] / 255
    input = torch.tensor(torch.tensor([image])).float()
    output = model(input)
    if isinstance(output,tuple):
        output = list(output)
    # print("output[0]: ",output[0])
    # print("output[1]: ",output[1])
    # print("output[2]: ",output[2])
    
    dcfg = DetectConfiguration()
    z = []
    for i in range(3):
        bs, _, ny, nx = output[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
        output[i] = output[i].view(bs, dcfg.na, dcfg.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

        if dcfg.grid[i].shape[2:4] != output[i].shape[2:4] :
            dcfg.grid[i], dcfg.anchor_grid[i] = dcfg._make_grid(nx, ny, i)
            # print("[%d] grid:" % i,dcfg.grid[i])
            # print("[%d] anchor_grid:" % i,dcfg.anchor_grid[i])
        
        xy, wh, conf = output[i].sigmoid().split((2, 2, dcfg.nc + 1), 4)
        xy = (xy * 2 + dcfg.grid[i]) * dcfg.stride[i]
        wh = (wh * 2) ** 2 * dcfg.anchor_grid[i]  # wh
        y = torch.cat((xy, wh, conf), 4)
        z.append(y.view(bs, dcfg.na * nx * ny, dcfg.no))

    pred = torch.cat(z,1)
    pred = non_max_suppression(pred)

    return pred
    
if __name__=='__main__':
    # model = DetectMultiBackend("models/best.pt")
    img_path = "../1.jpg"
    result = evaluate(model, img_path)
    print(result)
    

