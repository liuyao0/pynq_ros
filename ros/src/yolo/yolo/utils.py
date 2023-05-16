import sys
sys.path.append("/usr/local/share/pynq-venv/lib/python3.10/site-packages")
import numpy as np
import pkg_resources as pkg
import torch

def pre_process(image, model_image_size):
    image = image[...,::-1]
    print('model image size: ', model_image_size)
    new_image = np.ones((640,640,3), np.uint8) * (2 ** 6)
    new_image = image
    image_data = np.array(new_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0) 	
    return image_data


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


class DetectModelConfiguration:
    def __init__(self):
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
        self.class_name = [
            "i10","i2","i2r","i4","i4l","i5","il100","il60","il80","il90","im","ip","p1","p10","p11","p12","p13","p14","p18","p19","p23","p25","p26","p27","p3","p5","p6","p9","pa14","pb","pbm","pbp","pcl","pdd","pg","ph4","ph4.5","ph5","pl10","pl100","pl110","pl120","pl15","pl20","pl30","pl40","pl5","pl50","pl60","pl70","pl80","pl90","pm10","pm20","pm30","pm55","pmb","pn","pne","pr30","pr40","pr50","pr60","ps","w13","w21","w22","w30","w32","w47","w55","w57","w58","w59","w63","wc"
        ]


    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, '1.10.0')):
        d = torch.device("cpu")
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing='ij') if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid
