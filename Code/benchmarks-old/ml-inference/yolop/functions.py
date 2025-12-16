
import uuid, os, sys
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(SCRIPT_DIR)
INPUT_DIR = os.path.join(SCRIPT_DIR, os.path.pardir, os.path.pardir, os.path.pardir, 'benchmark-inputs', 'ml-inference', 'yolop')

import cv2
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
from timeit import default_timer as timer

conf_thres = 0.25
iou_thres = 0.45
img_size = 640
device = 'cuda'

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

from plot import show_seg_result
from dataset import LoadImages
from general import non_max_suppression, scale_coords

model = None

def function(obj):

    # load model
    global model

    if model is None:
        model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)
        #model = torch.load(os.path.join(INPUT_DIR, 'yolop.pt'))
        before = timer()
        model.eval()
        model.to(device)
        after = timer()

    #print('model eval time:')
    #print(after - before)

    start = timer()
    u = uuid.uuid4().hex

    dataset = LoadImages(
        os.path.join(INPUT_DIR, 'images'),
        img_size=img_size
    )
    bs = 1  # batch_size

    for path, img, img_det, vid_cap,shapes in dataset:

        img = transform(img).to(device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        det_out, da_seg_out,ll_seg_out= model(img)

        inf_out, _ = det_out
        det_pred = non_max_suppression(
            inf_out,
            conf_thres=conf_thres, 
            iou_thres=iou_thres,
            classes=None, agnostic=False
        )
        det=det_pred[0]

        _, _, height, width = img.shape
        h,w,_ = img_det.shape
        pad_w, pad_h = shapes[1][1]
        pad_w = int(pad_w)
        pad_h = int(pad_h)
        ratio = shapes[1][0][1]

        da_predict = da_seg_out[:, :, pad_h:(height-pad_h),pad_w:(width-pad_w)]
        da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=int(1/ratio), mode='bilinear')
        _, da_seg_mask = torch.max(da_seg_mask, 1)
        da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
        # da_seg_mask = morphological_process(da_seg_mask, kernel_size=7)

        
        ll_predict = ll_seg_out[:, :,pad_h:(height-pad_h),pad_w:(width-pad_w)]
        ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=int(1/ratio), mode='bilinear')
        _, ll_seg_mask = torch.max(ll_seg_mask, 1)
        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()
        # Lane line post-processing
        #ll_seg_mask = morphological_process(ll_seg_mask, kernel_size=7, func_type=cv2.MORPH_OPEN)
        #ll_seg_mask = connect_lane(ll_seg_mask)

        img_det = show_seg_result(img_det, (da_seg_mask, ll_seg_mask), _, _, is_demo=True)

        save_path = os.path.join(INPUT_DIR, 'images', 'output', f'{u}-{Path(path).name}')
        #print(save_path)
        cv2.imwrite(save_path,img_det)

#inference

    end = timer()
    #print(end - start)

if __name__ == "__main__":
    function({})
