import cv2
import numpy as np
from PIL import Image

from classes import dinov2, megapose, mmdet_sam
from utils.choose import validate_preds
from utils.convert import Convert_YCB


device = 'cuda:0'
convert_YCB = Convert_YCB()
MMDet_SAM = mmdet_sam.MMDet_SAM(device)
DINOv2 = dinov2.DINOv2(device)
Megapose = megapose.Megapose(device)
desc_name = 'drill'

# read images
rgb = cv2.imread('./data/drill/image_rgb.png')
rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
depth = np.array(Image.open('./data/drill/image_depth.png'), dtype=np.float32) / 10000

# run mmdet_sam to get bbox and mask
pred = MMDet_SAM.run_detector(rgb.copy(), desc_name)
MMDet_SAM.draw_outcome(rgb.copy(), pred, show_result=True)

best_pred = 0
if len(pred['labels']) > 0:
    # run fbdinov2 to get the best prediction
    best_pred = validate_preds(rgb, pred, DINOv2, show_result=True)

mask = pred['masks'][best_pred].cpu().numpy().astype(np.uint8)
mask = np.transpose(mask, (1, 2, 0))

rgb = np.array(rgb, dtype=np.uint8)
rgb_masked = rgb * mask
mask = mask.squeeze(axis=-1)
depth_masked = depth * mask

bbox = np.round(pred['boxes'][best_pred].cpu().numpy()).astype(int)
ycb_name = convert_YCB.convert_name(pred['labels'][best_pred])

# run megapose
pose_estimation = Megapose.inference(rgb_masked, depth_masked, ycb_name, bbox)
Megapose.save_predictions('./data/drill', pose_estimation)
Megapose.visualise_output(rgb, pose_estimation, ycb_name)

