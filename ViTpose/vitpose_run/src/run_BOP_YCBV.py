"""
run_BOP_YCBV.py

Author: Wanqing Xia
Email: wxia612@aucklanduni.ac.nz

This script is the main entry point for the object detection and pose estimation pipeline.
It uses the MMDet_SAM, DINOv2, and Megapose models to process images from the BOP YCBV dataset.

The script iterates over each image in the dataset, runs the MMDet_SAM detector to get object predictions,
validates these predictions using DINOv2, and then estimates the pose of the validated objects using Megapose.

The results, including the scene ID, image ID, object ID, score, rotation matrix (R), translation vector (t),
and processing time, are saved in a CSV file.

The script also includes testing code for different combinations of the MMDet_SAM, DINOv2, and Megapose models.
"""
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from classes import dinov2, megapose, mmdet_sam
from utils.choose import validate_preds
from utils.convert import Convert_YCB
import time
import csv
import json
import torch

def main(root_path, device):
    folder_paths = sorted([p for p in (Path(root_path) / 'test').glob('*') if p.is_dir()])
    convert_YCB = Convert_YCB()

    MMDet_SAM = mmdet_sam.MMDet_SAM(device)
    DINOv2 = dinov2.DINOv2(device)
    Megapose = megapose.Megapose(device)

    # Initialize the list to store result
    data = []

    for count, folder in enumerate(folder_paths):
        print('Evaluating images from %s' % folder)
        rgb_files = sorted((folder / "rgb").rglob('*.png'))
        dep_files = sorted((folder / "depth").rglob('*.png'))
        mask_files = sorted((folder / "mask").rglob('*.png'))
        objs_in_scene = int(len(mask_files) / len(rgb_files))
        t = 0

        for num in tqdm(range(0, len(rgb_files)), desc=f'Processing images in {folder}'):
            rgb_path = rgb_files[num]
            dep_path = dep_files[num]
            rgb = cv2.imread(str(rgb_path))
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            depth = Image.open(dep_path)
            depth = np.array(depth, dtype=np.float32) / 10000

            with open((folder / "scene_gt.json"), 'r') as f:
                scene_data = json.load(f)
            first_row_key = list(scene_data.keys())[0]  # Get the key for the first row
            first_row_objects = scene_data[first_row_key]  # Get the list of objects in the first row
            obj_ids = [obj['obj_id'] for obj in first_row_objects]  # Extract all obj_id values

            for index, obj_id in enumerate(obj_ids):
                tic = time.time()
                success_flag = False
                pose_estimation = 0

                ycb_name = convert_YCB.convert_number(obj_id)
                desc_name = convert_YCB.convert_name(ycb_name)
                pred = MMDet_SAM.run_detector(rgb.copy(), desc_name)

                if len(pred['labels']) > 0:
                    """Testing code for Detic + SAM + DINOv2 scores"""
                    best_pred = validate_preds(rgb, pred, DINOv2)
                    bbox = np.round(pred['boxes'][best_pred].cpu().numpy()).astype(int)
                    ycb_name = convert_YCB.convert_name(pred['labels'][best_pred])

                    mask = pred['masks'][best_pred].cpu().numpy().astype(np.uint8)
                    mask = np.transpose(mask, (1, 2, 0))
                    rgb = np.array(rgb, dtype=np.uint8)
                    rgb_masked = rgb * mask
                    mask = mask.squeeze(axis=-1)
                    depth_masked = depth * mask

                    pose_estimation = Megapose.inference(rgb_masked, depth_masked, ycb_name, bbox)
                    success_flag = True

                    """Testing code for Detic + DINOv2 (need to modify the validate_preds function)"""
                    # best_pred = validate_preds(rgb, pred, DINOv2)
                    # bbox = np.round(pred['boxes'][best_pred].cpu().numpy()).astype(int)
                    # ycb_name = convert_YCB.convert_name(pred['labels'][best_pred])
                    # pose_estimation = Megapose.inference(rgb, depth, ycb_name, bbox)
                    # success_flag = True

                    """Testing code for Detic + SAM"""
                    # best_pred = np.argmax(pred['scores'])
                    # bbox = np.round(pred['boxes'][best_pred].cpu().numpy()).astype(int)
                    # ycb_name = convert_YCB.convert_name(pred['labels'][best_pred])
                    #
                    # mask = pred['masks'][best_pred].cpu().numpy().astype(np.uint8)
                    # mask = np.transpose(mask, (1, 2, 0))
                    # rgb = np.array(rgb, dtype=np.uint8)
                    # rgb_masked = rgb * mask
                    # mask = mask.squeeze(axis=-1)
                    # depth_masked = depth * mask
                    #
                    # pose_estimation = Megapose.inference(rgb_masked, depth_masked, ycb_name, bbox)
                    # success_flag = True

                    """Testing code for Detic"""
                    # best_pred = np.argmax(pred['scores'])
                    # bbox = np.round(pred['boxes'][best_pred].cpu().numpy()).astype(int)
                    # ycb_name = convert_YCB.convert_name(pred['labels'][best_pred])
                    # pose_estimation = Megapose.inference(rgb, depth, ycb_name, bbox)
                    # success_flag = True

                t += time.time() - tic
                if success_flag:
                    poses = pose_estimation.poses.cpu().numpy()
                    R = poses[0, :3, :3]
                    T = poses[0, :3, 3] * 1000
                    # Flatten R and t for CSV
                    R_str = ' '.join(map(str, R.flatten()))
                    T_str = ' '.join(map(str, T))
                    # Create the row
                    row = [int(folder.name), int(rgb_path.stem), obj_id, 1, R_str, T_str, t]
                    data.append(row)
                else:
                    R = np.zeros((3, 3))
                    T = np.zeros(3)
                    # Flatten R and t for CSV
                    R_str = ' '.join(map(str, R.flatten()))
                    T_str = ' '.join(map(str, T))
                    # Create the row
                    row = [int(folder.name), int(rgb_path.stem), obj_id, 0, R_str, T_str, t]
                    data.append(row)
                if index == objs_in_scene - 1:
                    for i in range(len(data) - objs_in_scene, len(data)):
                        data[i][-1] = t
                    t = 0

    return data

def save_results(data, file_path):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['scene_id', 'im_id', 'obj_id', 'score', 'R', 't', 'time'])
        # Write the data
        writer.writerows(data)

if __name__ == "__main__":
    device = torch.device('cuda:0')
    root_path = './bop_datasets/ycbv'
    result_file = 'outputs/Result_ycbv-test.csv'
    data_save = main(root_path, device)
    save_results(data_save, result_file)



