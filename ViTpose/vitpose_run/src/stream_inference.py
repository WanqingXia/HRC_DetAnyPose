import cv2
import os
import numpy as np
from PIL import Image
import json
import time
import matplotlib.pyplot as plt
import torch
import gc


from classes import dinov2, megapose, mmdet_sam, k4a_camera
from utils.choose import validate_preds
from utils.convert import Convert_YCB
from generate import generate_ycb


def stream(desc_name: str):
    cv2.namedWindow('Pose estimation result', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Pose estimation result', 1080, 720)

    # Variables for calculating FPS
    prev_frame_time = 0
    pose_estimation = None
    while True:
        ret_color, color, ret_depth, depth = K4A_Camera.get_capture()

        if not ret_color or not ret_depth:
            continue
        else:
            color = cv2.cvtColor(color, cv2.COLOR_BGRA2RGB)
            depth = np.array(depth, dtype=np.float32) / 1000
            if pose_estimation is None:
                print("Estimating pose for the first time...")
                tic = time.time()
                # first time estimation
                # run mmdet_sam to get bbox and mask
                pred = MMDet_SAM.run_detector(color.copy(), desc_name)
                if len(pred['labels']) > 0:
                    # run fbdinov2 to get the best prediction
                    best_pred = validate_preds(color, pred, DINOv2)

                    mask = pred['masks'][best_pred].cpu().numpy().astype(np.uint8)
                    mask = np.transpose(mask, (1, 2, 0))

                    color = np.array(color, dtype=np.uint8)
                    color_masked = color * mask
                    mask = mask.squeeze(axis=-1)
                    depth_masked = depth * mask

                    bbox = np.round(pred['boxes'][best_pred].cpu().numpy()).astype(int)
                    ycb_name = convert_YCB.convert_name(pred['labels'][best_pred])

                    # run megapose
                    pose_estimation = Megapose.inference(color_masked, depth_masked, ycb_name, bbox)
                    del mask, color_masked, depth_masked
            else:
                # print("Refining pose estimation...")
                tic = time.time()
                ycb_name = convert_YCB.convert_name(desc_name)
                color = np.array(color, dtype=np.uint8)
                # refine the pose estimation
                pose_estimation = Megapose.refinement_color(color, pose_estimation)
                # pose_estimation = Megapose.refinement_depth(color, depth, pose_estimation)

            # print("Estimation took {} seconds".format(time.time() - tic))

            if pose_estimation is None:
                print("No pose estimation found")
                continue
            else:
                # Calculate FPS
                new_frame_time = time.time()
                fps = 1 / (new_frame_time - prev_frame_time)
                prev_frame_time = new_frame_time

                contour_image, mesh_image = Megapose.get_output_image(color, pose_estimation, ycb_name)
                mesh_image = cv2.cvtColor(mesh_image, cv2.COLOR_RGB2BGR)
                # Put FPS text on the image
                cv2.putText(mesh_image, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.imshow('Pose estimation result', mesh_image)

                # Press q key to stop
                if cv2.waitKey(1) == ord('q'):
                    break

                del color, depth, contour_image, mesh_image

    cv2.destroyAllWindows()


if __name__ == '__main__':
    device = 'cuda:0'
    global convert_YCB
    convert_YCB = Convert_YCB()
    global MMDet_SAM
    MMDet_SAM = mmdet_sam.MMDet_SAM(device)
    global K4A_Camera
    K4A_Camera = k4a_camera.K4A_Camera()
    calibration = K4A_Camera.get_calibration()
    # modify the calibration to match the camera data
    color_img = None
    while color_img is None:
        _, color_img, _, _ = K4A_Camera.get_capture()

    # Define the path to your JSON file
    json_file_path = './data/ycbv_camera_data.json'

    # Read the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Modify the parameters in the JSON data
    data['K'][0][0] = calibration.color_params.fx
    data['K'][0][2] = calibration.color_params.cx
    data['K'][1][1] = calibration.color_params.fy
    data['K'][1][2] = calibration.color_params.cy
    data['resolution'] = [color_img.shape[0], color_img.shape[1]]

    # Save the updated JSON data back to the file
    with open(json_file_path, 'w') as file:
        json.dump(data, file, indent=4)

    if not os.path.exists('./data/ycbv_generated'):
        print("Generating viewpoints for YCB objects...")
        generate_ycb.generate()

    global Megapose
    Megapose = megapose.Megapose(device)
    global DINOv2
    DINOv2 = dinov2.DINOv2(device)
    name = 'drill'
    stream(name)
