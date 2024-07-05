# Standard Library
import json
import os
from pathlib import Path
from typing import List, Tuple, Union

# Third Party
import numpy as np
from bokeh.io import export_png
from bokeh.plotting import gridplot
from bokeh.io.export import get_screenshot_as_png

import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torch
from utils.convert import Convert_YCB

# MegaPose
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.datasets.scene_dataset import CameraData, ObjectData
from megapose.inference.types import (
    DetectionsType,
    ObservationTensor,
    PoseEstimatesType,
)
from megapose.inference.utils import make_detections_from_object_data
from megapose.lib3d.transform import Transform
from megapose.panda3d_renderer import Panda3dLightData
from megapose.panda3d_renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from megapose.utils.conversion import convert_scene_observation_to_panda3d
from megapose.utils.load_model import NAMED_MODELS, load_named_model
from megapose.utils.logging import get_logger
from megapose.visualization.bokeh_plotter import BokehPlotter
from megapose.visualization.utils import make_contour_overlay

logger = get_logger(__name__)

from megapose.config import PROJECT_ROOT

class Megapose:
    def __init__(self, device):
        self.device = device
        self.Convert_YCB = Convert_YCB()
        self.model_name = "megapose-1.0-RGB-multi-hypothesis"
        self.model_info = NAMED_MODELS[self.model_name]
        self.camera_data = CameraData.from_json((PROJECT_ROOT / 'data/ycbv_camera_data.json').read_text())
        self.models_path = PROJECT_ROOT / "models/megapose-models"
        self.cad_path = PROJECT_ROOT / "bop_datasets/ycbv/models"
        self.object_dataset = self.make_ycb_object_dataset(self.cad_path)
        self.renderer = Panda3dSceneRenderer(self.object_dataset)
        logger.info(f"Loading model {self.model_name}.")
        self.pose_estimator = load_named_model(self.model_name, self.models_path, self.object_dataset).to(self.device)
        self.renders_path = PROJECT_ROOT / "data/ycbv_generated"
        renders = self.load_renders(self.renders_path)
        self.pose_estimator.attach_renders(renders)

    def inference(self, rgb, depth, label, bbox):
        """
        :param rgb: np array of the RGB image, np.uint8 type
        :param depth: np array of the depth image, np.float32 type or None
        :param label: object name in string format
        :param bbox: bounding box of the object [xmin, ymin, xmax, ymax] format
        :return: prediction result in RT
        """
        # make sure the size of camera input and images are same
        assert rgb.shape[:2] == self.camera_data.resolution
        assert depth.shape[:2] == self.camera_data.resolution
        observation = ObservationTensor.from_numpy(rgb, depth, self.camera_data.K).to_cuda(device=self.device)

        object_data = [ObjectData(label=label, bbox_modal=bbox)]
        detections = make_detections_from_object_data(object_data).to(self.device)

        output, _ = self.pose_estimator.run_inference_pipeline(
            observation, detections=detections, run_detector=False, **self.model_info["inference_parameters"]
        )

        return output

    def refinement_color(self, rgb, estimation):
        """
        :param rgb: np array of the RGB image, np.uint8 type
        :param depth: np array of the depth image, np.float32 type or None
        :param label: object name in string format
        :param bbox: bounding box of the object [xmin, ymin, xmax, ymax] format
        :return: prediction result in RT
        """
        # make sure the size of camera input and images are same
        assert rgb.shape[:2] == self.camera_data.resolution
        observation = ObservationTensor.from_numpy(rgb, K=self.camera_data.K).to_cuda(device=self.device)
        refiner_iterations = 1
        output, _ = self.pose_estimator.forward_refiner(observation, estimation, n_iterations=refiner_iterations)
        del observation, estimation, _

        return output[f"iteration={refiner_iterations}"]

    def refinement_depth(self, rgb, depth, estimation):
        """
        :param rgb: np array of the RGB image, np.uint8 type
        :param depth: np array of the depth image, np.float32 type or None
        :param label: object name in string format
        :param bbox: bounding box of the object [xmin, ymin, xmax, ymax] format
        :return: prediction result in RT
        """
        # make sure the size of camera input and images are same
        assert rgb.shape[:2] == self.camera_data.resolution
        assert depth.shape[:2] == self.camera_data.resolution
        observation = ObservationTensor.from_numpy(rgb, depth, self.camera_data.K).to_cuda(device=self.device)
        output, _ = self.pose_estimator.run_depth_refiner(observation, estimation)

        return output

    def get_output_image(self, rgb, pose_estimate, label):
        self.camera_data.TWC = Transform(np.eye(4))
        pose = pose_estimate.poses.cpu().numpy()
        del pose_estimate

        object_data = [ObjectData(label=label, TWO=Transform(pose.squeeze()))]

        camera_data, object_datas = convert_scene_observation_to_panda3d(self.camera_data, object_data)
        light_datas = [
            Panda3dLightData(
                light_type="ambient",
                color=((1.0, 1.0, 1.0, 1)),
            ),
        ]
        renderings = self.renderer.render_scene(
            object_datas,
            [camera_data],
            light_datas,
            render_depth=False,
            render_binary_mask=False,
            render_normals=False,
            copy_arrays=True,
        )[0]

        plotter = BokehPlotter()

        fig_mesh_overlay = plotter.plot_overlay(rgb, renderings.rgb)
        contour_overlay = make_contour_overlay(
            rgb, renderings.rgb, dilate_iterations=1, color=(0, 255, 0)
        )["img"]
        fig_contour_overlay = plotter.plot_image(contour_overlay)

        mesh_overlay_image = get_screenshot_as_png(fig_mesh_overlay)
        mesh_overlay_image = np.array(mesh_overlay_image)

        contour_overlay_image = get_screenshot_as_png(fig_contour_overlay)
        contour_overlay_image = np.array(contour_overlay_image)
        del plotter, renderings, contour_overlay, fig_mesh_overlay, fig_contour_overlay, camera_data, object_datas
        return contour_overlay_image, mesh_overlay_image


    def visualise_output(self, rgb, pose_estimate, label):
        self.camera_data.TWC = Transform(np.eye(4))
        pose = pose_estimate.poses.cpu().numpy()

        object_data = [ObjectData(label=label, TWO=Transform(pose.squeeze()))]
        renderer = Panda3dSceneRenderer(self.object_dataset)

        camera_data, object_datas = convert_scene_observation_to_panda3d(self.camera_data, object_data)
        light_datas = [
            Panda3dLightData(
                light_type="ambient",
                color=((1.0, 1.0, 1.0, 1)),
            ),
        ]
        renderings = renderer.render_scene(
            object_datas,
            [camera_data],
            light_datas,
            render_depth=False,
            render_binary_mask=False,
            render_normals=False,
            copy_arrays=True,
        )[0]

        plotter = BokehPlotter()

        fig_mesh_overlay = plotter.plot_overlay(rgb, renderings.rgb)
        contour_overlay = make_contour_overlay(
            rgb, renderings.rgb, dilate_iterations=1, color=(0, 255, 0)
        )["img"]
        fig_contour_overlay = plotter.plot_image(contour_overlay)

        mesh_overlay_image = get_screenshot_as_png(fig_mesh_overlay)
        mesh_overlay_image = np.array(mesh_overlay_image)
        # Plotting with Matplotlib
        plt.figure()
        plt.imshow(mesh_overlay_image)
        plt.title('Mesh Overlay Result')
        plt.axis('off')  # Hide axes
        plt.show()

        contour_overlay_image = get_screenshot_as_png(fig_contour_overlay)
        contour_overlay_image = np.array(contour_overlay_image)
        plt.figure()
        plt.imshow(contour_overlay_image)
        plt.title('Contour Overlay Result')
        plt.axis('off')  # Hide axes
        plt.show()

    def make_ycb_object_dataset(self, cad_model_dir: Path) -> RigidObjectDataset:
        rigid_objects = []
        mesh_units = "mm"
        object_plys = sorted(cad_model_dir.rglob('*.ply'))
        print("Loading all CAD models from {}, default unit {}.".
              format(cad_model_dir, mesh_units))
        for num, object_ply in enumerate(object_plys):
            label = self.Convert_YCB.convert_number(num + 1)
            rigid_objects.append(RigidObject(label=label, mesh_path=object_ply, mesh_units=mesh_units))
        rigid_object_dataset = RigidObjectDataset(rigid_objects)
        return rigid_object_dataset

    def save_predictions(self, out_dir, pose_estimates):
        labels = pose_estimates.infos["label"]
        poses = pose_estimates.poses.cpu().numpy()
        object_data = [
            ObjectData(label=label, TWO=Transform(pose)) for label, pose in zip(labels, poses)
        ]
        object_data_json = json.dumps([x.to_json() for x in object_data])
        output_fn = Path(out_dir) / "object_data.json"
        output_fn.parent.mkdir(exist_ok=True)
        output_fn.write_text(object_data_json)
        logger.info(f"Wrote predictions: {output_fn}")

    def load_renders(self, renders_path):
        # Dictionary to hold the tensors for each sub-folder
        folder_tensors = {}

        # Transform to convert images to tensors and normalize by 255
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts to [C, H, W] and scales pixel values to [0, 1]
        ])

        # Iterate over each sub-folder in the root directory
        for sub_folder in os.listdir(renders_path):
            sub_folder_path = renders_path / sub_folder

            if os.path.isdir(sub_folder_path):
                # Initialize list to hold pairs of rgb and normal images
                png_files = sorted(list(sub_folder_path.rglob('*.png')))
                images_list = []
                num_files = int(len(png_files) / 2)

                for i in range(num_files):
                    # Load rgb image
                    padded_num = "{:03d}".format(i)
                    rgb_path = os.path.join(sub_folder_path, f'rgb_{padded_num}.png')
                    rgb_image = Image.open(rgb_path)
                    rgb_tensor = transform(rgb_image)

                    # Load normal image
                    normal_path = os.path.join(sub_folder_path, f'normal_{padded_num}.png')
                    normal_image = Image.open(normal_path)
                    normal_tensor = transform(normal_image)

                    # Concatenate the rgb and normal tensors along the channel dimension
                    combined_tensor = torch.cat((rgb_tensor, normal_tensor), dim=0)
                    images_list.append(combined_tensor)

                # Stack all image tensors to create a single tensor of shape [576, 6, H, W]
                folder_tensor = torch.stack(images_list)
                folder_tensors[sub_folder] = folder_tensor

        return folder_tensors
