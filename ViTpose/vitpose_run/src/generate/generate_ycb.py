import os
from pathlib import Path
import json
# Third Party
import numpy as np

# MegaPose
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.datasets.scene_dataset import CameraData
from megapose.utils.load_model import load_named_model
from utils import convert
from megapose.config import PROJECT_ROOT



def make_ycb_object_dataset(cad_model_dir: Path, Convert_YCB) -> RigidObjectDataset:
    rigid_objects = []
    mesh_units = "mm"
    object_plys = sorted(cad_model_dir.rglob('*.ply'))
    print("Loading all CAD models from {}, default unit {}, this may take a long time".
          format(cad_model_dir, mesh_units))
    for num, object_ply in enumerate(object_plys):
        label = Convert_YCB.convert_number(num + 1)
        rigid_objects.append(RigidObject(label=label, mesh_path=object_ply, mesh_units=mesh_units))
    rigid_object_dataset = RigidObjectDataset(rigid_objects)
    return rigid_object_dataset

def generate():
    Convert_YCB = convert.Convert_YCB()
    device = 'cuda:0'
    model_name = "megapose-1.0-RGB-multi-hypothesis-icp"
    camera_data_path = PROJECT_ROOT / "data/ycbv_camera_data.json"
    camera_data = CameraData.from_json(camera_data_path.read_text())
    models_path = PROJECT_ROOT / "models/megapose-models"
    cad_path = PROJECT_ROOT / "bop_datasets/ycbv/models"
    save_dir_root = PROJECT_ROOT / "data/ycbv_generated"
    object_dataset = make_ycb_object_dataset(cad_path, Convert_YCB)
    pose_estimator = load_named_model(model_name, models_path, object_dataset).to(device)

    # Read the JSON file
    with open(camera_data_path, 'r') as file:
        data = json.load(file)

    # Extract resolution from JSON data
    height, width = data['resolution']

    # Calculate the center of the new resolution
    center_x = width // 2
    center_y = height // 2

    # Calculate the new detection coordinates (100x100) centered in the new resolution
    half_size = 50
    detection = np.array([
        center_x - half_size,
        center_y - half_size,
        center_x + half_size,
        center_y + half_size
    ], dtype=int)

    for label in Convert_YCB.get_object_list():
        save_dir = save_dir_root / label
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        pose_estimator.image_generation(
            save_dir=save_dir, detection=detection, K=camera_data.K, label=label, device=device
        )


if __name__ == "__main__":
    generate()
