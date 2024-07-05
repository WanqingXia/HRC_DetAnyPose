#!/usr/bin/env python
#!\file
#
# \author  Wanqing Xia wxia612@aucklanduni.ac.nz
# \date    2024-07-02
#
#
# ---------------------------------------------------------------------

import cv2
import sys
import os
import numpy as np
import json
import time
import torch
import pyaudio
import signal
import threading

import speech_recognition as sr
from openai import OpenAI
from classes import dinov2, megapose, mmdet_sam, k4a_camera
from utils.choose import validate_preds
from utils.convert import Convert_YCB
from generate import generate_ycb
import base64
from scipy.spatial.transform import Rotation as R

import rospy
from geometry_msgs.msg import Pose
from std_msgs.msg import String

def sig_handler(signal, frame):
    print("Existing Program...")
    sys.exit(0)


# Define the directory you want to add to sys.path
project_root_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '/megapose'))

# Append the directory to sys.path
if project_root_directory not in sys.path:
    sys.path.insert(0, project_root_directory)

from megapose.config import PROJECT_ROOT

dinov2_dir = str(PROJECT_ROOT / 'dinov2')
if dinov2_dir not in sys.path:
    sys.path.insert(1, dinov2_dir)

# Initialize OpenAI API
client = OpenAI(
    # This is the default and can be omitted
    api_key="sk-proj-YROABCgnjm08g5hMZvYGT3BlbkFJC1Eaj1Awq5VOhZB6mMCj",
)

def gpt4_api_call(system_message, text_message, image_message=None):
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": [
                {"type": "text", "text": system_message}
            ]},
            {"role": "user", "content": [
                {"type": "text", "text": text_message},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{image_message}"}
                 }
            ]}
        ],
        model="gpt-4o",
        temperature=1.0,
    )
    return chat_completion.choices[0].message.content


def stream(desc_name: str):
    pose_estimation = None
    try_count = 0

    while pose_estimation is None and try_count < 3:
        ret_color, color, ret_depth, depth = K4A_Camera.get_capture()
        if not ret_color or not ret_depth:
            continue
        else:
            color = cv2.cvtColor(color, cv2.COLOR_BGRA2RGB)
            depth = np.array(depth, dtype=np.float32) / 1000
            print("Estimating {} pose...".format(desc_name))
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

            if pose_estimation is None:
                try_count += 1
                print("No pose estimation found")
                continue
            else:
                pose_matrix = pose_estimation.poses.cpu().numpy()
                rotation = R.from_matrix(pose_matrix[0, :3, :3]).as_quat()
                translation = pose_matrix[0, :3, 3]
                print(f'Pose estimated for {ycb_name} is translation: {translation} and rotation: {rotation}')
                contour_image, mesh_image = Megapose.get_output_image(color, pose_estimation, ycb_name)
                mesh_image = cv2.cvtColor(mesh_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f'{str(os.path.abspath(os.path.dirname(__file__)))}/Pose estimation result {ycb_name}.jpg', mesh_image)
                cv2.imshow('Pose estimation result', mesh_image)
                # Wait for a key press indefinitely or for a specified amount of time in milliseconds
                cv2.waitKey(0)
                # Destroy all the windows created by OpenCV
                cv2.destroyAllWindows()
                return np.hstack([translation, rotation])
    return None

def recognize_speech_from_microphone(system_command: str):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        try:
            print("Adjusting for ambient noise, please wait...")
            recognizer.adjust_for_ambient_noise(source, duration=5)
            print("Microphone adjusted. Listening...")

            while True:
                print("Say something!")
                audio = recognizer.listen(source, phrase_time_limit=4)
                print("Recognizing...")

                # Recognize speech using Google Web Speech API
                try:
                    user_command = recognizer.recognize_google(audio)
                    print(f"Recognized text: {user_command}")

                    _, color_img, _, _ = K4A_Camera.get_capture()
                    while color_img is None:
                        _, color_img, _, _ = K4A_Camera.get_capture()
                    color_img = cv2.cvtColor(color_img, cv2.COLOR_BGRA2RGB)
                    color_img = cv2.resize(color_img, (640, 480))
                    # Encode the image to PNG format
                    success, encoded_image = cv2.imencode('.png', color_img)

                    base64img = None
                    if success:
                        # Convert the encoded image to bytes
                        img_bytes = encoded_image.tobytes()
                        base64img = base64.b64encode(img_bytes).decode('utf-8')

                    response = gpt4_api_call(system_command, user_command, base64img)
                    print(f"GPT-4 response: {response}")

                    if str(response).lower() != "none":
                        return response
                    else:
                        print("Going back to voice recognition...")

                except sr.UnknownValueError:
                    print("Google Web Speech API could not understand the audio.")
                except sr.RequestError as e:
                    print(f"Could not request results from Google Web Speech API; {e}")

        except KeyboardInterrupt:
            print("Listening stopped.")


def main():
    while True:
        # Initialize the GPT-4 API
        system_command = ('You are an AI assistant. Here is a list of object names: \n'
                           'blue cylindrical can\n'
                           'red cracker cardbox\n'
                           'yellow sugar cardbox\n'
                           'red cylindrical can\n'
                           'yellow mustard bottle\n'
                           'tuna fish tin can\n'
                           'brown jelly cardbox\n'
                           'red jelly cardbox\n'
                           'spam rectangular can\n'
                           'banana\n'
                           'blue cup\n'
                           'white bleach bottle\n'
                           'red bowl\n'
                           'red cup\n'
                           'drill\n'
                           'wooden block\n'
                           'scissors\n'
                           'marker pen\n'
                           'black clamp\n'
                           'bigger black clamp\n'
                           'red rectangular block\n'
                           'I will later give you a command and a picture, directly give the object name objects from the list that are visible in the provided image based on the user command without additional context. If no objects from the list are visible in the image or fit user requirements, respond \'None\'. The name of an object also implies it\'s content, for example, tuna fish tin can means it contains tuna fish, so when I ask you to pass me something to eat and tuna fish tin can is in the image, you should replyn \'tuna fish tin can\'.')

        # Start the voice recognition loop
        gpt_response = recognize_speech_from_microphone(system_command)
        pose_estimation = stream(str(gpt_response))

        if pose_estimation is not None:
            object_pose.position.x = pose_estimation[0]
            object_pose.position.y = pose_estimation[1]
            object_pose.position.z = pose_estimation[2]
            object_pose.orientation.x = pose_estimation[3]
            object_pose.orientation.y = pose_estimation[4]
            object_pose.orientation.z = pose_estimation[5]
            object_pose.orientation.w = pose_estimation[6]
            object_name.data = str(gpt_response)


def vision_publish():
    r = rospy.Rate(100)
    while not rospy.is_shutdown():
        vision_pub.publish(object_pose)
        name_pub.publish(object_name)
        r.sleep()


if __name__ == "__main__":
    device = 'cuda:0'
    signal.signal(signal.SIGINT, sig_handler)
    convert_YCB = Convert_YCB()
    MMDet_SAM = mmdet_sam.MMDet_SAM(device)
    K4A_Camera = k4a_camera.K4A_Camera()
    calibration = K4A_Camera.get_calibration()
    # modify the calibration to match the camera data
    color_img = None
    while color_img is None:
        _, color_img, _, _ = K4A_Camera.get_capture()

    # Define the path to your JSON file
    json_file_path = PROJECT_ROOT / 'data/ycbv_camera_data.json'

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
        json.dump(data, file)

    if not os.path.exists(PROJECT_ROOT / 'data/ycbv_generated'):
        print("Generating viewpoints for YCB objects...")
        generate_ycb.generate()

    Megapose = megapose.Megapose(device)
    DINOv2 = dinov2.DINOv2(device)
    object_pose = Pose()
    object_name = String()
    object_pose.position.x = 0.0
    object_pose.position.y = 0.0
    object_pose.position.z = 0.0
    object_pose.orientation.x = 0.0
    object_pose.orientation.y = 0.0
    object_pose.orientation.z = 0.0
    object_pose.orientation.w = 1.0
    object_name.data = "none"
    rospy.init_node("object")
    vision_pub = rospy.Publisher('/object/pose', Pose, queue_size=10)
    name_pub = rospy.Publisher('/object/name', String, queue_size=10)
    vision_thread = threading.Thread(target=vision_publish)
    vision_thread.daemon = True  # This ensures the thread will exit when the main program exits
    vision_thread.start()

    main()
