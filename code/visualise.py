import math
import pickle
import re
from collections import defaultdict

from tqdm import tqdm
import os
import copy
import numpy as np
import cv2

activity2id = {
    "BG": 0,  # background
    "activity_walking": 1,
    "activity_standing": 2,
    "activity_carrying": 3,
    "activity_gesturing": 4,
    "Closing": 5,
    "Opening": 6,
    "Interacts": 7,
    "Exiting": 8,
    "Entering": 9,
    "Talking": 10,
    "Transport_HeavyCarry": 11,
    "Unloading": 12,
    "Pull": 13,
    "Loading": 14,
    "Open_Trunk": 15,
    "Closing_Trunk": 16,
    "Riding": 17,
    "specialized_texting_phone": 18,
    "Person_Person_Interaction": 19,
    "specialized_talking_phone": 20,
    "activity_running": 21,
    "PickUp": 22,
    "specialized_using_tool": 23,
    "SetDown": 24,
    "activity_crouching": 25,
    "activity_sitting": 26,
    "Object_Transfer": 27,
    "Push": 28,
    "PickUp_Person_Vehicle": 29,
}


def get_person_box_at_frame(tracking_results, target_frame_idx):
    """Get all the box from the tracking result at frameIdx."""
    data = []
    for frame_idx, track_id, left, top, width, height in tracking_results:
        if frame_idx == target_frame_idx:
            data.append({
                "track_id": track_id,
                "bbox": [left, top, width, height]
            })
    return data


def get_video_meta(vcap):
    """Given the cv2 opened video, get video metadata."""
    if cv2.__version__.split(".")[0] != "2":
        frame_width = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)
        frame_height = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        fps = vcap.get(cv2.CAP_PROP_FPS)
        frame_count = vcap.get(cv2.CAP_PROP_FRAME_COUNT)
    else:
        frame_width = vcap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
        frame_height = vcap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)

        fps = vcap.get(cv2.cv.CV_CAP_PROP_FPS)
        frame_count = vcap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    return {
        "frame_height": frame_height,
        "frame_width": frame_width,
        "fps": fps,
        "frame_count": frame_count}


def plot_traj(img, traj, color):
    """Plot a trajectory on image."""
    traj = np.array(traj, dtype="int")
    points = zip(traj[:-1], traj[1:])

    for p1, p2 in points:
        img = cv2.arrowedLine(img, tuple(p1), tuple(p2), color=color, thickness=1,
                              line_type=cv2.LINE_AA, tipLength=0.2)

    return img


def get_video_dict(pred_data):
    video_dict = defaultdict(list)

    for idx, seq_id in enumerate(pred_data['seq_ids']):
        seq_info = seq_id.split('_')
        video_basename = '_'.join(seq_info[:-2])
        video_dict[video_basename].append(idx)
    return video_dict

def filter_pred(pred_data, pattern ):
    match_list = []
    for idx, seq_id in enumerate(pred_data['seq_ids']):
        pattern = re.compile(pattern)
        if pattern.match(seq_id) is not None:
            match_list.append(idx)

    return match_list

def get_frame_range(pred_data):
    frames = []
    for idx, seq_id in enumerate(pred_data['seq_ids']):
        seq_info = seq_id.split('_')
        start_frame = int(seq_info[-2])
        frames.append(start_frame)
    return frames, min(frames), max(frames)



def visualise(args):
    with open(args.pred_path, "rb") as f:
        pred_data = pickle.load(f)
        pred_data = {
            k: v for k, v in pred_data.items() if k in ['seq_ids', 'obs_list', 'pred_gt_list', 'pred_list', 'pred_act']
        }

    video_dict = get_video_dict(pred_data)
    id2activity = {v:k for k,v in activity2id.items()}
    for video_name in video_dict:
        video_file = os.path.join(args.video_base, video_name + '.mp4')

        # ------------- 1. Get video frames
        tqdm.write("1. Getting video frames...")
        try:
            vcap = cv2.VideoCapture(video_file)
            if not vcap.isOpened():
                raise ValueError("Cannot open %s" % video_file)
        except ValueError as e:
            # skipping this video
            tqdm.write("Skipping %s due to %s" % (video_file, e))
            continue
        video_meta = get_video_meta(vcap)
        video_pred_data = {
            k: [v[i] for i in video_dict[video_name]] for k, v in pred_data.items()
        }
        frames, min_frame, max_frame = get_frame_range(video_pred_data)

            # for seq_id, obs_traj, pred_traj, gt_traj in zip(
            #         pred_data['seq_ids'], pred_data['obs_traj_list'], pred_data['pred_traj_list'], pred_data['gt_traj_list']):
            #
            #
            #
            #     print(seq_id, video_meta)
        cur_frame = - 8 * args.drop_frame
        next_signal = False
        while cur_frame < video_meta["frame_count"]:
            cur_frame += 1
            suc, frame = vcap.read()
            if not suc:
                break
            if cur_frame not in frames: continue

            if cur_frame < min_frame: continue
            if cur_frame > max_frame + args.drop_frame * 20: break

            for idx in filter_pred(video_pred_data, video_name + f'_{cur_frame}_+'):
                seq_id = video_pred_data['seq_ids'][idx]
                obs_traj = video_pred_data['obs_list'][idx]
                pred_traj = video_pred_data['pred_list'][idx]
                gt_traj = video_pred_data['pred_gt_list'][idx]
                # pred_act =sigmoid(np.array( video_pred_data['pred_act'][idx] * 20 ))
                pred_act = video_pred_data['pred_act'][idx]
                frame = cv2.resize(frame, (1920, 1080))
                frame = plot_traj(frame, obs_traj, (255, 0, 0))
                frame = plot_traj(frame, pred_traj, (0, 255, 0))
                frame = plot_traj(frame, gt_traj, (0, 0, 255))
                act_ids = np.nonzero( pred_act > 0.7)[0]
                print(act_ids)
                msg = ','.join([id2activity[id] for id in act_ids])

                frame = cv2.putText(frame,msg, (int(obs_traj[-1,0]), int(obs_traj[-1,1])),cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 0, 0), 1, cv2.LINE_AA )
            # center = obs_traj[-1].astype("int")
            # frame = frame[center[1] - 200: center[1] + 200, center[0] - 200: center[0] + 200]
            frame = cv2.putText(frame, video_name + ' ' + 'frame ' + str(cur_frame), (30,30), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 0, 0), 2, cv2.LINE_AA)
            while True:

                cv2.imshow("windows", frame)
                pressed_keys = cv2.waitKey(1) & 0xFF
                if pressed_keys == ord('q'):
                    exit(0)
                elif pressed_keys == ord('f'):
                    break
                elif pressed_keys == ord('n'):
                    next_signal = True
                    break

            if next_signal: break
        # cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("video_base")
    parser.add_argument("pred_path")
    parser.add_argument("save_dir")
    parser.add_argument("--drop_frame", default=1, type=int)

    args = parser.parse_args()
    visualise(args)
