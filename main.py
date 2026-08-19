
import cv2
import numpy as np
from ultralytics import YOLO
import os
import pandas as pd
import matplotlib.ticker as ticker
from scipy.signal import find_peaks 
from scipy.signal import savgol_filter

from longJump import tracking
from longJump.config import ROOT, PATHS, DETECTION, ATHLETE, TRACKING
from longJump.video import Video
from longJump import visualization
from longJump.athlete import Athlete
from longJump import detection


video_path = str(ROOT / PATHS["videos"] / "professionalLongJumpVid.mp4")
save_path = str(ROOT / PATHS["output"] / "professionalLongJumpVid.mp4")
model_path = str(ROOT / PATHS["model"])
tracker_path = str(ROOT / PATHS["tracker"])


video = Video(video_path)



pose_results = tracking.run_tracking(model_path, tracker_path, video_path, conf=TRACKING["conf"], iou=TRACKING["iou"],imgsz=TRACKING["imgsz"])

bounding_boxes_df, video_keypoints_df = tracking.process_tracking_results(pose_results)

athlete_id = -1
mostMovement = 0

video_keypoints_df = video_keypoints_df.sort_values(by=["person_id", "frame"]) # sort by frame and person_id

for pid in video_keypoints_df["person_id"].unique():
    anklePositions = video_keypoints_df.loc[video_keypoints_df["person_id"] == pid, ['left_y', "left_hip_y"]]
    if (len(anklePositions) < video.fps):
        continue # Discard ppl not on frame for more than 30 frames
    
    totalMovement = (anklePositions["left_hip_y"] - anklePositions["left_y"]).abs().diff().abs().sum() # total the movement of the ankle between frames, because the athlete probably has a cyclical motion with lots of ankle movement, while spectators don't


    if (totalMovement.item() > mostMovement):
        mostMovement = totalMovement.item()
        athlete_id = pid
print(athlete_id)




import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


athlete = Athlete(bounding_boxes_df, video_keypoints_df, athlete_id, DETECTION["minimum_movement_threshold"], video.fps)

direction_score = athlete.CalculateDirectionScore(ATHLETE["knee_peaks_prominence"], ATHLETE["minimum_direction_score"])

runningLeftToRight = True


# 4. Make the final directional decision
if direction_score > 0:
    runningLeftToRight = True
    print(f"Direction: Left-to-Right (Score: {direction_score:.3f})")
else:
    runningLeftToRight = False
    print(f"Direction: Right-to-Left (Score: {direction_score:.3f})")


if (not runningLeftToRight):
    athlete.df["left_ankle_x_rel"] *= -1
    athlete.df["right_ankle_x_rel"] *= -1




sprint_peaks_left, sprint_peaks_right = athlete.findPeaks(DETECTION["leg_extension_peak_prominence"], DETECTION["leg_extension_peak_height"], DETECTION["leg_extension_min_gap"], DETECTION["leg_extension_max_gap"], DETECTION["consecutive_extension_peak"])

sprint_peaks = np.concatenate((np.array(sprint_peaks_left), np.array(sprint_peaks_right)))
sprint_peaks.sort()




frames_before_max_extension = int(DETECTION["possible_initial_contact_frames_before_leg_extension"])
frames_after_max_extension = int(round(video.fps * DETECTION["possible_initial_contact_frames_after_leg_extension"]))
initial_contact_frames = []

for max_extension_frame in sprint_peaks:
    isLeft = False
    if (max_extension_frame in sprint_peaks_left):
        isLeft = True

    ankle_df = athlete.GetAnkleVerticalDeacceleration(max_extension_frame, isLeft, frames_before_max_extension, frames_after_max_extension)



    peak_deaccel = find_peaks((ankle_df["deaccel"]) * -1)[0]

    detected_frames = ankle_df["frame"].iloc[peak_deaccel].values

    if (len(detected_frames) <= 0):
        raise detection.InitialContactNotFound()
    else:
        candidate_rows = ankle_df[ankle_df["frame"].isin(detected_frames)]

        best_row_idx = candidate_rows["x_dist"].idxmin()

        chosen_frame = candidate_rows.loc[best_row_idx, "frame"]
        initial_contact_frames.append(chosen_frame)

print(initial_contact_frames)



if (video == None):
    video = Video(video_path)

with video as cap:
    visualization.processVideo(cap, video, save_path, bounding_boxes_df, video_keypoints_df, athlete_id, initial_contact_frames, sprint_peaks, sprint_peaks_left)


print(f"Output image saved to {save_path}")


