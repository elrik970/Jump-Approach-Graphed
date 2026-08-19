from ultralytics import YOLO
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
from longJump.video import Video

def RunTrackingBoundingBox(model_path, tracker_path, video_path, conf, iou, imgsz):
    """ Run YOLO model and returns YOLO structured output

    Args: 
        model_path: String. The path from the root to the YOLO pose model
        video_path: String. The path from the root to the input video

    Returns:
        YOLO pose model results

    """

    video = Path(video_path)
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video}")

    model = YOLO(model_path)

    pose_results = model.track(
        video_path,
        conf=conf,          # Lowered detection floor to catch rapid motion blur
        iou=iou,           # Increased NMS overlap to prevent duplicate box drops
        imgsz=imgsz,
        classes=[0],
        device='cuda:0',
        half=True,
        verbose=False,
        #persist=True,
        tracker=tracker_path,  # fastTrack with custom parameters
        stream=True

    )

    return pose_results

def ProcessBoundingBoxResults(model_results):
    """ Extracts bounding box informaiton out of model.track results
    
    Args:
        pose_results: Model results

    Returns:
         Tuple of (bounding_boxes_df, keypoints_df)

    """
    # print(model_results)

    person_bboxes = []

    for frame_num, result in enumerate(model_results):


        if result.boxes is None: # If the checker doesn't have bounding Boxes then we can't salvedge it
            print("NO BOXES!!!", frame_num)
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()


        if (result.boxes.id is not None):
            
            track_ids = result.boxes.id.int().cpu().numpy()
        else:
            track_ids = None

        for person_num in range(len(boxes)):

            if (track_ids is not None and person_num < len(track_ids)):
                personid = int(track_ids[person_num])
            else:
                personid = -1

            # print(boxes)
            
            boundingBox = {
                    "frame": frame_num,
                    "person_id": personid,
                    "x1": boxes[person_num][0],
                    "y1": boxes[person_num][1],
                    "x2": boxes[person_num][2],
                    "y2": boxes[person_num][3],
                    "conf":  float(confs[person_num])
            }
            person_bboxes.append(boundingBox)

    bounding_boxes_df = pd.DataFrame(person_bboxes)
    return bounding_boxes_df





def RunTrackingPose(model_path, tracker_path, video_path, conf, iou, imgsz):
    """ Run YOLO model and returns YOLO structured output

    
        0. Nose
        1. Left Eye
        2. Right Eye
        3. Left Ear
        4. Right Ear
        5. Left Shoulder
        6. Right Shoulder
        7. Left Elbow
        8. Right Elbow
        9. Left Wrist
        10. Right Wrist
        11. Left Hip
        12. Right Hip
        13. Left Knee
        14. Right Knee
        15. Left Ankle
        16. Right Ankle


    Args: 
        model_path: String. The path from the root to the YOLO pose model
        video_path: String. The path from the root to the input video

    Returns:
        YOLO pose model results

    """

    video = Path(video_path)
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video}")

    pose_model = YOLO(model_path)

    pose_results = pose_model.track(
        video_path,
        conf=conf,          # Lowered detection floor to catch rapid motion blur
        iou=iou,           # Increased NMS overlap to prevent duplicate box drops
        imgsz=imgsz,
        classes=[0],
        device='cuda:0',
        half=True,
        verbose=False,
        #persist=True,
        tracker=tracker_path,  # fastTrack with custom parameters
        stream=True

    )

    return pose_results

def ProcessTrackingResults(pose_results) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    """ Extracts bounding box and keypoint information as 2 dataframes out of model results
    
    Args:
        pose_results: Model results

    Returns:
         Tuple of (bounding_boxes_df, keypoints_df)

    """

    video_keypoints = []
    person_bboxes = []

    for frame_num, result in enumerate(pose_results):


        if result.boxes is None: # If the checker doesn't have bounding Boxes then we can't salvedge it
            print("NO BOXES!!!", frame_num)
            continue

        boxes = result.boxes.xyxy.cpu().numpy()

        if (result.boxes.id is not None):
            
            track_ids = result.boxes.id.int().cpu().numpy()
        else:
            track_ids = None

        for person_num in range(len(boxes)):

            if (track_ids is not None and person_num < len(track_ids)):
                personid = int(track_ids[person_num])
            else:
                personid = -1


            
            boundingBox = {
                    "frame": frame_num,
                    "person_id": personid,
                    "x1": boxes[person_num][0],
                    "y1": boxes[person_num][1],
                    "x2": boxes[person_num][2],
                    "y2": boxes[person_num][3]
            }
            person_bboxes.append(boundingBox)

            if (result.keypoints.xy is not None):



                left_ankle = result.keypoints.xy[person_num][15].cpu().numpy() # list len 2 with x position and y position
                right_ankle = result.keypoints.xy[person_num][16].cpu().numpy()
                left_hip = result.keypoints.xy[person_num][11].cpu().numpy()
                right_hip = result.keypoints.xy[person_num][12].cpu().numpy()
                left_knee = result.keypoints.xy[person_num][13].cpu().numpy()
                right_knee = result.keypoints.xy[person_num][14].cpu().numpy()
                left_shoulder = result.keypoints.xy[person_num][5].cpu().numpy()
                right_shoulder = result.keypoints.xy[person_num][6].cpu().numpy()


                
                # 2. Create one flat dictionary per detection row
                row_data = {
                    "frame": frame_num,
                    "person_id": personid,
                    "left_x": float(left_ankle[0]),
                    "left_y": float(left_ankle[1]),
                    "right_x": float(right_ankle[0]),
                    "right_y": float(right_ankle[1]),
                    "left_hip_x": float(left_hip[0]),
                    "left_hip_y": float(left_hip[1]),
                    "right_hip_x": float(right_hip[0]),
                    "right_hip_y": float(right_hip[1]),
                    "left_knee_x": float(left_knee[0]),
                    "left_knee_y": float(left_knee[1]),
                    "right_knee_x": float(right_knee[0]),
                    "right_knee_y": float(right_knee[1]),
                    "left_shoulder_x": float(left_shoulder[0]),
                    "left_shoulder_y": float(left_shoulder[1]),
                    "right_shoulder_x": float(right_shoulder[0]),
                    "right_shoulder_y": float(right_shoulder[1]),

                    "left_conf": float(result.keypoints.conf[person_num][15]), # confidence score of left ankle
                    "right_conf": float(result.keypoints.conf[person_num][16]), # confidence score of right ankle
                    "left_hip_conf":   float(result.keypoints.conf[person_num][11]),
                    "right_hip_conf":  float(result.keypoints.conf[person_num][12]),
                    "left_knee_conf":  float(result.keypoints.conf[person_num][13]),
                    "right_knee_conf": float(result.keypoints.conf[person_num][14]),
                    "left_shoulder_conf": float(result.keypoints.conf[person_num][5]),
                    "right_shoulder_conf": float(result.keypoints.conf[person_num][6])

                }
                
                # Append this specific row to our master list
                video_keypoints.append(row_data)
            else:
                print("NO KEYPOINTS", frame_num)



    video_keypoints_df = pd.DataFrame(video_keypoints)

    bounding_boxes_df = pd.DataFrame(person_bboxes)
    
    return bounding_boxes_df, video_keypoints_df

def InterpolateLowConfidenceAnkles(video_keypoints_df, joints_to_conf, conf_threshold):

    

    # for pid in video_keypoints_df["person_id"].unique():
    #     mask = video_keypoints_df["person_id"] == pid
        
    #     for joint, (x_col, y_col) in joints.items():
    #         low_conf = mask & (video_keypoints_df[f'{joint}_conf'] < conf_threshold)
            
    #         video_keypoints_df.loc[low_conf, x_col] = np.nan
    #         video_keypoints_df.loc[low_conf, y_col] = np.nan
            
    #         video_keypoints_df.loc[mask, x_col] = video_keypoints_df.loc[mask, x_col].interpolate(method='linear').ffill().bfill()
    #         video_keypoints_df.loc[mask, y_col] = video_keypoints_df.loc[mask, y_col].interpolate(method='linear').ffill().bfill()


    # video_keypoints_df = video_keypoints_df.set_index("frame", drop=False)

    video_keypoints_df = video_keypoints_df.sort_values("frame").reset_index(drop=True)


    for joint in joints_to_conf.keys():

        video_keypoints_df.loc[video_keypoints_df[joints_to_conf[joint]] < conf_threshold, joint] = np.nan

        video_keypoints_df.loc[video_keypoints_df[joints_to_conf[joint]] < conf_threshold, joints_to_conf[joint]] = conf_threshold


        video_keypoints_df[joint] = video_keypoints_df[joint].interpolate(method='linear').ffill().bfill()

    return video_keypoints_df




def run_pose_on_athlete_crops(cap, video, athlete_boxes, pose_model, padding_fraction=0.1, debug=False):
    """
    Reads video frame by frame, crops to athlete bounding box,
    runs pose estimation on crop, remaps keypoints to full frame coordinates.
    
    Args:
        cap: cap
        video: Video class
        athlete_boxes: DataFrame with columns [frame, x1, y1, x2, y2] for athlete only
        pose_model: loaded YOLO pose model
        padding_fraction: padding as fraction of athlete height
    
    Returns:
        Tuple of (bounding_boxes_df, keypoints_df)
    """
    
    # Pre-index by frame for O(1) lookup
    boxes_by_frame = athlete_boxes.set_index("frame")
    
    video_keypoints = []
    person_bboxes = []

    frame_num = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_num in boxes_by_frame.index:
            row = boxes_by_frame.loc[frame_num]
            
            x1 = int(row["x1"])
            y1 = int(row["y1"])
            x2 = int(row["x2"])
            y2 = int(row["y2"])
            
            # Scale padding to athlete size
            pad = int((y2 - y1) * padding_fraction)
            height = frame.shape[0]
            width = frame.shape[1]
            
            x1_pad = max(0, x1 - pad) # Don't let x1 - pad become negative
            y1_pad = max(0, y1 - pad)
            x2_pad = min(width, x2 + pad)
            y2_pad = min(height, y2 + pad)
            
            crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]

            
            
            # Run pose on crop — single person so always index [0]
            result = pose_model.predict(crop, verbose=False)[0]
            
            if result.keypoints is not None and len(result.keypoints.xy) > 0:
                kps = result.keypoints.xy[0].cpu().numpy()
                conf = result.keypoints.conf[0].cpu().numpy()

                if debug:
                    plt.figure(figsize=(5,3))
                    plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)) # Convert Blue Green Red color channels to Red Green Blue using cv2
                    circle1 = plt.Circle((kps[15][0], kps[15][1]), 3, color='g')
                    circle2 = plt.Circle((kps[16][0], kps[16][1]), 3, color='g')

                    circle3 = plt.Circle((kps[11][0], kps[11][1]), 3, color='g')
                    circle4 = plt.Circle((kps[12][0], kps[12][1]), 3, color='g')

                    circle5 = plt.Circle((kps[13][0], kps[13][1]), 3, color='g')
                    circle6 = plt.Circle((kps[14][0], kps[14][1]), 3, color='g')

                    plt.gca().add_patch(circle1)
                    plt.gca().add_patch(circle2)
                    plt.gca().add_patch(circle3)
                    plt.gca().add_patch(circle4)
                    plt.gca().add_patch(circle5)
                    plt.gca().add_patch(circle6)

                    plt.axis('off')  
                    plt.show()
                
                # Remap back to full frame coordinates
                # kps[:, 0] += x1_pad # [:, ] empty column means get every row, 0 means just the first column, the x coordinate of the keypoints
                # kps[:, 1] += y1_pad # [:, ] empty column means get every row, 1 means just the second column, the y coordinate of the keypoints
                
                video_keypoints.append({
                    "frame": frame_num,
                    "person_id": 1,  # always the athlete
                    "left_x":       float(kps[15][0]),
                    "left_y":       float(kps[15][1]),
                    "right_x":      float(kps[16][0]),
                    "right_y":      float(kps[16][1]),
                    "left_hip_x":   float(kps[11][0]),
                    "left_hip_y":   float(kps[11][1]),
                    "right_hip_x":  float(kps[12][0]),
                    "right_hip_y":  float(kps[12][1]),
                    "left_knee_x":  float(kps[13][0]),
                    "left_knee_y":  float(kps[13][1]),
                    "right_knee_x": float(kps[14][0]),
                    "right_knee_y": float(kps[14][1]),
                    "left_shoulder_x":  float(kps[5][0]),
                    "left_shoulder_y":  float(kps[5][1]),
                    "right_shoulder_x": float(kps[6][0]),
                    "right_shoulder_y": float(kps[6][1]),


                    "left_conf":    float(conf[15]),
                    "right_conf":   float(conf[16]),
                    "left_hip_conf":   float(conf[11]),
                    "right_hip_conf":  float(conf[12]),
                    "left_knee_conf":  float(conf[13]),
                    "right_knee_conf": float(conf[14]),
                    "left_shoulder_conf": float(conf[5]),
                    "right_shoulder_conf": float(conf[6])
                })
                
                person_bboxes.append({
                    "frame": frame_num,
                    "person_id": 1,
                    "x1": x1, "y1": y1,
                    "x2": x2, "y2": y2
                })
        
        frame_num += 1
    
    cap.release()
    
    return pd.DataFrame(person_bboxes), pd.DataFrame(video_keypoints)




def run_pose_on_athlete_crops_RTM_POSE(cap, video, athlete_boxes, pose_model, padding_fraction=0.1, debug=False):
    """
    Reads video frame by frame, crops to athlete bounding box,
    runs pose estimation on crop, remaps keypoints to full frame coordinates.
    
    Args:
        cap: cap
        video: Video class
        athlete_boxes: DataFrame with columns [frame, x1, y1, x2, y2] for athlete only
        pose_model: loaded YOLO pose model
        padding_fraction: padding as fraction of athlete height
    
    Returns:
        Tuple of (bounding_boxes_df, keypoints_df)
    """
    
    # Pre-index by frame for O(1) lookup
    boxes_by_frame = athlete_boxes.set_index("frame")
    
    video_keypoints = []
    person_bboxes = []

    frame_num = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_num in boxes_by_frame.index:
            row = boxes_by_frame.loc[frame_num]
            
            x1 = int(row["x1"])
            y1 = int(row["y1"])
            x2 = int(row["x2"])
            y2 = int(row["y2"])
            
            # Scale padding to athlete size
            pad = int((y2 - y1) * padding_fraction)
            height = frame.shape[0]
            width = frame.shape[1]
            
            x1_pad = max(0, x1 - pad) # Don't let x1 - pad become negative
            y1_pad = max(0, y1 - pad)
            x2_pad = min(width, x2 + pad)
            y2_pad = min(height, y2 + pad)
            
            crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]

            
            
            # Run pose on crop — single person so always index [0]
            result = pose_model.predict(crop, verbose=False)[0]
            
            if result.keypoints is not None and len(result.keypoints.xy) > 0:
                kps = result.keypoints.xy[0].cpu().numpy()
                conf = result.keypoints.conf[0].cpu().numpy()

                if debug:
                    plt.figure(figsize=(5,3))
                    plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)) # Convert Blue Green Red color channels to Red Green Blue using cv2
                    circle1 = plt.Circle((kps[15][0], kps[15][1]), 3, color='g')
                    circle2 = plt.Circle((kps[16][0], kps[16][1]), 3, color='g')

                    circle3 = plt.Circle((kps[11][0], kps[11][1]), 3, color='g')
                    circle4 = plt.Circle((kps[12][0], kps[12][1]), 3, color='g')

                    circle5 = plt.Circle((kps[13][0], kps[13][1]), 3, color='g')
                    circle6 = plt.Circle((kps[14][0], kps[14][1]), 3, color='g')

                    plt.gca().add_patch(circle1)
                    plt.gca().add_patch(circle2)
                    plt.gca().add_patch(circle3)
                    plt.gca().add_patch(circle4)
                    plt.gca().add_patch(circle5)
                    plt.gca().add_patch(circle6)

                    plt.axis('off')  
                    plt.show()
                
                # Remap back to full frame coordinates
                # kps[:, 0] += x1_pad # [:, ] empty column means get every row, 0 means just the first column, the x coordinate of the keypoints
                # kps[:, 1] += y1_pad # [:, ] empty column means get every row, 1 means just the second column, the y coordinate of the keypoints
                
                video_keypoints.append({
                    "frame": frame_num,
                    "person_id": 1,  # always the athlete
                    "left_x":       float(kps[15][0]),
                    "left_y":       float(kps[15][1]),
                    "right_x":      float(kps[16][0]),
                    "right_y":      float(kps[16][1]),
                    "left_hip_x":   float(kps[11][0]),
                    "left_hip_y":   float(kps[11][1]),
                    "right_hip_x":  float(kps[12][0]),
                    "right_hip_y":  float(kps[12][1]),
                    "left_knee_x":  float(kps[13][0]),
                    "left_knee_y":  float(kps[13][1]),
                    "right_knee_x": float(kps[14][0]),
                    "right_knee_y": float(kps[14][1]),
                    "left_conf":    float(conf[15]),
                    "right_conf":   float(conf[16]),
                    "left_hip_conf":   float(conf[11]),
                    "right_hip_conf":  float(conf[12]),
                    "left_knee_conf":  float(conf[13]),
                    "right_knee_conf": float(conf[14]),
                })
                
                person_bboxes.append({
                    "frame": frame_num,
                    "person_id": 1,
                    "x1": x1, "y1": y1,
                    "x2": x2, "y2": y2
                })
        
        frame_num += 1
    
    cap.release()
    
    return pd.DataFrame(person_bboxes), pd.DataFrame(video_keypoints)