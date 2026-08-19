# processVideo() draw boxes, keypoints
import cv2
import numpy as np


def processVideo(cap, video, save_path, bounding_boxes_df, video_keypoints_df, athlete_id, initial_contact_frames, sprint_peaks, sprint_peaks_left):
    """ Draws bounding boxes, keypoints, and flashes bounding boxes dpeending on athletes state

    Args: 
        cap: cv2 capture object from the video file
        video: Video class object
        save_path: path to save the edited file
        bounding_boxes_df: 
            <class 'pandas.DataFrame'>
            RangeIndex: 158 entries, 0 to 157
            Data columns (total 6 columns):
            #   Column     Non-Null Count  Dtype  
            ---  ------     --------------  -----  
            0   frame      158 non-null    int64  
            1   person_id  158 non-null    int64  
            2   x1         158 non-null    float32
            3   y1         158 non-null    float32
            4   x2         158 non-null    float32
            5   y2         158 non-null    float32
            dtypes: float32(4), int64(2)
            memory usage: 5.1 KB
        video_keypoints_df:
            <class 'pandas.DataFrame'>
            Index: 158 entries, 145 to 157
            Data columns (total 16 columns):
            #   Column        Non-Null Count  Dtype  
            ---  ------        --------------  -----  
            0   frame         158 non-null    int64  
            1   person_id     158 non-null    int64  
            2   left_x        158 non-null    float64
            3   left_y        158 non-null    float64
            4   right_x       158 non-null    float64
            5   right_y       158 non-null    float64
            6   left_hip_x    158 non-null    float64
            7   left_hip_y    158 non-null    float64
            8   right_hip_x   158 non-null    float64
            9   right_hip_y   158 non-null    float64
            10  left_knee_x   158 non-null    float64
            11  left_knee_y   158 non-null    float64
            12  right_knee_x  158 non-null    float64
            13  right_knee_y  158 non-null    float64
            14  left_conf     158 non-null    float64
            15  right_conf    158 non-null    float64
            dtypes: float64(14), int64(2)
            memory usage: 21.0 KB
        athlete_id: person_id of the athlete
        initial_contact_frames: list[int] containing frame numbers of initial contact of each step of the athlete
        sprint_peaks: list[int] containing frame numbers of maximum extension of each stride of the athlete
        sprint_peaks_left: list[int] containing frame numbers of maximum extenion of each stride of the athlete's LEFT FOOT
    
    """

    fps = video.fps
    width = video.width
    height = video.height

    output = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*'mp4v'), 
        fps, (width, height))
    
    frame_num = 0
    
    while(True):
        ret, frame = cap.read() # ret is if frame is sucesfully captured

        
        if(ret):

            cv2.putText(frame, "Frame: " + str(frame_num), (20, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)   
            
            
            left_knee_angle = video_keypoints_df.loc[video_keypoints_df["frame"] == frame_num, "left_knee_angle"]
            if len(left_knee_angle) == 1:
                cv2.putText(frame, "Left angle: " + str(left_knee_angle.item()), (20, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3) 
                
            right_knee_angle = video_keypoints_df.loc[video_keypoints_df["frame"] == frame_num, "right_knee_angle"]
            if len(right_knee_angle) == 1:
                cv2.putText(frame, "Right angle: " + str(right_knee_angle.item()), (20, 110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3) 
                
            left_hip_angle = video_keypoints_df.loc[video_keypoints_df["frame"] == frame_num, "left_hip_angle"]
            if len(left_hip_angle) == 1:
                cv2.putText(frame, "Left hip angle: " + str(left_hip_angle.item()), (20, 140), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3) 
                

            right_hip_angle = video_keypoints_df.loc[video_keypoints_df["frame"] == frame_num, "right_hip_angle"]
            if len(right_hip_angle) == 1:
                cv2.putText(frame, "right hip angle: " + str(right_hip_angle.item()), (20, 170), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3) 
            

            processBoundingBoxes(frame=frame, frame_num=frame_num, bounding_boxes_df=bounding_boxes_df, width=int(width/280), athlete_id=athlete_id, initial_contact_frames=initial_contact_frames, sprint_peaks=sprint_peaks, sprint_peaks_left=sprint_peaks_left)
            processKeypointPositions(frame=frame, frame_num=frame_num, video_keypoints_df=video_keypoints_df, circle_size=int(width/440))

            output.write(frame)

            
        else:
            break

        frame_num += 1


    cv2.destroyAllWindows()
    output.release()
    

def processBoundingBoxes(frame, frame_num, bounding_boxes_df, width, athlete_id, initial_contact_frames, sprint_peaks, sprint_peaks_left):

    # print(bounding_boxes_df)
    
    frame_df  = bounding_boxes_df.loc[bounding_boxes_df['frame'] == frame_num]
    # print("frame_df: \n ", frame_df)
    if len(frame_df) <= 0: # If the athlete is not found on this frame
        return


    # Convert just the coordinate columns straight to a NumPy array.
    athlete = frame_df.iloc[0][["x1", "y1", "x2", "y2"]].to_numpy().astype(int)
    

    # print("athlete : \n", athlete)

    if (np.isin(frame_num, initial_contact_frames)):
        cv2.rectangle(frame, (athlete[0], athlete[1]), (athlete[2], athlete[3]),
                    (255, 255, 255), width)
    elif (np.isin(frame_num, sprint_peaks)):
        if (np.isin(frame_num, sprint_peaks_left)):
            cv2.rectangle(frame, (athlete[0], athlete[1]), (athlete[2], athlete[3]),
                    (255, 0, 0), width)
        else:
            cv2.rectangle(frame, (athlete[0], athlete[1]), (athlete[2], athlete[3]),
                    (0, 255, 0), width)
    else:
        cv2.rectangle(frame, (athlete[0], athlete[1]), (athlete[2], athlete[3]),
                    (0, 0, 255), width)
    # else:
    #     cv2.rectangle(frame, (athlete[0], athlete[1]), (athlete[2], athlete[3]),
    #                 (0, 255, 0), width)
    #     cv2.putText(frame, "Person " + str(athlete[4]), (athlete[0], athlete[1] - 10), 
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
        
    


def processKeypointPositions(frame, frame_num, video_keypoints_df, circle_size):

    frame_df = video_keypoints_df.loc[video_keypoints_df['frame'] == frame_num]
        
    # Convert just the coordinate columns straight to a NumPy array.
    ankles = frame_df[["left_x", "left_y", "right_x", "right_y", "left_hip_x", "left_hip_y", "right_hip_x", "right_hip_y", "left_knee_x", "left_knee_y", "right_knee_x", "right_knee_y", "left_shoulder_x", "left_shoulder_y", "right_shoulder_x", "right_shoulder_y"]].fillna(0).to_numpy().astype(int)
    
    # Loop through the NumPy array to draw the circles
    for lx, ly, rx, ry, lhx, lhy, rhx, rhy, lkx, lky, rkx, rky, lsx, lsy, rsx, rsy in ankles:
        # print(lx, ly, rx, ry, lhx, lhy, rhx, rhy, lkx, lky, rkx, rky)

        # Draw Left Ankle (Green circle, radius 6, filled in)
        cv2.circle(frame, (lx, ly), circle_size, (0, 255, 0), -1)
        
        # Draw Right Ankle (Red circle, radius 6, filled in)
        cv2.circle(frame, (rx, ry), circle_size, (0, 0, 255), -1)

        cv2.circle(frame, (lhx, lhy), circle_size, (0, 255, 0), -1)
        
        cv2.circle(frame, (rhx, rhy), circle_size, (0, 0, 255), -1)

        cv2.circle(frame, (lkx, lky), circle_size, (0, 255, 0), -1)
        
        cv2.circle(frame, (rkx, rky), circle_size, (0, 0, 255), -1)

        cv2.circle(frame, (lsx, lsy), circle_size, (0, 255, 0), -1)
        
        cv2.circle(frame, (rsx, rsy), circle_size, (0, 0, 255), -1)
