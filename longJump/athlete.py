# identify_athlete(), direction score, InsufficientDirectional
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.signal import find_peaks 
from longJump import detection

class Athlete:
    def __init__(self, bounding_boxes_df, video_keypoints_df, athlete_id, minimum_movement_threshold, fps):
        self.fps = fps

        # boxPositions = bounding_boxes_df.loc[bounding_boxes_df["person_id"] == athlete_id, ["x1", "x2", "y1", "y2", "frame"]]
        # anklePositions = video_keypoints_df.loc[video_keypoints_df["person_id"] == athlete_id, ["left_hip_x", "right_hip_x", "left_hip_y", "right_hip_y", "left_x", "right_x", 'left_y', "right_y", "left_knee_x", "left_knee_y", "right_knee_x", "right_knee_y", "frame", "left_conf", "right_conf"]]

        # self.df = pd.merge(anklePositions, boxPositions, on="frame")

        self.df = pd.merge(bounding_boxes_df, video_keypoints_df, on="frame")


        self.df["time"] = self.df["frame"] / self.fps
        self.df["hip_center_x"] = (self.df["left_hip_x"] + self.df["right_hip_x"]) / 2
        self.df["hip_center_y"] = (self.df["left_hip_y"] + self.df["right_hip_y"]) / 2

        self.createNormalizedColumns(minimum_movement_threshold)
        self.createBodyAngleColumns()
        self.createNormalizedColumnsByLegLength()

    def createNormalizedColumns(self, minimum_movement_threshold):

        rolling_window_size = int(self.fps)
        bounding_box_height = (self.df["y2"] - self.df["y1"]).abs()

        minimum_movement_threshold = bounding_box_height * minimum_movement_threshold

        # Left Ankle Calculations
        self.df["left_ankle_x_rel"] = self.NormalizeColumnByRollingWindow(
            series=self.df["left_x"] - self.df["left_hip_x"],
            rolling_window_size=rolling_window_size,
            minimum_movement_threshold=minimum_movement_threshold
        )

        self.df["left_ankle_y_rel"] = self.NormalizeColumnByRollingWindow(
            series=self.df["left_y"] - self.df["left_hip_y"],
            rolling_window_size=rolling_window_size,
            minimum_movement_threshold=minimum_movement_threshold
        )

        # Right Ankle Calculations
        self.df["right_ankle_x_rel"] = self.NormalizeColumnByRollingWindow(
            series=self.df["right_x"] - self.df["right_hip_x"],
            rolling_window_size=rolling_window_size,
            minimum_movement_threshold=minimum_movement_threshold
        )

        self.df["right_ankle_y_rel"] = self.NormalizeColumnByRollingWindow(
            series=self.df["right_y"] - self.df["right_hip_y"],
            rolling_window_size=rolling_window_size,
            minimum_movement_threshold=minimum_movement_threshold
        )

        self.df["left_knee_x_rel"] = self.NormalizeColumnByRollingWindow(
            series=self.df["left_knee_x"] - self.df["left_hip_x"],
            rolling_window_size=rolling_window_size,
            minimum_movement_threshold=minimum_movement_threshold
        )

        self.df["left_knee_y_rel"] = self.NormalizeColumnByRollingWindow(
            series=self.df["left_knee_y"] - self.df["left_hip_y"],
            rolling_window_size=rolling_window_size,
            minimum_movement_threshold=minimum_movement_threshold
        )


        self.df["right_knee_x_rel"] = self.NormalizeColumnByRollingWindow(
            series=self.df["right_knee_x"] - self.df["right_hip_x"],
            rolling_window_size=rolling_window_size,
            minimum_movement_threshold=minimum_movement_threshold
        )

        self.df["right_knee_y_rel"] = self.NormalizeColumnByRollingWindow(
            series=self.df["right_knee_y"] - self.df["right_hip_y"],
            rolling_window_size=rolling_window_size,
            minimum_movement_threshold=minimum_movement_threshold
        )

        self.df["left_knee_y_rel"] = savgol_filter(self.df["left_knee_y_rel"], window_length=5, polyorder=2)
        self.df["right_knee_y_rel"] = savgol_filter(self.df["right_knee_y_rel"], window_length=5, polyorder=2)

        self.df["left_ankle_x_rel"] = savgol_filter(self.df["left_ankle_x_rel"], window_length=5, polyorder=2)
        self.df["right_ankle_x_rel"] = savgol_filter(self.df["right_ankle_x_rel"], window_length=5, polyorder=2)
        

    def createNormalizedColumnsByLegLength(self):

        legLength = self.LegLengthOfRollingWindow(round((self.fps / 10)))

        self.df["left_ankle_x_rel_leg_length"] = (self.df["left_x"] - self.df["left_hip_x"]) / legLength
        self.df["left_ankle_y_rel_leg_length"] = (self.df["left_y"] - self.df["left_hip_y"]) / legLength

        self.df["right_ankle_x_rel_leg_length"] = (self.df["right_x"] - self.df["right_hip_x"]) / legLength
        self.df["right_ankle_y_rel_leg_length"] = (self.df["right_y"] - self.df["right_hip_y"]) / legLength

        self.df["left_knee_x_rel_leg_length"] = (self.df["left_knee_x"] - self.df["left_hip_x"]) / legLength
        self.df["left_knee_y_rel_leg_length"] = (self.df["left_knee_y"] - self.df["left_hip_y"]) / legLength

        self.df["right_knee_x_rel_leg_length"] = (self.df["right_knee_x"] - self.df["right_hip_x"]) / legLength
        self.df["right_knee_y_rel_leg_length"] = (self.df["right_knee_y"] - self.df["right_hip_y"]) / legLength




        # self.df["left_knee_y_rel"] = savgol_filter(self.df["left_knee_y_rel"], window_length=5, polyorder=2)
        # self.df["right_knee_y_rel"] = savgol_filter(self.df["right_knee_y_rel"], window_length=5, polyorder=2)

        # self.df["left_ankle_x_rel"] = savgol_filter(self.df["left_ankle_x_rel"], window_length=5, polyorder=2)
        # self.df["right_ankle_x_rel"] = savgol_filter(self.df["right_ankle_x_rel"], window_length=5, polyorder=2)

    def createBodyAngleColumns(self):


        v_hip_knee_x = self.df['left_hip_x'] - self.df['left_knee_x']
        v_hip_knee_y = self.df['left_hip_y'] - self.df['left_knee_y']
        v_knee_ankle_x = self.df['left_x'] - self.df['left_knee_x']
        v_knee_ankle_y = self.df['left_y'] - self.df['left_knee_y']
        
        # 2. Compute the dot product
        dot_product_left = (v_hip_knee_x * v_knee_ankle_x) + (v_hip_knee_y * v_knee_ankle_y)
        
        # 3. Compute the lengths (magnitudes) of each bone
        mag_hip_knee_left = np.sqrt(v_hip_knee_x**2 + v_hip_knee_y**2)
        mag_knee_ankle_left = np.sqrt(v_knee_ankle_x**2 + v_knee_ankle_y**2)
        
        # 4. Calculate the raw interior angle
        cos_angle_left = dot_product_left / (mag_hip_knee_left * mag_knee_ankle_left)
        
        # Clip to prevent floating-point errors (e.g., 1.0000001) from crashing arccos
        cos_angle_left = np.clip(cos_angle_left, -1.0, 1.0)
        self.df['left_knee_angle'] = np.degrees(np.arccos(cos_angle_left))

        # --- RIGHT SIDE ---
        # 1. Get vectors for the bone segments
        v_hip_knee_x_r = self.df['right_hip_x'] - self.df['right_knee_x']
        v_hip_knee_y_r = self.df['right_hip_y'] - self.df['right_knee_y']
        v_knee_ankle_x_r = self.df['right_x'] - self.df['right_knee_x']
        v_knee_ankle_y_r = self.df['right_y'] - self.df['right_knee_y']
        
        # 2. Compute the dot product
        dot_product_right = (v_hip_knee_x_r * v_knee_ankle_x_r) + (v_hip_knee_y_r * v_knee_ankle_y_r)
        
        # 3. Compute lengths
        mag_hip_knee_right = np.sqrt(v_hip_knee_x_r**2 + v_hip_knee_y_r**2)
        mag_knee_ankle_right = np.sqrt(v_knee_ankle_x_r**2 + v_knee_ankle_y_r**2)
        
        # 4. Calculate the raw interior angle
        cos_angle_right = dot_product_right / (mag_hip_knee_right * mag_knee_ankle_right)
        cos_angle_right = np.clip(cos_angle_right, -1.0, 1.0)
        self.df['right_knee_angle'] = np.degrees(np.arccos(cos_angle_right))






        # === LEFT HIP ANGLE ===
        v_shoulder_hip_x = self.df['left_shoulder_x'] - self.df['left_hip_x']
        v_shoulder_hip_y = self.df['left_shoulder_y'] - self.df['left_hip_y']
        v_hip_knee_x = self.df['left_hip_x'] - self.df['left_knee_x']
        v_hip_knee_y = self.df['left_hip_y'] - self.df['left_knee_y']
        
        # Compute the dot product
        dot_product_left_hip = (v_shoulder_hip_x * v_hip_knee_x) + (v_shoulder_hip_y * v_hip_knee_y)
        
        # Compute the proper lengths (magnitudes)
        mag_shoulder_hip_left = np.sqrt(v_shoulder_hip_x**2 + v_shoulder_hip_y**2)
        mag_hip_knee_left = np.sqrt(v_hip_knee_x**2 + v_hip_knee_y**2)
        
        # Calculate the raw interior angle
        cos_angle_left_hip = dot_product_left_hip / (mag_shoulder_hip_left * mag_hip_knee_left)
        cos_angle_left_hip = np.clip(cos_angle_left_hip, -1.0, 1.0)
        self.df['left_hip_angle'] = np.degrees(np.arccos(cos_angle_left_hip))

        # === RIGHT HIP ANGLE ===
        v_shoulder_hip_x_r = self.df['right_shoulder_x'] - self.df['right_hip_x']
        v_shoulder_hip_y_r = self.df['right_shoulder_y'] - self.df['right_hip_y']
        v_hip_knee_x_r = self.df['right_hip_x'] - self.df['right_knee_x']
        v_hip_knee_y_r = self.df['right_hip_y'] - self.df['right_knee_y']
        
        # Compute the dot product
        dot_product_right_hip = (v_shoulder_hip_x_r * v_hip_knee_x_r) + (v_shoulder_hip_y_r * v_hip_knee_y_r)
        
        # Compute the proper lengths (magnitudes)
        mag_shoulder_hip_right = np.sqrt(v_shoulder_hip_x_r**2 + v_shoulder_hip_y_r**2)
        mag_hip_knee_right = np.sqrt(v_hip_knee_x_r**2 + v_hip_knee_y_r**2)
        
        # Calculate the raw interior angle
        cos_angle_right_hip = dot_product_right_hip / (mag_shoulder_hip_right * mag_hip_knee_right)
        cos_angle_right_hip = np.clip(cos_angle_right_hip, -1.0, 1.0)
        self.df['right_hip_angle'] = np.degrees(np.arccos(cos_angle_right_hip))


    def NormalizeColumnByRollingWindow(self, series: pd.Series, rolling_window_size, minimum_movement_threshold) -> pd.Series:

        """
        Normalizes a series by its rolling range, suppressing noise 
        below a minimum movement threshold. 

        Best for ZOOMING OUT on the data, creates beutiful peaks and valleys where every step looks almost identical and easily identifiable. However, if we zoom into just one stride using this approach we lose all the data that makes the stride unique

        Args: 
            series: series to perform adjustments on
            rolling_window_size: Size of each window (in frames)
            minimum_movement_threshold: If there's not this minimum amount of movement, set the divisor to very high, squashing the little bit of movement to avoid huge artificial peaks and dividing by 0
        """

        local_max = (series).rolling(window=rolling_window_size, center=True, min_periods=1).max()
        local_min = (series).rolling(window=rolling_window_size, center=True, min_periods=1).min()

        raw_rolling_range = (local_max - local_min)

        # Apply the safety ceiling to prevent division by zero/tiny noise
        raw_rolling_range[raw_rolling_range < (minimum_movement_threshold)] = 1000000

        return series / raw_rolling_range
    
    def LegLengthOfRollingWindow(self, rolling_window_size) -> pd.Series:

        """
        Normalizes a series by its skeletal size per rolling window

        Less effective if we ZOOM OUT on the data, but if we ZOOM IN this normalization technique preserves the data that makes each stride unique and is much less of a "black box", we can easily read and understand the data in it's context

        """


        leftThighLength = np.sqrt((self.df['left_hip_x'] - self.df['left_knee_x'])**2 + (self.df['left_hip_y'] - self.df['left_knee_y'])**2)
        leftLowerLegLength = np.sqrt((self.df['left_knee_x'] - self.df['left_x'])**2 + (self.df['left_knee_y'] - self.df['left_y'])**2)
        leftLegLength = leftThighLength + leftLowerLegLength

        rightThighLength = np.sqrt((self.df['right_hip_x'] - self.df['right_knee_x'])**2 + (self.df['right_hip_y'] - self.df['right_knee_y'])**2)
        rightLowerLegLength = np.sqrt((self.df['right_knee_x'] - self.df['right_x'])**2 + (self.df['right_knee_y'] - self.df['right_y'])**2)
        rightLegLength = rightThighLength + rightLowerLegLength

        avgLegLength = (leftLegLength + rightLegLength) / 2

        rollingLegLength = avgLegLength.rolling(window=rolling_window_size, center=True, min_periods=1).median()

        return rollingLegLength
    def CalculateDirectionScore(self, prominence, minimum_direction_score, start_frame, end_frame):

        knee_peaks_tuple_left = find_peaks(self.df["left_knee_y_rel"] * -1, prominence=prominence)
        knee_peaks_tuple_right = find_peaks(self.df["right_knee_y_rel"] * -1, prominence=prominence)

        raw_left_frames = knee_peaks_tuple_left[0]
        raw_right_frames = knee_peaks_tuple_right[0]

        # 3. Filter frames to only keep those inside the approach bounds
        filtered_left_frames = raw_left_frames[(raw_left_frames >= start_frame) & (raw_left_frames <= end_frame)]
        filtered_right_frames = raw_right_frames[(raw_right_frames >= start_frame) & (raw_right_frames <= end_frame)]

        

        left_knee_apex = self.df.loc[self.df["frame"].isin(filtered_left_frames), "left_knee_x_rel"]
        right_knee_apex = self.df.loc[self.df["frame"].isin(filtered_right_frames), "right_knee_x_rel"]

        

        all_apex_values = pd.concat([left_knee_apex, right_knee_apex])

        # 3. Calculate the grand mean
        direction_score = all_apex_values.median()

        if (abs(direction_score) < minimum_direction_score):
            print("Directoin_score: ", abs(direction_score))
            raise InsufficientDirectionalScore("Direction_score.abs() < " + str(minimum_direction_score))

        return direction_score
    
    def findPeaks(self, left_ankle_sprint, right_ankle_sprint, prominence, height):
        """ Find peaks in horizontal displacement of the ankles from the hips (normalized)

            Args:
                prominence: prominence of peaks
                height: minimum height of peaks
                min_gap: minimum gap between peaks
                max_gap: maximum gap between peaks
            Returns:
                tuple[list, list]: sprint peaks for the left foot, sprint peaks for the right foot 
        """

        peaks_tuple_left = find_peaks(left_ankle_sprint,height=height, prominence=prominence)
        peaks_tuple_right = find_peaks(right_ankle_sprint,height=height, prominence=prominence)


        

        sprint_peaks_right = peaks_tuple_right[0]
        sprint_peaks_left = peaks_tuple_left[0]

        # print(consecutive_peaks)


        

        return sprint_peaks_left, sprint_peaks_right
    def GetAnkleVerticalDeacceleration(self, max_extension_frame, isLeft, frames_before_max_extension, frames_after_max_extension):
        """
        
        Returns:
            ankle_df: 
                <class 'pandas.DataFrame'>
                RangeIndex: 7 entries, 14 to 20
                Data columns (total 8 columns):
                #   Column        Non-Null Count  Dtype  
                ---  ------        --------------  -----  
                0   frame         7 non-null      int64  
                1   right_y       7 non-null      float64
                2   hip_center_y  7 non-null      float64
                3   right_x       7 non-null      float64
                4   right_hip_x   7 non-null      float64
                5   y_pos         7 non-null      float64
                6   x_dist        7 non-null      float64
                7   deaccel       5 non-null      float64
                dtypes: float64(7), int64(1)
        """
        ankle_df = pd.DataFrame()
        if (isLeft):
            ankle_df = self.df.loc[((max_extension_frame-(frames_before_max_extension)) <= self.df["frame"]) & (self.df["frame"] <= (max_extension_frame+frames_after_max_extension)), ["frame", "left_y", "hip_center_y", "left_x", "left_hip_x"]].copy()
            # ankle_df = ankle_df.rename(columns={"left_y": "y_pos"})
            ankle_df['x_dist'] = (ankle_df['left_x'] - ankle_df['left_hip_x']).abs()
            ankle_df['y_pos'] = ankle_df['left_y'] - ankle_df['hip_center_y']

        else:
            ankle_df = self.df.loc[((max_extension_frame-(frames_before_max_extension)) <= self.df["frame"]) & (self.df["frame"] <= (max_extension_frame+frames_after_max_extension)), ["frame", "right_y", "hip_center_y", "right_x", "right_hip_x"]].copy()
            # ankle_df = ankle_df.rename(columns={"right_y": "y_pos"})
            ankle_df['y_pos'] = ankle_df['right_y'] - ankle_df['hip_center_y']
            ankle_df['x_dist'] = (ankle_df['right_x'] - ankle_df['right_hip_x']).abs()
        # print((max_extension_frame-(frames_before_max_extension)), " : ", (max_extension_frame+frames_after_max_extension))



        # print(ankle_df["y_pos"].diff().abs().diff())

        ankle_df["deaccel"] = ankle_df["y_pos"].diff().abs().diff()

        return ankle_df
    def CorrelationValues(self, windowSizeScalar):
        """ Finds the frames during which the athlete is sprinting as their approach.
            Checks the .corr() value of the .diff() value of the left and right ankle in a rolling window. (Ankles are moving in opposite direction during that rolling window). 
            If the .corr() value is close to -1, that means the athlete is sprinting

            Returns:
                 series representing correlatin values

        """
        window = int(round(self.fps * windowSizeScalar))

        left_vel = self.df["left_ankle_x_rel_leg_length"].diff()
        right_vel = self.df["right_ankle_x_rel_leg_length"].diff()
        # left_vel = self.df["left_ankle_x_rel"].diff()
        # right_vel = self.df["right_ankle_x_rel"].diff()

        left_std  = left_vel.rolling(window=window, center=True).std()
        right_std = right_vel.rolling(window=window, center=True).std()

        avg_std = (left_std + right_std) / 2

        correlation = left_vel.rolling(window=window, center=True).corr(right_vel)

        

        # correlation = correlation.rolling(window=window, center=True).mean()

        correlation = correlation.fillna(0)



        correlation_values = savgol_filter(correlation, window_length=15, polyorder=2)
        smoothed_correlation = pd.Series(correlation_values, index=correlation.index)

        




        # print(correlation.head(200))

        return smoothed_correlation

    
class InsufficientDirectionalScore(Exception):
    """Raised when unable to confidently detect the direction the athlete is running in."""
    pass