# Jump-Approach-Graphed
Automating graphing and understanding the approach of a long jump / triple jump using Computer Vision and Biomechanics 

## Features
- **Bounding Box Normalization:** Normalizes ankle-to-hip coordinate vectors to account for camera angle and zoom.
- **Step Preperation Identification:** Identifies the moment where the athlete is most extended in preperation to make a ground contact
- **Velocity & Deceleration Detection:** Identifies stride frequency and frame-by-frame deceleration spikes to identify moment of contact.
- **Automated Plot Generation:** Exports kinematic data directly to structured NumPy/Pandas dataframes and matplotlib visuals.

## Tech Stack
- **Language:** Python 3.10+
- **Computer Vision:** OpenCV, YOLOv8
- **Data Analysis:** NumPy, Pandas, Matplotlib

Achieved ±2 frames detection of ground contact across 12 videos on 30FPS 90 degrees monocular footage

<img width="2036" height="1139" alt="image" src="https://github.com/user-attachments/assets/c1647021-fa8e-4ac0-8af7-ed131ca47fa5" />

<img width="1624" height="1104" alt="image" src="https://github.com/user-attachments/assets/cd3a51b6-638e-43d6-8031-ddea53f724f6" />

Red Bounding Box = Detected Athlete, as shown below

<img width="1534" height="1029" alt="image" src="https://github.com/user-attachments/assets/8733f346-7b2e-4272-b00d-5030750cbd89" />


White bounding box = Detected Contact Frame, as shown below

<img width="569" height="695" alt="image" src="https://github.com/user-attachments/assets/4171f453-940e-426d-b250-b003ad72705c" />

<img width="1776" height="1021" alt="image" src="https://github.com/user-attachments/assets/1941ae35-325c-4a72-b9f7-740ed392a518" />

