import cv2
import numpy as np 
import time



class SpeedAnalysis:
    def __init__(self , ratio_px2meter:float=5.0):
        self.c1_time = None
        self.c2_time = None
        self.ratio_px2meter = ratio_px2meter

    
    def calculate_distance(self , c1:float , c2:float ):
        """
        This si teh Calcutae Distabce Methods  between the two cones.

        Args:
            c1 (float): the Postion of the First Cone
            c2 (float): the Postion of teh second Cone
        """
        return np.linalg.norm(np.array(c1) - np.array(c2))
    
    def detect_speed(self , frame , c1 , c2 , time_c1 , time_c2):
        """
        Docstring for detect_speed between Two Cones.
        
        :param frame: Frame of the Video
        :param c1: the postion of the First Cone 
        :param c2: the postion of the seocnd cone.
        """    
        if c1 is None or c2 is None:
            return None
        
        if time_c1 is None or time_c2 is None:
            return None
        
        # Calculate pixel  Ecludian distance
        pixel_distance = self.calculate_distance(c1, c2)

        # Convert pixels → meters
        distance_meters = pixel_distance / self.ratio_px2meter

        # Calculate time difference
        time_diff = time_c2 - time_c1
        if time_diff <= 0:
            return None

        # Calculate speed
        speed_mps = distance_meters / time_diff
        speed_kmph = speed_mps * 3.6

        return speed_mps, speed_kmph
        
         
              
        
        
         