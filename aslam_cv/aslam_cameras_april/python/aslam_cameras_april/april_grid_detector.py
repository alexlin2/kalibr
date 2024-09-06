#!/usr/bin/env python

import os
import aslam_cameras_april as acv_april
import aslam_cv as acv
import argparse
import kalibr_common as kc
import sm
import aslam_backend as aopt
import numpy as np

import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage, CameraInfo

from scipy.spatial.transform import Rotation as R


class AprilGridDetector:

    def __init__(self, chainConfig, targetConfig):

        self.use_compressed = rospy.get_param("~use_compressed", True)
        self.bridge = CvBridge()

        camConfig = chainConfig.getCameraParameters(0)

        self.camera_matrix = np.array([1103.9770623220127, 0.0, 961.8260524258075, 0.0, 1103.3975429981613, 728.5408945117463, 0.0, 0.0, 1.0]).reshape((3, 3))
        self.dist_coeffs = np.array([-0.01832523469977359, 0.025427758318680566, -0.00013429022676927387, -0.0021799323412372983])
        
        self.camera = kc.ConfigReader.AslamCamera.fromParameters(camConfig)
        targetParams = targetConfig.getTargetParams()
        grid = acv_april.GridCalibrationTargetAprilgrid(targetParams['tagRows'], 
                                                            targetParams['tagCols'], 
                                                            targetParams['tagSize'], 
                                                            targetParams['tagSpacing'])

        options = acv.GridDetectorOptions() 
        options.filterCornerOutliers = True
        self.detector = acv.GridDetector(self.camera.geometry, grid, options)

        if self.use_compressed:
            self.image_sub = rospy.Subscriber("/left/pixelink_camera2/image/compressed", CompressedImage, self.image_callback)
        else:
            self.image_sub = rospy.Subscriber("/left/pixelink_camera2/image/compressed", Image, self.image_callback)

        self.overlay_pub = rospy.Publisher("/april_grid_overlay", Image, queue_size=1)

    def image_callback(self, data):

        try:
            if self.use_compressed:
                cv_image = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")

            else:
                cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

        except CvBridgeError as e:
            print(e)

        overlay = cv_image.copy()

        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        timestamp = acv.Time(data.header.stamp.secs, data.header.stamp.nsecs)
        success, observation = self.detector.findTarget(timestamp, gray_image)
        
        if success:
            observations = observation.getCornersImageFrame()
            T_t_c = observation.T_t_c().inverse()
            tvec = T_t_c.t()
            tvec = np.expand_dims(tvec, axis=-1)
            q = T_t_c.q()
            rvec = R.from_quat([q[0], q[1], q[2], q[3]]).as_rotvec()
            rvec = np.expand_dims(rvec, axis=-1)

            for obs in observations:
                cv2.drawFrameAxes(overlay, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.5, 5)
                cv2.circle(overlay, (int(obs[0]), int(obs[1])), 5, (0, 255, 0), -1)

        img_msg = self.bridge.cv2_to_imgmsg(overlay, "bgr8")
        img_msg.header = data.header
        self.overlay_pub.publish(img_msg)


if __name__ == "__main__":        
    parser = argparse.ArgumentParser(description='Detect AprilGrids in images')
    parser.add_argument('--target', dest='targetYaml', help='Calibration target configuration as yaml file', required=True)
    parser.add_argument('--cam', dest='chainYaml', help='Camera configuration as yaml file', required=True)
    parsed = parser.parse_args()

    targetConfig = kc.ConfigReader.CalibrationTargetParameters(parsed.targetYaml)
    camchain = kc.ConfigReader.CameraChainParameters(parsed.chainYaml)

    detector = AprilGridDetector(camchain, targetConfig)
    
    rospy.init_node('april_grid_detector')
    rospy.spin()
