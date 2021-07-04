#!/usr/bin/env python

import cv2
import os
import numpy as np
from collections import deque

import rospy
from data_utils import *
from publish_utils import *
from kitti_util import *
from utils import *


ROOT_PATH = '/mnt/jiali/data/kitti_dataset/2011_09_26/'
DATA_PATH = os.path.join(ROOT_PATH, '2011_09_26_drive_0005_sync/')



if __name__=='__main__':
    rospy.init_node('kitt_node', anonymous=True)
    cam_pub = rospy.Publisher('kitti_cam', Image, queue_size=10)
    pcl_pub = rospy.Publisher('kitti_point_cloud', PointCloud2, queue_size=10)
    ego_pub = rospy.Publisher('kitti_ego_car', MarkerArray, queue_size=10)
    imu_pub = rospy.Publisher('kitti_imu', Imu, queue_size=10)
    gps_pub = rospy.Publisher('kitti_gps', NavSatFix, queue_size=10)
    box3d_pub = rospy.Publisher('kitti_3d', MarkerArray, queue_size=10)
    loc_pub = rospy.Publisher('kitti_loc', MarkerArray, queue_size=10)
    bridge = CvBridge()

    rate = rospy.Rate(10)
    frame = 0

    df_tracking = read_tracking(os.path.join(DATA_PATH, 'label_02/0000.txt')) #not sure why is in this file
    calib = Calibration(ROOT_PATH, from_video=True)

    tracker = {} # tracker_id: Object, all objects until current frame
    prev_imu_data = None

    while not rospy.is_shutdown():
        # read box and type
        df_tracking_frame = df_tracking[df_tracking.frame==frame]
        boxes_2d = np.array(df_tracking_frame[['bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom']])
        types = np.array(df_tracking_frame['type'])
        track_ids = np.array(df_tracking_frame['track_id'])
        # 3d data
        corners_3d_velos = []
        centers = {} # track_id: center, objects in the current frame
        boxes_3d = np.array(df_tracking_frame[['height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y']])
        for track_id, box_3d in zip(track_ids, boxes_3d):
            corners_3d_cam2 = compute_3d_box_cam2(*box_3d)
            corners_3d_velo = calib.project_rect_to_velo(corners_3d_cam2.T)
            corners_3d_velos += [corners_3d_velo]  # nx(8x3)
            centers[track_id] = np.mean(corners_3d_velo, axis=0)[:2]
        centers[-1] = np.array([0,0]) # ego carw

        # sensor info
        image = read_camera(os.path.join(DATA_PATH, 'image_02/data/%010d.png' % frame))
        point_cloud = read_point_cloud(os.path.join(DATA_PATH, 'velodyne_points/data/%010d.bin' % frame))
        imu_data = read_imu(os.path.join(DATA_PATH, 'oxts/data/%010d.txt' % frame))

        # track_id info
        if prev_imu_data is None:
            for track_id in centers:
                tracker[track_id] = Object(centers[track_id])
        else:
            displacement = 0.1 * np.linalg.norm(imu_data[['vf', 'vl']])
            yaw_change = float(imu_data.yaw - prev_imu_data.yaw)
            for track_id in centers:
                if track_id in tracker:
                    tracker[track_id].update(centers[track_id], displacement, yaw_change)
                else:
                    tracker[track_id] = Object(centers[track_id])
            for track_id in tracker: #not detected, but already in tracker
                if track_id not in centers:
                    tracker[track_id].update(None, displacement, yaw_change)
        prev_imu_data = imu_data


        # publish info
        publish_camera(cam_pub, bridge, image, boxes_2d, types)
        publish_point_cloud(pcl_pub, point_cloud)
        publish_ego_car(ego_pub)
        publish_imu_data(imu_pub, imu_data)
        publish_gps_data(gps_pub, imu_data)
        publish_3dbox(box3d_pub, corners_3d_velos, types, track_ids)
        publish_loc(loc_pub, tracker, centers)

        rospy.loginfo('published')
        rate.sleep()
        frame += 1
        if frame == 154:
            frame = 0
            for track_id in tracker:
                tracker[track_id].reset()


