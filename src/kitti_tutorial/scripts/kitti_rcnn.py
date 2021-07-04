#!/usr/bin/env python
''''
This script helps visualize predictions from point-rcnn
'''
import os

import rospy
from data_utils import *
from publish_utils import *
from kitti_util import *
from utils import *

DATA_PATH = '/mnt/jiali/data/kitti_dataset/training'
DET_PATH = '/mnt/jiali/internship_2020/remote/PointRCNN/output/rpn/default/eval/epoch_5/val/detections/data'

if __name__=='__main__':
    rospy.init_node('kitt_node', anonymous=True)
    pcl_pub = rospy.Publisher('kitti_point_cloud', PointCloud2, queue_size=10)
    box3d_pub = rospy.Publisher('kitti_3d', MarkerArray, queue_size=10)

    rate = rospy.Rate(10)
    files = os.listdir(DET_PATH)
    num_frames = len(files)
    frame = 0


    while not rospy.is_shutdown():
        # read box and type
        frame_name = files[frame]
        frame_num = os.path.splitext(frame_name)[0]
        df_tracking_frame = read_tracking_v2(DET_PATH, frame_name)
        types = np.array(df_tracking_frame['type'])
        calib = Calibration(os.path.join(DATA_PATH, 'calib/%s' % frame_name))
        # 3d data
        corners_3d_velos = []
        boxes_3d = np.array(df_tracking_frame[['height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y']])
        for boxes_3d in boxes_3d:
            corners_3d_cam2 = compute_3d_box_cam2(*boxes_3d)
            corners_3d_velo = calib.project_rect_to_velo(corners_3d_cam2.T)
            corners_3d_velos += [corners_3d_velo]

        #sensor info
        point_cloud = read_point_cloud(os.path.join(DATA_PATH, 'velodyne/%s.bin' % frame_num))

        # publish info
        publish_point_cloud(pcl_pub, point_cloud)
        publish_3dbox(box3d_pub, corners_3d_velos, types)

        rospy.loginfo('published')
        # rate.sleep()
        rospy.sleep(3)
        frame += 1
        if frame == num_frames:
            frame = 0








