import numpy as np
import cv2
import os
import math

# camera calibration to get the rvec and tvec
def camera_calibration(camera):
    if (camera == "left"):
        object_3d_points = np.array(([3805, 4226, 0.],
                                     [2186, 1902, 0.],
                                     [3818, 674, 0],
                                     [5835, 3049, 0],
                                     [7225, 1888, 0]), dtype=np.double)

        object_2d_point = np.array(([84, 147],
                                    [188, 293],
                                    [502, 215],
                                    [318, 104],
                                    [464, 85]), dtype=np.double)

        fz_camera_matrix = np.array(([672.696475436197, 0.0, 336.091575679676],
                                     [0.0, 631.740668979164, 232.912518319968],
                                     [0.0, 0.0, 1.0]), dtype="double")

        dist_coefs = np.array(
            [-0.0753413793419683, 0.242546341519954, -0.00219587406317966, 0.00396341324028028, -0.107908652593460],
            dtype=np.double)

    else:
        object_3d_points = np.array(([4315, 238, 0.],
                                     [5829, 2601, 0],
                                     [4342, 3782, 0],
                                     [2196, 1459, 0],
                                     [794, 2596, 0]), dtype=np.double)

        object_2d_point = np.array(([60, 153],
                                    [177, 312],
                                    [493, 231],
                                    [308, 102],
                                    [460, 78]), dtype=np.double)

        fz_camera_matrix = np.array(([654.690854219901, 0.0, 331.747624399371],
                                     [0.0, 615.075013458089, 257.729982008233],
                                     [0.0, 0.0, 1.0]), dtype="double")

        dist_coefs = np.array(
            [-0.0721106179904230, 0.178067038866243, 0.00122551662740768, 0.00414688113770697, -0.0844652346664083],
            dtype=np.double)

    (success, rotation_vector, translation_vector) = cv2.solvePnP(object_3d_points, object_2d_point, fz_camera_matrix,
                                                                  dist_coefs,
                                                                  flags=cv2.SOLVEPNP_ITERATIVE)
    fz_rvec = cv2.Rodrigues(rotation_vector)[0]
    fz_tvec = translation_vector

    return fz_rvec, fz_tvec, fz_camera_matrix

# 3D points to image points
def getimage(points3d, rvec, tvec, camera_matrix):
    newpoints3d = np.vstack((points3d, 1.0))
    Zc = np.dot(np.hstack((rvec, tvec)), newpoints3d)[-1]
    imagepoints = np.dot(camera_matrix, np.dot(np.hstack((rvec, tvec)), newpoints3d)) / Zc
    return imagepoints

# get the 8 points of 3D box in image
def get3Dlocation(objectx, objecty, dimesions, angle, rvec, tvec, camera_matrix):
    high = float(dimesions[0])
    width = float(dimesions[1])
    length = float(dimesions[2])

    true_angle  = angle
    pose1 = np.array(([length / 2], [width / 2]), dtype="double") * 1000
    pose2 = np.array(([length / 2], [-width / 2]), dtype="double") * 1000
    pose3 = np.array(([-length / 2], [-width / 2]), dtype="double") * 1000
    pose4 = np.array(([-length / 2], [width / 2]), dtype="double") * 1000

    ro_matrix = np.array(([math.cos(true_angle), -math.sin(true_angle)],
                         [math.sin(true_angle), math.cos(true_angle)]), dtype="double")

    points3d = []
    result = []

    points3d.append(np.vstack((np.dot(ro_matrix, pose1) + np.array([[objectx],[objecty]]), high * 800)))
    points3d.append(np.vstack((np.dot(ro_matrix, pose2) + np.array([[objectx],[objecty]]), high * 800)))
    points3d.append(np.vstack((np.dot(ro_matrix, pose3) + np.array([[objectx],[objecty]]), high * 800)))
    points3d.append(np.vstack((np.dot(ro_matrix, pose4) + np.array([[objectx],[objecty]]), high * 800)))
    points3d.append(np.vstack((np.dot(ro_matrix, pose1) + np.array([[objectx],[objecty]]), 0)))
    points3d.append(np.vstack((np.dot(ro_matrix, pose2) + np.array([[objectx],[objecty]]), 0)))
    points3d.append(np.vstack((np.dot(ro_matrix, pose3) + np.array([[objectx],[objecty]]), 0)))
    points3d.append(np.vstack((np.dot(ro_matrix, pose4) + np.array([[objectx],[objecty]]), 0)))

    for i in points3d:
        result.append(getimage(i, rvec, tvec, camera_matrix))

    return result

def main(logpath, image_path, savepath, camera):

    # read the log
    flog_list = open(logpath)
    lines = flog_list.readlines()

    # get the imageslist
    frame_list = os.listdir(image_path)
    frame_list.sort()

    for i in range(len(lines)):

        line = lines[i].split(" ")

        red1_x = float(line[1])
        red1_y = float(line[2])
        red1_yaw = float(line[3])

        red2_x = float(line[5])
        red2_y = float(line[6])
        red2_yaw = float(line[7])

        blue1_x = float(line[9])
        blue1_y = float(line[10])
        blue1_yaw = float(line[11])

        blue2_x = float(line[13])
        blue2_y = float(line[14])
        blue2_yaw = float(line[15])

        # the car's high width length
        dimensions = [0.505, 0.499, 0.592]
        img_name = frame_list[i].split('.')[0]


        copyimg = cv2.imread(image_path + frame_list[i])

        rvec, tvec, camera_matrix = camera_calibration(camera)

        b1_img_points = get3Dlocation(blue1_x * 1000, blue1_y * 1000, dimensions, blue1_yaw, rvec, tvec, camera_matrix)
        b2_img_points = get3Dlocation(blue2_x * 1000, blue2_y * 1000, dimensions, blue2_yaw, rvec, tvec, camera_matrix)
        r1_img_points = get3Dlocation(red1_x * 1000, red1_y * 1000, dimensions, red1_yaw, rvec, tvec, camera_matrix)
        r2_img_points = get3Dlocation(red2_x * 1000, red2_y * 1000, dimensions, red2_yaw, rvec, tvec, camera_matrix)

        b1_up1 = [b1_img_points[0][0], b1_img_points[0][1]]
        b1_up2 = [b1_img_points[1][0], b1_img_points[1][1]]
        b1_up3 = [b1_img_points[2][0], b1_img_points[2][1]]
        b1_up4 = [b1_img_points[3][0], b1_img_points[3][1]]
        b1_down1 = [b1_img_points[4][0], b1_img_points[4][1]]
        b1_down2 = [b1_img_points[5][0], b1_img_points[5][1]]
        b1_down3 = [b1_img_points[6][0], b1_img_points[6][1]]
        b1_down4 = [b1_img_points[7][0], b1_img_points[7][1]]

        b2_up1 = [b2_img_points[0][0], b2_img_points[0][1]]
        b2_up2 = [b2_img_points[1][0], b2_img_points[1][1]]
        b2_up3 = [b2_img_points[2][0], b2_img_points[2][1]]
        b2_up4 = [b2_img_points[3][0], b2_img_points[3][1]]
        b2_down1 = [b2_img_points[4][0], b2_img_points[4][1]]
        b2_down2 = [b2_img_points[5][0], b2_img_points[5][1]]
        b2_down3 = [b2_img_points[6][0], b2_img_points[6][1]]
        b2_down4 = [b2_img_points[7][0], b2_img_points[7][1]]

        r1_up1 = [r1_img_points[0][0], r1_img_points[0][1]]
        r1_up2 = [r1_img_points[1][0], r1_img_points[1][1]]
        r1_up3 = [r1_img_points[2][0], r1_img_points[2][1]]
        r1_up4 = [r1_img_points[3][0], r1_img_points[3][1]]
        r1_down1 = [r1_img_points[4][0], r1_img_points[4][1]]
        r1_down2 = [r1_img_points[5][0], r1_img_points[5][1]]
        r1_down3 = [r1_img_points[6][0], r1_img_points[6][1]]
        r1_down4 = [r1_img_points[7][0], r1_img_points[7][1]]

        r2_up1 = [r2_img_points[0][0], r2_img_points[0][1]]
        r2_up2 = [r2_img_points[1][0], r2_img_points[1][1]]
        r2_up3 = [r2_img_points[2][0], r2_img_points[2][1]]
        r2_up4 = [r2_img_points[3][0], r2_img_points[3][1]]
        r2_down1 = [r2_img_points[4][0], r2_img_points[4][1]]
        r2_down2 = [r2_img_points[5][0], r2_img_points[5][1]]
        r2_down3 = [r2_img_points[6][0], r2_img_points[6][1]]
        r2_down4 = [r2_img_points[7][0], r2_img_points[7][1]]

        # 3D box
        b1_pts1 = np.array([b1_up1, b1_up2, b1_down2, b1_down1], np.int32)
        b1_pts2 = np.array([b1_up2, b1_up3, b1_down3, b1_down2], np.int32)
        b1_pts3 = np.array([b1_up1, b1_up4, b1_down4, b1_down1], np.int32)
        b1_pts4 = np.array([b1_up3, b1_up4, b1_down4, b1_down3], np.int32)
        b1_pts1 = b1_pts1.reshape((-1, 1, 2))
        b1_pts2 = b1_pts2.reshape((-1, 1, 2))
        b1_pts3 = b1_pts3.reshape((-1, 1, 2))
        b1_pts4 = b1_pts4.reshape((-1, 1, 2))

        b2_pts1 = np.array([b2_up1, b2_up2, b2_down2, b2_down1], np.int32)
        b2_pts2 = np.array([b2_up2, b2_up3, b2_down3, b2_down2], np.int32)
        b2_pts3 = np.array([b2_up1, b2_up4, b2_down4, b2_down1], np.int32)
        b2_pts4 = np.array([b2_up3, b2_up4, b2_down4, b2_down3], np.int32)
        b2_pts1 = b2_pts1.reshape((-1, 1, 2))
        b2_pts2 = b2_pts2.reshape((-1, 1, 2))
        b2_pts3 = b2_pts3.reshape((-1, 1, 2))
        b2_pts4 = b2_pts4.reshape((-1, 1, 2))

        r1_pts1 = np.array([r1_up1, r1_up2, r1_down2, r1_down1], np.int32)
        r1_pts2 = np.array([r1_up2, r1_up3, r1_down3, r1_down2], np.int32)
        r1_pts3 = np.array([r1_up1, r1_up4, r1_down4, r1_down1], np.int32)
        r1_pts4 = np.array([r1_up3, r1_up4, r1_down4, r1_down3], np.int32)
        r1_pts1 = r1_pts1.reshape((-1, 1, 2))
        r1_pts2 = r1_pts2.reshape((-1, 1, 2))
        r1_pts3 = r1_pts3.reshape((-1, 1, 2))
        r1_pts4 = r1_pts4.reshape((-1, 1, 2))

        r2_pts1 = np.array([r2_up1, r2_up2, r2_down2, r2_down1], np.int32)
        r2_pts2 = np.array([r2_up2, r2_up3, r2_down3, r2_down2], np.int32)
        r2_pts3 = np.array([r2_up1, r2_up4, r2_down4, r2_down1], np.int32)
        r2_pts4 = np.array([r2_up3, r2_up4, r2_down4, r2_down3], np.int32)
        r2_pts1 = r2_pts1.reshape((-1, 1, 2))
        r2_pts2 = r2_pts2.reshape((-1, 1, 2))
        r2_pts3 = r2_pts3.reshape((-1, 1, 2))
        r2_pts4 = r2_pts4.reshape((-1, 1, 2))

        cv2.polylines(copyimg, [b1_pts1], True, (0, 255, 0), 2)
        cv2.polylines(copyimg, [b1_pts2], True, (0, 255, 0), 2)
        cv2.polylines(copyimg, [b1_pts3], True, (0, 255, 0), 2)
        cv2.polylines(copyimg, [b1_pts4], True, (0, 255, 0), 2)

        cv2.polylines(copyimg, [b2_pts1], True, (0, 255, 0), 2)
        cv2.polylines(copyimg, [b2_pts2], True, (0, 255, 0), 2)
        cv2.polylines(copyimg, [b2_pts3], True, (0, 255, 0), 2)
        cv2.polylines(copyimg, [b2_pts4], True, (0, 255, 0), 2)

        cv2.polylines(copyimg, [r1_pts1], True, (0, 255, 0), 2)
        cv2.polylines(copyimg, [r1_pts2], True, (0, 255, 0), 2)
        cv2.polylines(copyimg, [r1_pts3], True, (0, 255, 0), 2)
        cv2.polylines(copyimg, [r1_pts4], True, (0, 255, 0), 2)

        cv2.polylines(copyimg, [r2_pts1], True, (0, 255, 0), 2)
        cv2.polylines(copyimg, [r2_pts2], True, (0, 255, 0), 2)
        cv2.polylines(copyimg, [r2_pts3], True, (0, 255, 0), 2)
        cv2.polylines(copyimg, [r2_pts4], True, (0, 255, 0), 2)

        # 2D box
        b1_x_min = np.array(min(b1_up1[0], b1_up2[0], b1_up3[0], b1_up4[0]))
        b1_y_min = np.array(min(b1_up1[1], b1_up2[1], b1_up3[1], b1_up4[1]))
        b1_x_max = np.array(max(b1_down1[0], b1_down2[0], b1_down3[0], b1_down4[0]))
        b1_y_max = np.array(max(b1_down1[1], b1_down2[1], b1_down3[1], b1_down4[1]))

        b2_x_min = np.array(min(b2_up1[0], b2_up2[0], b2_up3[0], b2_up4[0]))
        b2_y_min = np.array(min(b2_up1[1], b2_up2[1], b2_up3[1], b2_up4[1]))
        b2_x_max = np.array(max(b2_down1[0], b2_down2[0], b2_down3[0], b2_down4[0]))
        b2_y_max = np.array(max(b2_down1[1], b2_down2[1], b2_down3[1], b2_down4[1]))

        r1_x_min = np.array(min(r1_up1[0], r1_up2[0], r1_up3[0], r1_up4[0]))
        r1_y_min = np.array(min(r1_up1[1], r1_up2[1], r1_up3[1], r1_up4[1]))
        r1_x_max = np.array(max(r1_down1[0], r1_down2[0], r1_down3[0], r1_down4[0]))
        r1_y_max = np.array(max(r1_down1[1], r1_down2[1], r1_down3[1], r1_down4[1]))
        #
        r2_x_min = np.array(min(r2_up1[0], r2_up2[0], r2_up3[0], r2_up4[0]))
        r2_y_min = np.array(min(r2_up1[1], r2_up2[1], r2_up3[1], r2_up4[1]))
        r2_x_max = np.array(max(r2_down1[0], r2_down2[0], r2_down3[0], r2_down4[0]))
        r2_y_max = np.array(max(r2_down1[1], r2_down2[1], r2_down3[1], r2_down4[1]))

        cv2.imshow('line', copyimg)

        # cv2.imwrite(savepath + img_name + ".jpg", copyimg)
        cv2.waitKey(2)

if __name__ == '__main__':
    camera = "left"

    savepath = "/media/mmj/casia/202109/0907/img/4cars/" + camera + "/"

    imgpath = "/media/mmj/casia/202109/0907/4cars/0907_camera_raw_" + camera + "/"
    logpath = "/media/mmj/casia/202109/0907/4cars/0907log.log"
    main(logpath, imgpath, savepath, camera)
