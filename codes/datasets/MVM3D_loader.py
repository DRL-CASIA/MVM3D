import json
from PIL import Image
import torch
from torchvision.transforms import ToTensor
from codes.datasets.MVM3D import *
import warnings

from codes.EX_CONST import Const
warnings.filterwarnings("ignore")

class MVM3D_loader(VisionDataset):
    def __init__(self, base,  train=True, transform=ToTensor(), target_transform=ToTensor(),
                 reID=False, grid_reduce=Const.reduce, img_reduce=Const.reduce):
        super().__init__(base.root, transform=transform, target_transform=target_transform)

        self.reID, self.grid_reduce, self.img_reduce = reID, grid_reduce, img_reduce
        self.base = base
        self.train = train
        self.root, self.num_cam = base.root, base.num_cam
        self.img_shape, self.worldgrid_shape = base.img_shape, base.worldgrid_shape  # H,W; N_row,N_col
        self.reducedgrid_shape = list(map(lambda x: int(x / self.grid_reduce), self.worldgrid_shape))
        self.extrinsic_matrix = base.extrinsic_matrices
        self.intrinsic_matrix = base.intrinsic_matrices

        # split the dataset according to the
        if train == 1:
            frame_range = list(range(0, 1800)) + list(range(2100, 3500)) + list(range(3600, 4330))
        elif train == 3:
            frame_range = list (range(2000, 2100)) + list(range(3500, 3600))
        elif train == 2:
            frame_range = list(range(1800, 2100)) + list(range(3500, 3600))
        elif train == 4:
            frame_range = list(range(0, 1625))


        self.upsample_shape = list(map(lambda x: int(x / self.img_reduce), self.img_shape))
        img_reduce_local = np.array(self.img_shape) / np.array(self.upsample_shape)
        imgcoord2worldgrid_matrices = get_imgcoord2worldgrid_matrices(base.intrinsic_matrices,
                                                                           base.extrinsic_matrices,
                                                                           base.worldgrid2worldcoord_mat)
        img_zoom_mat = np.diag(np.append(img_reduce_local, [1]))
        map_zoom_mat = np.diag(np.append(np.ones([2]) / self.grid_reduce, [1]))

        self.proj_mats = [torch.from_numpy(map_zoom_mat @ imgcoord2worldgrid_matrices[cam] @ img_zoom_mat)
                          for cam in range(2)]

        # create angle bins
        bins = Const.bins
        overlap = 0.1
        self.bins = bins
        self.angle_bins = np.zeros(bins)
        self.interval = 2 * np.pi / bins
        for i in range(1, bins):
            self.angle_bins[i] = i * self.interval
        self.angle_bins += self.interval / 2  # center of the bin

        self.overlap = overlap
        # ranges for confidence
        self.bin_ranges = []
        for i in range(0, bins):
            self.bin_ranges.append(((i * self.interval - overlap) % (2 * np.pi), \
                                    (i * self.interval + self.interval + overlap) % (2 * np.pi)))

        self.bev_bboxes = {}
        self.left_bboxes = {}
        self.right_bboxes = {}
        self.left_dir = {}
        self.right_dir = {}
        self.left_angle = {}
        self.right_angle = {}
        self.left_orientation = {}
        self.left_conf = {}
        self.right_orientation = {}
        self.right_conf = {}
        self.world_xy = {}
        self.bev_angle = {}
        self.mark = {}

        self.img_fpaths = self.base.get_image_fpaths(frame_range)
        if train:
            self.gt_fpath = os.path.join(self.root, 'res/train_gt.txt')
        else:
            self.gt_fpath = os.path.join(self.root, 'res/test_gt.txt')
        self.prepare_gt(frame_range)
        self.prepare_bbox(frame_range)
        self.prepare_dir(frame_range)
        self.prepare_bins(frame_range)

    def get_bin(self, angle):

        bin_idxs = []

        def is_between(min, max, angle):
            max = (max - min) if (max - min) > 0 else (max - min) + 2*np.pi
            angle = (angle - min) if (angle - min) > 0 else (angle - min) + 2*np.pi
            return angle < max

        for bin_idx, bin_range in enumerate(self.bin_ranges):
            if is_between(bin_range[0], bin_range[1], angle):
                bin_idxs.append(bin_idx)

        return bin_idxs

    def prepare_bins(self, frame_range):
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations'))):
            frame_left_dir = []
            frame_right_dir = []
            frame_left_ang = []
            frame_right_ang = []
            frame_wxy = []
            frame_bev_angle = []
            frame_left_orientation = []
            frame_left_conf = []
            frame_right_orientation = []
            frame_right_conf = []

            frame = int(fname.split('.')[0])
            if frame in frame_range:
                with open(os.path.join(self.root, 'annotations', fname)) as json_file:
                    cars = [json.load(json_file)][0]
                for i, car in enumerate(cars):
                    wx = int(car["wx"]) // 10
                    wy = int(car["wy"]) // 10
                    mk = int(car["mark"])
                    # left_dir = int(car["direc_left"])
                    # right_dir = int(car["direc_right"])
                    left_dir = 0
                    right_dir = 0
                    bev_angle = float(car["angle"])

                    frame_wxy.append([wx, wy])

                    if Const.roi_classes != 1:
                        frame_left_dir.append(left_dir)
                        frame_right_dir.append(right_dir)
                    else:
                        frame_left_dir.append(0)
                        frame_right_dir.append(0)

                    # 0~360
                    if bev_angle < 0:
                        bev_angle += 2 * np.pi
                    # 左角度标签
                    alpha = np.arctan((Const.grid_height - wy) / wx)
                    left_target = bev_angle - alpha if bev_angle - alpha > 0 else 2 * np.pi + (bev_angle - alpha)
                    # if frame in range(500, 600) and i == 2:
                        # print(wx, wy)
                        # print(np.rad2deg(bev_angle))
                        # print(np.rad2deg(alpha))
                        # print(np.rad2deg(left_target))
                        # print(np.arctan(np.sin(left_target) / np.cos(left_target)))
                    # frame_left_ang.append([np.sin(left_target), np.cos(left_target)]) # 方案1, 回归sin cos
                    left_orientation = np.zeros((self.bins, 2))
                    left_confidence = np.zeros(self.bins)
                    left_bin_idxs = self.get_bin(left_target)
                    for bin_idx in left_bin_idxs:
                        angle_diff = left_target - self.angle_bins[bin_idx]
                        left_orientation[bin_idx, :] = np.array([np.cos(angle_diff), np.sin(angle_diff)])
                        left_confidence[bin_idx] = 1
                    # print("left conf", left_confidence)
                    frame_left_orientation.append(left_orientation)
                    frame_left_conf.append(left_confidence)


                    # 右角度标签, 颠倒一下正方向
                    bev_angle -= np.pi
                    if bev_angle < 0:
                        bev_angle += 2 * np.pi
                    frame_bev_angle.append(bev_angle)
                    alpha = np.arctan(wy / (Const.grid_width - wx))
                    right_target = bev_angle - alpha if bev_angle - alpha > 0 else 2 * np.pi + (bev_angle - alpha)
                    # frame_right_ang.append([np.sin(right_target), np.cos(right_target)]) # 方案1, 回归sin cos

                    right_orientation = np.zeros((self.bins, 2))
                    right_confidence = np.zeros(self.bins)
                    right_bin_idxs = self.get_bin(right_target)
                    for bin_idx in right_bin_idxs:
                        angle_diff = right_target - self.angle_bins[bin_idx]
                        right_orientation[bin_idx, :] = np.array([np.cos(angle_diff), np.sin(angle_diff)])
                        right_confidence[bin_idx] = 1
                    # print("right conf", right_confidence)
                    frame_right_orientation.append(right_orientation)
                    frame_right_conf.append(right_confidence)

                # print(frame_left_orientation)
                self.left_orientation[frame] = frame_left_orientation
                self.left_conf[frame] = frame_left_conf
                self.right_orientation[frame] = frame_right_orientation
                self.right_conf[frame] = frame_right_conf

    def prepare_gt(self,frame_range):
        og_gt = []
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations'))):
            frame = int(fname.split('.')[0])
            if frame in frame_range:
                with open(os.path.join(self.root, 'annotations', fname)) as json_file:
                    all_pedestrians = [json.load(json_file)][0]
                for single_pedestrian in all_pedestrians:
                    def is_in_cam(cam):
                        return not (single_pedestrian['views'][cam]['xmin'] == -1 and
                                    single_pedestrian['views'][cam]['xmax'] == -1 and
                                    single_pedestrian['views'][cam]['ymin'] == -1 and
                                    single_pedestrian['views'][cam]['ymax'] == -1)

                    in_cam_range = sum(is_in_cam(cam) for cam in range(self.num_cam))
                    if not in_cam_range:
                        continue

                    wx = single_pedestrian['wx']
                    wy = single_pedestrian['wy']

                    if wx > Const.grid_width * 10:
                        wx = Const.grid_width * 10 - 1
                    if wy > Const.grid_height * 10:
                        wy = Const.grid_height * 10 - 1

                    grid_x, grid_y= [wx //10, wy//10]
                    og_gt.append(np.array([frame, grid_x, grid_y]))
        og_gt = np.stack(og_gt, axis=0)
        os.makedirs(os.path.dirname(self.gt_fpath), exist_ok=True)
        print(self.gt_fpath)
        np.savetxt(self.gt_fpath, og_gt, '%d')

    def prepare_bbox(self, frame_range):
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations'))):
            frame_bev_box = []
            frame_left_box = []
            frame_right_box = []
            frame = int(fname.split('.')[0])
            if frame in frame_range:
                with open(os.path.join(self.root, 'annotations', fname)) as json_file:
                    cars = [json.load(json_file)][0]
                for i, car in enumerate(cars):
                    ymin_od = int(car["ymin_od"])
                    xmin_od = int(car["xmin_od"])
                    ymax_od = int(car["ymax_od"])
                    xmax_od = int(car["xmax_od"])
                    frame_bev_box.append([ymin_od, xmin_od, ymax_od, xmax_od])

                    for j in range(self.num_cam):
                        ymin = car["views"][j]["ymin"]
                        xmin = car["views"][j]["xmin"]
                        ymax = car["views"][j]["ymax"]
                        xmax = car["views"][j]["xmax"]
                        if j == 0:
                            frame_left_box.append([ymin, xmin, ymax, xmax])
                        else:
                            frame_right_box.append([ymin, xmin, ymax, xmax])

                self.bev_bboxes[frame] = frame_bev_box
                self.left_bboxes[frame] = frame_left_box
                self.right_bboxes[frame] = frame_right_box

    def prepare_dir(self, frame_range):
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations'))):
            frame_left_dir = []
            frame_right_dir = []
            frame_left_ang = []
            frame_right_ang = []
            frame_wxy = []
            frame_bev_angle = []
            frame = int(fname.split('.')[0])
            if frame in frame_range:
                with open(os.path.join(self.root, 'annotations', fname)) as json_file:
                    cars = [json.load(json_file)][0]
                for i, car in enumerate(cars):
                    wx = int(car["wx"]) // 10
                    wy = int(car["wy"]) // 10
                    mk = int(car["mark"])
                    # left_dir = int(car["direc_left"])
                    # right_dir = int(car["direc_right"])
                    left_dir = 0
                    right_dir = 0
                    bev_angle = float(car["angle"])

                    frame_wxy.append([wx, wy])

                    if Const.roi_classes != 1:
                        frame_left_dir.append(left_dir)
                        frame_right_dir.append(right_dir)
                    else:
                        frame_left_dir.append(0)
                        frame_right_dir.append(0)

                    # 0~360
                    if bev_angle < 0:
                        bev_angle += 2 * np.pi
                    # 左角度标签
                    alpha = np.arctan((Const.grid_height - wy) / wx)
                    left_target = bev_angle - alpha if bev_angle - alpha > 0 else 2 * np.pi + (bev_angle - alpha)
                    # if frame in range(500, 600) and i == 2:
                        # print(wx, wy)
                        # print(np.rad2deg(bev_angle))
                        # print(np.rad2deg(alpha))
                        # print(np.rad2deg(left_target))
                        # print(np.arctan(np.sin(left_target) / np.cos(left_target)))
                    frame_left_ang.append([np.sin(left_target), np.cos(left_target)]) # 方案1, 回归sin cos

                    # 右角度标签, 颠倒一下正方向
                    bev_angle -= np.pi
                    if bev_angle < 0:
                        bev_angle += 2 * np.pi
                    frame_bev_angle.append(bev_angle)
                    alpha = np.arctan(wy / (Const.grid_width - wx))
                    right_target = bev_angle - alpha if bev_angle - alpha > 0 else 2 * np.pi + (bev_angle - alpha)
                    frame_right_ang.append([np.sin(right_target), np.cos(right_target)]) # 方案1, 回归sin cos

                self.world_xy[frame] = frame_wxy
                self.left_dir[frame] = frame_left_dir
                self.right_dir[frame] = frame_right_dir
                self.bev_angle[frame] = frame_bev_angle
                self.left_angle[frame] = frame_left_ang
                self.right_angle[frame] = frame_right_ang
                self.mark[frame] = mk

    def __getitem__(self, index):
        frame = list(self.bev_bboxes.keys())[index]
        imgs = []
        for cam in range(self.num_cam):
            fpath = self.img_fpaths[cam][frame]
            img = Image.open(fpath).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)
        imgs = torch.stack(imgs)
        bev_bboxes = torch.tensor(self.bev_bboxes[frame])
        left_bboxes = torch.tensor(self.left_bboxes[frame])
        right_bboxes = torch.tensor(self.right_bboxes[frame])
        left_dirs = torch.tensor(self.left_dir[frame])
        right_dirs = torch.tensor(self.right_dir[frame])
        left_angles = torch.tensor(self.left_angle[frame])
        right_angles = torch.tensor(self.right_angle[frame])
        bev_xy =torch.tensor(self.world_xy[frame])
        bev_angle = torch.tensor(self.bev_angle[frame])
        mark = self.mark[frame]

        left_orientation = torch.tensor(self.left_orientation[frame])
        left_conf = torch.tensor(self.left_conf[frame])
        right_orientation = torch.tensor(self.right_orientation[frame])
        right_conf = torch.tensor(self.right_conf[frame])

        return imgs, bev_xy, bev_angle, bev_bboxes, \
               left_bboxes, right_bboxes,\
               left_dirs, right_dirs, \
               left_angles, right_angles, \
                left_orientation, right_orientation, \
                left_conf, right_conf, \
               frame, \
               self.extrinsic_matrix, self.intrinsic_matrix, \
               mark

    def __len__(self):
        return len(self.bev_bboxes.keys())

def get_imgcoord2worldgrid_matrices(intrinsic_matrices, extrinsic_matrices, worldgrid2worldcoord_mat):
    projection_matrices = {}
    for cam in range(2):
        worldcoord2imgcoord_mat = intrinsic_matrices[cam] @ np.delete(extrinsic_matrices[cam], 2, 1)

        worldgrid2imgcoord_mat = worldcoord2imgcoord_mat @ worldgrid2worldcoord_mat
        imgcoord2worldgrid_mat = np.linalg.inv(worldgrid2imgcoord_mat)
        permutation_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        projection_matrices[cam] = permutation_mat @ imgcoord2worldgrid_mat
    return projection_matrices


if __name__ == "__main__":
    data_path = os.path.expanduser('/home/dzc/Data/4carreal_0318blend')
    world_shape = Const.grid_size
    base = Robomaster_1_dataset(data_path, None, worldgrid_shape = world_shape)
    dataset = oftFrameDataset(base)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                               num_workers=8, pin_memory=True, drop_last=True)
    left_result = np.zeros((36,))
    right_result = np.zeros((36,))
    for batch_idx, data in enumerate(data_loader):
        # print(batch_idx)
        imgs, bev_xy, bev_angle, gt_bbox, gt_left_bbox, gt_right_bbox, left_dirs, right_dirs, left_sincos, right_sincos, frame, extrin, intrin = data
        for i in range(4):
            sin = left_sincos.squeeze()[i, 0]
            cos = left_sincos.squeeze()[i, 1]
            angle = np.arctan(sin / cos)
            if (sin > 0 and cos < 0) or (sin < 0 and cos < 0):
                angle += np.pi
            if sin < 0 and cos > 0:
                angle += np.pi * 2

            angle = np.rad2deg(angle)
            left_result[int(angle.item() // 10)] += 1

            if frame in range(600, 700) and i == 0:
                print("------------------")
                print(frame)
                print(angle.item())

            sin = right_sincos.squeeze()[i, 0]
            cos = right_sincos.squeeze()[i, 1]
            angle = np.arctan(sin / cos)
            if (sin > 0 and cos < 0) or (sin < 0 and cos < 0):
                angle += np.pi
            if sin < 0 and cos > 0:
                angle += np.pi * 2

            angle = np.rad2deg(angle)

            right_result[int(angle.item() // 10)] += 1

    import matplotlib.mlab as mlab
    import matplotlib.pyplot as plt

    X = np.arange(0, 36)
    Y = left_result
    fig = plt.figure()
    plt.bar(X, Y, 0.4, color="green")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("left")

    # plt.show()
    # plt.savefig("/home/dzc/Desktop/CASIA/proj/mvRPN-det/images/left_result.jpg")

    X = np.arange(0, 36)
    Y = right_result
    fig = plt.figure()
    plt.bar(X, Y, 0.4, color="green")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("right")

    # plt.show()
    # plt.savefig("/home/dzc/Desktop/CASIA/proj/mvRPN-det/images/right_result.jpg")































