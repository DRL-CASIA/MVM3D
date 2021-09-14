import numpy as np
from scipy.optimize import linear_sum_assignment
import math
import shapely
from shapely.geometry import Polygon, MultiPoint    # 多边形计算的库

def wh2bottomleft(bbox):
    x, y, w, h = bbox
    xmin = x - w/2
    xmax = x + w/2
    ymin = y - h/2
    ymax = y + h/2
    return xmin, ymin, xmax, ymax


# 求任意四边形iou
def compute_IOU(line1,line2):
    # 四边形四个点坐标的一维数组表示，[x,y,x,y....]
    # 如：line1 = [728, 252, 908, 215, 934, 312, 752, 355]
    # 返回iou的值，如 0.7
    line1_box = np.array(line1).reshape(4, 2)  # 四边形二维坐标表示
    # 凸多边形与凹多边形
    poly1 = Polygon(line1_box)
    # .convex_hull  # python四边形对象，会自动计算四个点，最后四个点顺序为：左上 左下  右下 右上 左上
    line2_box = np.array(line2).reshape(4, 2)
    # 凸多边形与凹多边形
    poly2 = Polygon(line2_box)

    union_poly = np.concatenate((line1_box, line2_box))  # 合并两个box坐标，变为8*2
    if not poly1.intersects(poly2):  # 如果两四边形不相交
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area  # 相交面积
            union_area = MultiPoint(union_poly).convex_hull.area
            if union_area == 0:
                iou = 0
            else:
                iou = float(inter_area) / union_area
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    return iou

def bbox_iou(box1, box2):
    '''
    两个框（二维）的 iou 计算
    注意：边框以左上为原点
    box:[x1,y2,x2,y2],依次为左上右下坐标
    '''
    h = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
    w = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
    area_box1 = ((box1[2] - box1[0]) * (box1[3] - box1[1]))
    area_box2 = ((box2[2] - box2[0]) * (box2[3] - box2[1]))
    inter = w * h
    union = area_box1 + area_box2 - inter
    iou = inter / union
    return iou

def getDistance(x1, y1, x2, y2):
    return math.sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2))

def CLEAR_MOD_HUN2(gt, det, thresh):
    F = int(max(gt[:, 0])) + 1
    # F = 1
    precs = 0
    aos = 0
    all_infolist = None
    for t in range(1, F + 1):
        gt_results = gt[gt[:, 0] == t - 1]
        det_results = det[det[:, 0] == t - 1]
        frame_infolist = cal_frame_TPFP_iou(thresh, gt_results, det_results)
        if all_infolist is None:
            all_infolist = frame_infolist
        else:
            all_infolist = np.concatenate((all_infolist, frame_infolist), axis=0)

    idx = np.argsort(all_infolist[:,0], axis=0)
    idx = idx[::-1]
    all_infolist = all_infolist[idx]
    TP = 0
    FP = 0
    all_P = gt.shape[0]
    for i, data in enumerate(all_infolist):
        flag = data[-4]
        if flag == 1:
            TP += 1
        else:
            FP += 1

        all_infolist[i, -3] = TP / (TP + FP)
        all_infolist[i, -2] = TP / all_P
        cur_aos = 0
        for m in range(i + 1):
            cur_aos += all_infolist[m, -4] * (1 + np.cos(np.deg2rad(all_infolist[m, 3]))) / 2
        cur_aos /= (i + 1)
        all_infolist[i, -1] = cur_aos
    recall_threshold = np.arange(0, 1.1, 0.1)
    accu_precisions = 0
    for thresh in recall_threshold:
        max_prec = 0
        for k in range(all_infolist.shape[0]):
            if all_infolist[k][-2] >= thresh:
                max_prec = max(all_infolist[k:,-3])
                break
        accu_precisions += max_prec


    final_11_precision = accu_precisions / 11

    # AOS
    accu_aos = 0
    for thresh in recall_threshold:
        max_aos = 0
        for k in range(all_infolist.shape[0]):
            if all_infolist[k][-2] >= thresh:
                max_aos = max(all_infolist[k:,-1])
                break
        accu_aos += max_aos
    final_11_aos = accu_aos / 11

    return final_11_precision, final_11_aos

def cal_frame_TPFP(dist_threshold, gt_res, pred_res):
    # prec = TP / (TP + FP)
    # recall = TP / label_P
    # 根据距离打标签，超出39厘米就是FP
    # 0     1           2           3           4       5    6       7
    # score, frame_idx, delta_dist, delta_ori, TP/FP?, prec, recall, aos
    frame_gt_det_match = np.zeros(shape=(pred_res.shape[0], 8)) - 1
    frame_gt_det_match[:, -4:] += 1
    for i, pred in enumerate(pred_res):
        min_dist = -1
        min_idx = -1
        cur_gt_ori = -1
        _, _, x_pred, y_pred, w_pred, h_pred, score, ori_pred = pred
        for j, gt in enumerate(gt_res):
            _, _, x_gt, y_gt, w_pred, h_pred, ori_gt = gt
            dist = math.sqrt(pow(x_pred - x_gt, 2)+ pow(y_pred - y_gt, 2))
            if (dist < min_dist or min_dist == -1) and dist <= dist_threshold:
                # 找到距离最近的gt分配给那个pred，始终没分配到gt的pred认为是FN
                min_dist = dist
                min_idx = j
                cur_gt_ori = ori_gt

        # 将这个检测对应的gt信息存到数组里，如果没有gt（前两行都是-1），那就是FN
        frame_gt_det_match[i][0] = score
        frame_gt_det_match[i][1] = min_idx
        frame_gt_det_match[i][2] = min_dist
        frame_gt_det_match[i][3] = ori_pred - cur_gt_ori

    TP = 0
    FP = 0
    passed_index = []
    # 这一帧的TP，FP
    for k in range(pred_res.shape[0]):
        # 判断是TP还是FP
        if -1 not in frame_gt_det_match[k, :] and frame_gt_det_match[k, :][1] not in passed_index:
            TP += 1
            passed_index.append(frame_gt_det_match[k, :][1])
            frame_gt_det_match[k, 4] = 1
        elif -1 not in frame_gt_det_match[k, :] and frame_gt_det_match[k, :][1] in passed_index:
            FP += 1
            frame_gt_det_match[k, 4] = 0
        elif -1 in frame_gt_det_match[k, :]:
            FP += 1
            frame_gt_det_match[k, 4] = 0
    return frame_gt_det_match

def cal_frame_TPFP_iou(dist_threshold, gt_res, pred_res):
    # 0     1           2           3   4       5     6       7
    # score, frame_idx, iou, delta_ori, TP/FP?, prec, recall, aos
    frame_gt_det_match = np.zeros(shape=(pred_res.shape[0], 8)) - 1
    frame_gt_det_match[:, -4:] += 1
    for i, pred in enumerate(pred_res):
        max_iou = -1
        max_idx = -1
        cur_gt_ori = -1
        _, _, x1_rot, y1_rot, x2_rot, y2_rot, x3_rot, y3_rot, x4_rot, y4_rot, score, ori_pred = pred
        for j, gt in enumerate(gt_res):
            _, _, gt_x1_rot, gt_y1_rot, gt_x2_rot, gt_y2_rot, gt_x3_rot, gt_y3_rot, gt_x4_rot, gt_y4_rot, ori_gt = gt
            iou = compute_IOU([x1_rot, y1_rot, x2_rot, y2_rot, x3_rot, y3_rot, x4_rot, y4_rot],\
                                      [gt_x1_rot, gt_y1_rot, gt_x2_rot, gt_y2_rot, gt_x3_rot, gt_y3_rot, gt_x4_rot, gt_y4_rot])
            if max_iou != 0 and iou >= dist_threshold and iou > max_iou:
                # 找到距离最近的gt分配给那个pred，始终没分配到gt的pred认为是FN
                max_iou = iou
                max_idx = j
                cur_gt_ori = ori_gt

        frame_gt_det_match[i][0] = score
        frame_gt_det_match[i][1] = max_idx
        frame_gt_det_match[i][2] = max_iou
        frame_gt_det_match[i][3] = ori_pred - cur_gt_ori


    TP = 0
    FP = 0
    passed_index = []
    for k in range(pred_res.shape[0]):
        if -1 not in frame_gt_det_match[k, :]:
            TP += 1
            passed_index.append(frame_gt_det_match[k, :][1])
            frame_gt_det_match[k, 4] = 1
        elif -1 in frame_gt_det_match[k, :]:
            FP += 1
            frame_gt_det_match[k, 4] = 0
    return frame_gt_det_match

def evaluateDetectionAPAOS(res_fpath, gt_fpath):
    gtRaw = np.loadtxt(gt_fpath)
    detRaw = np.loadtxt(res_fpath)

    frames = np.unique(detRaw[:, 0]) if detRaw.size else np.zeros(0)
    frame_ctr = 0
    gt_flag = True
    det_flag = True

    gtAllMatrix = 0
    detAllMatrix = 0
    if detRaw is None or detRaw.shape[0] == 0:
        MODP, MODA, recall, precision = 0, 0, 0, 0
        return MODP, MODA, recall, precision

    for t in frames:
        idxs = np.where(gtRaw[:, 0] == t)
        idx = idxs[0]
        idx_len = len(idx)
        tmp_arr = np.zeros(shape=(idx_len, 11))

        tmp_arr[:, 0] = np.array([frame_ctr for n in range(idx_len)])
        tmp_arr[:, 1] = np.array([i for i in range(idx_len)])
        tmp_arr[:, 2] = np.array([j for j in gtRaw[idx, -8]])
        tmp_arr[:, 3] = np.array([k for k in gtRaw[idx, -7]])
        tmp_arr[:, 4] = np.array([k for k in gtRaw[idx, -6]])
        tmp_arr[:, 5] = np.array([k for k in gtRaw[idx, -5]])
        tmp_arr[:, 6] = np.array([j for j in gtRaw[idx, -4]])
        tmp_arr[:, 7] = np.array([k for k in gtRaw[idx, -3]])
        tmp_arr[:, 8] = np.array([k for k in gtRaw[idx, -2]])
        tmp_arr[:, 9] = np.array([k for k in gtRaw[idx, -1]])
        tmp_arr[:, 10] = np.array([m for m in gtRaw[idx, -9]])
        if gt_flag:
            gtAllMatrix = tmp_arr
            gt_flag = False
        else:
            gtAllMatrix = np.concatenate((gtAllMatrix, tmp_arr), axis=0)
        idxs = np.where(detRaw[:, 0] == t)
        idx = idxs[0]
        idx_len = len(idx)
        tmp_arr = np.zeros(shape=(idx_len, 12))
        tmp_arr[:, 0] = np.array([frame_ctr for n in range(idx_len)])
        tmp_arr[:, 1] = np.array([i for i in range(idx_len)])
        tmp_arr[:, 2] = np.array([j for j in detRaw[idx, -8]])
        tmp_arr[:, 3] = np.array([k for k in detRaw[idx, -7]])
        tmp_arr[:, 4] = np.array([k for k in detRaw[idx, -6]])
        tmp_arr[:, 5] = np.array([k for k in detRaw[idx, -5]])
        tmp_arr[:, 6] = np.array([j for j in detRaw[idx, -4]])
        tmp_arr[:, 7] = np.array([k for k in detRaw[idx, -3]])
        tmp_arr[:, 8] = np.array([k for k in detRaw[idx, -2]])
        tmp_arr[:, 9] = np.array([k for k in detRaw[idx, -1]])
        tmp_arr[:, 10] = np.array([m for m in detRaw[idx, -9]])
        tmp_arr[:, 11] = np.array([p for p in detRaw[idx, -10]])

        if det_flag:
            detAllMatrix = tmp_arr
            det_flag = False
        else:
            detAllMatrix = np.concatenate((detAllMatrix, tmp_arr), axis=0)
        frame_ctr += 1

    AP_50, AOS_50 = CLEAR_MOD_HUN2(gtAllMatrix, detAllMatrix, 0.5)
    AP_25, AOS_25 = CLEAR_MOD_HUN2(gtAllMatrix, detAllMatrix, 0.25)
    return AP_50 * 100, AOS_50 * 100, AOS_50/AP_50, AP_25* 100, AOS_25* 100, AOS_25/AP_25

