import math
class Const:
    grid_height = 449
    grid_width = 800

    ori_img_height = 480
    ori_img_width = 640
    grid_size = [grid_height, grid_width]
    reduce =4
    dataset = "MVM3D"

    # please change this to your data folder path
    data_root = '/home/dzc/Data/MVM3D'

    car_dist = math.sqrt(50/2 * 50/2 + 60/2 * 60/2)

    car_width = 50

    car_length = 60

    car_height = 40

    #------------------------------
    nms_left = 4

    roi_classes = 1

    bins = 8