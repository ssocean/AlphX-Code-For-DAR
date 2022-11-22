import csv
import os
import sys
import time
import zipfile

import cv2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from dataset.pan_pp.pan_pp_coco import Single_Det_Dataset
from Infer_Utils import *


GLOBAL_ORDER_ID = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
H = 64
seq_vis_dir = 'your path'
out_dir = 'your path'
DEBUG = False
USE_PREVIOUS_PALN = False
is_RGB = True
auto_make_directory(seq_vis_dir)
auto_make_directory(out_dir)
ORDER_WEIGHTS_PTH = 'weights/order-cnt.pth'
REC_MODEL_WEIGHTS_PTH = 'weights/64_512_best_model_128_256_57.7acc.pt'
PROCESS_ON_SMALLER_PIC = True

#说明：为减少空间占用，本队的三次提交均在代码中。三次提交分别对应代号PLAN-A、PLAN-B、PLAN-C。请通过更改PLAN值的方式，依次进行测试。
# 即，若需要测试PLAN-C,则令下方的PLAN='C'即可。
PLAN = 'A'

if PLAN == 'A':
    # ------------------PLAN-A----------------------------------
    FIXED_EDGE_LENGTH = 1024
    DET_MODEL_WEIGHTS_PTH = 'weights/det-R50-best.pth.tar'
    DET_CONFIG_FILE = 'config/pan_pp/R50-AUG.py'
    #-----------------------------------------------------------
elif PLAN == 'B':
    # ------------------PLAN-B----------------------------------
    FIXED_EDGE_LENGTH = 1024
    DET_MODEL_WEIGHTS_PTH = 'weights/det-R18-best.pth.tar'
    DET_CONFIG_FILE = 'config/pan_pp/R18-AUG.py'
    #-----------------------------------------------------------
elif PLAN == 'C':
    # ------------------PLAN-C----------------------------------
    FIXED_EDGE_LENGTH = 2048
    DET_MODEL_WEIGHTS_PTH = 'weights/det-R50-best.pth.tar'
    DET_CONFIG_FILE = 'config/pan_pp/R50-AUG.py'
    #-----------------------------------------------------------







class Infer(object):

    def __init__(self):
        self.det_model_weights_pth = DET_MODEL_WEIGHTS_PTH
        self.det_config_pth = DET_CONFIG_FILE
        cfg = Config.fromfile(self.det_config_pth)
        for d in [cfg, cfg.data.test]:
            d.update(dict(report_speed=True))
        print(json.dumps(cfg._cfg_dict, indent=4))
        self.cfg = cfg
        self.device = device
        self.output_dir = 'outputs/' #不要修改
        auto_make_directory(self.output_dir)
        model = build_model(self.cfg.model)
        model = model.cuda()
        if os.path.isfile(self.det_model_weights_pth):
            print("Loading model and optimizer from checkpoint '{}'".format(
                self.det_model_weights_pth))

            checkpoint = torch.load(self.det_model_weights_pth)

            d = dict()
            for key, value in checkpoint['state_dict'].items():
                tmp = key[7:]
                d[tmp] = value
            model.load_state_dict(d)
        else:
            print("No checkpoint found at '{}'".format(self.det_model_weights_pth))
            raise
        model = fuse_module(model)
        model_structure(model)
        self.model=model

        num_class = len(LABEL2CHAR) + 1
        img_height = config['img_height']
        img_width = config['img_width']
        iuput_channel = 3 if is_RGB else 1

        self.rec_model_weights_pth = REC_MODEL_WEIGHTS_PTH
        crnn = CRNN_CBAM(iuput_channel, H, 512, 13981, 128, 256, False)
        crnn.load_state_dict(torch.load(self.rec_model_weights_pth, map_location=self.device))
        self.rec_model = crnn
        order_model = UNet(n_channels=1).to(device)
        order_model.load_state_dict(torch.load(ORDER_WEIGHTS_PTH))
        self.order_model = order_model
        self.GID = 0
    def eval(self, image_name):# image_name 传入的是相对于本文件的图片路径，如'images/image_1000.jpg' # ,writer
        # print('Reading', image_name)
        writer = None # 如需可视化，请将writer从main.py传过来
        img=cv2.imread(image_name)
        assert img is not None, image_name
        print(image_name)
        # image_name = '/opt/data/private/data/test/image_1028.jpg'
        data_loader = Single_Det_Dataset(image_name)
        test_loader = torch.utils.data.DataLoader(
            data_loader,
            batch_size=1,
            shuffle=False,
            num_workers=1)
        self.GID = self.GID + 1
        test(test_loader, self.model, self.rec_model,self.cfg,self.output_dir,self.order_model,writer,self.GID)


def test(test_loader, model, rec_model,cfg,output_dir,order_model,writer,GID):

    rec_model.eval()
    model.eval()

    # print('Start testing %d images' % len(test_loader))
    result = []

    for idx, data in enumerate(test_loader):
        # print(f'Testing {idx}/{len(test_loader)}\r', end='', flush=True)
        # sleep(0.1)
        # prepare input
        idx = GID
        data['imgs'] = data['imgs'].cuda()
        img_pth = data['img_metas']['img_path'][0]
        img_name = data['img_metas']['img_name'][0]

        imgs = data['imgs'] # BCHW

        # writer.add_images('aug_vis', imgs, global_step=idx, dataformats='NCHW')
        data.update(dict(cfg=cfg))
        filename, file_extension = os.path.splitext(img_name)
        result_name='res_%s.jpg' % filename
        # time_1 = time.time()
        with torch.no_grad():
            outputs = model(**data)
        # time_2 = time.time()
        # print(f'detect time:{time_2-time_1}')
        bboxes = outputs['bboxes']
        ori_img_fullsize = cv2.imread(img_pth,1)
        if is_img_bg_black(ori_img_fullsize):
            ori_img_fullsize = 255 - ori_img_fullsize
        ori_h,ori_w = ori_img_fullsize.shape[:2]

        # order_img用于排序，尺寸固定896×896
        order_img = cv2.resize(ori_img_fullsize, dsize=(896, 896))
        # time_1 = time.time()
        ordered_bboxes = order_it_by_unet(order_img,bboxes,order_model,device,writer,idx)
        # print(len(ordered_bboxes))
        # time_2 = time.time()
        # print(f'orderit time:{time_2-time_1}')
        # print(f'bbox num after order:{len(ordered_bboxes)}')



        if PROCESS_ON_SMALLER_PIC:
            #用于绘图
            fixed_edge = FIXED_EDGE_LENGTH
            min_edge = min(ori_h,ori_w) # 找到短边
            # print(ori_h,ori_w)
            # print(f'min_edge:{min_edge}')
            if min_edge>FIXED_EDGE_LENGTH+200:
                # 如果短边是高
                if min_edge == ori_h:
                #短边为1024，计算放缩比例ratio
                    ratio = fixed_edge/ori_h
                #长边×ratio
                    resized_other_edge = int(ori_w*ratio)
                    adaptive_shrink_img = cv2.resize(ori_img_fullsize, (resized_other_edge, fixed_edge))
                else:
                    ratio = fixed_edge/ori_w
                    resized_other_edge = int(ori_h*ratio)
                    adaptive_shrink_img = cv2.resize(ori_img_fullsize, (fixed_edge,resized_other_edge))
                #缩放图像
            else:
                adaptive_shrink_img = ori_img_fullsize.copy()
            resized_h,resized_w = adaptive_shrink_img.shape[:2]
            ori_img_for_paint = adaptive_shrink_img.copy()
        else:
            ori_img_for_paint = ori_img_fullsize.copy()


        # print(f'ori_img_for_paint.shape:{ori_img_for_paint.shape}')
        with torch.no_grad():
            result = []
            all_patches = []
            cnts_dict={}
            cnts_list = []
            # print(f'len(ordered_bboxes):{len(ordered_bboxes)}')


            #-------------------------------------BEFORE REC-----------------------------------
            three_times = []
            for_times = []
            # time_1_1 = time.time()
            for i, bbox in enumerate(ordered_bboxes):
                # print(ori_img.shape)
                # time_for_start = time.time()
                # time_3 = time.time()
                full_size_cnts = resize_contour(bbox.copy(),(896,896),(ori_h,ori_w))
                rst = np.round(full_size_cnts.reshape((-1))).tolist() # 点的坐标
                if len(rst)<=6:
                    continue
                cnts_dict[f'{i}'] = rst
                cnts_list.append(rst)
                # time_4 = time.time()
                # cr_time = time_4-time_3
                # print(f'Contour_Resize time:{time_4-time_3}')
                # time_3 = time.time()
                if PROCESS_ON_SMALLER_PIC:# 如果在小图上进行提取

                    # print(f'auto shrinked shape:{adaptive_shrink_img.shape}')
                    shrinked_cnts = resize_contour(bbox.copy(),(896,896),(resized_h,resized_w))
                    img_for_roi = adaptive_shrink_img.copy()
                    # cnt边缘平滑
                    approx_cnt = np.array(shrinked_cnts.reshape(-1).reshape(-1, 2).reshape(-1, 1, 2),dtype=np.float32)
                    epsilon = 0.005*cv2.arcLength(approx_cnt,True)
                    shrinked_cnts = cv2.approxPolyDP(approx_cnt,epsilon,True).reshape(-1).reshape(-1, 2)


                    CRNN_ROI = extract_roi_by_cnt(img_for_roi.copy(),shrinked_cnts.copy())
                    # CRNN_ROI = extract_roi_by_cnt(ori_img_fullsize.copy(),cnts)

                    draw_cnt_poly = shrinked_cnts.copy()
                    pass
                else:#在原图上操作
                # print(cnts) #ndarray
                    full_size_cnts = resize_contour(bbox.copy(),(896,896),(ori_h,ori_w))
                    ori_img_for_roi = ori_img_fullsize.copy()

                    # 让cnt边缘平滑些
                    approx_cnt = np.array(full_size_cnts.reshape(-1).reshape(-1, 2).reshape(-1, 1, 2),dtype=np.float32)
                    epsilon = 0.005*cv2.arcLength(approx_cnt,True)
                    full_size_cnts = cv2.approxPolyDP(approx_cnt,epsilon,True).reshape(-1).reshape(-1, 2)

                    CRNN_ROI = extract_roi_by_cnt(ori_img_for_roi.copy(),full_size_cnts.copy())
                    # CRNN_ROI = extract_roi_by_cnt(ori_img_fullsize.copy(),cnts)
                    draw_cnt_poly = full_size_cnts.copy()
                # print(.shape)

                # time_4 = time.time()
                # erbc_time = time_4-time_3
                # print(f'extract_roi_by_cnt:{time_4-time_3}')
                # cv2.imwrite(f'{seq_vis_dir}/{filename}_{i}_BEFOREDESKEW.png',CRNN_ROI)
                # time_3 = time.time()
                CRNN_ROI = deskew(CRNN_ROI)
                # time_4 = time.time()
                # deskew_time = time_4-time_3
                # print(f'Deskew-time:{time_4-time_3}')
                # cv2.imwrite(f'{seq_vis_dir}/{filename}_{i}_CRNN_ROI.png',CRNN_ROI)
                # print(f'CRNN_ROI.shape:{CRNN_ROI.shape}')
                if CRNN_ROI is None:
                    # print('-----------------------None CRNN ROI------------------------')
                    continue
                if is_RGB:
                    CRNN_ROI = cv2.cvtColor(CRNN_ROI.copy(), cv2.COLOR_BGR2RGB)

                # patches是以长度256 步长128的方式在32*N (N>1.5*256)的图像上切出来的
                # time_3 = time.time()
                patches = overlapping_seg(CRNN_ROI)
                # time_4 = time.time()
                # overlap_time = time_4-time_3
                # print(f'Overlapseg-time:{time_4-time_3}')
                all_patches += [(i,patch) for patch in patches]
                # print(f'Three-time:{erbc_time+deskew_time+overlap_time}')
                # three_times.append(float(erbc_time+deskew_time+overlap_time+cr_time))
                if DEBUG:
                    # DEBUG模式在识别部分采用单张推理，以便可视化

                    with torch.no_grad():
                        rec_seq = []
                        # print(f'len(patches):{len(patches)}')
                        for j,patch_cv in enumerate(patches):
                            # print(f'patch尺寸:{patch_cv.shape}')
                            if is_RGB:
                                patch = Image.fromarray(patch_cv)
                            else:
                                patch = Image.fromarray(patch_cv).convert('L')
                            # 获得序列识别结果识别结果
                            output = crnn_rec(rec_model,patch,LABEL2CHAR,tfs,device)
                            # print(f'crnn输出：{output}')
                            rec_seq.append(output)
                            # writer.add_image(f'Seq-Vis_{GID}', patch_cv, global_step=j, dataformats='HWC')
                            # cv2.imwrite(f'{seq_vis_dir}/{filename}_{i}_{j}_{output}.png',patch_cv)

                        # print(f'strs waiting for being merged:{rec_seq}')
                        # 字符串合并算法
                        rec_rst = merge_strs(rec_seq)

                        rst.append(rec_rst)
                        if rec_rst == '' or len(rst)<=7:
                            continue
                        # print(rst)
                        result.append(tuple(rst))
                        # print(outputs['words'][i])
                        poly = np.array(draw_cnt_poly.copy()).astype(np.int32).reshape((-1))
                        poly = poly.reshape(-1, 2)
                        # cx,cy = find_cnt_center(poly)
                        cv2.polylines(ori_img_for_paint, [poly.reshape((-1, 1, 2))], True, color=(0, 190, 0), thickness=1)
                        # print(outputs['bboxes'][i][:2])
                        print(f'{i},'+rec_rst)
                        ori_img_for_paint = cv2_chinese_text(ori_img_for_paint, f'{i},'+rec_rst, draw_cnt_poly[0], textColor=(255, 0, 0), textSize=20)
                    # writer.add_image('Rec-Vis', ori_img_for_paint, global_step=GID, dataformats='HWC')
                    cv2.imwrite(os.path.join(out_dir ,result_name), ori_img_for_paint)
                # time_for_end = time.time()
                # for_times.append(float(time_for_end-time_for_start))
            # time_1_2 = time.time()

            # print(f'For time:{sum(for_times)}')
            # print(f'Three time:{sum(three_times)}')
            # print(f'ROI Processing Before Rec time:{time_1_2-time_1_1}')
            if not DEBUG:
                # 比赛时，禁用DEBUG，采用batch方式加速推理
                # time_1 = time.time()
                patch_dataset = PatchDataset(all_patches,tfs,opt)
                demo_loader = torch.utils.data.DataLoader(patch_dataset, batch_size=128,shuffle=False,num_workers=8, pin_memory=True)
                # time_1_1 = time.time()
                rec_rsts_in_single_img = seq_rec(rec_model,demo_loader,device)
                # time_1_2 = time.time()
                # print(f'rec per batch time:{time_1_2-time_1_1}')

                cur_textline_rst = []
                rec_rsts = []
                last_pivot = 0
                for j,(pivot,cnts) in enumerate(all_patches):
                    if pivot!=last_pivot:
                        # print(merge_strs(cur_textline_rst))
                        rec_rsts.append(merge_strs(cur_textline_rst))
                        cur_textline_rst = []
                    # print(f'当前添加字符串：{rec_rsts_in_single_img[j]}')
                    cur_textline_rst.append(rec_rsts_in_single_img[j])
                    last_pivot = pivot
                rec_rsts.append(merge_strs(cur_textline_rst))

                result = []
                for id in range(len(rec_rsts)):
                    temp = cnts_dict[f'{id}'] + [rec_rsts[id]]
                    result.append(temp)

                out_pth = os.path.join(output_dir,f'{filename}.csv')
                # print(out_pth)
                write_csv(result, file_pth=out_pth, overwrite=True)
                # time_2 = time.time()
                # print(f'Rec time:{time_2-time_1}')
    # writer.close()
    # time_2 = time.time()
    # print(f'rec time:{time_2-time_1}')
