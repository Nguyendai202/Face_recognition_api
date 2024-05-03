import numpy as np
import tensorflow
from numba import njit,prange
import os, time, cv2
import matplotlib.pyplot as plt
from facedetector.ssd_face_detector import FaceMaskDetection
from facerecognition.face_recongition import model_restore_from_pb
if tensorflow.__version__.startswith('1.'):
    import tensorflow as tf
    from tensorflow.python.platform import gfile
else:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    import tensorflow.compat.v1.gfile as gfile

print("Tensorflow version: ",tf.__version__)

def img_alignment(root_dir,output_dir,margin=44,GPU_ratio = 0.1,img_show=False,dataset_range=None):
    # ----record the start time
    # Hàm này sẽ tìm từng ảnh trong thư mục data rồi đưa vào mô hình dectection nhận dạng , sau đó sẽ cắt để 
    # tạo ra hình ảnh chỉ mỗi mặt (re_mask_id),kích thước ảnh phải lớn hơn ngưỡng quy địnhj và cuối cùng là lưu vào output_dir
    d_t = time.time()
    # ----var
    face_mask_model_path = r'./model/face_mask_detection.pb'
    img_format = {'png','bmp','jpg'}
    width_threshold = 100 + margin // 2
    height_threshold = 100 + margin // 2
    quantity = 0# số lượng 

    # ----collect all folders
    dirs = [obj.path for obj in os.scandir(root_dir) if obj.is_dir()]
    if len(dirs) == 0:
        print("No sub folders in ",root_dir)
    else:
        dirs.sort()# sắp xếp 
        print("Total class number: ", len(dirs))
        if dataset_range is not None:
            dirs = dirs[dataset_range[0]:dataset_range[1]]
            print("Working classes: {} to {}".format(dataset_range[0], dataset_range[1]))
        else:
            print("Working classes:All")

        #----init of face detection model
        fmd = FaceMaskDetection(face_mask_model_path,margin,GPU_ratio)

        # ----handle images of each dir
        for dir_path in dirs:
            paths = [file.path for file in os.scandir(dir_path) if file.name.split(".")[-1] in img_format]
            if len(paths) == 0:
                print("No images in ",dir_path)
            else:
                #----create the save dir
                save_dir = os.path.join(output_dir,dir_path.split("\\")[-1])
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                #----
                quantity += len(paths)
                for idx,path in enumerate(paths):
                    img = cv2.imread(path)
                    if img is None:
                        print("Read failed:",path)
                    else:
                        ori_height,ori_width = img.shape[:2]
                        img_ori = img.copy()
                        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                        # print(fmd.img_size)
                        img = cv2.resize(img, (260,260))
                        img = img.astype(np.float32)
                        img /= 255
                        img_4d = np.expand_dims(img,axis=0)# tạo mảng 4 chiều cho đầu vào mô hình ( thêm 1 vào đầu )
                        bboxes, re_confidence, re_classes, re_mask_id = fmd.inference(img_4d,ori_height,ori_width)

                        for num,bbox in enumerate(bboxes):
                            if bbox[2] > width_threshold and bbox[3] > height_threshold:
                                img_crop = img_ori[bbox[1]:bbox[1] + bbox[3],bbox[0]:bbox[0] + bbox[2], :]# crop ảnh theo toạ độ này 
                                save_path = os.path.join(save_dir,str(idx) + '_' + str(num) + ".png")
                                # print("save_path:",save_path)

                                cv2.imwrite(save_path,img_crop)

                                #----display images
                                if img_show is True:
                                    plt.subplot(1,2,1)
                                    plt.imshow(img_ori[:,:,::-1])

                                    plt.subplot(1,2,2)
                                    plt.imshow(img_crop[:,:,::-1])

                                    plt.show()

    # ----statistics(to know the average process time of each image)
    if quantity != 0:
        d_t = time.time() - d_t
        print("ave process time of each image:", d_t / quantity)



if __name__ == "__main__":
    #----alignment
    root_dir =  r"D:/dataset/lfw"#r"D:/dataset/test_aligmetn"
    output_dir =r"D:/dataset/lfw_aglined" #r"D:/face_mask_done/real_time_recongize_10/my_database"
    margin = 40
    GPU_ratio = None
    img_show = False
    dataset_range = None
    img_alignment(root_dir, output_dir, margin=margin, GPU_ratio=GPU_ratio, img_show=img_show,dataset_range=dataset_range)
    print('done')