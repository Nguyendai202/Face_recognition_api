import cv2
import os

img_format = {'png','jpg','bmp'}
def video_init(camera_source=0,resolution="480",to_write=False,save_dir=None):
    #----var
    writer = None
    resolution_dict = {"480":[480,640],"720":[720,1280],"1080":[1080,1920]}

    #----camera source connection
    cap = cv2.VideoCapture(camera_source)

    #----resolution decision
    if resolution in resolution_dict.keys():
        width = resolution_dict[resolution][1]
        height = resolution_dict[resolution][0]
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    else:
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)#default 480
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)#default 640

    if to_write is True:
        #fourcc = cv2.VideoWriter_fourcc('x', 'v', 'i', 'd')
        #fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = fourcc.get(cv2.CAP_PROP_FPS)
        save_path = 'dai_0308_30fps.avi'

        if save_dir is not None:
            save_path = os.path.join(save_dir,save_path)
        writer = cv2.VideoWriter(save_path, fourcc,fps, (int(width), int(height)))# đầu ra giữ như đầu vào  

    return cap,height,width,writer
def get_image_paths_recursive(directory, img_format):
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(img_format):
                image_path = os.path.join(root, file)
                image_paths.append(image_path)

    return image_paths