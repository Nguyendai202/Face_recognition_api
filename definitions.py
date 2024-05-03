MAX_FPS = 30
processed_videos_dir = "video_process"
# Load model
pb_path = "./model/2.3 pb_model_select_num=15.pb"
node_dict = {'input': 'input:0',
             'phase_train': 'phase_train:0',
             'embeddings': 'embeddings:0',
             'keep_prob': 'keep_prob:0'
             }
face_mask_model_path = r'./model/face_mask_detection.pb'
margin = 40
ref_dir = "my_database"
img_format = ('.png', '.jpg', '.bmp')
id2class = {0: '', 1: ''}  # 0: mask , 1: nomask
batch_size = 32
threshold = 0.7  # 0.8