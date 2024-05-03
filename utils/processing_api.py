import numpy as np
from numba import njit,prange
@njit(fastmath=True, cache=True)
def merge_images(images):  # gộp các ảnh đã xử lí vào 1 ảnh
    # Lấy kích thước của ảnh đầu tiên trong danh sách
    height, width, _ = images[0].shape
    # Tạo một ảnh trống có kích thước tương tự
    merged_image = np.zeros((height, width * len(images), 3), dtype=np.uint8)
    # Gán từng ảnh vào vị trí tương ứng trên ảnh gộp
    for i, image in enumerate(images):
        merged_image[:, i * width:(i + 1) * width, :] = image
    return merged_image
