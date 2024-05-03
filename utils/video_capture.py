import cv2 as cv
import os
cap = cv.VideoCapture(0)
FourCC = cv.VideoWriter_fourcc(*"XVID")
result = cv.VideoWriter("dai_0308_20fps.avi",FourCC,40.0,(640,480))
# cap.set(3,1280)
# cap.set(4,720)# set độ phân giải khung hình lớn hơn 
ref_dir = "Realtime_face_recognition"

while (cap.isOpened()):
    ret,frame = cap.read()
    if ret ==True:
        print(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        print(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        result.write(frame)
        cv.imshow("Frame",frame)
        key = cv.waitKey(1) & 0xFF
        if key == ord('a'):
                break
        elif key == ord('s'):
                    save_path = "img_crop.jpg"
                    save_path = os.path.join(ref_dir,save_path)
                    cv.imwrite(save_path,frame)
                    print("An image is saved to ",save_path)
        

cap.release()
result.release()
cv.destroyAllWindows()