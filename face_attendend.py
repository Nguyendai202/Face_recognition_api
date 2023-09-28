from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUiType
import sys
import sqlite3
from datetime import date
import cv2, os
from face_alignment import FaceMaskDetection
import numpy
from  real_time_face_recognition import stream_attendend

ui,_=loadUiType('face_recongtion.ui')
print(ui)

class MainApp(QMainWindow, ui):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.tabWidget.setCurrentIndex(0)# tap hiển thị trong tiên trong tap widget của app 
        self.LOGIN.clicked.connect(self.login)
        self.CLOSE.clicked.connect(self.close)
        self.LOGOUT.clicked.connect(self.logout)
        self.train_user.clicked.connect(self.show_training)
        self.attendt_entry.clicked.connect(self.show_attendent)
        self.reports_tab2.clicked.connect(self.show_reportstab2)
        self.goback_training.clicked.connect(self.show_mainfrom)
        self.face_recong_back.clicked.connect(self.show_training)
        self.reports_back.clicked.connect(self.show_face_recongize)
        self.train.clicked.connect(self.start_training)
        self.Record.clicked.connect(lambda: self.record_attendance(0, "video_output"))
        self.dateEdit.setDate(date.today())
        self.dateEdit.dateChanged.connect(self.show_selected_date_reports)# thay đổi ngày
        self.tabWidget.setStyleSheet("QTabWidget::pane{border:0;}")# xoá đường viền 

        try:
            con = sqlite3.connect("face-reco.db")
            con.execute("CREATE TABLE IF NOT EXISTS attendance(attendanceid INTEGER, name TEXT, attendancedate TEXT)")
            #nếu bảng đã tồn tại thì ko thực thi câu lệnh , gồm 3 cột và kiểu data ở phía sau 
            con.commit()# lưu thay đổi 
            print("Table created successfully")
        except:
            print("error in database!")

        ## login process
    def login(self):
        pw = self.PASSWORD.text()
        if(pw=="114"):
            self.PASSWORD.setText("")
            self.tabWidget.setCurrentIndex(1)# chuyển tab tiếp theo 
        else:
            self.FAILWORD.setText("Invalid password!")
            self.PASSWORD.setText("")
    def logout (self):
        self.tabWidget.setCurrentIndex(0)
    def close(self):
        self.close()
    def show_training(self):
         self.tabWidget.setCurrentIndex(2)
    def show_attendent(self):
         self.tabWidget.setCurrentIndex(3)
    def show_reportstab2(self):
         self.tabWidget.setCurrentIndex(4)
         self.REPORTTAB4.setRowCount(0)# xoá tất cả hàng
         self.REPORTTAB4.clear()# thực hiện xoá
         con = sqlite3.connect("face-reco.db")
         cursor = con.execute("SELECT * FROM attendance")
         result = cursor.fetchall()
         r = 0
         c = 0 
         for row_number, row_data in enumerate(result):
             r += 1
             c = 0 
             for colum_number, data in enumerate(row_data):
                 c+=1
         self.REPORTTAB4.setColumnCount(c)
         for row_number, row_data in enumerate(result):
             self.REPORTTAB4.insertRow(row_number)
             for colum_number, data in enumerate(row_data):
                   self.REPORTTAB4.setItem(row_number,colum_number,QTableWidgetItem(str(data)))# đặt phàn tử data vào vị trí đc chỉ định hàng ? và cột ? 


         self.REPORTTAB4.setHorizontalHeaderLabels(['Id','Name','Date']) # nhãn cho các cột 
         self.REPORTTAB4.setColumnWidth(0,10)# chiều rộng từng cột 
         self.REPORTTAB4.setColumnWidth(1,30)
         self.REPORTTAB4.setColumnWidth(2,90)
         self.REPORTTAB4.verticalHeader().setVisible(False)# ẩn tiêu đề các hàng trong bảng 


    def show_mainfrom(self):
         self.tabWidget.setCurrentIndex(1)
    def show_training(self):
         self.tabWidget.setCurrentIndex(2)
    def show_face_recongize(self):
         self.tabWidget.setCurrentIndex(3)
    # Training process 
    # sử dụng haar phát hiện mặt và crop nó vào thư mục lưu trữ , có thể dùng cho database hoặc lấy dữ liệu training facenet ,arcface
    def start_training(self):
        face_mask_model_path = r'./model/face_mask_detection.pb'
        margin = 40
        img_format = {'png','jpg','bmp'}
        id2class = {0: '', 1: ''}# 0: mask , 1: nomask
        batch_size = 32
        threshold = 0.7#0.8
        # cap,height,width,writer = video_init(camera_source=camera_source, resolution=resolution, to_write=to_write, save_dir=save_dir)
            # ----face detection init
        ref_dir = 'datasets'
        sub_data = self.personame_training.text()
        path = os.path.join(ref_dir,sub_data)
        if not os.path.isdir(path):
            os.mkdir(path)
            print("The new directory is created")
            # (width,height) = (130,100)
            fmd = FaceMaskDetection(face_mask_model_path, margin, GPU_ratio=None)
            webcam = cv2.VideoCapture(0)
            count = 1
            while count < int(self.training_capture_counts.text()) + 1:
                print(count)
                (_,im) = webcam.read()
                img_format = {'png','bmp','jpg'}
                width_threshold = 100 + margin // 2
                height_threshold = 100 + margin // 2 
                    #----init of face detection model
                ori_height,ori_width = im.shape[:2]
                img_ori = im.copy()
                img = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
                # print(fmd.img_size)
                img = cv2.resize(img, (260,260))
                img = img.astype(numpy.float32)
                img /= 255
                img_4d = numpy.expand_dims(img,axis=0)# tạo mảng 4 chiều cho đầu vào mô hình ( thêm 1 vào đầu )
                bboxes, re_confidence, re_classes, re_mask_id = fmd.inference(img_4d,ori_height,ori_width)

                for num,bbox in enumerate(bboxes):
                    if bbox[2] > width_threshold and bbox[3] > height_threshold:
                        img_crop = img_ori[bbox[1]:bbox[1] + bbox[3],bbox[0]:bbox[0] + bbox[2], :]# crop ảnh theo toạ độ này 
                        cv2.imwrite('%s/%s.png'%(path,count),img_crop)
                count += 1
                cv2.imshow('OpenCV',im)
                key = cv2.waitKey(2000)# độ trễ 2s 
                if key == 27:# 27 =esc 
                    break
            webcam.release()
            cv2.destroyAllWindows()  
            path=""
            QMessageBox.information(self,"Attendance System","Training Completed Successfully") # thôg báo  
            self.personame_training.setText("")
            self.training_capture_counts.setText("10")
    
     ### RECORD ATTENDANCE ###
    def record_attendance(self,source,output_directory):
        self.Record.setText("Process started.. Waiting..")       
        cnt=0
        pb_path = r"./model/2.3 pb_model_select_num=15.pb"
        node_dict = {'input': 'input:0',
                 'keep_prob': 'keep_prob:0',
                 'phase_train': 'phase_train:0',
                 'embeddings': 'embeddings:0', 
                 }
        datasets = "datasets"
        names = stream_attendend(pb_path,node_dict,datasets)
        if names is not None: 
            self.Record.setText("Dected face " + str(names))  # show form       
            attendanceid =0
            available = False
            try:
                connection = sqlite3.connect("face-reco.db")
                cursor = connection.execute("SELECT MAX(attendanceid) from attendance")
                result = cursor.fetchall()# lấy tất cả hàng thoả mãn điều kiện truy vấn 
                if result:# nếu có rồi thêm tăng lên 1 hàng nữa vào sau cùng  , còn chưa có thì id của nó bắt đầu từ 1 
                    for maxid in result:
                        attendanceid = int(maxid[0])+1
            except:
                attendanceid=1
            print(attendanceid)    

            try:
                con = sqlite3.connect("face-reco.db")
                cursor = con.execute("SELECT * FROM attendance WHERE name='"+ str(names) +"' and attendancedate = '"+ str(date.today()) +"'")
                result = cursor.fetchall()
                if result:
                    available=True
                if(available==False):
                    con.execute("INSERT INTO attendance VALUES("+ str(attendanceid) +",'"+ str(names) +"','"+ str(date.today()) +"')")
                    con.commit()   
            except:
                print("Error in database insert")
            print("Attendance Registered successfully")
            self.Record.setText("Attence entered for " + names)            
            cnt=0
        else:
            cnt+=1# tăng lên 1 nếu khuôn mặt không đc nhận dạng đúng 
            # cv2.putText(im,'UnKnown',(x-10,y-10),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0))
            if(cnt>100):
                print("Unknown person")
                self.Record.setText("Unknown Person ")        
                # cv2.imwrite('unKnown.jpg',im)
                cnt=0
 ### SHOW SELECTED DATE REPORTS ###
    def show_selected_date_reports(self):
        self.REPORTTAB4.setRowCount(0)
        self.REPORTTAB4.clear()
        con = sqlite3.connect("face-reco.db")
        cursor = con.execute("SELECT * FROM attendance WHERE attendancedate = '"+ str((self.dateEdit.date()).toPyDate()) +"'")# lấy ngày ở dạng Qdate và chuyển sang obj datetime.date trong python = topydate
        result = cursor.fetchall()
        r=0
        c=0
        for row_number,row_data in enumerate(result):
            r+=1
            c=0
            for column_number,data in enumerate(row_data):
                c+=1
        self.REPORTTAB4.setColumnCount(c)

        for row_number,row_data in enumerate(result):
            self.REPORTTAB4.insertRow(row_number)
            for column_number, data in enumerate(row_data):
                self.REPORTTAB4.setItem(row_number,column_number,QTableWidgetItem(str(data)))

        self.REPORTTAB4.setHorizontalHeaderLabels(['Id','Name','Date'])        
        self.REPORTTAB4.setColumnWidth(0,10)
        self.REPORTTAB4.setColumnWidth(1,60)
        self.REPORTTAB4.setColumnWidth(2,70)
        self.REPORTTAB4.verticalHeader().setVisible(False)

def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    app.exec_()

if __name__ == '__main__':
    main()    