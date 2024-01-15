from os import path, listdir, makedirs
from PIL import Image
import numpy as np
import cv2
import av
import logging

from PySide6.QtCore import (QMetaObject, Qt)
from PySide6.QtWidgets import (QComboBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QProgressBar, QMessageBox,
    QPushButton, QFileDialog, QSizePolicy, QSpinBox, QVBoxLayout, QWidget, QApplication)
from PySide6.QtGui import QIcon

log = logging.getLogger()

class Export_Dialog(object):
    def setupUi(self, Dialog, cfg):
        self.cfg = cfg
        if not Dialog.objectName():
            Dialog.setObjectName(u"Dialog")
        Dialog.resize(400, 182)
        Dialog.setWindowIcon(QIcon('gui/cutie_r.ico'))
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Dialog.sizePolicy().hasHeightForWidth())
        Dialog.setSizePolicy(sizePolicy)
        Dialog.setWindowTitle(u"Export Video")
        self.verticalLayout = QVBoxLayout(Dialog)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.formLayout = QFormLayout()
        self.formLayout.setObjectName(u"formLayout")
        self.label_Type = QLabel(Dialog)
        self.label_Type.setObjectName(u"label_Type")
        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.label_Type.sizePolicy().hasHeightForWidth())
        self.label_Type.setSizePolicy(sizePolicy1)
        self.label_Type.setText(u"Export Type:")
        self.label_Type.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.formLayout.setWidget(0, QFormLayout.LabelRole, self.label_Type)

        self.comboBox_Type = QComboBox(Dialog)
        self.comboBox_Type.addItem(u"Matte")
        self.comboBox_Type.addItem(u"Composite on Alpha")
        self.comboBox_Type.addItem(u"Composite on Green Screen")        
        self.comboBox_Type.setObjectName(u"comboBox_Type")
        sizePolicy2 = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.comboBox_Type.sizePolicy().hasHeightForWidth())
        self.comboBox_Type.setSizePolicy(sizePolicy2)
        self.comboBox_Type.setCurrentText(u"Matte")

        self.formLayout.setWidget(0, QFormLayout.FieldRole, self.comboBox_Type)

        self.label_Codec = QLabel(Dialog)
        self.label_Codec.setObjectName(u"label_Codec")
        self.label_Codec.setText(u"Codec:")
        self.label_Codec.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.formLayout.setWidget(1, QFormLayout.LabelRole, self.label_Codec)

        self.comboBox_Codec = QComboBox(Dialog)
        self.comboBox_Codec.addItem(u"Prores")
        self.comboBox_Codec.addItem(u"FFV1")
        self.comboBox_Codec.addItem(u"x264")
        self.comboBox_Codec.addItem(u"x265")
        self.comboBox_Codec.setObjectName(u"comboBox_Codec")
        sizePolicy2.setHeightForWidth(self.comboBox_Codec.sizePolicy().hasHeightForWidth())
        self.comboBox_Codec.setSizePolicy(sizePolicy2)

        self.formLayout.setWidget(1, QFormLayout.FieldRole, self.comboBox_Codec)

        self.label_Quantizer = QLabel(Dialog)
        self.label_Quantizer.setObjectName(u"label_Quantizer")
        self.label_Quantizer.setText(u"Quantizer:")
        self.label_Quantizer.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.formLayout.setWidget(3, QFormLayout.LabelRole, self.label_Quantizer)

        self.spinBox_Quantizer = QSpinBox(Dialog)
        self.spinBox_Quantizer.setObjectName(u"spinBox_Quantizer")
        sizePolicy2.setHeightForWidth(self.spinBox_Quantizer.sizePolicy().hasHeightForWidth())
        self.spinBox_Quantizer.setSizePolicy(sizePolicy2)
        self.spinBox_Quantizer.setMaximum(51)
        self.spinBox_Quantizer.setValue(20)
        self.spinBox_Quantizer.setEnabled(False)

        self.formLayout.setWidget(3, QFormLayout.FieldRole, self.spinBox_Quantizer)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.lineEdit_OutputFolder = QLineEdit(Dialog)
        self.lineEdit_OutputFolder.setObjectName(u"lineEdit_OutputFolder")
        sizePolicy3 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.lineEdit_OutputFolder.sizePolicy().hasHeightForWidth())
        self.lineEdit_OutputFolder.setSizePolicy(sizePolicy3)
        self.lineEdit_OutputFolder.setText(u"")
        self.lineEdit_OutputFolder.setReadOnly(False)

        self.horizontalLayout.addWidget(self.lineEdit_OutputFolder)

        self.pushButton_OutputFolder = QPushButton(Dialog)
        self.pushButton_OutputFolder.setObjectName(u"pushButton_OutputFolder")
        sizePolicy2.setHeightForWidth(self.pushButton_OutputFolder.sizePolicy().hasHeightForWidth())
        self.pushButton_OutputFolder.setSizePolicy(sizePolicy2)
        self.pushButton_OutputFolder.setText(u"Select Folder")

        self.horizontalLayout.addWidget(self.pushButton_OutputFolder)


        self.formLayout.setLayout(4, QFormLayout.FieldRole, self.horizontalLayout)

        self.label_OutputFolder = QLabel(Dialog)
        self.label_OutputFolder.setObjectName(u"label_OutputFolder")
        self.label_OutputFolder.setText(u"Output Folder:")

        self.formLayout.setWidget(4, QFormLayout.LabelRole, self.label_OutputFolder)

        self.pushButton_Export = QPushButton(Dialog)
        self.pushButton_Export.setObjectName(u"pushButton_Export")
        sizePolicy.setHeightForWidth(self.pushButton_Export.sizePolicy().hasHeightForWidth())
        self.pushButton_Export.setSizePolicy(sizePolicy)
        self.pushButton_Export.setText(u"Export")

        self.formLayout.setWidget(6, QFormLayout.LabelRole, self.pushButton_Export)

        self.progressBar = QProgressBar(Dialog)
        self.progressBar.setObjectName(u"progressBar")
        self.progressBar.setValue(0)

        self.formLayout.setWidget(6, QFormLayout.FieldRole, self.progressBar)

        self.label_FPS = QLabel(Dialog)
        self.label_FPS.setObjectName(u"label_FPS")
        self.label_FPS.setText(u"FPS:")

        self.formLayout.setWidget(2, QFormLayout.LabelRole, self.label_FPS)

        self.comboBox_FPS = QComboBox(Dialog)
        self.comboBox_FPS.addItem(u"23.976")
        self.comboBox_FPS.addItem(u"24")
        self.comboBox_FPS.addItem(u"29.97")
        self.comboBox_FPS.addItem(u"30")
        self.comboBox_FPS.addItem(u"60")
        self.comboBox_FPS.setObjectName(u"comboBox_FPS")
        sizePolicy2.setHeightForWidth(self.comboBox_FPS.sizePolicy().hasHeightForWidth())
        self.comboBox_FPS.setSizePolicy(sizePolicy2)
        self.comboBox_FPS.setMinimumWidth(70)
        self.comboBox_FPS.setEditable(True)
        self.comboBox_FPS.setCurrentText(u"23.976")

        self.formLayout.setWidget(2, QFormLayout.FieldRole, self.comboBox_FPS)

        self.label_Filename = QLabel(Dialog)
        self.label_Filename.setObjectName(u"label_Filename")
        self.label_Filename.setText(u"Filename:")

        self.formLayout.setWidget(5, QFormLayout.LabelRole, self.label_Filename)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.lineEdit_Filename = QLineEdit(Dialog)
        self.lineEdit_Filename.setObjectName(u"lineEdit_Filename")

        self.horizontalLayout_2.addWidget(self.lineEdit_Filename)

        self.comboBox_Ext = QComboBox(Dialog)
        self.comboBox_Ext.addItem(u".mov")
        self.comboBox_Ext.addItem(u".mkv")
        self.comboBox_Ext.setObjectName(u"comboBox_Ext")
        sizePolicy2.setHeightForWidth(self.comboBox_Ext.sizePolicy().hasHeightForWidth())
        self.comboBox_Ext.setSizePolicy(sizePolicy2)

        self.horizontalLayout_2.addWidget(self.comboBox_Ext)


        self.formLayout.setLayout(5, QFormLayout.FieldRole, self.horizontalLayout_2)


        self.verticalLayout.addLayout(self.formLayout)

        QWidget.setTabOrder(self.comboBox_Type, self.comboBox_Codec)
        QWidget.setTabOrder(self.comboBox_Codec, self.comboBox_FPS)
        QWidget.setTabOrder(self.comboBox_FPS, self.spinBox_Quantizer)
        QWidget.setTabOrder(self.spinBox_Quantizer, self.lineEdit_OutputFolder)
        QWidget.setTabOrder(self.lineEdit_OutputFolder, self.pushButton_OutputFolder)
        QWidget.setTabOrder(self.pushButton_OutputFolder, self.lineEdit_Filename)
        QWidget.setTabOrder(self.lineEdit_Filename, self.comboBox_Ext)
        QWidget.setTabOrder(self.comboBox_Ext, self.pushButton_Export)

        #signals
        self.pushButton_OutputFolder.clicked.connect(self.on_selectOutputFolder)
        self.pushButton_Export.clicked.connect(self.on_exportButtonPressed)
        self.comboBox_Type.currentIndexChanged.connect(self.updateType)
        self.comboBox_Codec.currentIndexChanged.connect(self.updateCodec)

        #set initial filename
        self.lineEdit_OutputFolder.setText(path.abspath(cfg['workspace']))
        self.lineEdit_Filename.setText(path.basename(cfg['workspace']))

        #check if masks match images
        if not self.file_counts_match():
            QMessageBox.warning(None, "File Count Mismatch", "The number of masks does not match the number of images. Please make sure you propogated masks to all frames, and that no extra files have been copied to the workspace.")

        self.exporting = False
        QMetaObject.connectSlotsByName(Dialog)


    def on_selectOutputFolder(self):
        folder_name = QFileDialog.getExistingDirectory(None,
            "Output Folder",
            path.dirname(self.lineEdit_OutputFolder.text()) #starting directory
        )
        if folder_name:
            self.lineEdit_OutputFolder.setText(path.abspath(folder_name))

    def on_exportButtonPressed(self):
        if self.exporting:
            self.cancel_exort()
            return

        output_folder = self.lineEdit_OutputFolder.text()
        filename = self.lineEdit_Filename.text() + self.comboBox_Ext.currentText()
        file_path = path.join(output_folder, filename)

        if path.exists(file_path):
            reply = QMessageBox.question(None, "File Exists", "The file already exists. Do you want to overwrite it?", QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No:
                return
        
        self.exporting = True
        self.pushButton_Export.setText("Cancel")
        makedirs(path.join(self.cfg['workspace'], 'binary_masks'), exist_ok=True)
        self.progressBar.setFormat('Generating Binary Masks... %p%')
        self.convert_mask_to_binary(path.join(self.cfg['workspace'], 'masks'), output_folder, self.progressbar_update)
        
        if self.comboBox_Type.currentText() == "Matte":
            image_folder = path.join(self.cfg['workspace'], 'binary_masks')
        else:
            self.progressBar.setFormat('Generating Composite Images... %p%')
            image_folder = path.join(self.cfg['workspace'], 'images')
            mask_folder = path.join(self.cfg['workspace'], 'binary_masks')
            self.generate_composite_images(image_folder, mask_folder, output_folder, self.progressbar_update)
            image_folder = path.join(self.cfg['workspace'], 'composite')
            
        self.progressBar.setFormat('Exporting Video... %p%')
        self.convert_frames_to_video(image_folder, file_path, self.comboBox_FPS.currentText(), self.spinBox_Quantizer.value(), self.progressbar_update)
        self.progressBar.setFormat('%p%')
        if self.exporting:
            QMessageBox.information(None, "Done", "Exported to " + file_path)
        else:
            QMessageBox.warning(None, "Export Cancelled", "Export cancelled.")
        self.exporting = False
    
    def updateType(self):
        if self.comboBox_Type.currentText() == "Composite on Alpha":
            self.comboBox_Codec.clear()
            self.comboBox_Codec.addItems(["Prores", "FFV1"])
        else:
            self.comboBox_Codec.clear()
            self.comboBox_Codec.addItems(["Prores", "FFV1", "x264", "x265"])

    def updateCodec(self):
        self.comboBox_Ext.clear()
        if self.comboBox_Codec.currentText() == "FFV1":
            self.comboBox_Ext.addItems([".mkv", ".avi"])
            self.spinBox_Quantizer.setEnabled(False)
        elif self.comboBox_Codec.currentText() == "Prores":
            self.comboBox_Ext.addItems([".mov", ".mkv"])
            self.spinBox_Quantizer.setEnabled(False)
        elif self.comboBox_Codec.currentText() == "x264" or self.comboBox_Codec.currentText() == "x265":
            self.comboBox_Ext.addItems([".mp4", ".mov", ".mkv"])
            self.spinBox_Quantizer.setEnabled(True)
        else:
            self.comboBox_Ext.addItems([".avi", ".mp4", ".mov", ".mkv"])
            self.spinBox_Quantizer.setEnabled(False)
        

    def convert_frames_to_video(self,
            image_folder: str,
            output_path: str,
            fps: int = 24,
            crf: int = 20,
            progress_callback=None) -> None:
        images = [img for img in sorted(listdir(image_folder)) if (img.endswith(".jpg") or img.endswith(".png"))]
        frame = cv2.imread(path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        output = av.open(output_path, mode="w")

        if self.comboBox_Codec.currentText() == "x264":
            stream = output.add_stream("h264", rate=fps, pix_fmt='yuv420p')
            stream.options = {'crf':str(crf)}
        elif self.comboBox_Codec.currentText() == "x265":
            stream = output.add_stream("hevc", rate=fps, pix_fmt='yuv420p')
            stream.options = {'crf':str(crf)}
        elif self.comboBox_Codec.currentText() == "FFV1":
            stream = output.add_stream("ffv1", rate=fps, pix_fmt='bgra')
        elif self.comboBox_Codec.currentText() == "Prores":
            stream = output.add_stream("prores_ks", rate=fps, pix_fmt='yuva444p10le')
            stream.options = {'profile':'4'}
        else:
            raise ValueError("Invalid codec")

        stream.width = width
        stream.height = height

        for i, img_path in enumerate(images):
            if self.exporting is False:
                return

            if self.comboBox_Type.currentText() == "Composite on Alpha":
                img = cv2.imread(path.join(image_folder, img_path), cv2.IMREAD_UNCHANGED)
                frame = av.VideoFrame.from_ndarray(img, format='bgra')
            else:
                img = cv2.imread(path.join(image_folder, img_path))
                frame = av.VideoFrame.from_ndarray(img, format='bgr24')

            packet = stream.encode(frame)
            output.mux(packet)

            if progress_callback is not None and i % 10 == 0:
                progress_callback(i / len(images))

        # flush
        packet = stream.encode(None)
        output.mux(packet)
        self.progressbar_update(1.0)
        output.close()
    
    def convert_mask_to_binary(self, mask_folder: str, output_path: str, progress_callback=None) -> None:
        binary_mask_path = path.join(output_path, 'binary_masks')
        masks = [img for img in sorted(listdir(mask_folder)) if img.endswith(".png")]
        for i, mask_path in enumerate(masks):
            if self.exporting is False:
                return

            mask = Image.open(path.join(mask_folder, mask_path))
            mask = np.array(mask)
            mask = np.where(np.isin(mask, 1), 255, 0)
            
            cv2.imwrite(path.join(binary_mask_path, mask_path), mask)

            if progress_callback is not None and i % 10 == 0:
                progress_callback(i / len(masks))
        self.progressbar_update(1.0)

    def generate_composite_images(self, image_folder: str, mask_folder: str, output_path: str, progress_callback=None) -> None:
            images = [img for img in sorted(listdir(image_folder)) if (img.endswith(".jpg") or img.endswith(".png"))]
            masks = [img for img in sorted(listdir(mask_folder)) if img.endswith(".png")]
            images = [img for img in sorted(listdir(image_folder)) if (img.endswith(".jpg") or img.endswith(".png"))]
            frame = cv2.imread(path.join(image_folder, images[0])) #read 1 frame to get height and width
            height, width, layers = frame.shape
            makedirs(path.join(self.cfg['workspace'], 'composite'), exist_ok=True)
            for image, mask in zip(images, masks):
                if self.exporting is False:
                    return
                
                image_path = path.join(image_folder, image)
                mask_path = path.join(mask_folder, mask)
                img = cv2.imread(image_path)
                msk = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if self.comboBox_Type.currentText() == "Composite on Alpha":
                    img_with_alpha = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                    img_with_alpha[:, :, 3] = msk
                    msk = cv2.cvtColor(msk, cv2.COLOR_GRAY2BGRA)
                    comp = img_with_alpha * (msk/255)
                elif self.comboBox_Type.currentText() == "Composite on Green Screen": 
                    msk = cv2.cvtColor(msk, cv2.COLOR_GRAY2BGR)
                    comp = img * (msk/255)
                    green = np.zeros_like(img)
                    green[:, :] = [0, 255, 0]
                    comp = comp + (green * (cv2.bitwise_not(msk) / 255))
                cv2.imwrite(path.join(self.cfg['workspace'], 'composite', mask), comp)
                self.progressbar_update(images.index(image) / len(images))
            

    def file_counts_match(self):
        image_folder = path.join(self.cfg['workspace'], 'images')
        mask_folder = path.join(self.cfg['workspace'], 'masks')
    
        image_files = [f for f in listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')]
        mask_files = [f for f in listdir(mask_folder) if f.endswith('.png')]
    
        if len(image_files) == len(mask_files):
            return True
        else:
            return False

    def progressbar_update(self, progress: float):
        QApplication.processEvents() 
        self.progressBar.setValue(int(progress * 100))

    def cancel_exort(self):
        reply = QMessageBox.question(None, "Cancel Export", "Do you really want to cancel?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.No:
            return
        else:
            self.exporting = False
            self.pushButton_Export.setText("Export")
