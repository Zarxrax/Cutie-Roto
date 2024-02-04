from os import path
from PySide6.QtCore import (QMetaObject, QRect, QSize, Qt)

from PySide6.QtWidgets import (QHBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog, QSizePolicy, QVBoxLayout, QWidget, QMessageBox)

from PySide6.QtGui import QIcon
import av
#import logging
#from omegaconf import open_dict
from hydra import compose
from showinfm import show_in_file_manager

class Launcher_Dialog(object):
    def setupUi(self, Dialog):
        if not Dialog.objectName():
            Dialog.setObjectName(u"Dialog")
        Dialog.setWindowModality(Qt.ApplicationModal)
        Dialog.resize(540, 150)
        Dialog.setWindowTitle("Cutie Roto Launcher")
        Dialog.setWindowIcon(QIcon('gui/cutie_y.ico'))
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Dialog.sizePolicy().hasHeightForWidth())
        Dialog.setSizePolicy(sizePolicy)
        self.layoutWidget = QWidget(Dialog)
        self.layoutWidget.setObjectName(u"layoutWidget")
        self.layoutWidget.setGeometry(QRect(30, 10, 481, 121))
        self.verticalLayout = QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_1 = QHBoxLayout()
        self.horizontalLayout_1.setObjectName(u"horizontalLayout_1")
        self.label_video = QLabel(self.layoutWidget)
        self.label_video.setObjectName(u"label_video")
        self.label_video.setText(u"Video:")

        self.horizontalLayout_1.addWidget(self.label_video)

        self.lineEdit_videofile = QLineEdit(self.layoutWidget)
        self.lineEdit_videofile.setObjectName(u"lineEdit_videofile")
        sizePolicy1 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.lineEdit_videofile.sizePolicy().hasHeightForWidth())
        self.lineEdit_videofile.setSizePolicy(sizePolicy1)
        self.lineEdit_videofile.setText(u"")

        self.horizontalLayout_1.addWidget(self.lineEdit_videofile)

        self.pushButton_videofile = QPushButton(self.layoutWidget)
        self.pushButton_videofile.setObjectName(u"pushButton_videofile")
        sizePolicy2 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.pushButton_videofile.sizePolicy().hasHeightForWidth())
        self.pushButton_videofile.setSizePolicy(sizePolicy2)
        self.pushButton_videofile.setMinimumSize(QSize(75, 0))
        self.pushButton_videofile.setText(u"Select File")

        self.horizontalLayout_1.addWidget(self.pushButton_videofile)


        self.verticalLayout.addLayout(self.horizontalLayout_1)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.label_workspace_folder = QLabel(self.layoutWidget)
        self.label_workspace_folder.setObjectName(u"label_workspace_folder")
        self.label_workspace_folder.setText(u"Workspace folder:")

        self.horizontalLayout_2.addWidget(self.label_workspace_folder)

        self.lineEdit_workspace_folder = QLineEdit(self.layoutWidget)
        self.lineEdit_workspace_folder.setObjectName(u"lineEdit_workspace_folder")
        self.lineEdit_workspace_folder.setEnabled(True)
        sizePolicy1.setHeightForWidth(self.lineEdit_workspace_folder.sizePolicy().hasHeightForWidth())
        self.lineEdit_workspace_folder.setSizePolicy(sizePolicy1)
        self.lineEdit_workspace_folder.setText(u"./Workspace/example")
        self.lineEdit_workspace_folder.setReadOnly(True)

        self.horizontalLayout_2.addWidget(self.lineEdit_workspace_folder)


        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.label_workspace_status = QLabel(self.layoutWidget)
        self.label_workspace_status.setObjectName(u"label_workspace_status")
        sizePolicy3 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.label_workspace_status.sizePolicy().hasHeightForWidth())
        self.label_workspace_status.setSizePolicy(sizePolicy3)
        self.label_workspace_status.setText(u"Workspace status:")

        self.horizontalLayout_3.addWidget(self.label_workspace_status)

        self.label_workspace_status_text = QLabel(self.layoutWidget)
        self.label_workspace_status_text.setObjectName(u"label_workspace_status_text")
        sizePolicy4 = QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.label_workspace_status_text.sizePolicy().hasHeightForWidth())
        self.label_workspace_status_text.setSizePolicy(sizePolicy4)
        self.label_workspace_status_text.setText(u"Please select a video file")

        self.horizontalLayout_3.addWidget(self.label_workspace_status_text)


        self.verticalLayout.addLayout(self.horizontalLayout_3)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.pushButton_workspaces_folder = QPushButton(self.layoutWidget)
        self.pushButton_workspaces_folder.setObjectName(u"pushButton_workspaces_folder")
        self.pushButton_workspaces_folder.setText(u"Open Workspaces Folder")

        self.horizontalLayout_4.addWidget(self.pushButton_workspaces_folder)

        self.pushButton_start = QPushButton(self.layoutWidget)
        self.pushButton_start.setObjectName(u"pushButton_start")
        self.pushButton_start.setText(u"Start")

        self.horizontalLayout_4.addWidget(self.pushButton_start)


        self.verticalLayout.addLayout(self.horizontalLayout_4)

        QWidget.setTabOrder(self.lineEdit_videofile, self.pushButton_videofile)
        QWidget.setTabOrder(self.pushButton_videofile, self.lineEdit_workspace_folder)
        QWidget.setTabOrder(self.lineEdit_workspace_folder, self.pushButton_workspaces_folder)
        QWidget.setTabOrder(self.pushButton_workspaces_folder, self.pushButton_start)
        
        #signals
        self.pushButton_videofile.clicked.connect(self.on_selectVideo)
        self.pushButton_workspaces_folder.clicked.connect(self.on_openWorkspacesFolder)
        self.pushButton_start.clicked.connect(self.on_clickStart)
        self.lineEdit_videofile.editingFinished.connect(self.on_videoFileChanged)
        self.lineEdit_workspace_folder.textChanged.connect(self.on_workspaceChanged)
        
        # read data from the config
        self.cfg = compose(config_name="gui_config")
        
        #set initial filename from the file last.txt
        try:
            if not path.exists('cutie/config/last.txt'):
                with open('cutie/config/last.txt', 'w') as file:
                    file.write("./examples/example.mp4")
            with open('cutie/config/last.txt', 'r') as file:
                line = file.readline()
                self.lineEdit_videofile.setText(path.abspath(line))
        except FileNotFoundError:
            print('Error detecting last opened file. last.txt not found.')
        except Exception as e:
            print('Error detecting last opened file: ', e)
            
        self.on_videoFileChanged()
        QMetaObject.connectSlotsByName(Dialog)



    def on_selectVideo(self):
        file_name, _ = QFileDialog.getOpenFileName(self,
                                                   "Select Video",
                                                   path.dirname(self.lineEdit_videofile.text()), #starting directory
                                                   "Video Files (*.mp4 *.mkv *.mov *.avi);;All Files (*)"
                                                   )
        if file_name:
            self.lineEdit_videofile.setText(path.abspath(file_name))
        self.on_videoFileChanged()

    def on_videoFileChanged(self):
        if path.isfile(self.lineEdit_videofile.text()):
            basename = path.basename(self.lineEdit_videofile.text())[:-4]
            self.lineEdit_workspace_folder.setText(path.abspath(path.join(self.cfg['workspace_root'], basename)))
        else:
            self.lineEdit_workspace_folder.setText("")
            self.label_workspace_status_text.setText("Video file not found. Please enter a valid filename.")
    
    def on_workspaceChanged(self):
        if path.exists(self.lineEdit_workspace_folder.text()):
            self.label_workspace_status_text.setText("Workspace exists from a previous session.")
        else:
            self.label_workspace_status_text.setText("The workspace does not exist. A new workspace will be created.")

    def on_openWorkspacesFolder(self):
        show_in_file_manager(self.cfg['workspace_root'])
    
    def on_clickStart(self):
        if self.label_workspace_status_text.text() == "Video file not found. Please enter a valid filename.":
            return

        #store filename for resuming next time
        try:
            with open('cutie/config/last.txt', 'w') as file:
                file.write(self.lineEdit_videofile.text())
        except Exception as e:
            print(f"Error: {e}")

        #check for long video
        with av.open(self.lineEdit_videofile.text()) as container:
            total_frames = container.streams.video[0].frames
            if total_frames > 5000 and self.label_workspace_status_text.text() == "The workspace does not exist. A new workspace will be created.":
                reply = QMessageBox.question(self, 'Warning', "You have selected a long video. It is recommended to trim the video before importing. Are you sure you want to continue?", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                if reply == QMessageBox.No:
                    return
        
        #Close the dialog and return the filename
        self.accept()  
        self.result = self.lineEdit_videofile.text()
    

    