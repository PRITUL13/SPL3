import sys
import ntpath

import cv2
import numpy as np

from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *

from functools import partial

from img_modifier import img_helper
from img_modifier import color_filter

from PIL import ImageQt

from logging.config import fileConfig
import logging


import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import os
import numpy as np

import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

logger = logging.getLogger()

# original img, can't be modified
_img_original = None
_img_preview = None
_img_lbl_glob=None
# constants
THUMB_BORDER_COLOR_ACTIVE = "#3893F4"
THUMB_BORDER_COLOR = "#ccc"
BTN_MIN_WIDTH = 120
ROTATION_BTN_SIZE = (70, 30)
THUMB_SIZE = 120

SLIDER_MIN_VAL = -100
SLIDER_MAX_VAL = 100
SLIDER_DEF_VAL = 0

_file_name_global=None

_emotion_model_glob=None
class Operations:
    def __init__(self):
        self.color_filter = None

        self.flip_left = False
        self.flip_top = False
        self.rotation_angle = 0

        self.size = None

        self.brightness = 0
        self.sharpness = 0
        self.contrast = 0

    def reset(self):
        self.color_filter = None

        self.brightness = 0
        self.sharpness = 0
        self.contrast = 0

        self.size = None

        self.flip_left = False
        self.flip_top = False
        self.rotation_angle = 0

    def has_changes(self):
        return self.color_filter or self.flip_left\
                or self.flip_top or self.rotation_angle\
                or self.contrast or self.brightness\
                or self.sharpness or self.size

    def __str__(self):
        return f"size: {self.size}, filter: {self.color_filter}, " \
               f"b: {self.brightness} c: {self.contrast} s: {self.sharpness}, " \
               f"flip-left: {self.flip_left} flip-top: {self.flip_top} rotation: {self.rotation_angle}"


operations = Operations()


def _get_ratio_height(width, height, r_width):
    return int(r_width/width*height)


def _get_ratio_width(width, height, r_height):
    return int(r_height/height*width)


def _get_converted_point(user_p1, user_p2, p1, p2, x):
    """
    convert user ui slider selected value (x) to PIL value
    user ui slider scale is -100 to 100, PIL scale is -1 to 2
    example:
     - user selected 50
     - PIL value is 1.25
    """

    r = (x - user_p1) / (user_p2 - user_p1)
    return p1 + r * (p2 - p1)


def _get_img_with_all_operations():
    logger.debug(operations)

    b = operations.brightness
    c = operations.contrast
    s = operations.sharpness

    img = _img_preview
    if b != 0:
        img = img_helper.brightness(img, b)

    if c != 0:
        img = img_helper.contrast(img, c)

    if s != 0:
        img = img_helper.sharpness(img, s)

    if operations.rotation_angle:
        img = img_helper.rotate(img, operations.rotation_angle)

    if operations.flip_left:
        img = img_helper.flip_left(img)

    if operations.flip_top:
        img = img_helper.flip_top(img)

    if operations.size:
        img = img_helper.resize(img, *operations.size)

    return img





class ActionTabs(QTabWidget):
    """Action tabs widget"""

    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        self.filters_tab = FiltersTab(self)
        self.adjustment_tab = AdjustingTab(self)
        self.modification_tab = ModificationTab(self)
        self.rotation_tab = RotationTab(self)
        self.cartoon_tab = CartoonTab(self)
        self.emoji_tab = EmojiTab(self)

        self.addTab(self.filters_tab, "Filters")
        self.addTab(self.adjustment_tab, "Adjusting")
        self.addTab(self.modification_tab, "Modification")
        self.addTab(self.rotation_tab, "Rotation")
        self.addTab(self.cartoon_tab, "Caroonify")
        self.addTab(self.emoji_tab, "Emojify")

        self.setMaximumHeight(190)

class EmojiTab(QWidget):

    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        emojify_btn = QPushButton("Emojify")
        emojify_btn.setMinimumSize(*ROTATION_BTN_SIZE)
        emojify_btn.clicked.connect(self.on_emojify)

        btn_layout = QHBoxLayout()
        btn_layout.setAlignment(Qt.AlignCenter)
        btn_layout.addWidget(emojify_btn)

        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignCenter)
        main_layout.addLayout(btn_layout)

        self.setLayout(main_layout)

    def show_vid(self,show_text):    
        frame1 = cv2.imread(_file_name_global)
        frame1 = cv2.resize(frame1,(600,800))

        bounding_box = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        num_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)
        emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


        emoji_dist={0:"./emojis/angry.png",2:"./emojis/disgusted.png",2:"./emojis/fearful.png",3:"./emojis/happy.png",4:"./emojis/neutral.png",5:"./emojis/sad.png",6:"./emojis/surprised.png"}

        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame1, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            prediction = _emotion_model_glob.predict(cropped_img)
            
            maxindex = int(np.argmax(prediction))
            # cv2.putText(frame1, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            show_text[0]=maxindex
        
        global last_frame1
        last_frame1 = frame1.copy()
        pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)     
        # cv2.imshow("pritul ",pic)
        # img = Image.fromarray(pic)
        # imgtk = ImageTk.PhotoImage(image=img)
        # lmain.imgtk = imgtk
        # lmain.configure(image=imgtk)
        # lmain.after(10, show_vid)
        print("show_text ", show_text[0], emoji_dist[show_text[0]])

        frame2=cv2.imread(emoji_dist[show_text[0]])
        pic2=cv2.cvtColor(frame2,cv2.COLOR_BGR2RGB)
        # cv2.imshow("anika ",pic2)

        from PIL import Image
        img2 = Image.open(emoji_dist[show_text[0]])
        preview_pix = ImageQt.toqpixmap(img2)
        # logger.debug("amar matha",_file_name_global)
        logger.debug(preview_pix)

        _img_lbl_glob.setScaledContents(True)
        _img_lbl_glob.setPixmap(preview_pix)

    def on_emojify(self):


        emotion_model = Sequential()

        emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
        emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
        emotion_model.add(Dropout(0.25))

        emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
        emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
        emotion_model.add(Dropout(0.25))

        emotion_model.add(Flatten())
        emotion_model.add(Dense(1024, activation='relu'))
        emotion_model.add(Dropout(0.5))
        emotion_model.add(Dense(7, activation='softmax'))
        emotion_model.load_weights('emotion_model.h5')
        global _emotion_model_glob
        _emotion_model_glob=emotion_model

        cv2.ocl.setUseOpenCL(False)


        global last_frame1                                    
        last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        global cap1
        show_text=[0]
        self.show_vid(show_text)
        logger.debug("emojified")

class CartoonTab(QWidget):

    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        cartoonify_btn = QPushButton("Cartoonify")
        cartoonify_btn.setMinimumSize(*ROTATION_BTN_SIZE)
        cartoonify_btn.clicked.connect(self.on_cartoonify)

        btn_layout = QHBoxLayout()
        btn_layout.setAlignment(Qt.AlignCenter)
        btn_layout.addWidget(cartoonify_btn)

        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignCenter)
        main_layout.addLayout(btn_layout)

        self.setLayout(main_layout)
    def read_file(self,filename):
        img = cv2.imread(filename)
        # originalmage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ReSized1 = cv2.resize(img, (960, 540))
        # cv2.imshow("first Image",ReSized1)
        return ReSized1

    def color_quantization(self,img, k):

        data = np.float32(img).reshape((-1, 3))


        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)


        ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        result = center[label.flatten()]
        result = result.reshape(img.shape)
        return result

    def edge_mask(self,img, line_size, blur_value):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.medianBlur(gray, blur_value)
        edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)
        return edges


    def on_cartoonify(self):
        global  _file_name_global
        img = self.read_file(_file_name_global)
        #logger.debug("asdfdsdjjdgnsgb "+_file_name_global)
        import os
        logger.debug(os.getcwd())
        root_path=os.getcwd()
        
        line_size = 7
        blur_value = 7

        edges = self.edge_mask(img, line_size, blur_value)
        # cv2.imshow("sketch",edges)
        total_color = 9

        img = self.color_quantization(img, total_color)
        # cv2.imshow("grained image",img)
        blurred = cv2.bilateralFilter(img, d=7, sigmaColor=200,sigmaSpace=200)
        # cv2.imshow("blurred image",blurred)
        cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)

        # cv2.imwrite("cartooned_"+_file_name_global,cartoon)
        # cv2.imshow("cartooned",cartoon)
        cv2.imwrite(root_path+"\\cartooned.jpg",img)
        cv2.waitKey(0)

        #cv2.imwrite("C:\\Users\\Pritul\\Desktop\\emoji-creator-project-code\\"+"cartooned.jpg",img)
        #cv2.waitKey(0)

        from PIL import Image
        #img2 = Image.open("C:\\Users\\Pritul\\Desktop\\emoji-creator-project-code\\"+"cartooned.jpg")
        img2 = Image.open(root_path+"\\cartooned.jpg")
        # pix = QPixmap("C:\\Users\\Pritul\\Desktop\\sp3_918\\"+"cartooned.jpg")
        preview_pix = ImageQt.toqpixmap(img2)
        # logger.debug("amar matha",_file_name_global)
        logger.debug(preview_pix)

        _img_lbl_glob.setScaledContents(True)
        _img_lbl_glob.setPixmap(preview_pix)



    logger.debug("cartoonified")




class RotationTab(QWidget):
    """Rotation tab widget"""

    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        rotate_left_btn = QPushButton("↺ 90°")
        rotate_left_btn.setMinimumSize(*ROTATION_BTN_SIZE)
        rotate_left_btn.clicked.connect(self.on_rotate_left)

        rotate_right_btn = QPushButton("↻ 90°")
        rotate_right_btn.setMinimumSize(*ROTATION_BTN_SIZE)
        rotate_right_btn.clicked.connect(self.on_rotate_right)

        flip_left_btn = QPushButton("⇆")
        flip_left_btn.setMinimumSize(*ROTATION_BTN_SIZE)
        flip_left_btn.clicked.connect(self.on_flip_left)

        flip_top_btn = QPushButton("↑↓")
        flip_top_btn.setMinimumSize(*ROTATION_BTN_SIZE)
        flip_top_btn.clicked.connect(self.on_flip_top)

        rotate_lbl = QLabel("Rotate")
        rotate_lbl.setAlignment(Qt.AlignCenter)
        rotate_lbl.setFixedWidth(140)

        flip_lbl = QLabel("Flip")
        flip_lbl.setAlignment(Qt.AlignCenter)
        flip_lbl.setFixedWidth(140)

        lbl_layout = QHBoxLayout()
        lbl_layout.setAlignment(Qt.AlignCenter)
        lbl_layout.addWidget(rotate_lbl)
        lbl_layout.addWidget(flip_lbl)

        btn_layout = QHBoxLayout()
        btn_layout.setAlignment(Qt.AlignCenter)
        btn_layout.addWidget(rotate_left_btn)
        btn_layout.addWidget(rotate_right_btn)
        btn_layout.addWidget(QVLine())
        btn_layout.addWidget(flip_left_btn)
        btn_layout.addWidget(flip_top_btn)

        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignCenter)
        main_layout.addLayout(lbl_layout)
        main_layout.addLayout(btn_layout)

        self.setLayout(main_layout)

    def on_rotate_left(self):
        logger.debug("rotate left")

        operations.rotation_angle = 0 if operations.rotation_angle == 270 else operations.rotation_angle + 90
        self.parent.parent.place_preview_img()

    def on_rotate_right(self):
        logger.debug("rotate left")

        operations.rotation_angle = 0 if operations.rotation_angle == -270 else operations.rotation_angle - 90
        self.parent.parent.place_preview_img()

    def on_flip_left(self):
        logger.debug("flip left-right")

        operations.flip_left = not operations.flip_left
        self.parent.parent.place_preview_img()

    def on_flip_top(self):
        logger.debug("flip top-bottom")

        operations.flip_top = not operations.flip_top
        self.parent.parent.place_preview_img()


class ModificationTab(QWidget):
    """Modification tab widget"""

    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        self.width_lbl = QLabel('width:', self)
        self.width_lbl.setFixedWidth(100)

        self.height_lbl = QLabel('height:', self)
        self.height_lbl.setFixedWidth(100)

        self.width_box = QLineEdit(self)
        self.width_box.textEdited.connect(self.on_width_change)
        self.width_box.setMaximumWidth(100)

        self.height_box = QLineEdit(self)
        self.height_box.textEdited.connect(self.on_height_change)
        self.height_box.setMaximumWidth(100)

        self.unit_lbl = QLabel("px")
        self.unit_lbl.setMaximumWidth(50)

        self.ratio_check = QCheckBox('aspect ratio', self)
        self.ratio_check.stateChanged.connect(self.on_ratio_change)

        self.apply_btn = QPushButton("Apply")
        self.apply_btn.setFixedWidth(90)
        self.apply_btn.clicked.connect(self.on_apply)

        width_layout = QHBoxLayout()
        width_layout.addWidget(self.width_box)
        width_layout.addWidget(self.height_box)
        width_layout.addWidget(self.unit_lbl)

        apply_layout = QHBoxLayout()
        apply_layout.addWidget(self.apply_btn)
        apply_layout.setAlignment(Qt.AlignRight)

        lbl_layout = QHBoxLayout()
        lbl_layout.setAlignment(Qt.AlignLeft)
        lbl_layout.addWidget(self.width_lbl)
        lbl_layout.addWidget(self.height_lbl)

        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignCenter)

        main_layout.addLayout(lbl_layout)
        main_layout.addLayout(width_layout)
        main_layout.addWidget(self.ratio_check)
        main_layout.addLayout(apply_layout)

        self.setLayout(main_layout)

    def set_boxes(self):
        self.width_box.setText(str(_img_original.width))
        self.height_box.setText(str(_img_original.height))

    def on_width_change(self, e):
        logger.debug(f"type width {self.width_box.text()}")

        if self.ratio_check.isChecked():
            r_height = _get_ratio_height(_img_original.width, _img_original.height, int(self.width_box.text()))
            self.height_box.setText(str(r_height))

    def on_height_change(self, e):
        logger.debug(f"type height {self.height_box.text()}")

        if self.ratio_check.isChecked():
            r_width = _get_ratio_width(_img_original.width, _img_original.height, int(self.height_box.text()))
            self.width_box.setText(str(r_width))

    def on_ratio_change(self, e):
        logger.debug("ratio change")

    def on_apply(self, e):
        logger.debug("apply")

        operations.size = int(self.width_box.text()), int(self.height_box.text())

        self.parent.parent.update_img_size_lbl()
        self.parent.parent.place_preview_img()


class AdjustingTab(QWidget):
    """Adjusting tab widget"""

    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        contrast_lbl = QLabel("Contrast")
        contrast_lbl.setAlignment(Qt.AlignCenter)

        brightness_lbl = QLabel("Brightness")
        brightness_lbl.setAlignment(Qt.AlignCenter)

        sharpness_lbl = QLabel("Sharpness")
        sharpness_lbl.setAlignment(Qt.AlignCenter)

        self.contrast_slider = QSlider(Qt.Horizontal, self)
        self.contrast_slider.setMinimum(SLIDER_MIN_VAL)
        self.contrast_slider.setMaximum(SLIDER_MAX_VAL)
        self.contrast_slider.sliderReleased.connect(self.on_contrast_slider_released)
        self.contrast_slider.setToolTip(str(SLIDER_MAX_VAL))

        self.brightness_slider = QSlider(Qt.Horizontal, self)
        self.brightness_slider.setMinimum(SLIDER_MIN_VAL)
        self.brightness_slider.setMaximum(SLIDER_MAX_VAL)
        self.brightness_slider.sliderReleased.connect(self.on_brightness_slider_released)
        self.brightness_slider.setToolTip(str(SLIDER_MAX_VAL))

        self.sharpness_slider = QSlider(Qt.Horizontal, self)
        self.sharpness_slider.setMinimum(SLIDER_MIN_VAL)
        self.sharpness_slider.setMaximum(SLIDER_MAX_VAL)
        self.sharpness_slider.sliderReleased.connect(self.on_sharpness_slider_released)
        self.sharpness_slider.setToolTip(str(SLIDER_MAX_VAL))

        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignCenter)

        main_layout.addWidget(contrast_lbl)
        main_layout.addWidget(self.contrast_slider)

        main_layout.addWidget(brightness_lbl)
        main_layout.addWidget(self.brightness_slider)

        main_layout.addWidget(sharpness_lbl)
        main_layout.addWidget(self.sharpness_slider)

        self.reset_sliders()
        self.setLayout(main_layout)

    def reset_sliders(self):
        self.brightness_slider.setValue(SLIDER_DEF_VAL)
        self.sharpness_slider.setValue(SLIDER_DEF_VAL)
        self.contrast_slider.setValue(SLIDER_DEF_VAL)

    def on_brightness_slider_released(self):
        logger.debug(f"brightness selected value: {self.brightness_slider.value()}")

        self.brightness_slider.setToolTip(str(self.brightness_slider.value()))

        factor = _get_converted_point(SLIDER_MIN_VAL, SLIDER_MAX_VAL, img_helper.BRIGHTNESS_FACTOR_MIN,
                                      img_helper.BRIGHTNESS_FACTOR_MAX, self.brightness_slider.value())
        logger.debug(f"brightness factor: {factor}")

        operations.brightness = factor

        self.parent.parent.place_preview_img()

    def on_sharpness_slider_released(self):
        logger.debug(self.sharpness_slider.value())

        self.sharpness_slider.setToolTip(str(self.sharpness_slider.value()))

        factor = _get_converted_point(SLIDER_MIN_VAL, SLIDER_MAX_VAL, img_helper.SHARPNESS_FACTOR_MIN,
                                      img_helper.SHARPNESS_FACTOR_MAX, self.sharpness_slider.value())
        logger.debug(f"sharpness factor: {factor}")

        operations.sharpness = factor

        self.parent.parent.place_preview_img()

    def on_contrast_slider_released(self):
        logger.debug(self.contrast_slider.value())

        self.contrast_slider.setToolTip(str(self.contrast_slider.value()))

        factor = _get_converted_point(SLIDER_MIN_VAL, SLIDER_MAX_VAL, img_helper.CONTRAST_FACTOR_MIN,
                                      img_helper.CONTRAST_FACTOR_MAX, self.contrast_slider.value())
        logger.debug(f"contrast factor: {factor}")

        operations.contrast = factor

        self.parent.parent.place_preview_img()


class FiltersTab(QWidget):
    """Color filters widget"""

    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        self.main_layout = QHBoxLayout()
        self.main_layout.setAlignment(Qt.AlignCenter)

        self.add_filter_thumb("none")
        for key, val in color_filter.ColorFilters.filters.items():
            self.add_filter_thumb(key, val)

        self.setLayout(self.main_layout)

    def add_filter_thumb(self, name, title=""):
        logger.debug(f"create lbl thumb for: {name}")

        thumb_lbl = QLabel()
        thumb_lbl.name = name
        thumb_lbl.setStyleSheet("border:2px solid #ccc;")

        if name != "none":
            thumb_lbl.setToolTip(f"Apply <b>{title}</b> filter")
        else:
            thumb_lbl.setToolTip('No filter')

        thumb_lbl.setCursor(Qt.PointingHandCursor)
        thumb_lbl.setFixedSize(THUMB_SIZE, THUMB_SIZE)
        thumb_lbl.mousePressEvent = partial(self.on_filter_select, name)

        self.main_layout.addWidget(thumb_lbl)

    def on_filter_select(self, filter_name, e):
        logger.debug(f"apply color filter: {filter_name}")

        global _img_preview
        if filter_name != "none":
            _img_preview = img_helper.color_filter(_img_original, filter_name)
        else:
            _img_preview = _img_original.copy()

        operations.color_filter = filter_name
        self.toggle_thumbs()

        self.parent.parent.place_preview_img()


    def toggle_thumbs(self):
        for thumb in self.findChildren(QLabel):
            color = THUMB_BORDER_COLOR_ACTIVE if thumb.name == operations.color_filter else THUMB_BORDER_COLOR
            thumb.setStyleSheet(f"border:2px solid {color};")


class MainLayout(QVBoxLayout):
    """Main layout"""

    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        self.img_lbl = QLabel("Press <b>'Upload'</b> to to start<br>"
                              "<div style='margin: 30px 0'><img src='cover.jpg' /></div>"
                              "<b>Start Editing!</b> <span style='color:white'>&#10084;</span>")
        self.img_lbl.setAlignment(Qt.AlignCenter)
        global _img_lbl_glob
        _img_lbl_glob=self.img_lbl
        logger.debug("kljhgfdszdxcfgvhbjkl;kjhgfds",self.img_lbl)
        self.file_name = None

        self.img_size_lbl = None
        self.img_size_lbl = QLabel()
        self.img_size_lbl.setAlignment(Qt.AlignCenter)

        upload_btn = QPushButton("Upload")
        upload_btn.setMinimumWidth(BTN_MIN_WIDTH)
        upload_btn.clicked.connect(self.on_upload)
        upload_btn.setStyleSheet("font-weight:bold;")

        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setMinimumWidth(BTN_MIN_WIDTH)
        self.reset_btn.clicked.connect(self.on_reset)
        self.reset_btn.setEnabled(False)
        self.reset_btn.setStyleSheet("font-weight:bold;")

        self.save_btn = QPushButton("Save")
        self.save_btn.setMinimumWidth(BTN_MIN_WIDTH)
        self.save_btn.clicked.connect(self.on_save)
        self.save_btn.setEnabled(False)
        self.save_btn.setStyleSheet("font-weight:bold;")

        self.addWidget(self.img_lbl)
        self.addWidget(self.img_size_lbl)
        self.addStretch()

        self.action_tabs = ActionTabs(self)
        self.addWidget(self.action_tabs)
        self.action_tabs.setVisible(False)

        btn_layout = QHBoxLayout()
        btn_layout.setAlignment(Qt.AlignCenter)
        btn_layout.addWidget(upload_btn)
        btn_layout.addWidget(self.reset_btn)
        btn_layout.addWidget(self.save_btn)

        self.addLayout(btn_layout)

    def place_preview_img(self):
        img = _get_img_with_all_operations()

        preview_pix = ImageQt.toqpixmap(img)
        self.img_lbl.setPixmap(preview_pix)

    def on_save(self):
        logger.debug("open save dialog")
        new_img_path, _ = QFileDialog.getSaveFileName(self.parent, "QFileDialog.getSaveFileName()",
                                                      f"Edited_{self.file_name}",
                                                      "Images (*.png *.jpg)")

        if new_img_path:
            logger.debug(f"save output image to {new_img_path}")

            img = _get_img_with_all_operations()
            img.save(new_img_path)

    def on_upload(self):
        logger.debug("upload")
        img_path, _ = QFileDialog.getOpenFileName(self.parent, "Open image",
                                                  "/Users",
                                                  "Images (*.png *jpg)")

        if img_path:
            logger.debug(f"open file {img_path}")

            self.file_name = ntpath.basename(img_path)
            global _file_name_global
            _file_name_global = img_path
            pix = QPixmap(img_path)
            self.img_lbl.setPixmap(pix)
            self.img_lbl.setScaledContents(True)
            self.action_tabs.setVisible(True)
            self.action_tabs.adjustment_tab.reset_sliders()

            global _img_original
            _img_original = ImageQt.fromqpixmap(pix)
            # print("_img_original", _img_original)
            logger.debug(_img_original)
            self.update_img_size_lbl()

            if _img_original.width < _img_original.height:
                w = THUMB_SIZE
                h = _get_ratio_height(_img_original.width, _img_original.height, w)
            else:
                h = THUMB_SIZE
                w = _get_ratio_width(_img_original.width, _img_original.height, h)

            img_filter_thumb = img_helper.resize(_img_original, w, h)

            global _img_preview
            _img_preview = _img_original.copy()

            for thumb in self.action_tabs.filters_tab.findChildren(QLabel):
                if thumb.name != "none":
                    img_filter_preview = img_helper.color_filter(img_filter_thumb, thumb.name)
                else:
                    img_filter_preview = img_filter_thumb

                preview_pix = ImageQt.toqpixmap(img_filter_preview)
                thumb.setPixmap(preview_pix)

            self.reset_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
            self.action_tabs.modification_tab.set_boxes()

    def update_img_size_lbl(self):
        logger.debug("update img size lbl")

        self.img_size_lbl.setText(f"<span style='font-size:11px'>"
                                  f"image size {operations.size[0] if operations.size else _img_original.width} × "
                                  f"{operations.size[1] if operations.size else _img_original.height}"
                                  f"</span>")

    def on_reset(self):
        logger.debug("reset all")

        global _img_preview
        _img_preview = _img_original.copy()

        operations.reset()

        self.action_tabs.filters_tab.toggle_thumbs()
        self.place_preview_img()
        self.action_tabs.adjustment_tab.reset_sliders()
        self.action_tabs.modification_tab.set_boxes()
        self.update_img_size_lbl()


class FunSnap(QWidget):
    """Main widget"""

    def __init__(self):
        super().__init__()

        self.main_layout = MainLayout(self)
        self.setLayout(self.main_layout)

        self.setMinimumSize(600, 500)
        self.setMaximumSize(900, 900)
        self.setGeometry(600, 600, 600, 600)
        self.setWindowTitle('FunSnap')
        self.center()
        self.show()

    def center(self):
        """align window center"""

        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def closeEvent(self, event):
        logger.debug("close")

        if operations.has_changes():
            reply = QMessageBox.question(self, "",
                                         "You have unsaved changes<br>Are you sure?", QMessageBox.Yes |
                                         QMessageBox.No, QMessageBox.No)

            if reply == QMessageBox.Yes:
                event.accept()
            else:
                event.ignore()

    def resizeEvent(self, e):
        pass


class QVLine(QFrame):
    """Vertical line"""

    def __init__(self):
        super(QVLine, self).__init__()
        self.setFrameShape(QFrame.VLine)
        self.setFrameShadow(QFrame.Sunken)


if __name__ == '__main__':
    fileConfig('logging_config.ini')

    app = QApplication(sys.argv)
    win = FunSnap()
    sys.exit(app.exec_())

