import sys
import os
import torch
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QFrame
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from omegaconf import OmegaConf
import torchvision.transforms as transforms

# 用项目自带的函数
from utils.import_utils import instantiate_from_config, recurse_instantiate_from_config, get_obj_from_str


class CamoWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('寻找伪装目标')
        self.resize(1000, 600)
        self.current_img_path = None

        # 界面排版
        main_layout = QVBoxLayout()
        btn_layout = QHBoxLayout()

        self.btn_select = QPushButton('选一张图片')
        self.btn_select.clicked.connect(self.choose_picture)
        self.btn_predict = QPushButton('开始寻找')
        self.btn_predict.clicked.connect(self.process_picture)
        self.btn_predict.setEnabled(False)

        btn_layout.addWidget(self.btn_select)
        btn_layout.addWidget(self.btn_predict)
        main_layout.addLayout(btn_layout)

        img_layout = QHBoxLayout()
        self.label_orig = QLabel('原图会在这里显示')
        self.label_orig.setAlignment(Qt.AlignCenter)
        self.label_orig.setFrameShape(QFrame.Panel)

        self.label_res = QLabel('正在准备模型，请等一下...')
        self.label_res.setAlignment(Qt.AlignCenter)
        self.label_res.setFrameShape(QFrame.Panel)

        img_layout.addWidget(self.label_orig)
        img_layout.addWidget(self.label_res)
        main_layout.addLayout(img_layout)

        self.setLayout(main_layout)

        # 把模型准备好
        self.setup_model()

    def setup_model(self):
        # 确认一下你的文件位置是不是这两个
        self.config_path = 'config/camoDiffusion_352x352.yaml'
        self.model_path = 'model-best/model-best.pt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if not os.path.exists(self.config_path) or not os.path.exists(self.model_path):
            self.label_res.setText('没找到配置文件或模型文件，检查一下路径吧。')
            return

        cfg = OmegaConf.load(self.config_path)
        self.img_size = cfg.diffusion_model.params.image_size

        # 照着 sample.py 的样子组装模型
        cond_uvit = instantiate_from_config(
            cfg.cond_uvit,
            conditioning_klass=get_obj_from_str(cfg.cond_uvit.params.conditioning_klass)
        )
        model_arch = recurse_instantiate_from_config(cfg.model, unet=cond_uvit)
        self.diffusion_model = instantiate_from_config(cfg.diffusion_model, model=model_arch).to(self.device)

        # 读取存好的模型状态
        state_dict = torch.load(self.model_path, map_location=self.device)
        if 'model' in state_dict:
            self.diffusion_model.load_state_dict(state_dict['model'])
        else:
            self.diffusion_model.load_state_dict(state_dict)

        self.diffusion_model.eval()
        self.label_res.setText('模型准备好了，可以选图了。')

    def choose_picture(self):
        fname, _ = QFileDialog.getOpenFileName(self, '选张图', '', '图片 (*.jpg *.png *.jpeg)')
        if fname:
            self.current_img_path = fname
            pixmap = QPixmap(fname).scaled(450, 450, Qt.KeepAspectRatio)
            self.label_orig.setPixmap(pixmap)
            self.btn_predict.setEnabled(True)
            self.label_res.setText('等候指令...')

    def process_picture(self):
        if not self.current_img_path:
            return

        self.label_res.setText('正在找，稍等一下...')
        QApplication.processEvents()  # 刷新一下屏幕文字

        # 处理图片大小和数值
        img = Image.open(self.current_img_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()
        ])

        # 把数值调到 [-1, 1] 区间，配合扩散模型
        img_data = transform(img).unsqueeze(0).to(self.device)
        img_data = (img_data - 0.5) * 2

        # 让模型去算图
        with torch.no_grad():
            out_images = self.diffusion_model.sample(img_data)
            # 拿最后一步跑出来的结果
            final_pic = out_images[-1].clamp(0, 1).squeeze().cpu().numpy()

        # 把结果变成能在屏幕上显示的格式
        final_pic = (final_pic * 255).astype(np.uint8)
        h, w = final_pic.shape
        q_img = QImage(final_pic.data, w, h, w, QImage.Format_Indexed8)

        pixmap_res = QPixmap.fromImage(q_img).scaled(450, 450, Qt.KeepAspectRatio)
        self.label_res.setPixmap(pixmap_res)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CamoWindow()
    window.show()
    sys.exit(app.exec_())