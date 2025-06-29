from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget,
    QCheckBox, QListWidget, QScrollArea, QFileDialog
)
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QCheckBox,
    QLabel, QLineEdit, QKeySequenceEdit, QApplication
)

from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt
import sys
import os

# sorted in alphabetical order
eng_to_cn = {
    # map
    'ant_cave_2': '螞蟻洞2',
    'cloud_balcony': '雲彩露臺',
    'dragon_territory': '魔龍領地',
    'empty_house': '空屋',
    'fire_land_1': '火焰之地1',
    'fire_land_2': '火焰之地2',
    'first_barrack': '第一軍營',
    'land_of_wild_boar': '黑肥肥領土',
    'lost_time_1': '遺失的時間1',
    'monkey_swamp_3': '猴子沼澤地3',
    'mushroom_hills': '菇菇山丘',
    'north_forest_training_ground_2': '北部森林訓練場2',
    'north_forest_training_ground_8': '北部森林訓練場8',
    'pig_shores': '肥肥海岸',
    'the_path_of_time_1_for_mage': '時間之路1',
    # monster
    'angel_monkey': '天使猴',
    'black_axe_stump': '黑斧木妖',
    'brown_windup_bear': '褐色發條熊',
    'cold_eye': '冰獨眼獸',
    'evolved_ghost': '進化妖魔',
    'fire_boar': '火肥肥',
    'green_mushroom': '綠菇菇',
    'mushroom': '菇菇寶貝',
    'pig': '肥肥',
    'pink_windup_bear': '粉色發條熊',
    'ribbon_pig': '緞帶肥肥',
    'skeleton_officer': '骷髏隊長',
    'skeleton_soldier': '骷髏士兵',
    'spike_mushroom': '刺菇菇',
    'wild_boar': '黑肥肥',
    'wild_kargo': '魔龍',
    'wind_single_eye_beast': '風獨眼獸',
    'zombie_mushroom': '殭屍菇菇',
}

map_mobs_mapping = {
    'ant_cave_2': ('spike_mushroom', 'zombie_mushroom'),
    'cloud_balcony': ('pink_windup_bear', 'brown_windup_bear'),
    'dragon_territory': ('wild_kargo'),
    'empty_house': ('mushroom'),
    'fire_land_1': ('black_axe_stump', 'fire_boar'),
    'fire_land_2': ('black_axe_stump', 'fire_boar'),
    'first_barrack': ('skeleton_officer', 'skeleton_soldier'),
    'land_of_wild_boar': ('wild_boar'),
    'lost_time_1': ('evolved_ghost'),
    'monkey_swamp_3': ('angel_monkey'),
    'mushroom_hills': ('mushroom', 'green_mushroom'),
    'north_forest_training_ground_2': ('green_mushroom', 'spike_mushroom'),
    'north_forest_training_ground_8': ('wind_single_eye_beast'),
    'pig_shores': ('pig', 'ribbon_pig'),
    'the_path_of_time_1_for_mage': ('evolved_ghost'),
}

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("MapleStory AutoBot")
        self.setMinimumSize(600, 400)

        # Layout setup
        layout = QVBoxLayout()

        # Label
        self.label = QLabel("Press 'F1' to start the Bot\n"
                            "Press 'F2' to take screenshot")
        layout.addWidget(self.label)

        # Button
        self.button = QPushButton("Load Map")
        self.button.clicked.connect(self.load_image)
        layout.addWidget(self.button)

        # Auto add HP checkbox
        self.checkbox_auto_add_hp = QCheckBox("Auto Add HP")
        self.checkbox_auto_add_hp.stateChanged.connect(self.auto_add_hp_toggle)
        layout.addWidget(self.checkbox_auto_add_hp)

        # Container for extra inputs (threshold and key)
        self.mp_input_container = QHBoxLayout()

        self.label_threshold = QLabel("Threshold (0.0-1.0):")
        self.input_threshold = QLineEdit()
        self.input_threshold.setPlaceholderText("0.5")
        self.input_threshold.setFixedWidth(60)

        self.label_key = QLabel("Trigger Key:")
        self.input_key = QKeySequenceEdit()
        self.input_key.setFixedWidth(100)

        # Add widgets to input container
        self.mp_input_container.addWidget(self.label_threshold)
        self.mp_input_container.addWidget(self.input_threshold)
        self.mp_input_container.addWidget(self.label_key)
        self.mp_input_container.addWidget(self.input_key)

        # Wrap the layout in a QWidget to show/hide easily
        self.mp_input_widget = QWidget()
        self.mp_input_widget.setLayout(self.mp_input_container)
        # self.mp_input_widget.setVisible(False)
        self.mp_input_widget.setVisible(True)

        layout.addWidget(self.mp_input_widget)



        # Auto add MP checkbox
        self.checkbox_auto_add_mp = QCheckBox("Auto Add MP")
        self.checkbox_auto_add_mp.stateChanged.connect(self.auto_add_mp_toggle)
        layout.addWidget(self.checkbox_auto_add_mp)

        # Scroll List
        self.list_widget_maps = QListWidget()
        # Get all directory names in minimaps/
        minimap_dir = "minimaps"
        for name in os.listdir(minimap_dir):
            if name.startswith("."):
                continue  # Skip files starting with a period

            name_cn = ""
            if name in eng_to_cn:
                name_cn = eng_to_cn[name]
            full_path = os.path.join(minimap_dir, name)
            if os.path.isdir(full_path):
                self.list_widget_maps.addItem(f"{name}({name_cn})")
        layout.addWidget(self.list_widget_maps)

        # Image Display Area (Canvas)
        self.image_label = QLabel("Image not loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(200)
        self.image_label.setStyleSheet("border: 1px solid gray;")

        # Scroll wrapper for image
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.image_label)
        layout.addWidget(self.scroll_area)

        # Set central widget
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if path:
            pixmap = QPixmap(path)
            self.image_label.setPixmap(pixmap.scaledToWidth(400, Qt.SmoothTransformation))
            self.label.setText(f"Loaded: {path.split('/')[-1]}")

    def auto_add_hp_toggle(self, state):
        enabled = state == Qt.Checked
        self.mp_input_widget.setVisible(state == Qt.CheckState.Checked)
        print(f"[auto_add_hp_toggle] Checkbox toggled: {'Enabled' if enabled else 'Disabled'}")

    def auto_add_mp_toggle(self, state):
        enabled = state == Qt.Checked
        print(f"[auto_add_mp_toggle] Checkbox toggled: {'Enabled' if enabled else 'Disabled'}")

    def get_config(self):
        if self.checkbox_add_mp.isChecked():
            try:
                threshold = float(self.input_threshold.text())
                if not 0 <= threshold <= 1:
                    raise ValueError("Threshold out of bounds")
            except ValueError:
                print("Invalid threshold input.")
                return None
            key_seq = self.input_key.keySequence().toString()
            return {
                "enabled": True,
                "threshold": threshold,
                "key": key_seq
            }
        return {"enabled": False}


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
