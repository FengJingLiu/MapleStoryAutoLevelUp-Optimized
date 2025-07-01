'''
UI main
'''
# Standard import
import sys
import os

# PySide 6
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget,
    QCheckBox, QListWidget, QScrollArea, QFileDialog, QHBoxLayout, QLineEdit,
    QKeySequenceEdit, QPlainTextEdit, QTabWidget
)
from PySide6.QtGui import QPixmap, QKeySequence, QKeyEvent
from PySide6.QtCore import Qt, QObject, Signal

# Local import
from logger import logger
from ui.ui_utils import validate_numerical_input, validate_key_input

# TODO: store these information to other places
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
    '''
    MainWindow
    '''
    def __init__(self):
        super().__init__()

        self.setWindowTitle("MapleStory AutoBot")
        self.setMinimumSize(600, 400)

        self.tabs = QTabWidget()

        self.tab_main = self.setup_main_tab()
        self.tab_advanced = self.setup_advanced_tab()
        self.tab_debug = self.setup_debug_tab()

        self.tabs.addTab(self.tab_main, "Main Control")
        self.tabs.addTab(self.tab_advanced, "Advanced Settings")
        self.tabs.addTab(self.tab_debug, "Debug Window")

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.tabs)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def setup_main_tab(self):
        '''
        setup_main_tab
        '''
        # --- Main Control Tab ---
        tab_main = QWidget()
        layout = QVBoxLayout()
        tab_main.setLayout(layout)

        # Label
        layout.addWidget(QLabel("Press 'F1' to start the Bot\n"
                                 "Press 'F2' to take a screenshot\n"
                                 "Press 'F3' to start recording"))

        # Attack mode
        self.checkbox_attack = QCheckBox("Attack") # directional
        self.checkbox_aoe_skill = QCheckBox("AOE Skill")
        self.checkbox_attack.setChecked(True) # Use attack as default
        self.checkbox_attack.stateChanged.connect(
            lambda state: self.on_mode_checkbox_toggle(self.checkbox_attack)
        )
        self.checkbox_aoe_skill.stateChanged.connect(
            lambda state: self.on_mode_checkbox_toggle(self.checkbox_aoe_skill)
        )
        layout.addWidget(self.checkbox_attack)
        layout.addWidget(self.checkbox_aoe_skill)

        # Attack mode widget
        self.attack_widget, self.attack_key, self.attack_range_x, \
            self.attack_range_y, self.attack_cooldown = self.create_attack_widget()

        # Error message
        error_label = QLabel()
        error_label.setStyleSheet("color: red;")
        error_label.setVisible(False)  # hidden error message by default
        layout.addWidget(error_label)

        # Check attack widge
        self.attack_range_x.editingFinished.connect(
            lambda: validate_numerical_input(self.attack_range_x.text(), error_label, 0, 9999))
        self.attack_range_y.editingFinished.connect(
            lambda: validate_numerical_input(self.attack_range_y.text(), error_label, 0, 9999))
        self.attack_cooldown.editingFinished.connect(
            lambda: validate_numerical_input(self.attack_cooldown.text(), error_label, 0, 9999))
        layout.addWidget(self.attack_widget)

        # Key bindings row layout
        key_row_layout = QHBoxLayout()

        # Jump key
        key_row_layout.addWidget(QLabel("Jump Key:"))
        self.jump_key = SingleKeyEdit()
        self.jump_key.setFixedWidth(100)
        key_row_layout.addWidget(self.jump_key)

        # Return Home key
        key_row_layout.addWidget(QLabel("Return Home Key:"))
        self.return_home_key = SingleKeyEdit()
        self.return_home_key.setFixedWidth(100)
        key_row_layout.addWidget(self.return_home_key)

        # Party key
        key_row_layout.addWidget(QLabel("Party Key:"))
        self.party_key = SingleKeyEdit()
        self.party_key.setFixedWidth(100)
        key_row_layout.addWidget(self.party_key)

        # Add this horizontal layout to your main vertical layout
        layout.addLayout(key_row_layout)

        # Auto add HP option
        self.checkbox_auto_add_hp = QCheckBox("Auto Add HP")
        self.checkbox_auto_add_hp.stateChanged.connect(self.toggle_auto_add_hp)
        layout.addWidget(self.checkbox_auto_add_hp)

        self.hp_input_widget, self.add_hp_percent, self.add_hp_key = self.create_hp_mp_widget("HP")
        self.hp_input_widget.setVisible(False)
        self.add_hp_key.keySequenceChanged.connect(lambda: validate_key_input(self.add_hp_key))
        layout.addWidget(self.hp_input_widget)

        # Auto add MP option
        self.checkbox_auto_add_mp = QCheckBox("Auto Add MP")
        self.checkbox_auto_add_mp.stateChanged.connect(self.toggle_auto_add_mp)
        layout.addWidget(self.checkbox_auto_add_mp)

        self.mp_input_widget, self.add_mp_percent, self.add_mp_key = self.create_hp_mp_widget("MP")
        self.mp_input_widget.setVisible(False)
        self.add_mp_key.keySequenceChanged.connect(lambda: validate_key_input(self.add_mp_key))
        layout.addWidget(self.mp_input_widget)

        # --- Buff Skill Section Toggle ---
        self.checkbox_enable_buff = QCheckBox("Enable Buff Skills")
        self.checkbox_enable_buff.stateChanged.connect(self.toggle_auto_buff)
        layout.addWidget(self.checkbox_enable_buff)

        # Buff container widget (so we can show/hide the whole thing)
        self.buff_section_container = QWidget()
        self.buff_section_layout = QVBoxLayout()
        self.buff_section_container.setLayout(self.buff_section_layout)
        self.buff_section_container.setVisible(False)  # hidden by default

        # Inner layout to hold dynamic rows
        self.buff_layout = QVBoxLayout()
        self.buff_inputs = []

        # Initial row
        self.add_buff_row()

        # Add buff row layout and button to buff_section_layout
        self.buff_section_layout.addLayout(self.buff_layout)

        self.button_add_buff = QPushButton("+ Add Buff")
        self.button_add_buff.clicked.connect(self.add_buff_row)
        self.buff_section_layout.addWidget(self.button_add_buff)

        # Add the full buff section container to main layout
        layout.addWidget(self.buff_section_container)

        # Scroll List
        label_choose_map = QLabel("Choose a map:")
        layout.addWidget(label_choose_map)
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

        # Logger output window
        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(150)
        layout.addWidget(QLabel("Log Output:"))
        layout.addWidget(self.log_output)

        return tab_main

    def setup_advanced_tab(self):
        tab_advanced = QWidget()
        layout = QVBoxLayout()
        tab_advanced.setLayout(layout)
        return tab_advanced

    def setup_debug_tab(self):
        tab_debug = QWidget()
        layout = QVBoxLayout()
        tab_debug.setLayout(layout)
        return tab_debug

    def on_mode_checkbox_toggle(self, toggled_checkbox):
        '''
        Mutural exclusive checkbox
        '''
        if not toggled_checkbox.isChecked():
            # Prevent unchecking — always keep one mode selected
            toggled_checkbox.setChecked(True)
            return

        if toggled_checkbox == self.checkbox_attack:
            self.checkbox_aoe_skill.blockSignals(True)
            self.checkbox_aoe_skill.setChecked(False)
            self.checkbox_aoe_skill.blockSignals(False)

        elif toggled_checkbox == self.checkbox_aoe_skill:
            self.checkbox_attack.blockSignals(True)
            self.checkbox_attack.setChecked(False)
            self.checkbox_attack.blockSignals(False)

    def create_attack_widget(self):
        '''
        Create a wedge widget: "Press key ➜ [KEY] for Mode A/B"
        '''
        layout_main = QHBoxLayout()

        layout_main.addWidget(QLabel("Attack Key:"))
        key_input = SingleKeyEdit()
        key_input.setFixedWidth(100)
        layout_main.addWidget(key_input)

        # Horizontal Range
        layout_main.addWidget(QLabel("Horizontal Range:"))
        range_x = QLineEdit()
        range_x.setPlaceholderText("50")
        range_x.setFixedWidth(60)
        layout_main.addWidget(range_x)

        # Vertical Range
        layout_main.addWidget(QLabel("Vertical Range:"))
        range_y = QLineEdit()
        range_y.setPlaceholderText("50")
        range_y.setFixedWidth(60)
        layout_main.addWidget(range_y)

        # CoolDown
        layout_main.addWidget(QLabel("CoolDown (s):"))
        cooldown = QLineEdit()
        cooldown.setPlaceholderText("0.1")
        cooldown.setFixedWidth(60)
        layout_main.addWidget(cooldown)

        container = QWidget()
        container.setLayout(layout_main)

        return container, key_input, range_x, range_y, cooldown

    def create_hp_mp_widget(self, title):
        '''
        create_hp_mp_widget
        '''
        layout_main = QVBoxLayout()
        container = QWidget()

        # Error message
        error_label = QLabel()
        error_label.setStyleSheet("color: red;")
        error_label.setVisible(False)  # hidden error message by default

        input_line = QHBoxLayout()

        # User input percentage
        input_line.addWidget(QLabel(f"When {title} is lower than: "))
        percent = QLineEdit()
        percent.setPlaceholderText("50")
        percent.setFixedWidth(60)
        input_line.addWidget(percent)
        input_line.addWidget(QLabel("%"))

        # User input key
        input_line.addWidget(QLabel("press key:"))
        input_key = SingleKeyEdit()
        input_key.setFixedWidth(100)
        input_line.addWidget(input_key)

        # Combine error message into one layout_main
        layout_main.addWidget(error_label)
        layout_main.addLayout(input_line)
        container.setLayout(layout_main)

        # Connect user input validation function
        percent.editingFinished.connect(
            lambda: validate_numerical_input(percent.text(), error_label, 0, 100))

        return container, percent, input_key

    def toggle_auto_add_hp(self, state):
        '''
        Callback function for auto add hp checkbox
        '''
        enabled = Qt.CheckState(state) == Qt.Checked
        self.hp_input_widget.setVisible(enabled)
        logger.debug("[toggle_auto_add_hp] Checkbox toggled: "
                    f"{'Enabled' if enabled else 'Disabled'}")

    def toggle_auto_add_mp(self, state):
        '''
        Callback function for auto add mp checkbox
        '''
        enabled = Qt.CheckState(state) == Qt.Checked
        self.mp_input_widget.setVisible(enabled)
        logger.debug("[toggle_auto_add_mp] Checkbox toggled: "
                    f"{'Enabled' if enabled else 'Disabled'}")

    def toggle_auto_buff(self, state):
        '''
        Callback function for auto buff
        '''
        enabled = Qt.CheckState(state) == Qt.Checked
        self.buff_section_container.setVisible(enabled)
        logger.debug("[toggle_auto_buff] Buff section :"
                     f"{'Enabled' if enabled else 'Disabled'}")

    def add_buff_row(self):
        row_layout = QHBoxLayout()
        row_widget = QWidget()  # Wrapper to cleanly remove later
        row_widget.setLayout(row_layout)

        # Minus button
        button_remove = QPushButton("−")
        button_remove.setFixedWidth(25)

        # Key input
        label_key = QLabel("Buff Skill Key:")
        key_input = SingleKeyEdit()
        key_input.setFixedWidth(100)

        # Cooldown input
        label_cooldown = QLabel("Cooldown (s):")
        cooldown_input = QLineEdit()
        cooldown_input.setPlaceholderText("e.g. 30")
        cooldown_input.setFixedWidth(60)

        # Error label
        error_label = QLabel()
        error_label.setStyleSheet("color: red;")
        error_label.setVisible(False)

        # Add widgets to layout_main
        row_layout.addWidget(label_key)
        row_layout.addWidget(key_input)
        row_layout.addWidget(label_cooldown)
        row_layout.addWidget(cooldown_input)
        row_layout.addWidget(button_remove)

        # Add to UI
        self.buff_layout.addWidget(row_widget)
        self.buff_layout.addWidget(error_label)

        # Store row info for removal later
        self.buff_inputs.append((row_widget, error_label, key_input, cooldown_input))

        # Connect validation
        key_input.keySequenceChanged.connect(lambda: validate_key_input(key_input))

        cooldown_input.editingFinished.connect(
            lambda: validate_numerical_input(cooldown_input.text(), error_label, 0, 9999)
        )

        # Connect remove button
        button_remove.clicked.connect(lambda: self.remove_buff_row(row_widget, error_label))

    def remove_buff_row(self, row_widget: QWidget, error_label: QLabel):
        # Remove from layout_main
        self.buff_layout.removeWidget(row_widget)
        self.buff_layout.removeWidget(error_label)
        row_widget.setParent(None)
        error_label.setParent(None)

        # Remove from tracking list
        self.buff_inputs = [
            entry for entry in self.buff_inputs if entry[0] != row_widget
        ]

class SingleKeyEdit(QKeySequenceEdit):
    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        modifiers = event.modifiers()

        # Allow modifier-only keys
        if key in (Qt.Key_Shift, Qt.Key_Control, Qt.Key_Alt, Qt.Key_Meta):
            self.setKeySequence(QKeySequence(key))
            return

        # Otherwise, record only one key (replace previous)
        if modifiers:
            # This prevents combos like Ctrl+A
            self.setKeySequence(QKeySequence(key))
        else:
            self.setKeySequence(QKeySequence(key))

# class QtLogHandler(logging.Handler, QObject):
#     log_signal = Signal(str)

#     def __init__(self):
#         logging.Handler.__init__(self)
#         QObject.__init__(self)

#     def emit(self, record):
#         msg = self.format(record)
#         self.log_signal.emit(msg)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
