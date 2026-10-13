from PyQt5.QtWidgets import QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QMessageBox, \
    QHBoxLayout, QLineEdit
from utils.classes_no_torch import PTrack

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from main import VideoPlayer


class TrackWindow(QMainWindow):
    def __init__(self, track: PTrack, main_app: 'VideoPlayer'):
        super().__init__()
        self.track = track
        self.main_app = main_app
        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        label = QLabel('Это новое окно')
        remove_btn = QPushButton('Удалить трек')
        split_btn = QPushButton('Разделить трек здесь')
        merge_btn = QPushButton('Объединить с другим треком')
        self.set_code_btn = QPushButton('Установить код на весь трек')
        self.set_code_till_now_btn = QPushButton('Установить код до это кадра')
        self.set_code_btn.setEnabled(False)
        self.set_code_till_now_btn.setEnabled(False)

        hbox = QHBoxLayout()
        #
        self.code_input = QLineEdit(self.track.get_status().split(',')[0][1:])  # TODO добавить при нажатии enter вызов
        # set_code_for_track и вставлять самое вероятное значение
        self.code_input.textChanged.connect(self.validate_code_input)
        hbox.addWidget(self.code_input)
        hbox.addWidget(self.set_code_btn)

        layout.addWidget(label)
        layout.addLayout(hbox)  # Добавляем горизонтальный компоновщик в вертикальный
        layout.addWidget(self.set_code_till_now_btn)
        layout.addWidget(remove_btn)
        layout.addWidget(split_btn)
        layout.addWidget(merge_btn)

        remove_btn.clicked.connect(self.remove_track)
        split_btn.clicked.connect(self.split_track)
        merge_btn.clicked.connect(self.merge_track)
        self.set_code_btn.clicked.connect(self.set_code_for_track)
        self.set_code_till_now_btn.clicked.connect(self.set_code_till_now)

        central_widget.setLayout(layout)
        self.setWindowTitle(f'Track {self.track.track_id}')

    def validate_code_input(self, text):
        try:
            value = int(text)
            if 1 <= value <= 2047:
                self.code_input.setStyleSheet("QLineEdit { color: black; background-color: white;}")
                self.set_code_btn.setEnabled(True)
                self.set_code_till_now_btn.setEnabled(True)
            else:
                self.code_input.setStyleSheet("background-color: red;")
                self.set_code_btn.setEnabled(False)
                self.set_code_till_now_btn.setEnabled(False)
        except ValueError:
            self.code_input.setStyleSheet("background-color: red;")
            self.set_code_btn.setEnabled(False)
            self.set_code_till_now_btn.setEnabled(False)

    def remove_track(self):
        track_id = self.track.track_id
        self.track.remove_track(self.main_app.tracks, self.main_app.frame2track)
        QMessageBox.information(self, 'Удалить трек', f'Трек {track_id} удален')
        self.main_app.update_info()
        self.main_app.update_this_frame_again()
        self.close()

    def split_track(self):
        QMessageBox.information(self, 'Разделить трек', 'Трек разделен')

    def merge_track(self):
        # self.main_app.update_this_frame_again()
        QMessageBox.information(self, 'Объединить трек', 'Трек объединен')

    def set_code_for_track(self):
        self.track.set_code(int(self.code_input.text()))
        self.main_app.update_info()
        self.main_app.update_this_frame_again()
        self.main_app.save_overwrite_callback()
        QMessageBox.information(self, 'Установить код на весь трек', 'Код установлен на весь трек')

    def set_code_till_now(self):
        self.track.set_code_till_now(int(self.code_input.text()), self.main_app.frame_num)
        self.main_app.update_info()
        self.main_app.update_this_frame_again()
        QMessageBox.information(self, 'Установить код до этого кадра', 'Код установлен до этого кадра')
