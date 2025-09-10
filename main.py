import sys
import os
import re
import wave
import time
import threading
from PyQt5 import QtWidgets, QtGui, QtCore
import pyaudio
from g2p_en import G2p
import audioop  # built-in, used for converting channels


class TextToSpeech:
    CHUNK = 1024

    def __init__(self, character_folder):
        self.character_folder = character_folder
        # g2p without nltk tokenizer (we’ll tokenize manually with regex)
        self.g2p = G2p()
        self.word_pause = 0.01
        self.comma_pause = 0.25
        self.period_pause = 0.3

    def _generate_silence(self, duration_sec, frame_rate, sample_width, channels):
        if not frame_rate or not sample_width or not channels:
            frame_rate, sample_width, channels = 44100, 2, 1
        num_samples = int(frame_rate * duration_sec)
        silence_sample = b'\x00' * sample_width
        return silence_sample * num_samples * channels

    def _get_phonemes(self, word):
        phonemes = self.g2p(word)  # g2p returns list
        return [re.sub(r'\d+', '', p) for p in phonemes if re.match(r'[A-Z]+[0-9]*', p)]

    def get_pronunciation(self, str_input):
        # Manual regex tokenizer instead of nltk
        tokens = re.findall(r"[\w']+|[.,!?;]", str_input)
        playback_list = []

        for token in tokens:
            token_upper = token.upper()

            if token in [".", "!", "?"]:
                playback_list.append(("silence", self.period_pause))
                continue
            elif token in [",", ";"]:
                playback_list.append(("silence", self.comma_pause))
                continue

            word_wav = os.path.join(self.character_folder, 'words', f'{token_upper}.wav')
            if os.path.isfile(word_wav):
                playback_list.append(("file", word_wav))
            else:
                phonemes = self._get_phonemes(token)
                for phoneme in phonemes:
                    phoneme_wav = os.path.join(self.character_folder, f'{phoneme}.wav')
                    if os.path.isfile(phoneme_wav):
                        playback_list.append(("file", phoneme_wav))
                    else:
                        print(f"Missing phoneme WAV: {phoneme}")
            playback_list.append(("silence", self.word_pause))

        # Play everything in one background thread
        threading.Thread(target=self._play_sequence, args=(playback_list,)).start()

    def _play_sequence(self, playback_list):
        p = pyaudio.PyAudio()
        target_channels = 1        # Force mono
        target_rate = 44100        # Force 44.1kHz
        target_width = 2           # 16-bit

        stream = p.open(format=p.get_format_from_width(target_width),
                        channels=target_channels,
                        rate=target_rate,
                        output=True)

        for item_type, value in playback_list:
            if item_type == "silence":
                time.sleep(value)
            elif item_type == "file":
                try:
                    with wave.open(value, 'rb') as wf:
                        channels = wf.getnchannels()
                        width = wf.getsampwidth()
                        rate = wf.getframerate()
                        data = wf.readframes(wf.getnframes())

                        # Convert to mono if needed
                        if channels > 1:
                            data = audioop.tomono(data, width, 0.5, 0.5)

                        # Convert sample width to 16-bit if needed
                        if width != target_width:
                            data = audioop.lin2lin(data, width, target_width)

                        # Resample to 44100 if needed
                        if rate != target_rate:
                            data, _ = audioop.ratecv(data, target_width, 1, rate, target_rate, None)

                        stream.write(data)

                except Exception as e:
                    print(f"Error playing {value}: {e}")

        stream.stop_stream()
        stream.close()
        p.terminate()

    def render_to_file(self, str_input, output_path):
        tokens = re.findall(r"[\w']+|[.,!?;]", str_input)
        audio_segments = []

        target_channels = 1
        target_width = 2       # 16-bit
        target_rate = 44100

        for token in tokens:
            token_upper = token.upper()

            if token in [".", "!", "?"]:
                audio_segments.append(self._generate_silence(0.5, target_rate, target_width, target_channels))
                continue
            elif token in [",", ";"]:
                audio_segments.append(self._generate_silence(0.25, target_rate, target_width, target_channels))
                continue

            word_wav = os.path.join(self.character_folder, 'words', f'{token_upper}.wav')
            if os.path.isfile(word_wav):
                audio_segments.append(self._normalize_wav(word_wav, target_channels, target_width, target_rate))
            else:
                phonemes = self._get_phonemes(token)
                for phoneme in phonemes:
                    phoneme_wav = os.path.join(self.character_folder, f'{phoneme}.wav')
                    if os.path.isfile(phoneme_wav):
                        audio_segments.append(self._normalize_wav(phoneme_wav, target_channels, target_width, target_rate))
                    else:
                        print(f"Missing phoneme WAV: {phoneme}")

            # word pause
            audio_segments.append(self._generate_silence(0.01, target_rate, target_width, target_channels))

        if not audio_segments:
            print("Nothing to render.")
            return

        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(target_channels)
            wf.setsampwidth(target_width)
            wf.setframerate(target_rate)
            wf.writeframes(b''.join(audio_segments))

    def _normalize_wav(self, filepath, target_channels, target_width, target_rate):
        """Read a wav file and normalize it to target format"""
        with wave.open(filepath, 'rb') as wf:
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            rate = wf.getframerate()
            data = wf.readframes(wf.getnframes())

            # Convert to mono if needed
            if channels > 1 and target_channels == 1:
                data = audioop.tomono(data, width, 0.5, 0.5)

            # Convert sample width
            if width != target_width:
                data = audioop.lin2lin(data, width, target_width)

            # Resample to target_rate
            if rate != target_rate:
                data, _ = audioop.ratecv(data, target_width, target_channels, rate, target_rate, None)

            return data


class TTSGui(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sentence Mixing Generator")
        self.setGeometry(100, 100, 500, 500)
        self.layout = QtWidgets.QVBoxLayout()

        self.category_selector = QtWidgets.QComboBox()
        self.category_selector.currentTextChanged.connect(self.update_character_list)

        self.character_selector = QtWidgets.QListWidget()
        self.character_selector.setIconSize(QtCore.QSize(100, 100))

        self.text_input = QtWidgets.QLineEdit()
        self.text_input.setPlaceholderText("Type your phrase here...")

        self.play_button = QtWidgets.QPushButton("Speak")
        self.play_button.clicked.connect(self.speak)

        self.layout.addWidget(QtWidgets.QLabel("Select Category:"))
        self.layout.addWidget(self.category_selector)
        self.layout.addWidget(QtWidgets.QLabel("Select Character:"))
        self.layout.addWidget(self.character_selector)
        self.layout.addWidget(self.text_input)
        self.layout.addWidget(self.play_button)
        self.render_button = QtWidgets.QPushButton("Render to WAV")
        self.render_button.clicked.connect(self.render)
        self.layout.addWidget(self.render_button)

        self.setLayout(self.layout)
        self.load_categories()

    def load_categories(self):
        self.categories = {}
        base_path = 'assets/characters'
        for category in os.listdir(base_path):
            category_path = os.path.join(base_path, category)
            if os.path.isdir(category_path):
                self.categories[category] = {}
                for character in os.listdir(category_path):
                    char_path = os.path.join(category_path, character)
                    if os.path.isdir(char_path):
                        self.categories[category][character] = char_path

        self.category_selector.addItems(self.categories.keys())
        if self.categories:
            first_category = next(iter(self.categories))
            self.update_character_list(first_category)

    def update_character_list(self, selected_category):
        self.character_selector.clear()
        if selected_category in self.categories:
            for char_name, char_path in self.categories[selected_category].items():
                profile_pic = os.path.join(char_path, 'profile.png')
                item = QtWidgets.QListWidgetItem(QtGui.QIcon(profile_pic), char_name)
                self.character_selector.addItem(item)

    def speak(self):
        selected_items = self.character_selector.selectedItems()
        if not selected_items:
            QtWidgets.QMessageBox.warning(self, "Error", "Please select a character.")
            return

        char_name = selected_items[0].text()
        category = self.category_selector.currentText()
        text = self.text_input.text()

        if not text.strip():
            QtWidgets.QMessageBox.warning(self, "Error", "Please enter some text.")
            return

        char_folder = self.categories[category][char_name]
        tts = TextToSpeech(char_folder)
        threading.Thread(target=tts.get_pronunciation, args=(text,)).start()

    def render(self):
        selected_items = self.character_selector.selectedItems()
        if not selected_items:
            QtWidgets.QMessageBox.warning(self, "Error", "Please select a character.")
            return

        char_name = selected_items[0].text()
        category = self.category_selector.currentText()
        text = self.text_input.text()

        if not text.strip():
            QtWidgets.QMessageBox.warning(self, "Error", "Please enter some text.")
            return

        char_folder = self.categories[category][char_name]
        tts = TextToSpeech(char_folder)

        # Ask where to save file
        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Rendered Audio", "", "WAV Files (*.wav)")
        if not save_path:
            return

        try:
            tts.render_to_file(text, save_path)
            QtWidgets.QMessageBox.information(self, "Success", f"Rendered audio saved to:\n{save_path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to render: {e}")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    gui = TTSGui()
    gui.show()
    sys.exit(app.exec_())
