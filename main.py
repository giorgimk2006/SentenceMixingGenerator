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
import random


class TextToSpeech:
    CHUNK = 1024

    def __init__(self, character_folder):
        self.character_folder = character_folder
        self.g2p = G2p()  # g2p without nltk tokenizer
        self.word_pause = 0.01
        self.comma_pause = 0.25
        self.period_pause = 0.3

    # ------------------ Random WAV Picker ------------------
    def _pick_random_variant(self, base_path):
        """
        Picks a random wav:
        base.wav OR base_<any number>.wav
        Example: AH.wav, AH_2000.wav, AH_57.wav
        """
        directory = os.path.dirname(base_path)
        base_name = os.path.splitext(os.path.basename(base_path))[0]

        if not os.path.isdir(directory):
            return None

        pattern = re.compile(rf"^{re.escape(base_name)}(_\d+)?\.wav$", re.IGNORECASE)

        candidates = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if pattern.match(f)
        ]

        if not candidates:
            return None

        return random.choice(candidates)

    # ------------------ Get WAVs ------------------
    def _get_word_wav(self, word):
        word = word.upper()
        base = os.path.join(self.character_folder, "words", f"{word}.wav")
        return self._pick_random_variant(base)

    def _get_phoneme_wav(self, phoneme):
        base = os.path.join(self.character_folder, f"{phoneme}.wav")
        return self._pick_random_variant(base)

    # ------------------ Phoneme Extraction ------------------
    def _get_phonemes(self, word):
        raw_phonemes = self.g2p(word)
        cleaned = []

        for p in raw_phonemes:
            if not re.match(r'[A-Z]+[0-9]*', p):
                continue
            # Strip stress digits AFTER replacement
            p = re.sub(r'\d+', '', p)
            cleaned.append(p)

        return cleaned


    # ------------------ Silence Generator ------------------
    def _generate_silence(self, duration_sec, frame_rate=44100, sample_width=2, channels=1):
        num_samples = int(frame_rate * duration_sec)
        silence_sample = b'\x00' * sample_width
        return silence_sample * num_samples * channels

    # ------------------ Playback ------------------
    def get_pronunciation(self, str_input):
        tokens = re.findall(r"[\w']+|[.,!?;]", str_input)
        playback_list = []

        for token in tokens:
            if token in [".", "!", "?"]:
                playback_list.append(("silence", self.period_pause))
                continue
            elif token in [",", ";"]:
                playback_list.append(("silence", self.comma_pause))
                continue

            # Word-first playback
            word_wav = self._get_word_wav(token)
            if word_wav:
                playback_list.append(("file", word_wav))
            else:
                phonemes = self._get_phonemes(token)
                for phoneme in phonemes:
                    phoneme_wav = self._get_phoneme_wav(phoneme)
                    if phoneme_wav:
                        playback_list.append(("file", phoneme_wav))
                    else:
                        print(f"Missing phoneme WAV: {phoneme}")

            playback_list.append(("silence", self.word_pause))

        threading.Thread(target=self._play_sequence, args=(playback_list,)).start()

    def _play_sequence(self, playback_list):
        p = pyaudio.PyAudio()
        target_channels = 1
        target_rate = 44100
        target_width = 2

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

                        if channels > 1:
                            data = audioop.tomono(data, width, 0.5, 0.5)
                        if width != target_width:
                            data = audioop.lin2lin(data, width, target_width)
                        if rate != target_rate:
                            data, _ = audioop.ratecv(data, target_width, 1, rate, target_rate, None)

                        stream.write(data)

                except Exception as e:
                    print(f"Error playing {value}: {e}")

        stream.stop_stream()
        stream.close()
        p.terminate()

    # ------------------ Render to WAV File ------------------
    def render_to_file(self, str_input, output_path):
        tokens = re.findall(r"[\w']+|[.,!?;]", str_input)
        audio_segments = []

        target_channels = 1
        target_width = 2
        target_rate = 44100

        for token in tokens:
            if token in [".", "!", "?"]:
                audio_segments.append(self._generate_silence(0.5, target_rate, target_width, target_channels))
                continue
            elif token in [",", ";"]:
                audio_segments.append(self._generate_silence(0.25, target_rate, target_width, target_channels))
                continue

            word_wav = self._get_word_wav(token)
            if word_wav:
                audio_segments.append(self._normalize_wav(word_wav, target_channels, target_width, target_rate))
            else:
                phonemes = self._get_phonemes(token)
                for phoneme in phonemes:
                    phoneme_wav = self._get_phoneme_wav(phoneme)
                    if phoneme_wav:
                        audio_segments.append(self._normalize_wav(phoneme_wav, target_channels, target_width, target_rate))
                    else:
                        print(f"Missing phoneme WAV: {phoneme}")

            audio_segments.append(self._generate_silence(self.word_pause, target_rate, target_width, target_channels))

        if not audio_segments:
            print("Nothing to render.")
            return

        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(target_channels)
            wf.setsampwidth(target_width)
            wf.setframerate(target_rate)
            wf.writeframes(b''.join(audio_segments))

    def _normalize_wav(self, filepath, target_channels, target_width, target_rate):
        with wave.open(filepath, 'rb') as wf:
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            rate = wf.getframerate()
            data = wf.readframes(wf.getnframes())

            if channels > 1 and target_channels == 1:
                data = audioop.tomono(data, width, 0.5, 0.5)
            if width != target_width:
                data = audioop.lin2lin(data, width, target_width)
            if rate != target_rate:
                data, _ = audioop.ratecv(data, target_width, target_channels, rate, target_rate, None)

            return data


# ------------------ GUI ------------------
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

        self.render_button = QtWidgets.QPushButton("Render to WAV")
        self.render_button.clicked.connect(self.render)

        self.layout.addWidget(QtWidgets.QLabel("Select Category:"))
        self.layout.addWidget(self.category_selector)
        self.layout.addWidget(QtWidgets.QLabel("Select Character:"))
        self.layout.addWidget(self.character_selector)
        self.layout.addWidget(self.text_input)
        self.layout.addWidget(self.play_button)
        self.layout.addWidget(self.render_button)
        self.setLayout(self.layout)

        self.load_categories()

    def load_categories(self):
        self.categories = {}
        base_path = 'assets/characters'
        if not os.path.isdir(base_path):
            return

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
            self.update_character_list(next(iter(self.categories)))

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
        text = self.text_input.text().strip()
        if not text:
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
        text = self.text_input.text().strip()
        if not text:
            QtWidgets.QMessageBox.warning(self, "Error", "Please enter some text.")
            return

        char_folder = self.categories[category][char_name]
        tts = TextToSpeech(char_folder)

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
