import sys
import os
import re
import wave
import threading
from PyQt5 import QtWidgets, QtGui, QtCore
import pyaudio
from g2p_en import G2p
import audioop
import random

class TextToSpeech:
    CHUNK = 1024

    PHONEME_MAPPING = {
        'AW': ['AE', 'OW'],
        'DH': ['D'],
        'EY': ['EH', 'IY'],
        'JH': ['CH'],
        'SH': ['CH'],
        'TH': ['D'],
        'ZH': ['CH'],
        'AE': ['AA'],
        'AO': ['AA', 'OW'],
        'ER': ['AA'],
        'IH': ['IY'],
        'OY': ['OW', 'Y', 'IY'],
        'UH': ['UW']
    }

    def __init__(self, character_folder):
        self.character_folder = character_folder
        self.g2p = G2p()
        self.word_pause = 0.01
        self.fade_duration = 0.05 

    def _is_vowel(self, phoneme):
        vowels = {'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW'}
        return phoneme.strip().upper() in vowels

    def _pick_random_variant(self, base_path):
        directory = os.path.dirname(base_path)
        base_name = os.path.splitext(os.path.basename(base_path))[0]
        if not os.path.isdir(directory): return None
        pattern = re.compile(rf"^{re.escape(base_name)}(_\d+)?\.wav$", re.IGNORECASE)
        candidates = [os.path.join(directory, f) for f in os.listdir(directory) if pattern.match(f)]
        return random.choice(candidates) if candidates else None

    def _get_phoneme_data(self, phoneme, target_ch, target_w, target_r):
        if phoneme == "AH0":
            data = self._get_phoneme_data("EH", target_ch, target_w, target_r)
            if data:
                frame_size = target_w * target_ch
                if len(data) > frame_size * 2:
                    return data[frame_size:-frame_size]
            return data

        base = os.path.join(self.character_folder, f"{phoneme}.wav")
        path = self._pick_random_variant(base)
        if path:
            return self._normalize_wav(path, target_ch, target_w, target_r)
        
        if phoneme in self.PHONEME_MAPPING:
            combined_data = b""
            for sub_p in self.PHONEME_MAPPING[phoneme]:
                sub_data = self._get_phoneme_data(sub_p, target_ch, target_w, target_r)
                if sub_data: combined_data += sub_data
            return combined_data if combined_data else None
        return None

    def _normalize_wav(self, filepath, target_channels, target_width, target_rate):
        """Normalizes audio format without looping or extending length."""
        with wave.open(filepath, 'rb') as wf:
            channels, width, rate = wf.getnchannels(), wf.getsampwidth(), wf.getframerate()
            data = wf.readframes(wf.getnframes())
            if channels > 1: data = audioop.tomono(data, width, 0.5, 0.5)
            if width != target_width: data = audioop.lin2lin(data, width, target_width)
            if rate != target_rate: data, _ = audioop.ratecv(data, target_width, 1, rate, target_rate, None)
            return data

    def _apply_crossfade(self, current_data, next_data, rate, width):
        fade_size = int(rate * self.fade_duration) * width
        actual_fade = min(fade_size, len(current_data), len(next_data))
        if actual_fade <= 0: return current_data, next_data

        curr_main = current_data[:-actual_fade]
        ramp_out, ramp_in = bytearray(), bytearray()
        num_frames = actual_fade // width
        for i in range(num_frames):
            out_f, in_f = 1.0 - (i / num_frames), i / num_frames
            ramp_out += audioop.mul(current_data[len(curr_main) + (i*width) : len(curr_main) + ((i+1)*width)], width, out_f)
            ramp_in += audioop.mul(next_data[(i*width) : ((i+1)*width)], width, in_f)

        mixed_fade = audioop.add(bytes(ramp_out), bytes(ramp_in), width)
        return curr_main + mixed_fade, next_data[actual_fade:]

    def render_to_file(self, str_input, output_path):
        tokens = re.findall(r"[\w']+|[.,!?;]", str_input)
        raw_segments = []
        target_ch, target_w, target_r = 1, 2, 44100

        for token in tokens:
            if token in [".", "!", "?", ",", ";"]:
                dur = 0.5 if token in [".", "!", "?"] else 0.25
                raw_segments.append({"data": b'\x00' * int(target_r * dur * target_w * target_ch), "is_vowel": False, "is_pause": True})
                continue

            word_wav = self._pick_random_variant(os.path.join(self.character_folder, "words", f"{token.upper()}.wav"))
            if word_wav:
                raw_segments.append({"data": self._normalize_wav(word_wav, target_ch, target_w, target_r), "is_vowel": False, "is_pause": False})
            else:
                phonemes = self.g2p(token)
                valid_ps = [re.sub(r'\d+', '', p) if p != "AH0" else p for p in phonemes]
                valid_ps = [p for p in valid_ps if re.match(r'[A-Z]+[0-9]*', p)]

                if valid_ps:
                    last_p = valid_ps[-1]
                    if last_p in ["AH", "AE", "AH0"]:
                        valid_ps[-1] = "AA"

                for p_clean in valid_ps:
                    data = self._get_phoneme_data(p_clean, target_ch, target_w, target_r)
                    if data: 
                        raw_segments.append({
                            "data": data, 
                            "is_vowel": self._is_vowel(p_clean), 
                            "is_pause": False, 
                            "phoneme": p_clean
                        })

            raw_segments.append({"data": b'\x00' * int(target_r * self.word_pause * target_w * target_ch), "is_vowel": False, "is_pause": True})

        final_audio = b""
        total_segments = len(raw_segments)
        for i in range(total_segments):
            curr_data = raw_segments[i]["data"]
            can_fade = False
            
            if i + 1 < total_segments:
                curr_seg = raw_segments[i]
                next_seg = raw_segments[i+1]
                
                curr_is_consonant = not curr_seg.get("is_vowel") and not curr_seg.get("is_pause")
                next_is_vowel = next_seg.get("is_vowel")
                next_is_pause = next_seg.get("is_pause")

                # Crossfade if transitioning from consonant to vowel
                if curr_is_consonant and next_is_vowel and not next_is_pause:
                    can_fade = True

            if can_fade:
                faded_curr, leftover_next = self._apply_crossfade(curr_data, raw_segments[i+1]["data"], target_r, target_w)
                final_audio += faded_curr
                raw_segments[i+1]["data"] = leftover_next
            else:
                final_audio += curr_data

        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(target_ch); wf.setsampwidth(target_w); wf.setframerate(target_r)
            wf.writeframes(final_audio)

    def get_pronunciation(self, str_input):
        temp_path = "temp_playback.wav"
        self.render_to_file(str_input, temp_path)
        def play():
            if not os.path.exists(temp_path): return
            wf = wave.open(temp_path, 'rb')
            p = pyaudio.PyAudio()
            stream = p.open(format=p.get_format_from_width(wf.getsampwidth()), channels=wf.getnchannels(), rate=wf.getframerate(), output=True)
            data = wf.readframes(self.CHUNK)
            while data: stream.write(data); data = wf.readframes(self.CHUNK)
            stream.stop_stream(); stream.close(); p.terminate(); wf.close()
            try: os.remove(temp_path)
            except: pass
        threading.Thread(target=play).start()

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
        self.play_button = QtWidgets.QPushButton("Speak")
        self.play_button.clicked.connect(self.speak)
        self.render_button = QtWidgets.QPushButton("Render to WAV")
        self.render_button.clicked.connect(self.render)
        for w in [QtWidgets.QLabel("Select Category:"), self.category_selector, QtWidgets.QLabel("Select Character:"), self.character_selector, self.text_input, self.play_button, self.render_button]: self.layout.addWidget(w)
        self.setLayout(self.layout)
        self.load_categories()

    def load_categories(self):
        self.categories = {}
        base_path = 'assets/characters'
        if not os.path.isdir(base_path): return
        for cat in os.listdir(base_path):
            cat_path = os.path.join(base_path, cat)
            if os.path.isdir(cat_path):
                self.categories[cat] = {char: os.path.join(cat_path, char) for char in os.listdir(cat_path) if os.path.isdir(os.path.join(cat_path, char))}
        self.category_selector.addItems(self.categories.keys())
        if self.categories: self.update_character_list(next(iter(self.categories)))

    def update_character_list(self, selected_category):
        self.character_selector.clear()
        for char_name, char_path in self.categories.get(selected_category, {}).items():
            profile_pic = os.path.join(char_path, 'profile.png')
            self.character_selector.addItem(QtWidgets.QListWidgetItem(QtGui.QIcon(profile_pic), char_name))

    def speak(self):
        selected = self.character_selector.selectedItems()
        if not selected: return
        char_folder = self.categories[self.category_selector.currentText()][selected[0].text()]
        TextToSpeech(char_folder).get_pronunciation(self.text_input.text().strip())

    def render(self):
        selected = self.character_selector.selectedItems()
        if not selected: return
        char_folder = self.categories[self.category_selector.currentText()][selected[0].text()]
        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save WAV", "", "WAV Files (*.wav)")
        if save_path: TextToSpeech(char_folder).render_to_file(self.text_input.text().strip(), save_path)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    gui = TTSGui(); gui.show()
    sys.exit(app.exec_())