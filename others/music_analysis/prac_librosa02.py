# libosaで音声波形の表示
import librosa

file_name = 'sound/sound01.wav'
wav, sr = librosa.load(file_name, sr=44100)

import librosa.display
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
librosa.display.waveshow(wav, sr=sr)
plt.show()
