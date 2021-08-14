# Energy_Base_VAD

### key-feature

- energy based method, simple but effective and efficient.
- support signal or mel spectrum as input.
- support offline-mode and online mode.
- hpermerters has be proved in many speech tasks. 


### usage

An example to generate new signal which only contains speech part.

```python

import librosa
import numpy as np

from vad import SigVad

_vad = SigVad(top_db=25)
wav_path = "demo.wav"
res = _vad.get_speech_endpoint(wav_path=wav_path)
signal, sr = librosa.load(wav_path, sr=16000)

new_signal = np.zeros_like(signal)

for s, e in res:
  start_point = int(s * sr)
  end_point = int(e * sr)
  new_signal[start_point:end_point] = signal[start_point:end_point]

```