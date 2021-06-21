#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:liufeng
# datetime:2020/11/20 11:17 AM
# software: PyCharm

import librosa
import numpy as np


class MelVad:
  def __init__(self, global_ref_db=None, top_db=25):
    """ Vad Class for Mel as input

     Args:
       top_db : number > 0
           The threshold reference to consider as silence. Only valid for offline.
           Recommand set top_db 35 for clean dataset(e.g. tts data) and 25 for
           noise dataset.


       global_ref_db : number > 0
           The reference power, only valid for online.

     """
    self._global_ref_db = global_ref_db
    self._top_db = top_db
    return

  def get_vad_label(self, mels, using_global_ref=False):
    """ only for one spec, return True or False"""
    assert (self._global_ref_db or not using_global_ref)
    assert np.shape(mels)[-1] == 80
    mel_amp = self._get_mel_power(mels)
    if using_global_ref:
      ref = self._global_ref_db
    else:
      ref = np.max(mel_amp)

    mel_db = librosa.power_to_db(mel_amp, ref=ref, top_db=None)
    return mel_db[0] > -self._top_db

  def get_nosil_endpoints(self, mels, using_global_ref=False):
    """ get nosil index of mel spec.

    Args:
      mels: np.array,  shape=(seq_dims, feat_dims(80))
          mels from Audio signal

      using_global_ref: bool
          True for online, using global ref_db to compute mel_db. False
          for offline, using local max ref_db to compute mel_db

    Returns:
      nosil_index: list as [(start1, end1), (start2, end2) ...]

    """

    assert (self._global_ref_db or not using_global_ref)
    assert np.shape(mels)[-1] == 80
    mel_amp = self._get_mel_power(mels)

    if using_global_ref:
      ref = self._global_ref_db
    else:
      ref = np.max(mel_amp)

    mel_db = librosa.power_to_db(mel_amp, ref=ref, top_db=None)
    sil_index = np.where(mel_db > -self._top_db)[0]

    start, end = None, None
    nosil_duration = []
    for x in range(-1, len(mel_db)):
      if x not in sil_index and (x + 1) in sil_index:
        start = int(x + 1)
      if x in sil_index and x + 1 not in sil_index:
        end = int(x)
        nosil_duration.append((start, end))
    nosil_duration = [(_s, _e) for _s, _e in nosil_duration if _e - _s > 1]
    return nosil_duration

  @staticmethod
  def _get_mel_power(mels):
    """ get power of mel_spec.  """
    assert np.shape(mels)[-1] == 80
    ref_level_db = 20
    min_level_db = -100
    mels = np.clip(mels, 0, 1) * (-min_level_db) + min_level_db
    mel_amp = np.power(10.0, (mels + ref_level_db) * 0.05)
    mel_amp = np.sum(mel_amp, axis=1)
    return mel_amp


class SigVad:
  def __init__(self, top_db=25, frame_length=1024, hop_length=256, ref=np.max):
    """ Sig Vad

    Args:
      top_db : number > 0
          The threshold (in decibels) below reference to consider as silence.
          Recommand set top_db 35 for clean dataset(e.g. tts data) and 25 for
          noise dataset.

      frame_length : int > 0
          The number of samples per analysis frame

      hop_length : int > 0
          The number of samples between analysis frames

      ref : number or callable
          The reference power.  By default, it uses `np.max` and compares
          to the peak power in the signal.

    """
    self._top_db = top_db
    self._frame_length = frame_length
    self._hop_length = hop_length
    self._ref = ref
    return

  def _get_sil_endpoints(self, y, sil_type="sil"):
    """ get sil endpoints of signal.

    Args:
      y : np.array,  shape=(n)
          Audio signal

      sil_type: "sil" or "nosil"
          return sil and nosil endpoint

    Returns:
      sil_endpoints : list as [(start1, end1), (start2, end2) ...]

    """
    mse = librosa.feature.rms(y, frame_length=self._frame_length,
                              hop_length=self._hop_length) ** 2
    signal_db = librosa.power_to_db(mse.squeeze(), ref=self._ref, top_db=None)
    if sil_type == "sil":
      sil_index = np.where(signal_db < -self._top_db)[0]
    elif sil_type == "nosil":
      sil_index = np.where(signal_db > -self._top_db)[0]
    else:
      raise Exception("Error for sil type")

    start, end = None, None
    sil_endpoints = []
    for x in range(-1, len(signal_db)):
      if x not in sil_index and (x + 1) in sil_index:
        start = int(librosa.frames_to_samples(x + 1, self._hop_length))
      if x in sil_index and x + 1 not in sil_index:
        end = int(librosa.frames_to_samples(x, self._hop_length))
        sil_endpoints.append((start, end))
    sil_endpoints = [(_s, _e) for _s, _e in sil_endpoints if _e - _s > 1]
    return sil_endpoints

  def get_speech_endpoint(self, wav_path=None, signal=None, sr=None,
                          min_sil_dur=0.75, min_nosil_dur=0.40):
    """ get speech endpoints. input can be wav_path or (signal and sr)

    Args:
      wav_path: wav path
      signal: signal (np.array)
      sr: sample rate (int)
      min_sil_dur: Ignore sil duration which is shorter than min_sil_dur
      min_nosil_dur: Ignore nosil duration which is shorter than min_nosil_dur

    Returns:
      speech_endpoints : list as [(start1, end1), (start2, end2) ...]
    """

    if wav_path:
      signal, sr = librosa.load(wav_path, sr=16000)
    else:
      pass

    dur = len(signal) / sr
    sil_data = self._get_sil_endpoints(signal, sil_type="sil")
    sil_data = [(s / sr, e / sr) for s, e in sil_data
                if e - s > sr * min_sil_dur]
    if len(sil_data) == 0:
      return [(0.0, dur)]

    nosil_data = self._sil_to_nosil(sil_data, dur)
    nosil_data = [(s, e) for s, e in nosil_data if e - s > min_nosil_dur]
    return nosil_data

  def _sil_to_nosil(self, sil_data, dur):
    nosil_data = []
    if sil_data[0][0] > 0.001:
      nosil_data.append((0.0, sil_data[0][0]))

    for idx in range(len(sil_data) - 1):
      nosil_data.append((sil_data[idx][1], sil_data[idx + 1][0]))

    if dur - sil_data[-1][-1] > 0.001:
      nosil_data.append((sil_data[-1][-1], dur))
    return nosil_data
