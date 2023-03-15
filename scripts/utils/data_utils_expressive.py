import re

import librosa
import numpy as np
import torch
from scipy.interpolate import interp1d
from sklearn.preprocessing import normalize

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dir_vec_pairs = [
    (0, 1, 0.26), # 0, spine-neck
    (1, 2, 0.22), # 1, neck-left shoulder
    (1, 3, 0.22), # 2, neck-right shoulder
    (2, 4, 0.36), # 3, left shoulder-elbow
    (4, 6, 0.33), # 4, left elbow-wrist

    (6, 8, 0.137), # 5 wrist-left index 1
    (8, 9, 0.044), # 6
    (9, 10, 0.031), # 7

    (6, 11, 0.144), # 8 wrist-left middle 1
    (11, 12, 0.042), # 9
    (12, 13, 0.033), # 10

    (6, 14, 0.127), # 11 wrist-left pinky 1
    (14, 15, 0.027), # 12
    (15, 16, 0.026), # 13

    (6, 17, 0.134), # 14 wrist-left ring 1
    (17, 18, 0.039), # 15
    (18, 19, 0.033), # 16

    (6, 20, 0.068), # 17 wrist-left thumb 1
    (20, 21, 0.042), # 18
    (21, 22, 0.036), # 19

    (3, 5, 0.36), # 20, right shoulder-elbow
    (5, 7, 0.33), # 21, right elbow-wrist

    (7, 23, 0.137), # 22 wrist-right index 1
    (23, 24, 0.044), # 23
    (24, 25, 0.031), # 24

    (7, 26, 0.144), # 25 wrist-right middle 1
    (26, 27, 0.042), # 26
    (27, 28, 0.033), # 27

    (7, 29, 0.127), # 28 wrist-right pinky 1
    (29, 30, 0.027), # 29
    (30, 31, 0.026), # 30

    (7, 32, 0.134), # 31 wrist-right ring 1
    (32, 33, 0.039), # 32
    (33, 34, 0.033), # 33

    (7, 35, 0.068), # 34 wrist-right thumb 1
    (35, 36, 0.042), # 35
    (36, 37, 0.036), # 36

    (1, 38, 0.18), # 37, neck-nose
    (38, 39, 0.14), # 38, nose-right eye
    (38, 40, 0.14), # 39, nose-left eye
    (39, 41, 0.15), # 40, right eye-right ear
    (40, 42, 0.15), # 41, left eye-left ear
]

def normalize_string(s):
    """ lowercase, trim, and remove non-letter characters """
    s = s.lower().strip()
    s = re.sub(r"([,.!?])", r" \1 ", s)  # isolate some marks
    s = re.sub(r"(['])", r"", s)  # remove apostrophe
    s = re.sub(r"[^a-zA-Z,.!?]+", r" ", s)  # replace other characters with whitespace
    s = re.sub(r"\s+", r" ", s).strip()
    return s


def remove_tags_marks(text):
    reg_expr = re.compile('<.*?>|[.,:;!?]+')
    clean_text = re.sub(reg_expr, '', text)
    return clean_text


def extract_melspectrogram(y, sr=16000):
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, power=2)
    log_melspec = librosa.power_to_db(melspec, ref=np.max)  # mels x time
    log_melspec = log_melspec.astype('float16')
    return log_melspec


def calc_spectrogram_length_from_motion_length(n_frames, fps):
    ret = (n_frames / fps * 16000 - 1024) / 512 + 1
    return int(round(ret))


def resample_pose_seq(poses, duration_in_sec, fps):
    n = len(poses)
    x = np.arange(0, n)
    y = poses
    f = interp1d(x, y, axis=0, kind='linear', fill_value='extrapolate')
    expected_n = duration_in_sec * fps
    x_new = np.arange(0, n, n / expected_n)
    interpolated_y = f(x_new)
    if hasattr(poses, 'dtype'):
        interpolated_y = interpolated_y.astype(poses.dtype)
    return interpolated_y


def time_stretch_for_words(words, start_time, speech_speed_rate):
    for i in range(len(words)):
        if words[i][1] > start_time:
            words[i][1] = start_time + (words[i][1] - start_time) / speech_speed_rate
        words[i][2] = start_time + (words[i][2] - start_time) / speech_speed_rate

    return words


def make_audio_fixed_length(audio, expected_audio_length):
    n_padding = expected_audio_length - len(audio)
    if n_padding > 0:
        audio = np.pad(audio, (0, n_padding), mode='symmetric')
    else:
        audio = audio[0:expected_audio_length]
    return audio


def convert_dir_vec_to_pose(vec):
    # vec = np.array(vec)

    if vec.shape[-1] != 3:
        vec = vec.reshape(vec.shape[:-1] + (-1, 3))

    if len(vec.shape) == 2:
        joint_pos = np.zeros((43, 3))
        for j, pair in enumerate(dir_vec_pairs):
            joint_pos[pair[1]] = joint_pos[pair[0]] + [pair[2]] * vec[j]
    elif len(vec.shape) == 3:
        joint_pos = np.zeros((vec.shape[0], 43, 3))
        for j, pair in enumerate(dir_vec_pairs):
            joint_pos[:, pair[1]] = joint_pos[:, pair[0]] + [pair[2]] * vec[:, j]
    elif len(vec.shape) == 4:  # (batch, seq, 42, 3)
        joint_pos = np.zeros((vec.shape[0], vec.shape[1], 43, 3))
        for j, pair in enumerate(dir_vec_pairs):
            joint_pos[:, :, pair[1]] = joint_pos[:, :, pair[0]] + [pair[2]] * vec[:, :, j]
    else:
        assert False

    return joint_pos

def convert_pose_seq_to_dir_vec(pose):
    if pose.shape[-1] != 3:
        pose = pose.reshape(pose.shape[:-1] + (-1, 3))

    if len(pose.shape) == 3:
        dir_vec = np.zeros((pose.shape[0], len(dir_vec_pairs), 3))
        for i, pair in enumerate(dir_vec_pairs):
            dir_vec[:, i] = pose[:, pair[1]] - pose[:, pair[0]]
            dir_vec[:, i, :] = normalize(dir_vec[:, i, :], axis=1)  # to unit length
    elif len(pose.shape) == 4:  # (batch, seq, ...)
        dir_vec = np.zeros((pose.shape[0], pose.shape[1], len(dir_vec_pairs), 3))
        for i, pair in enumerate(dir_vec_pairs):
            dir_vec[:, :, i] = pose[:, :, pair[1]] - pose[:, :, pair[0]]
        for j in range(dir_vec.shape[0]):  # batch
            for i in range(len(dir_vec_pairs)):
                dir_vec[j, :, i, :] = normalize(dir_vec[j, :, i, :], axis=1)  # to unit length
    else:
        assert False

    return dir_vec