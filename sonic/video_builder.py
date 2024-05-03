import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import numpy as np


def run_command(path):
    try:
        subprocess.run(['python', '-m', 'retro.scripts.playback_movie', path], timeout=20)
    except subprocess.TimeoutExpired:
        pass


def get_top(path_scores, path_record, top=None):
    # Замените на путь к вашей папке
    directory = path_record

    if top is None:
        files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.bk2')]
    else:
        # Загрузка файла
        data = np.load(path_scores)

        # топ 100
        top_100 = np.partition(data, -top)[-top:]
        # индексы топа 10
        top_100_index = np.argpartition(data, -top)[-top:]

        # Создаем словарь
        top_dict = {index: value for index, value in zip(top_100_index, top_100)}

        # Сортируем словарь по значениям
        sorted_dict = dict(sorted(top_dict.items(), key=lambda item: item[1], reverse=True))

        # Получаем список всех файлов .bk2
        files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.bk2')]

        # Отфильтровываем список файлов, чтобы оставить только те, которые указаны в словаре sorted_dict
        files = [f for f in files if int(os.path.splitext(os.path.basename(f))[0].split('-')[-1]) in sorted_dict.keys()]

    return files


path_scores = 'E:\GitHub\RL-Sonic-Hedgehog\sonic\checkpoints\ddqn\scores\scores_dqn20600.npy'
path_record = 'E:\GitHub\RL-Sonic-Hedgehog\sonic\\record_ddqn'

files = get_top(path_scores, path_record, top=100)

# Запускаем команду для каждого файла
with ThreadPoolExecutor(max_workers=10) as executor:
    list(tqdm(executor.map(run_command, files), total=len(files)))

# Перемещаем все файлы .mp4 в папку video
mp4_files = [f for f in os.listdir(path_record) if f.endswith('.mp4')]
for file in tqdm(mp4_files):
    old_path = os.path.join(path_record, file)
    new_path = os.path.join(path_record, 'video', file)
    os.rename(old_path, new_path)
