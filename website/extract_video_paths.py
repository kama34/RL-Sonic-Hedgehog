import os
import json

path_ddqn = '..\\sonic\\record_ddqn\\video'
path_dqn = '..\\sonic\\record_dqn\\video'

mp4_files_ddqn = [os.path.relpath(os.path.join(path_ddqn, f)) for f in os.listdir(path_ddqn) if f.endswith('.mp4')]
mp4_files_dqn = [os.path.relpath(os.path.join(path_dqn, f)) for f in os.listdir(path_dqn) if f.endswith('.mp4')]

video_paths = {
    'DQN': mp4_files_dqn,
    'DDQN': mp4_files_ddqn
}

with open('videoPaths.js', 'w') as file:
    file.write('const videoPaths = ' + json.dumps(video_paths) + ';')
