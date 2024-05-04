# RL-Sonic-Hedgehog
## About
The Sonic the Hedgehog Reinforcement Learning project, where we trained AI agents to play through the iconic Green Hill Zone using deep reinforcement learning techniques. The goal was to teach an agent to effectively navigate the first level of Sonic, which involves complex control tasks like speed, precision, and long-term planning.

## Website
[You can watch the video results on the demo site](https://kama34.github.io/RL-Sonic-Hedgehog/)

## Installation

Follow these steps to install and configure the environment for the project:

1. **Install Python 3.7.6 using Conda**:
    ```bash
    conda install python=3.7.6
    ```
2. **Install Matplotlib 3.5.3 using Conda**:
    ```bash
    conda install matplotlib==3.5.3
    ```
3. **Install PyTorch, torchvision and torchaudio using Conda**:
    ```bash
    conda install pytorch torchvision torchaudio cpuonly -c pytorch
    ```
4. **Install OpenCV 4.2.0.34 using pip**:
    ```bash
    pip install opencv-python==4.2.0.34
    ```
5. **Install Gym Retro 0.7.1 using pip**:
    ```bash
    pip install gym-retro==0.7.1
    ```
6. **Add the Sonic The Hedgehog (Japan, Europe, Korea) file (En).gen to your project**. This file should be added to the root directory of your project.
7. **Import games using Gym Retro**:
    ```bash
    python -m retro.import ./sonic
    ```
8. **Install Gym 0.17.1 using pip**:
    ```bash
    pip install gym==0.17.1
    ```

After completing these steps, your environment should be ready to work with our code:

- To work with the dqn algorithm, you need to run [DQN Script](./sonic/sonic_dqn.ipynb)
- To work with the ddqn algorithm, you need to run [DDQN Script](./sonic/sonic_ddqn.ipynb)

## Developers
- Kamyshnikov Dmitrii :
  - [GitHub](https://github.com/kama34)
  - [Email](mailto:d.kamyshnikov.offer@yandex.ru)
  - [Telegram](https://t.me/kama_34)
 
- Sofya Polozova :
   - [GitHub](https://github.com/Sofapss)
   - [Email](mailto:sofya_polozova@mail.ru)
   - [Telegram](https://t.me/Sofa_pss)
