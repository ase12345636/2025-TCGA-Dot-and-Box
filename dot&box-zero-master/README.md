# Board of Dot & Box
- Board Shape = 5 * 5
- Content of Board
    - blue(first move): -1
    - red(second move): 1
    - vertex: 5
    - legal edge: 0
    - empty box: 8
    - blue edge: -1
    - red edge: 1
    - blue box: 7
    - red box: 9

# Model Architecture

- AlphaZero
    - Input Shape: 17 * 17 * 9
    - Policy-Net: ResNet
    - Value-Net: CNN

- Training Method
    - Epoch: 128
    - Batch Size: 1024
    - Momentum: 0.9
    - L2 Weight: 1e-4
    - Learning Rate: 1e-2

- MCTS
    - Number of Simulation: 400, or 800, or 1600
    - Noise_Alpha = 0.5
    - Noise_Weight = 0.25

# Usage

- Train Model
    - python dot_and_box.py -l
    - python dot_and_box.py --learning-loop

- Play with Human
    - python dot_and_box.py -m
    - python dot_and_box.py --play-with-human
    - Decide First move, or Second Move
        - First move: -1
        - Second Move: 1

- See Argumment
    - python dot_and_box.py --help

- Show Learning Curve
    - python log/plot.py

# Training Log
- Iteration 1 ~ 25
    - simulations_num: 400

- Iteration 25 ~ 40
    - simulations_num: 800

- Iteration 40 ~ 47
    - simulations_num: 400

---

- ADD MODEL's PARAMETER

---

- Iteration 1 ~ 
    - simulations_num: 400