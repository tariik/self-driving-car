# TFM_DRL_CARLA

## Description
This project implements and evaluates Deep Reinforcement Learning (DRL) algorithms for autonomous driving in the CARLA simulator. It compares variants such as DQN and Dueling DQN, integrating Convolutional Neural Networks (CNNs) to process visual inputs and train agents to make decisions in dynamic urban environments.

## Features
- **DRL Agents:** Implementation of DQN and Dueling DQN algorithms.
- **Visual Processing:** Use of CNNs for image interpretation.
- **Integration with CARLA:** Adapter and communication with the CARLA simulator.
- **Training and Evaluation:** Scripts for running, monitoring, and validating agent performance.
- **Visualization:** Tools to plot results and analyze trajectories.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/TFM_DRL_CARLA.git
   cd TFM_DRL_CARLA
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   # Windows: venv\Scripts\activate
   # macOS/Linux: source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure the config.yaml file according to your needs (training parameters, paths, etc.).

## Usage
To start training the agent:
```bash
python src/main.py
```
To evaluate the model performance:
```bash
bash scripts/run_evaluation.sh
```

Interactive examples and exploratory analysis can be found in the notebooks/ directory.

## Repository Structure
- src/: Main source code.
- agents/: Implementation of agents (DQN, Dueling DQN).
- env/: Integration and adapter for the CARLA simulator.
- models/: Definition of neural network architectures.
- training/: Scripts and routines for training and evaluation.
- utils/: Utility functions (logging, visualization, reward computation).
- notebooks/: Jupyter Notebooks for experimentation and analysis.
- docs/: Project documentation.
- experiments/: Logs, checkpoints, and experimental results.
- scripts/: Automation and execution scripts.

## Contributing
To contribute:
- Fork the repository.
- Create a branch for your changes.
- Submit a Pull Request with a detailed description of your improvements.
- Report any issues or suggestions in the repository.

## License
This project is licensed under the MIT License (or your chosen license).

## Contact
For more information, questions, or suggestions, please contact [khalfaoui.t@hotmail.com] or visit your GitHub profile.
