# Commodities Trading through Deep Policy Gradient Methods

This repository contains the Python/PyTorch implementation of the research paper:  
**"Commodities Trading through Deep Policy Gradient Methods"**  
by Jonas Hanetho.  
[Read the Paper](https://arxiv.org/pdf/2309.00630v1)

## Overview

Algorithmic trading leverages computational methods to make trading decisions, often outperforming traditional approaches. This research explores the application of Deep Reinforcement Learning (DRL) techniques for commodities trading, specifically focusing on natural gas futures. The paper formulates the trading problem as a continuous, discrete-time stochastic dynamical system and introduces novel techniques to optimize trading strategies while accounting for transaction costs and risk sensitivity.

The key highlights of the paper are:
1. **Time-Discretization Scheme**: A volatility-adaptive subsampling method enhances the statistical properties of financial time series data.
2. **Policy Gradient Algorithms**: Two DRL approaches are introduced:
   - Actor-Based (Policy Optimization directly)
   - Actor-Critic-Based (Policy Optimization with a value estimation component)
3. **Neural Network Models**:
   - Convolutional Neural Networks (CNNs)
   - Long Short-Term Memory Networks (LSTMs)
4. **Customizable Risk Profiles**: A hyperparameter in the reward function allows for tuning the risk sensitivity of trading agents.
5. **Performance**: The approach demonstrates an **83% improvement in Sharpe ratio** over a buy-and-hold baseline, with CNN-based models slightly outperforming LSTM-based models.

## Repository Structure

This repository provides the implementation of the proposed DRL framework for commodities trading. Below is an outline of the main components:

```
├── data/
│   ├── historical_prices.csv      # Example dataset with natural gas futures prices
│   └── processed_data.pkl         # Preprocessed and subsampled data
├── models/
│   ├── cnn_policy.py              # CNN-based policy network implementation
│   ├── lstm_policy.py             # LSTM-based policy network implementation
├── algorithms/
│   ├── actor_based.py             # Actor-based policy gradient algorithm
│   ├── actor_critic_based.py      # Actor-Critic-based policy gradient algorithm
├── utils/
│   ├── data_preprocessing.py      # Volatility-based time discretization and data preprocessing
│   ├── evaluation.py              # Backtesting and performance evaluation utilities
│   ├── plotting.py                # Tools for generating performance charts
├── main.py                        # Main entry point for training and evaluation
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

## Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.8 or higher
- PyTorch 1.12 or higher
- Other dependencies specified in `requirements.txt`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/commodities-trading-drl.git
   cd commodities-trading-drl
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Data Preparation

1. Place your historical commodities price data in the `data/` directory (e.g., `historical_prices.csv`).
2. Run the preprocessing script to prepare the data:
   ```bash
   python utils/data_preprocessing.py
   ```

#### Training

To train a trading agent using the actor-based approach with a CNN policy:
```bash
python main.py --algorithm actor_based --model cnn --epochs 100
```

To train with the actor-critic method using an LSTM policy:
```bash
python main.py --algorithm actor_critic_based --model lstm --epochs 100
```

#### Backtesting and Evaluation

After training, evaluate the agent's performance using the backtesting framework:
```bash
python utils/evaluation.py --model_path models/saved_model.pth --data_path data/processed_data.pkl
```

#### Visualization

You can generate performance metrics and visualizations (e.g., equity curves, drawdowns) using:
```bash
python utils/plotting.py --results_path results/trading_metrics.json
```

## Key Components

1. **Data Preprocessing**:
   - Implements the volatility-adaptive time-discretization scheme.
   - Normalizes and subsamples historical price data based on market conditions.

2. **Policy Gradient Algorithms**:
   - **Actor-Based**: Directly optimizes the policy to maximize the expected return.
   - **Actor-Critic-Based**: Combines policy optimization with a value function to improve stability.

3. **Neural Network Architectures**:
   - **CNN**: Captures spatial dependencies in price data.
   - **LSTM**: Models temporal dependencies in sequential price data.

4. **Reward Function**:
   - Incorporates transaction costs and risk sensitivity.
   - A hyperparameter allows for customizing the agent's risk-return tradeoff.

5. **Backtesting**:
   - Evaluates the performance of trained agents on historical data.
   - Outputs metrics such as Sharpe ratio, cumulative returns, and drawdowns.

## Experimental Results

The DRL framework was tested on front-month natural gas futures. Key findings include:
- **Sharpe Ratio Improvement**: An 83% increase compared to the buy-and-hold strategy.
- **Model Performance**:
  - CNN-based models slightly outperformed LSTM-based models.
  - Actor-based approaches were more effective than actor-critic-based methods.

## Contributing

Contributions are welcome. If you'd like to contribute, please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See `LICENSE` for more details.

## References

For more details, please refer to the original paper:  
[Commodities Trading through Deep Policy Gradient Methods](https://arxiv.org/pdf/2309.00630v1) by Jonas Hanetho.