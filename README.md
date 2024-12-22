# model-servers
Repository for machine learning and deep learning models used for anomaly detection in cybersecurity time-series data.

model-servers/
├── data/               # Raw and processed datasets
│   ├── sampled_data/   # Sampled data for testing
│   ├── labeled_data/   # Labeled data for supervised learning
├── models/             # Trained models or model configurations
├── notebooks/          # Jupyter notebooks for experimentation
├── src/                # Source code
│   ├── preprocess/     # Data preprocessing scripts
│   │   ├── unlabeledPreprocess.py
│   │   ├── labeledPreprocess.py
│   ├── models/         # Model definitions
│   │   ├── isolation_forest.py
│   │   ├── lstm_rnn.py
│   │   ├── gru.py
│   │   ├── transformer.py
│   ├── decision/       # Decision methods for anomaly detection
│   │   ├── majority_voting.py
│   │   ├── advanced_decision.py
│   ├── main.py         # Main script to load data, run models, and make decisions
├── README.md           # Documentation
├── requirements.txt    # Python dependencies
├── .gitignore          # Files to exclude from version control


## Preprocessing Scripts
- `src/unlabeledPreprocess.py`: Preprocesses LANL dataset for unsupervised learning. 
  Run the script using:
  ```bash
  python src/unlabeledPreprocess.py

- `src/labeledPreprocess.py`: Preprocesses LANL dataset for supervised and semi-supervised learning. 
  Run the script using:
  ```bash
  python src/labeledPreprocess.py
