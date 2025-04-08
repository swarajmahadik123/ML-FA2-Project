# Network Intrusion Detection Using Machine Learning

## Overview

This project focuses on analyzing the CICIDS dataset to develop a robust machine learning-based intrusion detection system. The study explores various attack types (e.g., DoS, DDoS, PortScan) and utilizes advanced techniques such as feature correlation analysis, dimensionality reduction, and machine learning models to identify and classify network intrusions effectively.

The key objectives of this project are:

- To analyze the CICIDS dataset for patterns in network traffic.
- To evaluate the performance of machine learning models like KNN, Random Forest, and others for intrusion detection.
- To provide insights into feature importance and attack-specific behaviors.

---

## Features

- **Dataset**: CICIDS2017 dataset with over 2.8 million records and 79 features.
- **Machine Learning Models**: Implemented models include KNN, Random Forest, SVM, CNN, and more.
- **Performance Metrics**: Accuracy, precision, recall, F1-score, and training time are evaluated.
- **Visualizations**: Includes attack distribution plots, correlation heatmaps, MDS visualizations, and more.

---

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/swarajmahadik123/ML-FA2-Project.git

   ```

2. Install required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the CICIDS2017 dataset from the official source and place it in the `data/` directory.

---

## Usage

### Preprocessing

Run the preprocessing script to clean and prepare the dataset:

```bash
python preprocess.py
```

### Model Training

Train the machine learning models by running:

```bash
python train_models.py
```

### Evaluation

Evaluate model performance using:

```bash
python evaluate_models.py
```

### Visualization

Generate visualizations for analysis:

```bash
python visualize_results.py
```

---

## Results

### Key Findings

- **KNN** achieved a mean cross-validation accuracy of **98%**.
- **Random Forest** showed superior performance with an accuracy of **99.83%**.
- Feature correlation analysis revealed strong relationships between timing-based features for detecting DoS attacks.

### Visualizations

- Attack distribution bar chart.
- Correlation heatmap of network flow features.
- Multi-dimensional scaling (MDS) plot for attack separation.

---

## Team Members

This project was collaboratively developed by:

- **Swaraj Nandkishor Mahadik** (122B1B159)
- **Anuj Vijay Loharkar** (122B1B154)
- **Aishwarya Marathe** (122B1B170)

---

## Acknowledgments

We would like to express our gratitude to the authors of the CICIDS dataset and other research studies that guided this work. Their contributions have been invaluable in enriching our understanding of network intrusion detection systems.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## References

1. Canadian Institute for Cybersecurity. CICIDS2017 Dataset Documentation.
2. Additional references related to intrusion detection research can be found in our [research paper](./research_paper.pdf).

---

