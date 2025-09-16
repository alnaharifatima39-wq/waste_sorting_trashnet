# waste_sorting_trashnet

A waste-sorting project that uses (dataset / model) to classify trash types and help automate recycling.

**Key Features:**
- High-accuracy waste classification using a trained model.
- Supports multiple waste categories for practical sorting applications.
- Easy integration with automation systems for smart waste management.
- Provides real-time predictions and visual feedback.

**Goals:**
- Reduce human error in waste sorting.
- Promote efficient recycling practices.
- Provide a foundation for smart waste management solutions.


# Team Members
| No. | Name         | Role           | Contribution                              |
|-----|--------------|----------------|-------------------------------------------|
| 1   |Fatima Osamah | Lead Developer | Integration,README, training              |
| 2   |Maha Jamal    | Data Engineer  | Dataset collection & preprocessing        |
| 3   |Hala Hesham   | ML Engineer    | Model development, testing                |
| 4   |Afaf Abdulbaqi| UI dev Engineer| gui and web interface/ UI script          |

# Project Requirements / Config
This project uses `uv`. The project config lives in `pyproject.toml` and `uv.lock`.

## Installation and Setup

### Prerequisites
- Python 3.11
- UV package manager

### Installation Steps
1. Clone the repository:
   ```bash
   git clone git clone https://github.com/yourusername/waste_sorting_trashnet.git
   cd waste_sorting_trashnet

2. Install dependencies using UV:
 ```bash
uv sync
 ```
3. Run the project:
 ```bash
uv run python main.py 
  ```
# Project Structure
```
waste_sorting_trashnet/
├── README.md
├── pyproject.toml     # project config (uv / dependencies)
├── uv.lock            # generated after `uv sync`
├── main.py            # live local camera detection
├── gui.py             # User interface local
├── app.py             # Web user interface (Streamlit)
├── src/
│   ├── data/          # Data processing modules
│   │   └── check_dataset.py
|   |   └── check_split.py
|   |   └── prepare_dataset.py
│   ├── models/        # ML model implementations
│   │   └── classifier.py
|   |   └── train_model.py
|   |   └── tta_predict.py
│   ├── utils/
│   |   
|   |
|   |
|   └── app/           
|       └──gui_app.py
├── data/
|   ├── trashnet 
│   |   └── cardboard/
│   |   └── glass/
│   |   └── metal/
│   |   └── paper/
│   |   └── plastic/
│   |   └── trash/
│   |
│   ├── trashnet_split/ # Pre-split dataset
│       └── train/
│       └── test/
│       └── val/
|── notebooks/         # Jupyter notebooks
    ├── waste_sorting_demo.ipynb
```
# Usage

## Basic Usage
```python
uv run python train_model.py

## Load and train model

model.train("data/trashnet_split/train")
```
# Running Experiments
```bash
uv run python load_and_use_model.py

```
# Running app
```bash
uv run python gui.py #internal gui
uv run python app.py #streamlit web UI

```
# Results and Performance

## Results
- Model Accuracy: 90%
- Training Time: 277.75 minutes
- Key Findings:
   1.Model Choice: ResNet18 was adapted for 6 waste categories (cardboard, glass, metal, paper, plastic, trash).

   2.Training Time: The model trained for 30 epochs in ~277.45 minutes on [CPU].

   3.Accuracy: Achieved ~90% accuracy on the test set, showing good generalization.

   4.Confusion Matrix Insights: Most misclassifications occurred between visually similar classes (e.g., cardboard vs. paper, plastic vs. glass).

   5.Custom Image Test: The classifier correctly predicted uploaded images, with clear probability distributions.

   6.Deployment Readiness: Both a desktop GUI and a Streamlit web app were implemented, allowing real-time classification from webcam or uploaded images.

# Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Commit changes: `git commit -m 'Add feature'`
5. Push to branch: `git push origin feature-name`
6. Submit a pull request








