# 🗂️ Waste Sorting AI Project — Command Cheat Sheet

This cheat sheet lists **all commands** you need to run, train, test, and deploy the project.  
Both **uv run** and plain **python** equivalents are included.  

---

## ⚙️ Environment Setup

### Using uv (recommended)
```bash
uv venv --python 3.11 .venv     # create virtual environment
uv sync                         # install dependencies
uv run python --version         # check Python version inside env
uv run pip list                 # check installed packages
```

### Using venv + pip
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # PowerShell activation (Windows)
pip install -r requirements.txt
python --version
```

---

## 📂 Dataset Preparation

Split TrashNet dataset into `train/val/test`:

```bash
uv run python src/data/prepare_dataset.py --input data/trashnet --output data/trashnet_split
# OR
python src/data/prepare_dataset.py --input data/trashnet --output data/trashnet_split
```

---

## 🧠 Model Training

Train the ResNet18 model:

```bash
uv run python src/models/train_model.py --data-dir data/trashnet_split --epochs 30 --batch-size 32
# OR
python src/models/train_model.py --data-dir data/trashnet_split --epochs 10 --batch-size 32
```

---

## ✅ Evaluation

Inside Jupyter or CLI:
```bash
python src/models/train_model.py --data-dir data/trashnet_split --epochs 5 --batch-size 32
```

---

## 🖥️ Desktop GUI (Tkinter)

```bash
uv run python gui_app.py
# OR
python gui_app.py
```

---

## 🌐 Web App (Streamlit)

```bash
uv run streamlit run app.py
# OR
streamlit run app.py
```

---

## 🚀 One-Line Launcher

Choose GUI or Web:

```bash
# Web app (default)
python app.py
# OR
uv run python app.py

# Desktop GUI
python gui.py 
# OR
uv run python gui.py
```

---

## 📒 Jupyter Notebook

```bash
uv run jupyter notebook waste_sorting_demo.ipynb
# OR
jupyter notebook waste_sorting_demo.ipynb
```

---

## 🔀 Git Workflow

Merging feature branches into develop and pushing:

```bash
git checkout develop
git merge feature/Maha-data-prepare
git merge feature/Afaf-app-visualization
git merge feature/Hala-model-dev
git push origin develop
```

Merge develop into main and force push workspace:

```bash
git checkout main
git merge develop
git add .
git commit -m "Push entire workspace with merged features"
git push origin main --force
```
