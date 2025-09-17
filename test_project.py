from pathlib import Path

def test_files():
    assert Path("src/models/train_model.py").exists()
    assert Path("src/models/classifier.py").exists()
    assert Path("app.py").exists()
