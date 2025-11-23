# 🏥 Diabetes Prediction - Multi-Model ML Pipeline with CI/CD

[![CI/CD Pipeline](https://github.com/ahtalebi/ml-model-comparison/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/ahtalebi/ml-model-comparison/actions)

Advanced ML project demonstrating **automated model comparison**, **Docker deployment**, and **complete CI/CD pipeline**.

---

## 🎯 Project Overview

This project:
- ✅ Trains **4 different models** automatically
- ✅ Compares performance and **selects the best model**
- ✅ Creates **interactive visualizations**
- ✅ Provides **Flask API** for predictions
- ✅ **Dockerized** for easy deployment
- ✅ **Full CI/CD** with GitHub Actions
- ✅ **Live dashboard** on GitHub Pages

---

## 📊 Dataset

**Diabetes Dataset** (from scikit-learn)
- 442 samples
- 10 features: age, sex, BMI, blood pressure, etc.
- Target: Disease progression (quantitative measure)

**Features:**
- `age`: Age in years
- `sex`: Gender
- `bmi`: Body mass index
- `bp`: Average blood pressure
- `s1-s6`: Six blood serum measurements

---

## 🤖 Models Compared

| Model | Description |
|-------|-------------|
| **Linear Regression** | Simple baseline model |
| **Ridge Regression** | L2 regularization |
| **Random Forest** | Ensemble of decision trees |
| **XGBoost** | Gradient boosting (usually wins!) |

**Auto-selection:** Best model is automatically chosen based on validation R² score.

---

## 🏗️ Project Structure
```
ml-model-comparison/
├── .github/workflows/
│   └── ci-cd.yml              # CI/CD pipeline
├── api/
│   ├── app.py                 # Flask API
│   └── Dockerfile             # Docker configuration
├── data/
│   └── diabetes.csv           # Dataset
├── models/
│   ├── linear_regression.pkl  # Trained models
│   ├── ridge.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── best_model.pkl         # Auto-selected best model
│   └── comparison_results.json
├── plots/
│   └── model_comparison.png   # Performance visualizations
├── src/
│   ├── load_data.py           # Load dataset
│   ├── train_models.py        # Train all models
│   └── visualize.py           # Create plots
├── tests/
│   └── test_pipeline.py       # Automated tests
├── requirements.txt
├── index.html                 # Web dashboard
└── README.md
```

---

## 🚀 Quick Start

### **Local Setup**
```bash
# 1. Clone repository
git clone https://github.com/ahtalebi/ml-model-comparison.git
cd ml-model-comparison

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Load dataset
python src/load_data.py

# 5. Train models
python src/train_models.py

# 6. Create visualizations
python src/visualize.py

# 7. Run tests
python tests/test_pipeline.py
```

---

## 🐳 Docker Deployment
```bash
# Build Docker image
docker build -t diabetes-api ./api

# Run container
docker run -p 5000:5000 diabetes-api

# API is now available at http://localhost:5000
```

---

## 🌐 API Usage

### **Start Flask API**
```bash
cd api
python app.py
```

### **Make Predictions**
```bash
# Using curl
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 0.05,
    "sex": 0.05,
    "bmi": 0.06,
    "bp": 0.02,
    "s1": -0.04,
    "s2": -0.03,
    "s3": -0.02,
    "s4": -0.03,
    "s5": 0.04,
    "s6": 0.01
  }'

# Response:
{
  "prediction": 206.3,
  "model_used": "xgboost",
  "timestamp": "2024-11-23T10:30:00"
}
```

### **Python Client**
```python
import requests

data = {
    "age": 0.05,
    "bmi": 0.06,
    "bp": 0.02,
    # ... other features
}

response = requests.post(
    "http://localhost:5000/predict",
    json=data
)

print(response.json())
```

---

## 📊 Model Performance

After running, check `models/comparison_results.json` for detailed metrics:
```json
{
  "xgboost": {
    "test": {
      "r2": 0.479,
      "rmse": 52.3,
      "mae": 41.2
    },
    "training_time": 0.15
  },
  "best_model": "xgboost"
}
```

View visualizations in `plots/model_comparison.png`

---

## 🔄 CI/CD Pipeline

**Triggers on every push to main:**
```
1. ✅ Install dependencies
2. ✅ Load dataset
3. ✅ Train all 4 models
4. ✅ Run tests (7 tests)
5. ✅ Generate visualizations
6. ✅ Build Docker image
7. ✅ Deploy to GitHub Pages
8. ✅ Send notifications
```

**View pipeline:** [GitHub Actions](https://github.com/ahtalebi/ml-model-comparison/actions)

---

## 🌐 Live Dashboard

Visit the live dashboard: **https://ahtalebi.github.io/ml-model-comparison/**

Features:
- Model performance comparison
- Interactive charts
- API playground
- Download trained models

---

## 🧪 Running Tests
```bash
# Run all tests
python tests/test_pipeline.py

# Expected output:
test_1_data_exists ... ok
test_2_data_valid ... ok
test_3_models_exist ... ok
test_4_results_exist ... ok
test_5_best_model_selected ... ok
test_6_model_performance ... ok
test_7_plots_created ... ok

Ran 7 tests in 0.5s
OK
```

---

## 📈 Performance Metrics

Typical results:

| Model | Test R² | RMSE | Training Time |
|-------|---------|------|---------------|
| Linear Regression | 0.452 | 53.7 | 0.01s |
| Ridge | 0.454 | 53.6 | 0.01s |
| Random Forest | 0.465 | 53.1 | 0.25s |
| **XGBoost** 🏆 | **0.479** | **52.3** | 0.15s |

---

## 🛠️ Tech Stack

- **ML**: scikit-learn, XGBoost
- **API**: Flask
- **Containerization**: Docker
- **CI/CD**: GitHub Actions
- **Visualization**: Matplotlib, Seaborn
- **Testing**: unittest
- **Deployment**: GitHub Pages

---

## 📚 What You'll Learn

1. ✅ **Multi-model comparison** - Train and compare multiple models
2. ✅ **Automated selection** - Let the pipeline pick the best model
3. ✅ **CI/CD practices** - Full automated pipeline
4. ✅ **API development** - Flask REST API
5. ✅ **Containerization** - Docker packaging
6. ✅ **Testing** - Automated test suite
7. ✅ **Visualization** - Performance dashboards

---

## 🎯 Next Steps

- [ ] Add more models (Neural Networks, SVM)
- [ ] Implement hyperparameter tuning
- [ ] Add feature engineering
- [ ] Deploy to cloud (Heroku, AWS, Azure)
- [ ] Add monitoring and logging
- [ ] Create Streamlit dashboard

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

---

## 📄 License

MIT License - feel free to use for learning!

---

## 📧 Contact

- **GitHub**: [@ahtalebi](https://github.com/ahtalebi)
- **Project Link**: https://github.com/ahtalebi/ml-model-comparison

---

## 🙏 Acknowledgments

- Dataset: scikit-learn diabetes dataset
- Inspiration: Real-world ML deployment practices
- Tools: GitHub Actions, Docker, Flask

---

**Built with ❤️ for learning ML/DevOps practices**

*Last updated: November 23, 2024*
