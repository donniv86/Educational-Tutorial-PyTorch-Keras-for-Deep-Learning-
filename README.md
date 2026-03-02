# AI for Healthcare & Life Sciences: Educational Tutorials

This repository is a hands-on, beginner-friendly resource for students and professionals who want to learn how to apply AI and machine learning to healthcare and life sciences problems.

## What You'll Find
- **Heart Disease Prediction**: End-to-end notebooks using real-world data
- **Multi-Framework Tutorials**: PyTorch, Keras, TensorFlow, and scikit-learn
- **Model Comparison**: Cross-validation, statistical analysis, and robust evaluation
- **Best Practices**: Reproducible research, model saving/loading, and deployment basics
- **Educator-Style Explanations**: Clear, step-by-step guidance for learners

## Getting Started
1. **Clone the repo:**
   ```bash
   git clone https://github.com/donniv86/Educational-Tutorial-PyTorch-Keras-for-Deep-Learning-
   cd Educational-Tutorial-PyTorch-Keras-for-Deep-Learning-
   ```
2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Open the notebooks in Jupyter or VS Code and start learning!**

## Notebooks Included
- `multi_framework_heart_disease.ipynb`: Compare PyTorch, Keras, and TensorFlow on heart disease data
- `model_comparison_inferential_stats.ipynb`: Model comparison with statistical plots and tests
- `model_crossval_external_test.ipynb`: Cross-validation, model selection, and external test evaluation

## Python Crash Course for Healthcare Professionals and Students

If you are new to Python, start with the `python_crash_course_healthcare.ipynb` notebook. This step-by-step crash course will take you from zero to hero, covering Python basics, must-know features, and essential data analysis skills—all in a healthcare context.

**Topics covered:**
- Python installation and setup
- Variables, data types, operators
- Control flow (if, loops)
- Functions, modules
- Lists, tuples, dictionaries
- File I/O (text, CSV)
- NumPy, pandas, matplotlib basics
- Data cleaning, statistics, and visualization
- Exporting results

**Who is this for?**
- Healthcare professionals and students with no prior coding experience
- Anyone looking to learn Python for data analysis in healthcare

**How to use:**
1. Follow the setup instructions above to install Python and required libraries.
2. Open the notebook in Jupyter or VS Code and run each cell in order.
3. Use the notebook as a reference for your own healthcare data projects.

## Who Is This For?
- Students with basic Python knowledge
- Anyone interested in AI for healthcare, life sciences, or medical data

## License
MIT

## Technical Information

- **Python Version:** 3.11+ recommended
- **Main Libraries:**
  - `numpy`, `pandas`: Data manipulation and analysis
  - `matplotlib`, `seaborn`: Visualization
  - `scikit-learn`: Classical machine learning models and utilities
  - `torch` (PyTorch), `tensorflow`, `keras`: Deep learning frameworks
  - `lifelines`: Survival analysis (if used)
  - `flask`: API deployment
  - `mlflow`, `tensorboard`: Experiment tracking (optional)
- **Notebook Environment:** Jupyter Notebook or VS Code (with Jupyter extension)
- **Reproducibility:** Use virtual environments and requirements.txt for consistent setup. Set random seeds in notebooks for reproducible results.
- **Hardware:** Deep learning notebooks may require a GPU for faster training (optional; CPU is sufficient for small datasets).
- **Experiment Tracking:** For advanced users, MLflow or TensorBoard can be used to log parameters, metrics, and model artifacts.
- **Deployment:** Flask notebook demonstrates how to serve models as an API. For production, consider Docker and cloud platforms (AWS, Azure, GCP).

---

# Project Structure

This repository is organized for step-by-step learning. Each folder contains notebooks and data for a specific stage:

```
Heart_disease/
│
├── 0_python_basics/
│   └── python_crash_course_healthcare.ipynb
│
├── 1_data_preprocessing/
│   ├── patient_records.csv
│   ├── cleaned_patient_records.csv
│   └── bp_by_patient.png
│
├── 2_classical_ml/
│   └── model_comparison_inferential_stats.ipynb
│
├── 3_deep_learning/
│   └── heart_disease_deep_learning.ipynb
│   └── multi_framework_heart_disease.ipynb
│
├── 4_evaluation/
│   └── model_crossval_external_test.ipynb
│
├── 5_deployment/
│   └── flask_deployment_guide.ipynb
│
├── saved_models/
│   └── (model files)
│
├── requirements.txt
├── README.md
└── .gitignore
```

- Work through the folders in order for a complete learning path.
- Each notebook contains explanations and code for its topic.
- Data files are grouped for easy access.
- Deployment and advanced topics are in their own folders.
