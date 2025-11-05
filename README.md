# Install dependencies:
pip install -r requirements.txt

# Training the model
python -m datatraining.results.model train --dataset "datatraining/dataset" --model-path "datatraining/results/trained_model.joblib"

# Start the app:
python -m backend.app

# For analysis
    - python model_analysis.py --model trained_model.joblib
    - python image_analysis.py --image "C:\Users\Parthiv\Downloads\VS Code\School Related\COE843-SmartTrashSorter\datatraining\dataset\compost\biological_5.jpg" --model "trained_model.joblib" --out "analysis_single_bio5"

# Data Trained From
https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2\
