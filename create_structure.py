import os

# Define folder and file structure
structure = {
    "data": ["raw_data.csv", "processed_data.csv"],
    "notebooks": [
        "01_EDA.py",
        "02_Feature_Engg.py",
        "03_Modeling.py",
        "04_Backtesting.py"
    ],
    "utils": ["indicators.py"],
    "models": ["xgboost_model.pkl"],
    "reports": [],
    "": ["README.md", "requirements.txt", ".gitignore"]
}

# Create folders and files
for folder, files in structure.items():
    folder_path = os.path.join("project_root", folder)
    os.makedirs(folder_path, exist_ok=True)
    for file in files:
        file_path = os.path.join(folder_path, file)
        with open(file_path, "w") as f:
            f.write("")  # Create empty file

print("📁 Project structure created successfully!")
