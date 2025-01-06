import os

# Run the Jupyter Notebook directly
notebook_name = "WHL_Win_Predictor.ipynb"

# Execute the notebook in-place
os.system(f'jupyter nbconvert --to notebook --execute {notebook_name}')