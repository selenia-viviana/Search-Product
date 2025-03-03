#! bash
#!/bin/bash

set -e  

# 1 Create a virtual environment named 'env'
echo "Creating virtual environment 'env'..."
python3 -m venv env

# 2️ Activate the environment
echo "Activating virtual environment..."
source env/bin/activate

# 3️ Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# 4️ Install dependencies from requirements.txt

if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "No requirements.txt found. Skipping package installation."
fi

# 5️ Install Jupyter Notebook inside the virtual environment
echo "Installing Jupyter Notebook..."
pip install ipykernel jupyter

# 6️ Add the environment to Jupyter as a kernel
python -m ipykernel install --user --name=env --display-name "Python (env)"

# 7️ Confirm installation
echo "Setup complete! To start Jupyter Notebook, run:"
echo "source env/bin/activate && jupyter notebook"

# 8 Start microservice api
uvicorn 02_microservice:app --host 0.0.0.0 --port 8088 --reload