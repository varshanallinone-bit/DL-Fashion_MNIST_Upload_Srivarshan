# Fashion-MNIST CNN: Train + Deploy on Render

## A. Train in Google Colab
1. Open `training/Fashion_MNIST_CNN_Train_and_Save.ipynb`.
2. Run the training cell.
3. The notebook creates `fashion_cnn_model.keras`.
4. Run the download cell and download that model file.

## B. Prepare the deployment repository
Copy everything from the `deployment/` folder into a new GitHub repository.

Then place your downloaded `fashion_cnn_model.keras` in the repository root.

Final repository:
- .python-version
- app.py
- fashion_cnn_model.keras
- index.html
- requirements.txt
- css/style.css
- js/script.js

## C. Render settings
Create a Render Web Service from the GitHub repository.

Language: Python 3
Branch: main
Root Directory: blank
Build Command:
    pip install -r requirements.txt

Start Command:
    uvicorn app:app --host 0.0.0.0 --port $PORT

Health Check Path:
    /health

## D. Test
Open:
- https://YOUR-SERVICE.onrender.com/health
- https://YOUR-SERVICE.onrender.com
- https://YOUR-SERVICE.onrender.com/docs

Important: Fashion-MNIST uses 28x28 grayscale clothing images on dark backgrounds. Real-world clothing photographs may produce poor predictions because they differ from the training data.
