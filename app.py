from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from traning import model_training
import pickle
from pred_variables import variables_dtypes


app = FastAPI()

# Placeholder for the trained model
trained_model = None

@app.post("/train-model/")
def train_model():
    global trained_model

    try:
        trained_model = model_training()
        return JSONResponse(content={"message": "Model trained successfully"}, status_code=200)
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Failed to train model. Error: {str(e)}")

@app.post("/predict/")
def predict(data: variables_dtypes):
    data = data.dict()
    n_days = int(data['N_Days'])
    drug = int(data['Drug'])
    age = int(data['Age'])
    sex = int(data['Sex'])
    Ascites = int(data['Ascites'])
    Hepatomegaly = int(data['Hepatomegaly'])
    Spiders = int(data['Spiders'])
    Edema = int(data['Edema'])
    Bilirubin = float(data['Bilirubin'])
    Cholesterol = float(data['Cholesterol'])
    Albumin = float(data['Albumin'])
    Copper = float(data['Copper'])
    Alk_Phos = float(data['Alk_Phos'])
    SGOT = float(data['SGOT'])
    Tryglicerides = float(data['Tryglicerides'])
    Platelets = float(data['Platelets'])
    Prothrombin = float(data['Prothrombin'])
    Stage = float(data['Stage'])

    with open('model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    if loaded_model is None:
        return JSONResponse(content={"message": "Model not trained yet. Please train the model first."}, status_code=400)

    try:
        # Your prediction logic goes here using the trained_rf_model
        prediction_result = trained_model.predict([[n_days,drug,age,sex,Ascites,Hepatomegaly,Spiders,Edema,Bilirubin,Cholesterol,Albumin,Copper,Alk_Phos,SGOT,Tryglicerides,Platelets,Prothrombin,Stage]])
        # Convert the NumPy array to a serializable Python list
        prediction_result_list = prediction_result.tolist()
        x = None
        if (prediction_result_list[0] == 0):
            return JSONResponse(content={"Death": prediction_result_list[0]}, status_code=200)
        elif (prediction_result_list[0] == 1):
            return JSONResponse(content={"Censoried": prediction_result_list[0]}, status_code=200)
        elif (prediction_result_list[0] == 2):
            return JSONResponse(content={"Censoried due to liver": prediction_result_list[0]}, status_code=200)

        #return JSONResponse(content={"prediction": prediction_result_list[0]}, status_code=200)
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Prediction failed. Error: {str(e)}")
