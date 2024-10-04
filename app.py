from fastapi import FastAPI,File, UploadFile, HTTPException
import uvicorn
import os
from pathlib import Path
from fastapi.responses import Response
from cough_segmentation.pipeline.prediction_pipeline import ModelPrediction
from cough_segmentation.entity.config_entity import ModelPredictionConfig, ModelTrainerConfig
from cough_segmentation.logger import logging
app = FastAPI()


@app.get("/train")
async def train():
    try:
        # Run the script using os.system
        exit_status = os.system("python main.py")
        if exit_status == 0:
            return Response(content="Successfully trained.", media_type="text/plain")
        else:
            return Response(content=f"Error Occurred.",media_type="text/plain")
    except Exception as e:
        return Response(content=f"Error Occurred! {e}", media_type="text/plain")
    
@app.post("/predict")
async def uploadfile(files: list[UploadFile]):
    try:
        for file in files:
            file_path = f"./testing_data/{file.filename}"
            filepath = Path(file_path)
            print(filepath)
            filedir, filename = os.path.split(filepath)

            if filedir != "":
                os.makedirs(filedir, exist_ok=True)
                logging.info(f"Creating directory: {filedir} for the file {filename}")
                
            with open(file_path, "wb") as f:
                f.write(file.file.read())

        model_prediction_config = ModelPredictionConfig()
        model_trainer_config = ModelTrainerConfig()
        model_pred = ModelPrediction(model_prediction_config=model_prediction_config,model_trainer_config=model_trainer_config)
        prediction = model_pred.predict_audio(file_path)
        return Response(content=str(prediction.tolist()), media_type="application/json")
    
    except Exception as e:
        return {"message": e.args}


if __name__=="__main__":
    uvicorn.run("app:app", host="localhost", port=8080,reload=True)