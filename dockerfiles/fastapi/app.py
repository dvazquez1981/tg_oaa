from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import model
from schema import Movie, Prediction

@asynccontextmanager
async def lifespan(app: FastAPI):
    model.load_model()
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Model Service"}


@app.get("/predict", response_model=Prediction)
def predict(movie: Movie):
    # The model may have been registered after startup, retry before failing
    if not model.is_loaded() and not model.load_model():
        raise HTTPException(status_code=503,
                            detail=f"Model {model.MODEL_NAME}@{model.MODEL_ALIAS} is not available")

    return Prediction(revenue=model.predict_revenue(movie),
                      model_name=model.MODEL_NAME,
                      model_version=model.model_state["version"])