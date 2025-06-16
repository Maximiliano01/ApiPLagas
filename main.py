from fastapi import FastAPI, File, UploadFile, Request # type: ignore
from fastapi.responses import JSONResponse # type: ignore
from tensorflow.keras.models import load_model # type: ignor
from fastapi.middleware.cors import CORSMiddleware
from prediccion import predict_img
import shutil
import os
import uuid

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def cargar_modelo():
    app.state.modelo = load_model("clasificador.h5")

@app.post("/clasificar-planta")
async def clasificar_planta(request: Request, imagen: UploadFile = File(...)):
    extension = os.path.splitext(imagen.filename)[1]
    nombre_temp = f"temp_{uuid.uuid4()}{extension}"
    ruta = os.path.join("imagenes_temp", nombre_temp)

    os.makedirs("imagenes_temp", exist_ok=True)
    with open(ruta, "wb") as buffer:
        shutil.copyfileobj(imagen.file, buffer)

    try:
        modelo = request.app.state.modelo
        prediccion, confianza = predict_img(ruta, modelo)
        print(f"Prediccion: {prediccion},\nConfianza: {confianza * 100:.2f}")
        return JSONResponse(content={
            "prediccion": prediccion,
            "confianza": f"{confianza * 100:.2f}%"
        })
    finally:
        os.remove(ruta) 
        
