

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
from datetime import datetime
import io
import base64
import logging
import os
import requests
from openai import OpenAI

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Config:
    #configuration
    BASE_DIR = Path(__file__).parent
    MODEL_DIR = BASE_DIR / "models"
    
    # get models
    DENSENET_MODEL_PATH = Path(os.getenv("DENSENET_MODEL_PATH", str(MODEL_DIR / "densenet_epoch_06.pth")))
    QWEN_MODEL = os.getenv("QWEN_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    
    # api configuration
    HF_API_KEY = os.getenv("HF_API_KEY", None)  # Set this environment variable
    HF_API_URL = f"https://api-inference.huggingface.co/models/{QWEN_MODEL}"
    
    # cpu or gpu
    USE_CPU = os.getenv("USE_CPU", "false").lower() == "true"
    DEVICE = torch.device('cpu' if USE_CPU else ('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # parameters
    IMAGE_SIZE = 224
    DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", "0.2"))
    
    # class names
    CLASS_NAMES = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
        'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
        'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
        'Pleural_Thickening', 'Hernia'
    ]
    
    INDONESIAN_NAMES = {
        'Atelectasis': 'Atelektasis',
        'Cardiomegaly': 'Kardiomegali',
        'Effusion': 'Efusi Pleura',
        'Infiltration': 'Infiltrat',
        'Mass': 'Massa',
        'Nodule': 'Nodul',
        'Pneumonia': 'Pneumonia',
        'Pneumothorax': 'Pneumotoraks',
        'Consolidation': 'Konsolidasi',
        'Edema': 'Edema Paru',
        'Emphysema': 'Emfisema',
        'Fibrosis': 'Fibrosis',
        'Pleural_Thickening': 'Penebalan Pleura',
        'Hernia': 'Hernia'
    }


# schema
class DiseaseDetection(BaseModel):
    # detection
    disease_name: str
    indonesian_name: str
    probability: float
    location: str
    bbox: Optional[Dict[str, float]] = None


class AnalysisResponse(BaseModel):
    # analysis response
    success: bool
    timestamp: str
    detected_diseases: List[DiseaseDetection]
    all_predictions: Dict[str, float]
    medical_report: str
    threshold_used: float
    model_info: Dict[str, str]


class HealthResponse(BaseModel):
    # test
    status: str
    device: str
    models_loaded: bool


# ==================== MODEL DEFINITIONS ====================
class ChestXRayDenseNet(nn.Module):
    # densenet121 for chest xrays
    def __init__(self, num_classes=14):
        super().__init__()
        from torchvision.models import densenet121
        densenet = densenet121(weights=None) 
        
        self.features = densenet.features
        num_features = densenet.classifier.in_features
        self.classifier = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        features = self.features(x)
        out = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
    
    def get_features(self, x):
        # features for cam
        return self.features(x)


# ==================== CAM FUNCTIONS ====================
def get_cam_and_location(model, image_tensor, class_idx):
    # get the class activation and location
    model.eval()
    
    features = model.get_features(image_tensor)
    features = features.detach()
    
    weights = model.classifier.weight[class_idx].detach()
    
    # compute CAM
    cam = torch.zeros(features.shape[2:], device=features.device)
    for i, w in enumerate(weights):
        cam += w * features[0, i, :, :]
    
    # normalize
    cam = torch.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    
    cam_np = cam.cpu().numpy()
    
    # find the bounding box
    threshold = 0.5 * cam_np.max()
    mask = (cam_np > threshold).astype('uint8')
    
    import numpy as np
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if rows.any() and cols.any():
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        h, w = cam_np.shape
        bbox = {
            'x': float(x_min / w),
            'y': float(y_min / h),
            'width': float((x_max - x_min) / w),
            'height': float((y_max - y_min) / h)
        }
        
        center_x = (x_min + x_max) / (2 * w)
        center_y = (y_min + y_max) / (2 * h)
    else:
        bbox = None
        center_x, center_y = 0.5, 0.5
    
    location = determine_location(center_x, center_y)
    
    return cam_np, bbox, location


def determine_location(x, y):
    # location for the disease in indo
    if y < 0.33:
        vertical = 'lapang paru atas'
    elif y < 0.66:
        vertical = 'lapang paru tengah'
    else:
        vertical = 'lapang paru bawah'
    
    if x < 0.4:
        horizontal = 'kiri'
    elif x > 0.6:
        horizontal = 'kanan'
    else:
        horizontal = 'sentral'
    
    return f"{vertical} {horizontal}"


class QwenReportGenerator:
    # slm report using qwen and hugging face api
    
    def __init__(self, model_name, api_key=None):
       
        #Initialize Qwen API client using OpenAI with HF router
        
        self.model_name = model_name
        
        # interferance used (featherless-ai provided by hugging face)
        self.model_with_provider = f"{model_name}:featherless-ai"
        
        # initialize openai client with huging face router endpoint
        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=api_key
        )
        
        logger.info(f"Qwen API ready via HF router: {self.model_with_provider}")
    
    def generate_report(self, detected_diseases, locations, max_length=1000):
       # generate the report using qwen
        findings = []
        for disease, prob in detected_diseases:
            indo_name = Config.INDONESIAN_NAMES.get(disease, disease)
            location = locations.get(disease, "tidak dapat ditentukan")
            findings.append(f"- {indo_name} ({prob:.1%} confidence) di {location}")
        
        findings_text = "\n".join(findings) if findings else "Tidak ada temuan signifikan"
        
        user_prompt = f"""Buatlah laporan medis dalam Bahasa Indonesia yang profesional dan detail.

DATA PEMERIKSAAN:
Modalitas: Foto Toraks (Chest X-Ray)
Proyeksi: PA (Postero-Anterior)
Tanggal: {datetime.now().strftime('%d %B %Y')}

TEMUAN DARI ANALISIS AI:
{findings_text}

Buatlkan laporan radiologi yang lengkap dalam format berikut:

HASIL PEMERIKSAAN RADIOLOGI TORAKS

TEKNIK PEMERIKSAAN:
[Jelaskan teknik pemeriksaan yang digunakan]

KESAN:
[Deskripsikan temuan radiologi secara detail, termasuk lokasi anatomi yang spesifik]

KESIMPULAN:
[Berikan kesimpulan diagnosis berdasarkan temuan]

SARAN:
[Berikan rekomendasi pemeriksaan lanjutan atau tindakan klinis jika diperlukan]

Gunakan terminologi medis yang tepat dalam Bahasa Indonesia. Laporan harus profesional, objektif, dan informatif."""

        try:
            logger.info(f"Calling Qwen API: {self.model_with_provider}")
            
            # Use OpenAI client with HF router (provider-specific model name)
            completion = self.client.chat.completions.create(
                model=self.model_with_provider,
                messages=[
                    {
                        "role": "system",
                        "content": "Anda adalah dokter radiologi yang ahli dalam membaca foto rontgen toraks."
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                max_tokens=max_length,
                temperature=0.7,
                top_p=0.9
            )
            
            report = completion.choices[0].message.content
            
            if report and report.strip():
                logger.info("✓ Report generated successfully via Qwen API")
                return report.strip()
            else:
                logger.warning("Empty report from API")
                return self._get_fallback_report(findings_text)
                
        except Exception as e:
            logger.error(f"Error calling Qwen API: {e}")
           # fallback incasse the interferance we used doesnt work 
            try:
                logger.info("Retrying with fireworks-ai provider...")
                alt_model = f"{self.model_name}:fireworks-ai"
                completion = self.client.chat.completions.create(
                    model=alt_model,
                    messages=[
                        {"role": "system", "content": "Anda adalah dokter radiologi yang ahli dalam membaca foto rontgen toraks."},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=max_length,
                    temperature=0.7
                )
                report = completion.choices[0].message.content
                if report and report.strip():
                    logger.info("✓ Report generated with fireworks-ai provider")
                    return report.strip()
            except Exception as e2:
                logger.error(f"Fireworks-ai provider also failed: {e2}")
            
            return self._get_fallback_report(findings_text)
    
    def _get_fallback_report(self, findings_text):
        """Generate simple fallback report if API fails"""
        return f"""HASIL PEMERIKSAAN RADIOLOGI TORAKS

TEKNIK PEMERIKSAAN:
Pemeriksaan foto toraks proyeksi PA (Postero-Anterior) dalam posisi tegak.

KESAN:
Berdasarkan analisis AI, ditemukan:
{findings_text}

KESIMPULAN:
Ditemukan kelainan radiologi pada foto toraks sesuai temuan di atas.

SARAN:
Disarankan korelasi klinis dan pemeriksaan lanjutan sesuai indikasi.

(Catatan: Laporan ini dibuat dengan template sederhana karena layanan AI report generator sedang tidak tersedia)"""


# ==================== IMAGE PREPROCESSING ====================
def preprocess_image(image: Image.Image):
    # preprocess the image mainly just resizing and nromalizing
    transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(image).unsqueeze(0)


# ==================== FASTAPI APPLICATION ====================
app = FastAPI(
    title="Chest X-Ray Analysis API",
    description="Medical image analysis with DenseNet and Qwen for Indonesian reports",
    version="1.0.0"
)

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variables
densenet_model = None
qwen_generator = None


def load_models_sync():
    # load both models
    global densenet_model, qwen_generator
    
    try:
        logger.info("Loading DenseNet model...")
        densenet_model = ChestXRayDenseNet(num_classes=14)
        
        checkpoint = torch.load(
            Config.DENSENET_MODEL_PATH, 
            map_location=Config.DEVICE,
            weights_only=False 
        )
        
        if 'model_state_dict' in checkpoint:
            densenet_model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"DenseNet loaded (epoch: {checkpoint.get('epoch', 'unknown')})")
        else:
            densenet_model.load_state_dict(checkpoint)
            logger.info("DenseNet loaded")
        
        densenet_model = densenet_model.to(Config.DEVICE)
        densenet_model.eval()
        
        logger.info("Initializing Qwen API client...")
        qwen_generator = QwenReportGenerator(
            model_name=Config.QWEN_MODEL,
            api_key=Config.HF_API_KEY
        )
        
        logger.info("✓ All models loaded successfully!")
        
    except Exception as e:
        logger.error(f"✗ Error loading models: {e}")
        logger.error("Server will start but /analyze endpoint will not work")
        # Don't raise - let server start anyway


# Load models at import time (before server starts)
logger.info("=" * 80)
logger.info("INITIALIZING MODELS...")
logger.info("=" * 80)
load_models_sync()


@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        device=str(Config.DEVICE),
        models_loaded=densenet_model is not None and qwen_generator is not None
    )


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_xray(
    file: UploadFile = File(..., description="Chest X-ray image (PNG, JPG, JPEG)"),
    threshold: Optional[float] = Form(None, description="Detection threshold (0.0-1.0)"),
    model_epoch: Optional[str] = Form(None, description="Model epoch filename (e.g., densenet_epoch_06.pth)")
):

    if densenet_model is None or qwen_generator is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Use provided threshold or default
    detection_threshold = threshold if threshold is not None else Config.DEFAULT_THRESHOLD
    
    # Validate threshold
    if not 0.0 <= detection_threshold <= 1.0:
        raise HTTPException(status_code=400, detail="Threshold must be between 0.0 and 1.0")
    
    # Load different model epoch if specified
    current_model = densenet_model
    if model_epoch and model_epoch != Config.DENSENET_MODEL_PATH.name:
        try:
            logger.info(f"Loading custom model epoch: {model_epoch}")
            custom_model = ChestXRayDenseNet(num_classes=14)
            custom_model_path = Config.MODEL_DIR / model_epoch
            
            if not custom_model_path.exists():
                raise HTTPException(status_code=404, detail=f"Model file not found: {model_epoch}")
            
            checkpoint = torch.load(custom_model_path, map_location=Config.DEVICE, weights_only=False)
            
            if 'model_state_dict' in checkpoint:
                custom_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                custom_model.load_state_dict(checkpoint)
            
            custom_model = custom_model.to(Config.DEVICE)
            custom_model.eval()
            current_model = custom_model
            logger.info(f"Using model: {model_epoch}")
        except Exception as e:
            logger.error(f"Error loading custom model: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        logger.info(f"Processing image: {file.filename}, size: {image.size}")
        
        # Preprocess
        image_tensor = preprocess_image(image).to(Config.DEVICE)
        
        # Detect diseases
        with torch.no_grad():
            outputs = current_model(image_tensor)
            probabilities = torch.sigmoid(outputs).squeeze().cpu().numpy()
        
        predictions = {
            name: float(prob) 
            for name, prob in zip(Config.CLASS_NAMES, probabilities)
        }
        
        # Filter detected diseases
        detected_diseases = [
            (name, prob) 
            for name, prob in predictions.items() 
            if prob >= detection_threshold
        ]
        detected_diseases.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Detected {len(detected_diseases)} diseases above threshold {detection_threshold}")
        
        # Localize with CAM
        detections = []
        locations = {}
        
        for disease_name, prob in detected_diseases:
            class_idx = Config.CLASS_NAMES.index(disease_name)
            cam, bbox, location = get_cam_and_location(current_model, image_tensor, class_idx)
            
            locations[disease_name] = location
            
            detections.append(DiseaseDetection(
                disease_name=disease_name,
                indonesian_name=Config.INDONESIAN_NAMES.get(disease_name, disease_name),
                probability=prob,
                location=location,
                bbox=bbox
            ))
        
        # Generate report and base report for no finding for api efficiency
        if detected_diseases:
            report = qwen_generator.generate_report(detected_diseases, locations)
        else:
            report = """HASIL PEMERIKSAAN RADIOLOGI TORAKS

TEKNIK PEMERIKSAAN:
Pemeriksaan foto toraks proyeksi PA (Postero-Anterior) dalam posisi tegak.

KESAN:
Tidak tampak kelainan radiologi yang signifikan pada foto toraks ini.
Cor dan pulmo dalam batas normal.

KESIMPULAN:
Foto toraks dalam batas normal.

SARAN:
Tidak diperlukan pemeriksaan lanjutan saat ini."""
        
        return AnalysisResponse(
            success=True,
            timestamp=datetime.now().isoformat(),
            detected_diseases=detections,
            all_predictions=predictions,
            medical_report=report,
            threshold_used=detection_threshold,
            model_info={
                "densenet": model_epoch if model_epoch else str(Config.DENSENET_MODEL_PATH),
                "qwen": Config.QWEN_MODEL,
                "device": str(Config.DEVICE)
            }
        )
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/models/info")
async def get_model_info():
    return {
        "densenet_path": Config.DENSENET_MODEL_PATH,
        "qwen_model": Config.QWEN_MODEL,
        "qwen_mode": "Hugging Face Inference API",
        "qwen_authenticated": Config.HF_API_KEY is not None,
        "device": str(Config.DEVICE),
        "default_threshold": Config.DEFAULT_THRESHOLD,
        "num_classes": len(Config.CLASS_NAMES),
        "class_names": Config.CLASS_NAMES
    }


if __name__ == "__main__":
    import uvicorn
    
    # Run server
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
