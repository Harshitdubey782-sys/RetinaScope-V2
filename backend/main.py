
import base64
import json
from contextlib import asynccontextmanager
from io import BytesIO

import numpy as np
import torch

from PIL import Image, UnidentifiedImageError

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from backend.model import load_model, CLASS_NAMES
from backend.preprocessing import preprocess_image
from backend.gradcam import generate_gradcam

from backend.reports import (
    generate_screening_report,
    generate_dual_eye_report
)


# ============================================================
# APPLICATION LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Loading RetinaScope EfficientNet-B0...")

    app.state.model = load_model()

    print("RetinaScope model loaded successfully!")

    yield

    print("RetinaScope API shutting down...")


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="RetinaScope API",
    description="AI-powered Diabetic Retinopathy Screening API",
    version="1.0.0",
    lifespan=lifespan
)


# ============================================================
# CORS CONFIGURATION
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# IMAGE PROCESSING
# ============================================================

async def process_eye_upload(uploaded_file: UploadFile):

    allowed_extensions = {
        ".jpg",
        ".jpeg",
        ".png"
    }

    filename = uploaded_file.filename or ""
    extension = filename.lower().rsplit(".", 1)[-1]

    if f".{extension}" not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG and PNG images are supported."
        )

    try:

        image_bytes = await uploaded_file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty."
            )

        original_image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

    except HTTPException:
        raise

    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image file."
        )

    rgb_image, input_tensor = preprocess_image(
        original_image
    )

    return original_image, rgb_image, input_tensor


# ============================================================
# GRAD-CAM ENCODING
# ============================================================

def encode_heatmap(heatmap):

    buffer = BytesIO()

    Image.fromarray(
        heatmap.astype(np.uint8)
    ).save(
        buffer,
        format="PNG"
    )

    buffer.seek(0)

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


# ============================================================
# PREDICTION RESULT FORMATTER
# ============================================================

def create_prediction_result(
    filename,
    predicted_class,
    confidence,
    probabilities,
    heatmap
):

    probability_values = probabilities.detach().cpu().numpy()

    return {
        "filename": filename,

        "predicted_class": predicted_class,

        "predicted_grade": CLASS_NAMES[predicted_class],

        "confidence": round(
            confidence * 100,
            2
        ),

        "referable_dr": predicted_class >= 2,

        "probabilities": {
            CLASS_NAMES[i]: round(
                float(probability_values[0, i]) * 100,
                2
            )
            for i in range(len(CLASS_NAMES))
        },

        "heatmap_base64": encode_heatmap(heatmap)
    }


# ============================================================
# PATIENT DATA VALIDATION
# ============================================================

def validate_patient_data(patient, required_fields):

    if not isinstance(patient, dict):

        raise HTTPException(
            status_code=400,
            detail="Patient information must be a JSON object."
        )

    missing_fields = [
        field
        for field in required_fields
        if field not in patient
    ]

    if missing_fields:

        raise HTTPException(
            status_code=400,
            detail={
                "message": "Missing patient information.",
                "missing_fields": missing_fields
            }
        )

    return patient


def parse_patient_data(patient_json):

    try:

        patient = json.loads(patient_json)

    except (json.JSONDecodeError, TypeError):

        raise HTTPException(
            status_code=400,
            detail="Patient field must contain valid JSON."
        )

    if not isinstance(patient, dict):

        raise HTTPException(
            status_code=400,
            detail="Patient information must be a JSON object."
        )

    return patient


# ============================================================
# PDF RESPONSE
# ============================================================

def pdf_response(pdf_bytes, filename):

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{filename}"'
            )
        }
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
async def health_check():

    return {
        "status": "healthy",
        "application": "RetinaScope",
        "model_loaded": hasattr(app.state, "model")
    }


# ============================================================
# SINGLE-EYE PREDICTION
# ============================================================

@app.post("/api/predict")
async def predict_single_eye(
    file: UploadFile = File(...)
):

    model = app.state.model

    original_image, rgb_image, input_tensor = (
        await process_eye_upload(file)
    )

    with torch.inference_mode():

        output = model(input_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        predicted_class = int(
            torch.argmax(
                probabilities,
                dim=1
            ).item()
        )

        confidence = float(
            probabilities[0, predicted_class].item()
        )

    heatmap = generate_gradcam(
        model,
        input_tensor,
        predicted_class,
        rgb_image
    )

    result = create_prediction_result(
        filename=file.filename,
        predicted_class=predicted_class,
        confidence=confidence,
        probabilities=probabilities,
        heatmap=heatmap
    )

    return {
        "status": "success",
        "mode": "single_eye",
        "result": result
    }


# ============================================================
# DUAL-EYE PREDICTION
# ============================================================

@app.post("/api/predict/dual")
async def predict_dual_eye(
    right_file: UploadFile = File(...),
    left_file: UploadFile = File(...)
):

    model = app.state.model

    # Process both images

    right_original, right_rgb, right_tensor = (
        await process_eye_upload(right_file)
    )

    left_original, left_rgb, left_tensor = (
        await process_eye_upload(left_file)
    )

    # Combine tensors for batched inference

    batch = torch.cat(
        [right_tensor, left_tensor],
        dim=0
    )

    with torch.inference_mode():

        output = model(batch)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

    right_pred = int(predictions[0].item())

    left_pred = int(predictions[1].item())

    right_confidence = float(
        probabilities[0, right_pred].item()
    )

    left_confidence = float(
        probabilities[1, left_pred].item()
    )

    # Generate individual Grad-CAM heatmaps

    right_heatmap = generate_gradcam(
        model,
        right_tensor,
        right_pred,
        right_rgb
    )

    left_heatmap = generate_gradcam(
        model,
        left_tensor,
        left_pred,
        left_rgb
    )

    # Format results

    right_result = create_prediction_result(
        filename=right_file.filename,
        predicted_class=right_pred,
        confidence=right_confidence,
        probabilities=probabilities[0:1],
        heatmap=right_heatmap
    )

    left_result = create_prediction_result(
        filename=left_file.filename,
        predicted_class=left_pred,
        confidence=left_confidence,
        probabilities=probabilities[1:2],
        heatmap=left_heatmap
    )

    grade_difference = abs(
        right_pred - left_pred
    )

    return {
        "status": "success",
        "mode": "dual_eye",

        "right_eye": right_result,

        "left_eye": left_result,

        "comparison": {
            "grade_difference": grade_difference,

            "same_grade": grade_difference == 0,

            "higher_severity_eye": (
                "Right Eye"
                if right_pred > left_pred
                else "Left Eye"
                if left_pred > right_pred
                else "Same"
            )
        }
    }


# ============================================================
# SINGLE-EYE PDF REPORT
# ============================================================

@app.post("/api/report")
async def create_single_eye_report(
    file: UploadFile = File(...),
    patient: str = Form(...)
):

    patient_data = parse_patient_data(patient)

    required_fields = [
        "Case ID",
        "Screening Date",
        "Patient Name",
        "Age",
        "Sex",
        "Screening Centre",
        "Eye Examined",
        "Image ID",
        "Diabetes Type",
        "Diabetes Duration",
        "Previous DR",
        "Previous Eye Exam"
    ]

    validate_patient_data(
        patient_data,
        required_fields
    )

    model = app.state.model

    original_image, rgb_image, input_tensor = (
        await process_eye_upload(file)
    )

    with torch.inference_mode():

        output = model(input_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        predicted_class = int(
            torch.argmax(
                probabilities,
                dim=1
            ).item()
        )

        confidence = float(
            probabilities[0, predicted_class].item()
        )

    heatmap = generate_gradcam(
        model,
        input_tensor,
        predicted_class,
        rgb_image
    )

    pdf_bytes = generate_screening_report(
        patient=patient_data,
        original_image=original_image,
        heatmap=heatmap,
        probabilities=probabilities.cpu().numpy(),
        predicted_class=predicted_class,
        confidence=confidence
    )

    return pdf_response(
        pdf_bytes,
        "RetinaScope_Single_Eye_Report.pdf"
    )


# ============================================================
# DUAL-EYE PDF REPORT
# ============================================================

@app.post("/api/report/dual")
async def create_dual_eye_report(
    right_file: UploadFile = File(...),
    left_file: UploadFile = File(...),
    patient: str = Form(...)
):

    patient_data = parse_patient_data(patient)

    required_fields = [
        "Case ID",
        "Screening Date",
        "Patient Name",
        "Age",
        "Sex",
        "Screening Centre",
        "Right Image ID",
        "Left Image ID",
        "Diabetes Type",
        "Diabetes Duration",
        "Previous DR",
        "Previous Eye Exam"
    ]

    validate_patient_data(
        patient_data,
        required_fields
    )

    model = app.state.model

    # Process both eyes

    right_image, right_rgb, right_tensor = (
        await process_eye_upload(right_file)
    )

    left_image, left_rgb, left_tensor = (
        await process_eye_upload(left_file)
    )

    # Batched model inference

    batch = torch.cat(
        [right_tensor, left_tensor],
        dim=0
    )

    with torch.inference_mode():

        output = model(batch)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

    right_pred = int(
        predictions[0].item()
    )

    left_pred = int(
        predictions[1].item()
    )

    # Individual Grad-CAM generation

    right_heatmap = generate_gradcam(
        model,
        right_tensor,
        right_pred,
        right_rgb
    )

    left_heatmap = generate_gradcam(
        model,
        left_tensor,
        left_pred,
        left_rgb
    )

    # Keep probabilities in [1, 5] shape.
    # This matches the PDF report module's expected format.

    right_data = {

        "image": right_image,

        "heatmap": right_heatmap,

        "predicted_class": right_pred,

        "confidence": float(
            probabilities[0, right_pred].item()
        ),

        "probabilities": (
            probabilities[0:1].cpu().numpy()
        )
    }

    left_data = {

        "image": left_image,

        "heatmap": left_heatmap,

        "predicted_class": left_pred,

        "confidence": float(
            probabilities[1, left_pred].item()
        ),

        "probabilities": (
            probabilities[1:2].cpu().numpy()
        )
    }

    pdf_bytes = generate_dual_eye_report(
        patient=patient_data,
        right_data=right_data,
        left_data=left_data
    )

    return pdf_response(
        pdf_bytes,
        "RetinaScope_Dual_Eye_Report.pdf"
    )


# ============================================================
# END OF RETINASCOPE API
# ============================================================
