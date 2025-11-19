

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import shutil
import os
import json
import time

from roll_predictor import predict_roll_number
from omr_processor import get_default_processor  # ✅ Real OMR Processor


# ============================================
# Initialize FastAPI app
# ============================================
app = FastAPI(title="OMR + Roll Number Processor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploaded_sheets"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ============================================
# OMR Processing Endpoint
# ============================================
@app.post("/process-omr")
async def process_omr(
    files: List[UploadFile] = File(...),
    answer_key: Optional[str] = Form(None)
):
    results = {}
    total_time = 0.0

    # --- Parse answer key JSON ---
    key_dict = {}
    if answer_key:
        try:
            key_dict = json.loads(answer_key)
        except Exception as e:
            return {"error": f"Invalid answer key JSON: {str(e)}"}

    # ✅ Initialize OMR Processor once
    omr = get_default_processor(model_path="best.pt", template_path="template.json")

    # --- Loop through uploaded files ---
    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"✅ Received file: {file.filename}")
        start_time = time.time()

        # --- Step 1: OMR Detection ---
        try:
            omr_result = omr.process_image(file_path, key_dict)
            detected = omr_result.get("detected", {})
            score = omr_result.get("score", 0)
        except Exception as e:
            print(f"❌ OMR processing failed for {file.filename}: {e}")
            detected = {}
            score = 0

        # --- Step 2: Roll Number Prediction ---
        try:
            roll_number = predict_roll_number(file_path)
        except Exception as e:
            roll_number = f"Error: {str(e)}"

        # --- Step 3: Timing and Summary ---
        processing_time = round(time.time() - start_time, 3)
        total_time += processing_time

        results[file.filename] = {
            "roll_number": roll_number,
            "score": score,
            "detected": detected,
            "processing_time_sec": processing_time
        }

    # --- Summary for all files ---
    results["_summary"] = {
        "files_processed": len(files),
        "total_time_sec": round(total_time, 3),
        "avg_time_sec": round(total_time / len(files), 3) if files else 0
    }

    return results


# ============================================
# Root Route
# ============================================
@app.get("/")
def home():
    return {"message": "✅ OMR Processor API is running! Use /process-omr to upload sheets."}
