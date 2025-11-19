

"""
Standalone test script for roll_predictor.py
Run this to test roll number prediction independently.

Usage:
    python test_roll_predictor.py [image_path]

Example:
    python test_roll_predictor.py "test_omr_sheets/40 ques omr_1.jpg"
"""

import sys
import os
from roll_predictor import (
    save_canonical_sheet,
    predict_roll_number,
    generate_captcha,
    save_result_json
)

def main():
    # Default test image
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        test_images = [
            "test_omr_sheets/40 ques omr_1.jpg",
            "test_omr_sheets/40 ques omr_1(1).jpg"
        ]
        image_path = next((img for img in test_images if os.path.exists(img)), None)

    if not image_path:
        print("❌ No test image found. Please provide an image path.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"🧪 Testing Roll Predictor")
    print(f"{'='*60}")
    print(f"📸 Input image: {image_path}\n")

    required_files = {
        "mnist_cnn1.h5": "MNIST model for digit recognition",
        "template.json": "Template configuration"
    }

    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("❌ Missing required files:")
        for f in missing_files:
            print(f"  - {f} ({required_files[f]})")
        sys.exit(1)

    try:
        print("STEP 1: Creating canonical sheet...")
        canonical_path = save_canonical_sheet(image_path)
        print(f"✅ Canonical sheet ready: {canonical_path}\n")

        print("STEP 2: Predicting roll number...")
        roll_number = predict_roll_number(canonical_path)
        print(f"✅ Predicted Roll Number: {roll_number}\n")

        print("STEP 3: Generating CAPTCHA...")
        captcha = generate_captcha()
        print(f"✅ CAPTCHA: {captcha}\n")

        print("STEP 4: Saving results...")
        save_result_json(roll_number, captcha)

        print(f"\n{'='*60}")
        print("✅ TEST COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"📋 Results:")
        print(f"   Roll Number: {roll_number}")
        print(f"   CAPTCHA: {captcha}")
        print(f"   JSON saved: roll_result.json")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
