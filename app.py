import cv2
import numpy as np
import pytesseract
import re
import os
import sys
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI App
app = FastAPI(
    title="Automated PAN Card Reader API",
    description="Production-grade OCR service for extracting metadata from Indian PAN Cards.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Cross-Platform Tesseract Path Configuration ---
# Priority 1: explicit env-var override (works on any platform/deployment)
_tess_env = os.environ.get("TESSERACT_CMD", "")
if _tess_env and os.path.exists(_tess_env):
    pytesseract.pytesseract.tesseract_cmd = _tess_env

elif sys.platform.startswith('win'):
    # Windows: default Tesseract installer path
    _win_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if os.path.exists(_win_path):
        pytesseract.pytesseract.tesseract_cmd = _win_path

elif sys.platform == 'darwin':
    # macOS (Homebrew on Apple Silicon or Intel)
    for _mac_path in ['/opt/homebrew/bin/tesseract', '/usr/local/bin/tesseract']:
        if os.path.exists(_mac_path):
            pytesseract.pytesseract.tesseract_cmd = _mac_path
            # Point to Homebrew tessdata
            _mac_tessdata = os.path.dirname(_mac_path).replace('/bin', '/share/tessdata')
            if os.path.isdir(_mac_tessdata):
                os.environ['TESSDATA_PREFIX'] = _mac_tessdata + '/'
            break

else:
    # Linux (Ubuntu/Debian — used by Render, Railway, Heroku, etc.)
    # apt-get install tesseract-ocr  →  /usr/bin/tesseract
    for _linux_path in ['/usr/bin/tesseract', '/usr/local/bin/tesseract']:
        if os.path.exists(_linux_path):
            pytesseract.pytesseract.tesseract_cmd = _linux_path
            break
    # Set tessdata prefix for Linux (Tesseract 4/5 installed via apt)
    for _td in ['/usr/share/tesseract-ocr/5/tessdata',
                '/usr/share/tesseract-ocr/4.00/tessdata',
                '/usr/share/tessdata']:
        if os.path.isdir(_td):
            os.environ['TESSDATA_PREFIX'] = _td
            break


class PanCardEngine:
    def __init__(self):
        self.tesseract_config = r'--oem 3 --psm 3'

    def get_ocr_strategies(self, cv_img):
        """Returns a list of tuples: (strategy_name, processed_image, config)."""
        height, width = cv_img.shape[:2]
        scaling_factor = 1200 / width
        resized = cv2.resize(cv_img, (1200, int(height * scaling_factor)))
        
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # 1. Otsu Thresholding
        blurred_otsu = cv2.GaussianBlur(gray, (5, 5), 0)
        _, otsu = cv2.threshold(blurred_otsu, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 2. Adaptive Gaussian Thresholding
        blurred_adaptive = cv2.GaussianBlur(gray, (3, 3), 0)
        adaptive = cv2.adaptiveThreshold(
            blurred_adaptive, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        strategies = [
            ("Grayscale (PSM 3)", gray, r'--oem 3 --psm 3'),
            ("Grayscale (PSM 4)", gray, r'--oem 3 --psm 4'),
            ("Grayscale (PSM 6)", gray, r'--oem 3 --psm 6'),
            ("Otsu Threshold (PSM 3)", otsu, r'--oem 3 --psm 3'),
            ("Otsu Threshold (PSM 6)", otsu, r'--oem 3 --psm 6'),
            ("Adaptive Threshold (PSM 3)", adaptive, r'--oem 3 --psm 3'),
            ("Adaptive Threshold (PSM 4)", adaptive, r'--oem 3 --psm 4'),
            ("Color Resized (PSM 3)", resized, r'--oem 3 --psm 3'),
        ]
        return strategies

    def repair_pan(self, candidate):
        if len(candidate) != 10:
            return None
        letter_map = {
            '0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B', '6': 'G', '7': 'T', '9': 'J'
        }
        digit_map = {
            'O': '0', 'I': '1', 'L': '1', 'l': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6', 'T': '7', 'J': '9'
        }
        
        repaired = list(candidate)
        for i in range(5):
            if repaired[i] in letter_map:
                repaired[i] = letter_map[repaired[i]]
        for i in range(5, 9):
            if repaired[i] in digit_map:
                repaired[i] = digit_map[repaired[i]]
        if repaired[9] in letter_map:
            repaired[9] = letter_map[repaired[9]]
            
        final_pan = "".join(repaired)
        if re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]$', final_pan):
            return final_pan
        return None

    def find_pan_candidate(self, text):
        words = text.split()
        for word in words:
            cleaned = re.sub(r'[^A-Z0-9]', '', word.upper())
            for i in range(len(cleaned) - 9):
                candidate = cleaned[i:i+10]
                repaired = self.repair_pan(candidate)
                if repaired:
                    return repaired
        return "NOT FOUND"

    def find_dob_candidate(self, text):
        text_normalized = re.sub(r'(\d)\s*[|Il\\/]\s*(\d)', r'\1/\2', text)
        text_normalized = re.sub(r'\s+', ' ', text_normalized)
        
        dob_pattern = r'\b(\d{2})[/\-\.](\d{2})[/\-\.](\d{4})\b'
        match = re.search(dob_pattern, text_normalized)
        if match:
            day, month, year = match.groups()
            if 1 <= int(day) <= 31 and 1 <= int(month) <= 12 and 1900 <= int(year) <= 2030:
                return f"{day}/{month}/{year}"
                
        dob_space_pattern = r'\b(\d{2})\s+(\d{2})\s+(\d{4})\b'
        match = re.search(dob_space_pattern, text_normalized)
        if match:
            day, month, year = match.groups()
            if 1 <= int(day) <= 31 and 1 <= int(month) <= 12 and 1900 <= int(year) <= 2030:
                return f"{day}/{month}/{year}"
                
        dob_lax_pattern = r'\b(\d{2})[/\-\.\s](\d{2})[/\-\.\s]([0-9OIZS]{4})\b'
        match = re.search(dob_lax_pattern, text_normalized.upper())
        if match:
            day, month, year_str = match.groups()
            digit_map = {'O': '0', 'I': '1', 'Z': '2', 'S': '5'}
            year_repaired = "".join(digit_map.get(c, c) for c in year_str)
            if year_repaired.isdigit():
                if 1 <= int(day) <= 31 and 1 <= int(month) <= 12 and 1900 <= int(year_repaired) <= 2030:
                    return f"{day}/{month}/{year_repaired}"
                    
        return "NOT FOUND"

    def parse_names(self, text, pan_number, dob):
        raw_lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        header_idx = -1
        dob_idx = len(raw_lines)
        pan_idx = -1
        
        for idx, line in enumerate(raw_lines):
            upper_line = line.upper()
            if any(h in upper_line for h in ["INCOME", "TAX", "DEPARTMENT", "GOVT", "INDIA"]):
                header_idx = idx
            if pan_number != "NOT FOUND" and pan_number in upper_line:
                pan_idx = idx
            if dob != "NOT FOUND" and dob in upper_line:
                dob_idx = min(dob_idx, idx)
            if "BIRTH" in upper_line or "DATE" in upper_line:
                dob_idx = min(dob_idx, idx)
                
        pan_at_top = False
        if pan_idx != -1:
            if pan_idx < 4 or pan_idx < len(raw_lines) * 0.4:
                pan_at_top = True
                
        start_idx = header_idx + 1
        if pan_idx != -1 and pan_at_top:
            start_idx = max(start_idx, pan_idx + 1)
            
        end_idx = dob_idx
        candidate_lines = raw_lines[start_idx:end_idx]
        
        line_skip_keywords = [
            "INCOME", "TAX", "DEPARTMENT", "GOVT", "PERMANENT", "ACCOUNT", 
            "SIGNATURE", "DATE", "BIRTH", "UTI", "NSDL", "PHOTO", "PRNT", 
            "DIGITAL", "DOCUMENT", "CITIZEN", "VALID", "INDIVIDUAL", "MALE", "FEMALE"
        ]
        
        label_words = ["FATHER", "FATHERS", "GUARDIAN", "GUARDIANS", "NAME", "NAMES", "OF"]
        
        cleaned_lines = []
        for line in candidate_lines:
            upper_line = line.upper()
            
            words = re.findall(r'\b[A-Z]+\b', upper_line)
            if any(kw in words for kw in line_skip_keywords):
                continue
                
            cleaned_line = upper_line
            cleaned_line = re.sub(r'\'S\b', '', cleaned_line)
            for word in label_words:
                cleaned_line = re.sub(r'\b' + word + r'\b', '', cleaned_line)
                
            alpha_only = re.sub(r'[^A-Z\s\.\-]', '', cleaned_line)
            letters_count = len(re.sub(r'[^A-Z]', '', cleaned_line))
            total_count = len(re.sub(r'\s', '', cleaned_line))
            
            if total_count > 0 and (letters_count / total_count) > 0.75:
                cleaned_line = re.sub(r'\s+', ' ', alpha_only).strip()
                words_cleaned = cleaned_line.split()
                words_filtered = [w for w in words_cleaned if len(w) > 2 or w in ["K", "S", "R", "P", "D", "A", "V", "N", "L", "M"]]
                cleaned_line = " ".join(words_filtered)
                if len(cleaned_line) > 3:
                    cleaned_lines.append(cleaned_line)
                    
        name = "NOT FOUND"
        father_name = "NOT FOUND"
        
        if len(cleaned_lines) >= 1:
            name_parts = cleaned_lines[0].split()
            name = " ".join(name_parts[:3])
            
        if len(cleaned_lines) >= 2:
            father_parts = cleaned_lines[1].split()
            father_name = " ".join(father_parts[:3])
            
        name_parts = name.split()
        father_parts = father_name.split()
        
        has_valid_father = False
        if father_name != "NOT FOUND" and len(name_parts) >= 1 and len(father_parts) >= 1:
            if name_parts[-1] == father_parts[-1]:
                has_valid_father = True
                
        if not has_valid_father and len(name_parts) >= 3:
            father_name = " ".join(name_parts[1:])
            
        return name, father_name

    def parse_pan_metadata(self, raw_text):
        metadata = {
            "pan_number": "NOT FOUND",
            "name": "NOT FOUND",
            "father_name": "NOT FOUND",
            "date_of_birth": "NOT FOUND",
            "is_valid_individual": False
        }
        
        pan = self.find_pan_candidate(raw_text)
        if pan != "NOT FOUND":
            metadata["pan_number"] = pan
            if pan[3] == 'P':
                metadata["is_valid_individual"] = True

        dob = self.find_dob_candidate(raw_text)
        if dob != "NOT FOUND":
            metadata["date_of_birth"] = dob

        name, father_name = self.parse_names(raw_text, pan, dob)
        metadata["name"] = name
        metadata["father_name"] = father_name

        return metadata

    def score_metadata(self, metadata):
        score = 0
        if metadata["pan_number"] != "NOT FOUND":
            score += 5
        if metadata["date_of_birth"] != "NOT FOUND":
            score += 3
        if metadata["name"] != "NOT FOUND":
            score += 2
        if metadata["father_name"] != "NOT FOUND":
            score += 2
        return score

    def get_visibility_accuracy(self, cv_img):
        """Compute a composite scan visibility accuracy score (0–100).

        Three complementary signals are combined so the score reflects true
        image quality rather than just Tesseract's internal guess certainty:

        1. Sharpness  (30%) — Laplacian variance of the greyscale image.
           High variance = sharp text edges; low variance = blur/out-of-focus.

        2. Contrast   (20%) — Standard deviation of pixel intensities.
           High std-dev = well-separated text & background; low = washed-out.

        3. OCR confidence (50%) — Mean Tesseract word confidence, excluding
           conf ≤ 0 entries (layout-analysis rows, noise blobs, whitespace).

        All three are normalised to 0–100 and blended with the weights above.
        """
        import math

        # Always work on a standardised 1200-px-wide colour image so that
        # the sharpness and contrast thresholds below are resolution-agnostic.
        height, width = cv_img.shape[:2]
        scale = 1200 / width
        clean = cv2.resize(cv_img, (1200, int(height * scale)))
        gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)

        # --- Signal 1: Sharpness ---
        # Laplacian variance: blurry ≈ <50, acceptable ≈ 100–400, sharp > 500.
        # Sigmoid-like cap so very sharp images don't distort the scale.
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness = min(100.0, 100.0 * (1.0 - math.exp(-lap_var / 400.0)))

        # --- Signal 2: Contrast ---
        # Pixel std-dev: low contrast ≈ <30, good ≈ 60–80, excellent > 80.
        contrast = min(100.0, float(gray.std()) / 70.0 * 100.0)

        # --- Signal 3: OCR confidence ---
        # Only average real word detections; conf=-1 (layout) and conf=0
        # (unrecognised/noise) are excluded.
        try:
            data = pytesseract.image_to_data(
                clean, config=r'--oem 3 --psm 3',
                output_type=pytesseract.Output.DICT
            )
            conf_vals = [float(c) for c in data['conf']
                         if c is not None and int(c) > 0]
            ocr_conf = (sum(conf_vals) / len(conf_vals)) if conf_vals else 0.0
        except Exception as e:
            print(f"OCR confidence error: {e}")
            ocr_conf = 0.0

        # --- Composite ---
        score = (0.50 * ocr_conf) + (0.30 * sharpness) + (0.20 * contrast)
        score = round(min(100.0, max(0.0, score)), 2)

        print(f"📊 Visibility accuracy → sharpness={sharpness:.1f}% "
              f"contrast={contrast:.1f}% ocr={ocr_conf:.1f}% → composite={score}%")
        return score

    def scan_image(self, cv_img):
        """Scans image using multiple preprocessing and Tesseract configurations to get best metadata."""
        strategies = self.get_ocr_strategies(cv_img)
        
        best_metadata = {
            "pan_number": "NOT FOUND",
            "name": "NOT FOUND",
            "father_name": "NOT FOUND",
            "date_of_birth": "NOT FOUND",
            "is_valid_individual": False
        }
        best_score = -1
        best_strategy_name = "None"
        best_raw_text = ""

        for name, img, config in strategies:
            try:
                raw_text = pytesseract.image_to_string(img, config=config)
                metadata = self.parse_pan_metadata(raw_text)
                score = self.score_metadata(metadata)
                
                if score > best_score:
                    best_score = score
                    best_metadata = metadata
                    best_strategy_name = name
                    best_raw_text = raw_text
                elif score == best_score:
                    if len(raw_text) > len(best_raw_text):
                        best_metadata = metadata
                        best_strategy_name = name
                        best_raw_text = raw_text
            except Exception as e:
                print(f"Strategy {name} failed: {e}")
                
        print(f"\n🏆 Best preprocessing strategy: {best_strategy_name} (Score: {best_score}/12)")
        print("\n--- RAW TESSERACT OCR OUTPUT --- \n", best_raw_text, "\n------------------------------\n")
        
        # ---------------------------------------------------------------
        # STRICT PAN CARD IDENTITY CHECK
        # RULE: Must have BOTH a valid PAN regex match AND >= 2 keywords.
        # This prevents random images (selfies, receipts, IDs etc.) from
        # being processed as PAN cards.
        # ---------------------------------------------------------------
        upper_text = best_raw_text.upper()
        words = upper_text.split()
        cleaned_words = [re.sub(r'[^A-Z0-9]', '', w) for w in words]

        has_pan_regex = False
        for cw in cleaned_words:
            for i in range(len(cw) - 9):
                candidate = cw[i:i+10]
                if self.repair_pan(candidate):
                    has_pan_regex = True
                    break
            if has_pan_regex:
                break

        pan_keywords = [
            "INCOME", "TAX", "DEPARTMENT", "GOVT", "INDIA",
            "PERMANENT", "ACCOUNT", "NUMBER", "CARD", "NSDL", "UTI"
        ]
        keyword_matches = sum(1 for kw in pan_keywords if kw in upper_text)

        # Strict: require PAN regex AND at least 2 PAN keywords
        is_pan_card = has_pan_regex and (keyword_matches >= 2)

        print(f"🔍 PAN Detection → has_pan_regex={has_pan_regex}, keyword_matches={keyword_matches}, is_pan_card={is_pan_card}")

        # Compute composite scan quality (sharpness + contrast + OCR confidence)
        best_confidence = 0.0
        try:
            best_confidence = self.get_visibility_accuracy(cv_img)
        except Exception as e:
            print(f"Error computing visibility accuracy: {e}")

        # If not a PAN card, clear all extracted metadata
        if not is_pan_card:
            best_metadata = {
                "pan_number": "NOT FOUND",
                "name": "NOT FOUND",
                "father_name": "NOT FOUND",
                "date_of_birth": "NOT FOUND",
                "is_valid_individual": False
            }

        best_metadata["is_pan_card"] = is_pan_card
        best_metadata["visibility_accuracy"] = round(best_confidence, 2)

        return best_metadata


# Instantiate Core Engine
ocr_engine = PanCardEngine()


# --- FRONTEND ROUTE ---
@app.get("/luma.mp4")
async def get_background_video():
    video_path = os.path.join(os.path.dirname(__file__), "luma.mp4")
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Background video not found.")
    return FileResponse(video_path, media_type="video/mp4")


@app.get("/", response_class=HTMLResponse)
async def get_web_interface():
    """Serves a full AI-themed cyberpunk frontend to scan PAN cards visually."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PAN·AI — Intelligent PAN Card Verification</title>
        <meta name="description" content="AI-powered PAN card OCR verification system.">
        <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Share+Tech+Mono&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            :root{--cyan:#00f5ff;--violet:#bf00ff;--pink:#ff0080;--green:#39ff14;--dark:#020617;--card:rgba(2,10,30,0.85);}
            *,*::before,*::after{box-sizing:border-box;}
            html,body{margin:0;padding:0;height:100%;}
            body{font-family:'Inter',sans-serif;background:var(--dark);color:#e2e8f0;min-height:100vh;overflow-x:hidden;}
            #bgCanvas{position:fixed;inset:0;width:100%;height:100%;z-index:0;pointer-events:none;}
            .bg-video{position:fixed;inset:0;width:100%;height:100%;object-fit:cover;z-index:1;opacity:0.16;pointer-events:none;}
            .grid-overlay{position:fixed;inset:0;z-index:2;pointer-events:none;background-image:linear-gradient(rgba(0,245,255,0.04) 1px,transparent 1px),linear-gradient(90deg,rgba(0,245,255,0.04) 1px,transparent 1px);background-size:44px 44px;}
            .page-wrap{position:relative;z-index:10;min-height:100vh;display:flex;flex-direction:column;}
            /* HUD Bar */
            .hud-bar{display:flex;align-items:center;justify-content:space-between;padding:10px 28px;border-bottom:1px solid rgba(0,245,255,0.12);background:rgba(2,6,23,0.8);backdrop-filter:blur(16px);}
            .hud-logo{font-family:'Orbitron',sans-serif;font-size:1.1rem;font-weight:900;letter-spacing:0.18em;background:linear-gradient(90deg,var(--cyan),var(--violet));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
            .hud-logo span{-webkit-text-fill-color:var(--pink);}
            .hud-meta{display:flex;align-items:center;gap:16px;font-family:'Share Tech Mono',monospace;font-size:11px;color:rgba(0,245,255,0.55);}
            .hud-dot{width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 8px 3px rgba(57,255,20,0.7);animation:blink 1.3s ease-in-out infinite;}
            .hud-clock{color:var(--cyan);}
            @keyframes blink{0%,100%{opacity:1}50%{opacity:0.2}}
            /* Layout */
            main{flex-grow:1;display:flex;align-items:center;justify-content:center;padding:20px 16px;}
            /* Glass card */
            .ai-card{width:100%;max-width:490px;background:var(--card);border:1px solid rgba(0,245,255,0.15);border-radius:24px;padding:26px 22px;backdrop-filter:blur(28px);box-shadow:0 0 0 1px rgba(0,245,255,0.05),0 0 60px rgba(0,245,255,0.05),0 32px 64px rgba(0,0,0,0.65);position:relative;overflow:hidden;}
            .ai-card::before{content:'';position:absolute;top:0;left:0;width:38px;height:38px;border-top:2px solid var(--cyan);border-left:2px solid var(--cyan);border-radius:24px 0 0 0;}
            .ai-card::after{content:'';position:absolute;bottom:0;right:0;width:38px;height:38px;border-bottom:2px solid var(--violet);border-right:2px solid var(--violet);border-radius:0 0 24px 0;}
            .card-strip{position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,var(--cyan),var(--violet),var(--pink),transparent);background-size:200% 100%;animation:stripSlide 3.5s linear infinite;}
            @keyframes stripSlide{0%{background-position:-200% 0}100%{background-position:200% 0}}
            /* Title */
            .title-section{text-align:center;margin-bottom:18px;}
            .ai-heading{font-family:'Orbitron',sans-serif;font-size:1.38rem;font-weight:900;letter-spacing:0.1em;background:linear-gradient(135deg,var(--cyan) 0%,var(--violet) 45%,var(--pink) 80%,var(--cyan) 100%);background-size:300% 300%;-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;animation:hueShift 4s ease infinite;filter:drop-shadow(0 0 14px rgba(0,245,255,0.5));position:relative;display:inline-block;}
            @keyframes hueShift{0%,100%{background-position:0% 50%}50%{background-position:100% 50%}}
            .ai-heading::before{content:attr(data-text);position:absolute;inset:0;font-family:'Orbitron',sans-serif;font-weight:900;letter-spacing:0.1em;-webkit-text-fill-color:var(--cyan);clip-path:polygon(0 0,100% 0,100% 38%,0 38%);animation:glitchTop 3.8s infinite;opacity:0;}
            .ai-heading::after{content:attr(data-text);position:absolute;inset:0;font-family:'Orbitron',sans-serif;font-weight:900;letter-spacing:0.1em;-webkit-text-fill-color:var(--pink);clip-path:polygon(0 62%,100% 62%,100% 100%,0 100%);animation:glitchBot 3.8s infinite;opacity:0;}
            @keyframes glitchTop{0%,88%,100%{opacity:0;transform:translate(0,0)}90%{opacity:.8;transform:translate(-4px,-1px) skewX(-5deg)}92%{opacity:.5;transform:translate(4px,1px) skewX(4deg)}94%{opacity:.7;transform:translate(-2px,0) skewX(0)}96%{opacity:0;transform:translate(0,0)}}
            @keyframes glitchBot{0%,85%,100%{opacity:0;transform:translate(0,0)}87%{opacity:.7;transform:translate(4px,2px)}89%{opacity:.4;transform:translate(-4px,-2px)}91%{opacity:.6;transform:translate(3px,-1px)}93%{opacity:0;transform:translate(0,0)}}
            .ai-pill{display:inline-flex;align-items:center;gap:7px;background:rgba(0,245,255,0.07);border:1px solid rgba(0,245,255,0.22);border-radius:100px;padding:3px 14px;font-size:10px;letter-spacing:0.18em;color:var(--cyan);font-family:'Share Tech Mono',monospace;margin-top:8px;animation:pillGlow 2.2s ease-in-out infinite;}
            .ai-pill-dot{width:6px;height:6px;border-radius:50%;background:var(--cyan);box-shadow:0 0 6px 3px rgba(0,245,255,0.8);animation:blink 1.2s ease-in-out infinite;}
            @keyframes pillGlow{0%,100%{box-shadow:0 0 0 0 rgba(0,245,255,0.12)}50%{box-shadow:0 0 0 8px rgba(0,245,255,0)}}
            .ai-subtitle{font-size:0.74rem;color:rgba(148,163,184,0.65);letter-spacing:0.02em;line-height:1.55;margin-top:10px;}
            /* Scanner zone */
            .scanner-zone{position:relative;border:1px dashed rgba(0,245,255,0.22);border-radius:16px;min-height:195px;display:flex;flex-direction:column;align-items:center;justify-content:center;cursor:pointer;background:rgba(0,245,255,0.02);transition:border-color .3s,background .3s;overflow:hidden;margin-bottom:12px;}
            .scanner-zone:hover{border-color:rgba(0,245,255,0.5);background:rgba(0,245,255,0.04);}
            .brk{position:absolute;width:18px;height:18px;}
            .brk-tl{top:10px;left:10px;border-top:2px solid var(--cyan);border-left:2px solid var(--cyan);}
            .brk-tr{top:10px;right:10px;border-top:2px solid var(--cyan);border-right:2px solid var(--cyan);}
            .brk-bl{bottom:10px;left:10px;border-bottom:2px solid var(--violet);border-left:2px solid var(--violet);}
            .brk-br{bottom:10px;right:10px;border-bottom:2px solid var(--violet);border-right:2px solid var(--violet);}
            .zone-beam{position:absolute;left:10px;right:10px;height:2px;background:linear-gradient(90deg,transparent,var(--cyan),var(--violet),var(--cyan),transparent);box-shadow:0 0 10px 3px rgba(0,245,255,0.5);animation:zoneScan 2.5s ease-in-out infinite;pointer-events:none;}
            @keyframes zoneScan{0%{top:10px;opacity:0}8%{opacity:1}50%{top:calc(100% - 12px);opacity:1}60%{top:calc(100% - 12px);opacity:0}100%{top:10px;opacity:0}}
            .scan-icon{width:50px;height:50px;border-radius:14px;background:rgba(0,245,255,0.08);border:1px solid rgba(0,245,255,0.22);display:flex;align-items:center;justify-content:center;font-size:1.3rem;margin-bottom:12px;position:relative;animation:iconFloat 3s ease-in-out infinite;}
            @keyframes iconFloat{0%,100%{transform:translateY(0)}50%{transform:translateY(-5px)}}
            .scan-icon::after{content:'';position:absolute;inset:-5px;border-radius:18px;border:1px solid rgba(0,245,255,0.18);animation:iconPulse 2s ease-out infinite;}
            @keyframes iconPulse{0%{transform:scale(1);opacity:.6}100%{transform:scale(1.3);opacity:0}}
            .scan-text-main{font-size:0.8rem;color:rgba(226,232,240,0.75);font-weight:500;}
            .scan-text-main span{color:var(--cyan);text-decoration:underline;text-underline-offset:3px;}
            .scan-text-sub{font-size:0.68rem;color:rgba(100,116,139,0.75);margin-top:4px;font-family:'Share Tech Mono',monospace;}
            /* Camera */
            #cameraContainer{display:none;width:100%;padding:0 4px;}
            .cam-feed{width:100%;max-height:215px;border-radius:12px;object-fit:cover;background:#000;border:1px solid rgba(0,245,255,0.2);margin-bottom:10px;}
            .cam-btns{display:flex;gap:10px;width:100%;}
            /* Processing */
            #processingState{display:none;position:absolute;inset:0;background:rgba(2,6,23,0.93);border-radius:16px;flex-direction:column;align-items:center;justify-content:center;gap:14px;z-index:20;}
            .proc-ring{width:50px;height:50px;border-radius:50%;border:2px solid rgba(0,245,255,0.12);border-top-color:var(--cyan);animation:spin .85s linear infinite;}
            .proc-ring-2{position:absolute;width:36px;height:36px;border-radius:50%;border:2px solid rgba(191,0,255,0.12);border-bottom-color:var(--violet);animation:spin 1.3s linear infinite reverse;}
            @keyframes spin{to{transform:rotate(360deg)}}
            .proc-label{font-family:'Share Tech Mono',monospace;font-size:11px;color:var(--cyan);letter-spacing:0.14em;animation:textFlick 1.8s ease-in-out infinite;}
            @keyframes textFlick{0%,100%{opacity:1}50%{opacity:0.35}}
            .proc-steps{display:flex;gap:6px;}
            .proc-dot{width:5px;height:5px;border-radius:50%;background:var(--cyan);animation:dotHop .9s ease-in-out infinite;}
            .proc-dot:nth-child(2){animation-delay:.15s;background:var(--violet);}
            .proc-dot:nth-child(3){animation-delay:.3s;background:var(--pink);}
            @keyframes dotHop{0%,80%,100%{transform:translateY(0)}40%{transform:translateY(-8px)}}
            /* Camera button */
            .btn-camera{width:100%;padding:13px;border-radius:14px;border:1px solid rgba(0,245,255,0.22);background:rgba(0,245,255,0.05);color:var(--cyan);font-family:'Orbitron',sans-serif;font-size:0.68rem;font-weight:700;letter-spacing:0.14em;cursor:pointer;display:flex;align-items:center;justify-content:center;gap:10px;transition:all .3s;position:relative;overflow:hidden;margin-bottom:12px;}
            .btn-camera:hover{background:rgba(0,245,255,0.1);border-color:var(--cyan);box-shadow:0 0 20px rgba(0,245,255,0.14);}
            .btn-camera::before{content:'';position:absolute;top:-50%;left:-60%;width:35%;height:200%;background:linear-gradient(90deg,transparent,rgba(0,245,255,0.1),transparent);transform:skewX(-20deg);animation:btnSheen 3s ease-in-out infinite;}
            @keyframes btnSheen{0%{left:-60%}100%{left:160%}}
            .btn-capture{flex:1;padding:11px 16px;border-radius:12px;border:none;background:linear-gradient(135deg,#059669,#10b981);color:#fff;font-size:0.78rem;font-weight:700;cursor:pointer;transition:all .25s;font-family:'Inter',sans-serif;}
            .btn-capture:hover{filter:brightness(1.15);transform:scale(1.02);}
            .btn-cancel{padding:11px 16px;border-radius:12px;border:1px solid rgba(100,116,139,0.35);background:rgba(51,65,85,0.45);color:#94a3b8;font-size:0.78rem;font-weight:600;cursor:pointer;transition:all .25s;font-family:'Inter',sans-serif;}
            .btn-cancel:hover{background:rgba(51,65,85,0.75);}
            /* Error banner */
            #errorBanner{display:none;overflow:hidden;border-radius:14px;border:1px solid rgba(255,0,80,0.35);background:linear-gradient(135deg,rgba(30,0,10,0.92),rgba(60,0,20,0.8));margin-bottom:12px;}
            .err-body{display:flex;align-items:flex-start;gap:12px;padding:14px 16px;}
            .err-icon{width:32px;height:32px;flex-shrink:0;border-radius:50%;background:rgba(255,0,80,0.14);border:1px solid rgba(255,0,80,0.3);display:flex;align-items:center;justify-content:center;font-size:13px;color:var(--pink);}
            .err-label{font-size:10px;font-weight:700;letter-spacing:0.14em;color:var(--pink);text-transform:uppercase;margin-bottom:4px;font-family:'Share Tech Mono',monospace;}
            #errorMsg{font-size:0.78rem;color:rgba(255,200,210,0.85);line-height:1.5;}
            .err-close{width:26px;height:26px;flex-shrink:0;border-radius:50%;background:rgba(255,0,80,0.09);border:none;color:var(--pink);cursor:pointer;font-size:12px;display:flex;align-items:center;justify-content:center;transition:background .2s;}
            .err-close:hover{background:rgba(255,0,80,0.22);}
            .err-footer{padding:7px 16px;background:rgba(0,0,0,0.22);font-size:10px;color:rgba(255,100,130,0.55);font-family:'Share Tech Mono',monospace;border-top:1px solid rgba(255,0,80,0.1);}
            @keyframes errIn{from{opacity:0;transform:translateY(-14px) scale(.97)}to{opacity:1;transform:translateY(0) scale(1)}}
            @keyframes errOut{from{opacity:1;transform:translateY(0) scale(1)}to{opacity:0;transform:translateY(-14px) scale(.97)}}
            .err-in{animation:errIn .3s cubic-bezier(.22,1,.36,1) forwards;}
            .err-out{animation:errOut .25s ease-in forwards;}
            /* Results */
            #resultsCard{display:none;}
            .res-wrap{border:1px solid rgba(0,245,255,0.12);border-radius:16px;padding:16px;background:rgba(0,245,255,0.02);}
            .res-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;padding-bottom:12px;border-bottom:1px solid rgba(0,245,255,0.09);}
            .res-title{font-family:'Orbitron',sans-serif;font-size:0.6rem;font-weight:700;letter-spacing:0.18em;color:var(--green);text-transform:uppercase;}
            .res-badge{font-size:9px;font-family:'Share Tech Mono',monospace;letter-spacing:0.1em;padding:3px 10px;border-radius:100px;font-weight:700;}
            .badge-valid{background:rgba(57,255,20,.1);color:var(--green);border:1px solid rgba(57,255,20,.22);}
            .badge-amber{background:rgba(245,158,11,.08);color:#f59e0b;border:1px solid rgba(245,158,11,.18);}
            .field-grid{display:flex;flex-direction:column;gap:8px;}
            .field-row{background:rgba(0,245,255,0.025);border:1px solid rgba(0,245,255,0.08);border-radius:12px;padding:10px 14px;transition:border-color .25s;}
            .field-row:hover{border-color:rgba(0,245,255,0.18);}
            .field-label{font-size:9.5px;font-family:'Share Tech Mono',monospace;letter-spacing:0.12em;color:rgba(0,245,255,0.5);text-transform:uppercase;margin-bottom:4px;}
            .field-value{font-size:0.88rem;font-weight:600;color:#f1f5f9;}
            .field-value.mono{font-family:'Share Tech Mono',monospace;font-size:0.98rem;letter-spacing:0.12em;color:var(--cyan);}
            .field-value.green{color:var(--green);}
            .acc-wrap{display:flex;align-items:center;gap:10px;margin-top:4px;}
            .acc-val{font-family:'Share Tech Mono',monospace;font-size:0.98rem;color:var(--cyan);font-weight:700;min-width:46px;}
            .acc-track{flex-grow:1;height:6px;background:rgba(0,245,255,0.07);border-radius:4px;overflow:hidden;border:1px solid rgba(0,245,255,0.08);}
            .acc-fill{height:100%;border-radius:4px;transition:width .85s cubic-bezier(.22,1,.36,1);width:0%;}
            /* Footer */
            .hud-footer{text-align:center;padding:10px;font-family:'Share Tech Mono',monospace;font-size:10px;color:rgba(0,245,255,0.2);border-top:1px solid rgba(0,245,255,0.06);letter-spacing:0.08em;}
            ::-webkit-scrollbar{width:5px;}
            ::-webkit-scrollbar-track{background:var(--dark);}
            ::-webkit-scrollbar-thumb{background:rgba(0,245,255,0.18);border-radius:4px;}
        </style>
    </head>
    <body>
        <canvas id="bgCanvas"></canvas>
        <video class="bg-video" autoplay muted loop playsinline><source src="/luma.mp4" type="video/mp4"></video>
        <div class="grid-overlay"></div>

        <div class="page-wrap">
            <header class="hud-bar">
                <div class="hud-logo">PAN<span>·</span>AI</div>
                <div class="hud-meta">
                    <div class="hud-dot"></div>
                    <span>NEURAL ENGINE v2.0</span>
                    <span class="hud-clock" id="hudClock">--:--:--</span>
                    <span>OCR·ACTIVE</span>
                </div>
            </header>

            <main>
                <div class="ai-card">
                    <div class="card-strip"></div>

                    <div class="title-section">
                        <h1 class="ai-heading" data-text="PAN Card Verification">PAN Card Verification</h1>
                        <div style="display:flex;justify-content:center;margin-top:8px;">
                            <div class="ai-pill"><span class="ai-pill-dot"></span>AI ENGINE ACTIVE</div>
                        </div>
                        <p class="ai-subtitle">Upload or drop a clear PAN card image. The AI neural engine will automatically extract and validate all KYC parameters.</p>
                    </div>

                    <div class="scanner-zone" id="dropZone">
                        <input type="file" id="fileInput" accept="image/png,image/jpeg,image/jpg" style="display:none;">
                        <div class="brk brk-tl"></div><div class="brk brk-tr"></div>
                        <div class="brk brk-bl"></div><div class="brk brk-br"></div>
                        <div class="zone-beam"></div>

                        <div id="uploadPrompt" style="text-align:center;pointer-events:none;padding:18px 0;">
                            <div class="scan-icon">📄</div>
                            <p class="scan-text-main"><span>Click to select</span> or drag and drop</p>
                            <p class="scan-text-sub">PNG · JPG · JPEG &nbsp;|&nbsp; FULL RESOLUTION SUPPORTED</p>
                        </div>

                        <div id="cameraContainer">
                            <video id="webcamVideo" class="cam-feed" autoplay playsinline></video>
                            <div class="cam-btns">
                                <button id="captureBtn" class="btn-capture">⚡ Capture &amp; Scan</button>
                                <button id="cancelCameraBtn" class="btn-cancel">✕ Cancel</button>
                            </div>
                        </div>

                        <div id="processingState">
                            <div style="position:relative;width:50px;height:50px;display:flex;align-items:center;justify-content:center;">
                                <div class="proc-ring"></div>
                                <div class="proc-ring-2"></div>
                                <span style="font-size:16px;position:absolute;">🔍</span>
                            </div>
                            <div class="proc-label">NEURAL SCAN IN PROGRESS</div>
                            <div class="proc-steps">
                                <div class="proc-dot"></div><div class="proc-dot"></div><div class="proc-dot"></div>
                            </div>
                        </div>
                    </div>

                    <button class="btn-camera" id="cameraToggleBtn" type="button">
                        <span>📹</span> ACTIVATE LIVE CAMERA SCAN
                    </button>

                    <div id="errorBanner">
                        <div class="err-body">
                            <div class="err-icon">⚠</div>
                            <div style="flex-grow:1;min-width:0;">
                                <div class="err-label">⬡ PAN Card Not Detected</div>
                                <div id="errorMsg">Uploaded image does not match the formal PAN Card layout.</div>
                            </div>
                            <button class="err-close" id="errorClose">✕</button>
                        </div>
                        <div class="err-footer">💡 TIP: Good lighting · Full card visible · Not blurry · Try camera scan</div>
                    </div>

                    <div id="resultsCard">
                        <div class="res-wrap">
                            <div class="res-header">
                                <span class="res-title">⬡ Extracted KYC Parameters</span>
                                <span id="badge" class="res-badge badge-amber">SCANNING</span>
                            </div>
                            <div class="field-grid">
                                <div class="field-row">
                                    <div class="field-label">Scan Visibility Accuracy</div>
                                    <div class="acc-wrap">
                                        <span id="resAccuracy" class="acc-val">---</span>
                                        <div class="acc-track"><div id="accuracyBar" class="acc-fill"></div></div>
                                    </div>
                                </div>
                                <div class="field-row">
                                    <div class="field-label">Permanent Account Number (PAN)</div>
                                    <div id="resPan" class="field-value mono">---</div>
                                </div>
                                <div class="field-row">
                                    <div class="field-label">Full Legal Name</div>
                                    <div id="resName" class="field-value green">---</div>
                                </div>
                                <div class="field-row">
                                    <div class="field-label">Father's / Guardian Name</div>
                                    <div id="resFather" class="field-value">---</div>
                                </div>
                                <div class="field-row">
                                    <div class="field-label">Date of Birth</div>
                                    <div id="resDob" class="field-value mono" style="color:#f1f5f9;letter-spacing:.06em;">---</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </main>

            <footer class="hud-footer">POWERED BY OPENCV · FASTAPI · TESSERACT OCR &nbsp;|&nbsp; NO DATA STORED ON SERVER</footer>
        </div>

        <script>
            // Live clock
            function tick(){const n=new Date();document.getElementById('hudClock').textContent=n.toLocaleTimeString('en-IN',{hour12:false});}
            tick();setInterval(tick,1000);

            // Full-page neural canvas
            (function(){
                const C=document.getElementById('bgCanvas');
                const cx=C.getContext('2d');
                let W,H;
                function resize(){W=C.width=window.innerWidth;H=C.height=window.innerHeight;}
                resize();window.addEventListener('resize',resize);
                const N=55,COLS=['#00f5ff','#bf00ff','#ff0080','#7c3aed','#0ea5e9'];
                const nodes=Array.from({length:N},()=>({
                    x:Math.random()*1600,y:Math.random()*900,
                    vx:(Math.random()-.5)*.5,vy:(Math.random()-.5)*.5,
                    r:1.2+Math.random()*2,col:COLS[Math.floor(Math.random()*COLS.length)],
                    ph:Math.random()*Math.PI*2,sp:.018+Math.random()*.022
                }));
                function draw(){
                    cx.clearRect(0,0,W,H);
                    const sx=W/1600,sy=H/900;
                    for(let i=0;i<N;i++){for(let j=i+1;j<N;j++){
                        const dx=nodes[i].x-nodes[j].x,dy=nodes[i].y-nodes[j].y,d=Math.sqrt(dx*dx+dy*dy);
                        if(d<160){
                            const a=(1-d/160)*.28;
                            const g=cx.createLinearGradient(nodes[i].x*sx,nodes[i].y*sy,nodes[j].x*sx,nodes[j].y*sy);
                            g.addColorStop(0,nodes[i].col);g.addColorStop(1,nodes[j].col);
                            cx.beginPath();cx.moveTo(nodes[i].x*sx,nodes[i].y*sy);cx.lineTo(nodes[j].x*sx,nodes[j].y*sy);
                            cx.strokeStyle=g;cx.globalAlpha=a;cx.lineWidth=.8;cx.stroke();
                        }
                    }}
                    nodes.forEach(n=>{
                        n.x+=n.vx;n.y+=n.vy;n.ph+=n.sp;
                        if(n.x<0||n.x>1600)n.vx*=-1;if(n.y<0||n.y>900)n.vy*=-1;
                        const p=.5+.5*Math.sin(n.ph);
                        const grd=cx.createRadialGradient(n.x*sx,n.y*sy,0,n.x*sx,n.y*sy,n.r*4*sx);
                        grd.addColorStop(0,n.col);grd.addColorStop(1,'transparent');
                        cx.beginPath();cx.arc(n.x*sx,n.y*sy,n.r*4*sx,0,Math.PI*2);
                        cx.fillStyle=grd;cx.globalAlpha=p*.45;cx.fill();
                        cx.beginPath();cx.arc(n.x*sx,n.y*sy,n.r*p*sx,0,Math.PI*2);
                        cx.fillStyle=n.col;cx.globalAlpha=.8;cx.fill();
                    });
                    cx.globalAlpha=1;requestAnimationFrame(draw);
                }
                draw();
            })();

            // DOM refs
            const dropZone=document.getElementById('dropZone');
            const fileInput=document.getElementById('fileInput');
            const uploadPrompt=document.getElementById('uploadPrompt');
            const procState=document.getElementById('processingState');
            const resultsCard=document.getElementById('resultsCard');
            const errorBanner=document.getElementById('errorBanner');
            const errorMsg=document.getElementById('errorMsg');
            const errorClose=document.getElementById('errorClose');
            const camToggle=document.getElementById('cameraToggleBtn');
            const camContainer=document.getElementById('cameraContainer');
            const webcamVideo=document.getElementById('webcamVideo');
            const captureBtn=document.getElementById('captureBtn');
            const cancelCam=document.getElementById('cancelCameraBtn');
            let stream=null;

            // Error helpers
            function showErr(msg){
                errorMsg.textContent=msg;
                errorBanner.style.display='block';
                errorBanner.classList.remove('err-out');
                void errorBanner.offsetWidth;
                errorBanner.classList.add('err-in');
                resultsCard.style.display='none';
            }
            function hideErr(){
                if(errorBanner.style.display==='none')return;
                errorBanner.classList.remove('err-in');
                errorBanner.classList.add('err-out');
                errorBanner.addEventListener('animationend',()=>{
                    errorBanner.style.display='none';
                    errorBanner.classList.remove('err-out');
                },{once:true});
            }
            errorClose.addEventListener('click',hideErr);

            // Drop zone
            dropZone.addEventListener('click',e=>{if(!stream&&!e.target.closest('button')&&!e.target.closest('video'))fileInput.click();});
            dropZone.addEventListener('dragover',e=>{e.preventDefault();if(!stream)dropZone.style.borderColor='rgba(0,245,255,.55)';});
            dropZone.addEventListener('dragleave',()=>{dropZone.style.borderColor='';});
            dropZone.addEventListener('drop',e=>{e.preventDefault();dropZone.style.borderColor='';if(!stream&&e.dataTransfer.files.length)sendFile(e.dataTransfer.files[0]);});
            fileInput.addEventListener('change',e=>{if(e.target.files.length)sendFile(e.target.files[0]);fileInput.value='';});

            // Camera
            camToggle.addEventListener('click',async e=>{
                e.stopPropagation();hideErr();
                try{
                    stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:'environment'},width:{ideal:1280},height:{ideal:720}}});
                    webcamVideo.srcObject=stream;
                    uploadPrompt.style.display='none';camContainer.style.display='block';camToggle.style.display='none';
                }catch(err){
                    const m={NotFoundError:'No camera device found.',NotAllowedError:'Camera permission denied.',NotReadableError:'Camera in use by another app.'};
                    showErr('📷 '+(m[err.name]||'Camera error: '+err.message));
                }
            });
            cancelCam.addEventListener('click',e=>{e.stopPropagation();stopCam();});
            captureBtn.addEventListener('click',e=>{
                e.stopPropagation();if(!stream)return;
                const cv=document.createElement('canvas');
                cv.width=webcamVideo.videoWidth||640;cv.height=webcamVideo.videoHeight||480;
                cv.getContext('2d').drawImage(webcamVideo,0,0,cv.width,cv.height);
                cv.toBlob(blob=>{if(blob){stopCam();sendFile(new File([blob],'capture.jpg',{type:'image/jpeg'}));}},'image/jpeg',.95);
            });
            function stopCam(){
                if(stream){stream.getTracks().forEach(t=>t.stop());stream=null;}
                webcamVideo.srcObject=null;
                camContainer.style.display='none';
                uploadPrompt.style.display='';
                camToggle.style.display='';
            }

            // Main OCR handler
            async function sendFile(file){
                stopCam();hideErr();
                if(!file.type.match('image.*')){showErr('⛔ Invalid file. Upload PNG, JPG, or JPEG.');return;}
                procState.style.display='flex';resultsCard.style.display='none';
                const fd=new FormData();fd.append('file',file);
                try{
                    const res=await fetch('/api/v1/scan-pan',{method:'POST',body:fd});
                    if(!res.ok){
                        let detail='This image does not match a formal PAN card. Please upload a valid PAN card.';
                        try{const j=await res.json();if(j.detail)detail=j.detail;}catch(_){}
                        showErr(detail);return;
                    }
                    const d=await res.json();
                    document.getElementById('resPan').textContent=d.pan_number||'NOT FOUND';
                    document.getElementById('resName').textContent=d.name||'NOT FOUND';
                    document.getElementById('resFather').textContent=d.father_name||'NOT FOUND';
                    document.getElementById('resDob').textContent=d.date_of_birth||'NOT FOUND';
                    const acc=d.visibility_accuracy||0;
                    document.getElementById('resAccuracy').textContent=acc.toFixed(1)+'%';
                    const bar=document.getElementById('accuracyBar');
                    bar.style.width=acc+'%';
                    bar.style.background=acc>75?'linear-gradient(90deg,#10b981,#00f5ff)':acc>40?'linear-gradient(90deg,#f59e0b,#fbbf24)':'linear-gradient(90deg,#ef4444,#f87171)';
                    const badge=document.getElementById('badge');
                    if(d.is_valid_individual){badge.textContent='VALID INDIVIDUAL';badge.className='res-badge badge-valid';}
                    else{badge.textContent='NOT RECOGNIZED AS INDIVIDUAL';badge.className='res-badge badge-amber';}
                    resultsCard.style.display='block';
                }catch(err){showErr('🌐 Network error: '+err.message);}
                finally{procState.style.display='none';}
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)



# --- API BACKEND POST ROUTE ---
@app.post("/api/v1/scan-pan", summary="Scan an uploaded PAN card image")
async def scan_pan_card(file: UploadFile = File(...)):
    """Accepts an image file (JPEG/PNG), processes it, and returns structured data fields."""
    extension = file.filename.split(".")[-1].lower()
    if extension not in ["jpg", "jpeg", "png", "blob"]:
        raise HTTPException(status_code=400, detail="Invalid file format.")

    try:
        file_bytes = await file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Corrupted image file.")

        structured_data = ocr_engine.scan_image(img)
        
        if not structured_data.get("is_pan_card", False):
            raise HTTPException(
                status_code=400, 
                detail="Uploaded image does not match the formal PAN Card layout. Please scan or upload a valid PAN card."
            )

        return JSONResponse(status_code=200, content=structured_data)

    except HTTPException as he:
        raise he
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# --- UNIFIED STARTUP CONTROLLER LOGIC ---
if __name__ == "__main__":
    import uvicorn
    # Deployment platforms (Render, Railway, Heroku, etc.) inject PORT as an
    # environment variable and require the server to bind on 0.0.0.0.
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0"
    print(f"\n🚀 Starting Full-Stack Application Gateway on http://{host}:{port}")
    print("👉 Open your browser to the public URL to interact with the visual interface!\n")
    uvicorn.run("app:app", host=host, port=port, reload=False)
