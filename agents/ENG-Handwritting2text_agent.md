The following code was created by Grok using the following links of Issac Newtons handwritten
document on Hermes Tristmegistus
Text link: https://webapp1.dlib.indiana.edu/newton/mss/norm/ALCH00017
Handwritten Image link: https://webapp1.dlib.indiana.edu/newton/html/metsnav3.html#mets=https%3A%2F%2Fpurl.dlib.indiana.edu%2Fiudl%2Fnewton%2Fmets%2FALCH00017&page=3
Additional notes are located below the code snippet 





import cv2
import pytesseract
import re
import numpy as np
from PIL import Image

# Configure pytesseract path (update based on your system)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows example
# For Linux/Mac, ensure Tesseract is installed and accessible in PATH

def preprocess_image(image_path):
    """
    Preprocess the image to improve OCR accuracy for historical manuscripts.
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Increase contrast for old manuscripts
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # Adaptive thresholding to binarize the image
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Deskew the image to correct for slight rotations
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return deskewed

def extract_text(image):
    """
    Extract text using Tesseract with settings for historical English.
    """
    # Custom Tesseract configuration for better handling of old scripts
    custom_config = r'--oem 1 --psm 6 -l eng'
    text = pytesseract.image_to_string(image, config=custom_config)
    return text

def clean_text(text):
    """
    Clean and normalize text to modern plain English, handling 17th-century conventions.
    """
    # Remove extra whitespace and newlines
    cleaned = re.sub(r'\s+', ' ', text.strip())

    # Normalize common historical spellings and abbreviations
    replacements = {
        r'\bye\b': 'the',  # 'ye' to 'the'
        r'\byt\b': 'that',  # 'yt' to 'that'
        r'\bwch\b': 'which',  # 'wch' to 'which'
        r'\bwth\b': 'with',  # 'wth' to 'with'
        r'\b&b': 'and',  # Ampersand to 'and'
        r'[Ff]\s*?\b': 'f',  # Handle long 's' (often misread as 'f')
    }
    for old, new in replacements.items():
        cleaned = re.sub(old, new, cleaned, flags=re.IGNORECASE)

    # Fix common OCR errors (e.g., '1' for 'I', '0' for 'O')
    cleaned = cleaned.replace('1', 'I').replace('0', 'O')

    # Remove stray punctuation or symbols not typical in plain English
    cleaned = re.sub(r'[^\w\s.,!?]', '', cleaned)

    return cleaned

def handwriting_to_plain_english(image_path):
    """
    Process a handwritten manuscript image and return plain English text.
    """
    # Preprocess the image
    processed_image = preprocess_image(image_path)

    # Extract text
    raw_text = extract_text(processed_image)

    # Clean and normalize the text
    plain_text = clean_text(raw_text)

    return plain_text

# Example usage
if __name__ == "__main__":
    # Replace with the path to your manuscript image (e.g., downloaded from the provided link)
    image_path = "newton_manuscript_page.jpg"
    try:
        result = handwriting_to_plain_english(image_path)
        print("Extracted Plain English Text:")
        print(result)
    except Exception as e:
        print(f"Error processing image: {e}")



Additional notes from Grok:
Limitations and Next Steps
Historical Script Challenges: Tesseract’s default English model struggles with 17th-century scripts due to unique letter forms (e.g., long ‘s’, ligatures). For better accuracy, you could:
Use a pre-trained model like Tesseract’s script-specific models or train a custom model with historical English data.
Integrate a specialized OCR tool like Transkribus, which is designed for historical manuscripts.
Example-Specific Adjustments: Without seeing the exact text or image content, I included generic normalization rules. If you share a snippet of the expected text or specific challenges (e.g., certain abbreviations, symbols), I can refine the clean_text function.
Testing: If you have a specific page or text excerpt from ALCH00017, I can simulate the output or adjust the code to handle its unique features (e.g., alchemical symbols, Latin phrases).        