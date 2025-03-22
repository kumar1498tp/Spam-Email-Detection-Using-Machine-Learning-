import pandas as pd
import warnings
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from PIL import Image
from pytesseract import image_to_string, pytesseract

warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('spam.csv')

# Check for unexpected values
print(df['Category'].unique())  # Debugging step

# Remove rows with invalid Category values
df = df[df['Category'].isin(['ham', 'spam'])]

# Convert labels to numeric values
df['Category'] = df['Category'].map({'ham': 0, 'spam': 1})

# Ensure 'Category' is in integer format
y = df['Category'].astype(int)

# Text processing
v = CountVectorizer()
x = v.fit_transform(df['Message'].str.lower())

# Train the SVM model
model = SVC(kernel='linear')
model.fit(x, y)

# Load and process the image text
pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
img = Image.open('message.jpg')
text = image_to_string(img)

# Transform the text and predict
txt1 = v.transform([text])
res = model.predict(txt1)

if res[0] == 1:
    print('The email is a spam email')
else:
    print('The email is not a spam')
