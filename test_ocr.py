
import cv2
import pytesseract
import tkinter as tk
from tkinter import filedialog, messagebox

# ─── CONFIGURATION: SET YOUR TESSERACT PATH HERE ───
# If you are on Windows, uncomment the line below and fix the path:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# ───────────────────────────────────────────────────

# Open a file dialog
root = tk.Tk()
root.withdraw()  # Hide the main window

print("Opening file dialog...")
file_path = filedialog.askopenfilename(title="Select an Image to Test", filetypes=[("Images", "*.jpg;*.png;*.jpeg")])

if file_path:
    print(f"Selected: {file_path}")

    # Load Image
    img = cv2.imread(file_path)

    if img is None:
        messagebox.showerror("Error", "Could not open the image file.")
    else:
        print("Starting OCR (This might take 5-10 seconds)...")

        try:
            # Run OCR
            text = pytesseract.image_to_string(img, config='--psm 6')

            # Show result
            print("OCR Complete.")
            messagebox.showinfo("OCR Test Result", f"Computer Read:\n\n{text}")

        except Exception as e:
            print(f"Error: {e}")
            messagebox.showerror("OCR Test Failed", str(e))
else:
    print("No file selected.")