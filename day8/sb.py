import cv2
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk

def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            # Read the image using OpenCV
            image = cv2.imread(file_path)
            if image is None:
                raise ValueError("Could not open or find the image.")
            
            # Convert to black and white
            bw_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Display the images
            show_images(image, bw_image)
        except Exception as e:
            messagebox.showerror("Error", str(e))

def show_images(original, bw):
    # Convert images to PIL format for Tkinter
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    bw = cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)

    original_image = Image.fromarray(original)
    bw_image = Image.fromarray(bw)
    
    original_photo = ImageTk.PhotoImage(original_image)
    bw_photo = ImageTk.PhotoImage(bw_image)
    
    # Create new windows to display images
    original_window = tk.Toplevel(root)
    original_window.title("Original Image")
    original_label = tk.Label(original_window, image=original_photo)
    original_label.image = original_photo
    original_label.pack()

    bw_window = tk.Toplevel(root)
    bw_window.title("Black and White Image")
    bw_label = tk.Label(bw_window, image=bw_photo)
    bw_label.image = bw_photo
    bw_label.pack()

# Set up the main application window
root = tk.Tk()
root.title("Image Converter")

# Create a button to load an image
load_button = tk.Button(root, text="Load Image", command=load_image)
load_button.pack(pady=20)

# Start the GUI event loop
root.mainloop()