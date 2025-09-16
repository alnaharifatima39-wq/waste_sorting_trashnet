# src/app/gui_app.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import threading
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from src.models.classifier import WasteClassifier

MODEL_PATH = "models/resnet_trashnet.pth"

class WasteApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Waste Sorting Classifier")
        self.root.geometry("980x680")
        self.root.resizable(False, False)

        # load model (may be missing)
        self.clf = WasteClassifier(MODEL_PATH)

        # camera
        self.cap = None
        self.running = False
        self.frame_count = 0
        self.predict_every = 10  # predict every N frames when streaming

        # layout: left column = image, right column = controls + chart
        self.main_frame = ttk.Frame(root, padding=10)
        self.main_frame.pack(fill="both", expand=True)

        self.left_frame = ttk.Frame(self.main_frame)
        self.left_frame.grid(row=0, column=0, sticky="ns")

        self.right_frame = ttk.Frame(self.main_frame)
        self.right_frame.grid(row=0, column=1, padx=(10,0), sticky="n")

        self.build_left()
        self.build_right()

        # graceful close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def build_left(self):
        # Upload and camera buttons
        btn_frame = ttk.Frame(self.left_frame)
        btn_frame.pack(pady=(0,8))

        self.upload_btn = ttk.Button(btn_frame, text="Upload Image", command=self.upload_image)
        self.upload_btn.grid(row=0, column=0, padx=6)

        self.cam_btn = ttk.Button(btn_frame, text="Start Camera", command=self.toggle_camera)
        self.cam_btn.grid(row=0, column=1, padx=6)

        # Image display
        self.image_panel = ttk.Label(self.left_frame)
        self.image_panel.pack(pady=6)

        # Prediction label
        self.pred_label = ttk.Label(self.left_frame, text="Prediction: N/A", font=("Segoe UI", 14))
        self.pred_label.pack(pady=6)

        # Model ready warning
        if not self.clf.is_ready():
            self.pred_label.configure(text="Model not loaded. Place model at: models/resnet_trashnet.pth")

    def build_right(self):
        # Chart area
        chart_frame = ttk.LabelFrame(self.right_frame, text="Probabilities")
        chart_frame.pack(fill="both", padx=6, pady=6)

        self.fig, self.ax = plt.subplots(figsize=(5,3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

        # History
        hist_frame = ttk.LabelFrame(self.right_frame, text="Session history")
        hist_frame.pack(fill="both", padx=6, pady=6)

        self.history_box = tk.Text(hist_frame, width=30, height=10, state="disabled", wrap="none")
        self.history_box.pack()

        # Controls bottom
        ctl_frame = ttk.Frame(self.right_frame)
        ctl_frame.pack(pady=6)
        ttk.Button(ctl_frame, text="Clear History", command=self.clear_history).grid(row=0, column=0, padx=6)
        ttk.Button(ctl_frame, text="Snapshot (save)", command=self.save_snapshot).grid(row=0, column=1, padx=6)

    def upload_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not path:
            return
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Error", f"Cannot open image: {e}")
            return
        # display and predict
        self.display_and_predict(img, source=path)

    def toggle_camera(self):
        if not self.running:
            # start
            try:
                self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            except Exception:
                self.cap = cv2.VideoCapture(0)
            if not (self.cap and self.cap.isOpened()):
                messagebox.showerror("Camera", "Cannot open camera. Check permissions or camera index.")
                return
            self.running = True
            self.cam_btn.config(text="Stop Camera")
            self._camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self._camera_thread.start()
        else:
            # stop
            self.running = False
            self.cam_btn.config(text="Start Camera")
            if self.cap:
                self.cap.release()
                self.cap = None

    def camera_loop(self):
        while self.running and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                break
            # convert BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            self.frame_count += 1
            # show frame immediately (schedule in main thread)
            self.root.after(0, lambda img=pil_img: self._display_image(img))

            # predict every N frames (do prediction in camera thread to avoid blocking UI loop)
            if self.frame_count % self.predict_every == 0:
                try:
                    preds = self.clf.predict(pil_img, topk=6) if self.clf.is_ready() else None
                except Exception as e:
                    preds = None
                    # schedule error display
                    self.root.after(0, lambda e=e: messagebox.showerror("Prediction error", str(e)))
                # schedule UI update with preds
                self.root.after(0, lambda p=preds: self._update_prediction_ui(p))
            time.sleep(0.02)  # small sleep to reduce CPU use

    def _display_image(self, pil_img, size=(480,360)):
        imgtk = ImageTk.PhotoImage(pil_img.resize(size))
        self.image_panel.imgtk = imgtk
        self.image_panel.configure(image=imgtk)

    def display_and_predict(self, pil_img, source="upload"):
        # display
        self._display_image(pil_img)
        # predict (in separate thread to not block UI)
        threading.Thread(target=self._predict_and_update, args=(pil_img, source), daemon=True).start()

    def _predict_and_update(self, pil_img, source):
        if not self.clf.is_ready():
            self.root.after(0, lambda: messagebox.showinfo("Model missing", "Model not loaded. Place model at models/resnet_trashnet.pth"))
            return
        try:
            preds = self.clf.predict(pil_img, topk=6)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Predict error", str(e)))
            return
        # update UI on main thread
        self.root.after(0, lambda p=preds, s=source: self._update_prediction_ui(p, s))

    def _update_prediction_ui(self, preds, source="live"):
        if not preds:
            return
        top1, top1p = preds[0][0], preds[0][1]
        self.pred_label.config(text=f"Prediction: {top1} ({top1p:.2f})")
        # append history
        self.append_history(f"{time.strftime('%H:%M:%S')} | {source} | {top1} ({top1p:.2f})")

        # update bar chart
        classes, probs = zip(*preds)
        self.ax.clear()
        self.ax.bar(classes, probs, color="tab:blue")
        self.ax.set_ylim(0,1)
        self.ax.set_ylabel("Probability")
        self.ax.set_xticklabels(classes, rotation=30, ha='right')
        self.canvas.draw()

    def append_history(self, text):
        self.history_box.configure(state="normal")
        self.history_box.insert("1.0", text + "\n")
        self.history_box.configure(state="disabled")

    def clear_history(self):
        self.history_box.configure(state="normal")
        self.history_box.delete("1.0", tk.END)
        self.history_box.configure(state="disabled")

    def save_snapshot(self):
        # save current image panel as file
        if not hasattr(self.image_panel, "imgtk"):
            messagebox.showinfo("No image", "No image to save.")
            return
        # convert PhotoImage back to PIL by using last displayed image (we stored imgtk only)
        # easiest: just save using a temporary approach from camera if available
        if self.cap is None:
            # ask for filename
            fpath = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG","*.jpg")])
            if not fpath:
                return
            # grab current displayed image from image_panel (works because we have imgtk)
            pil = ImageTk.getimage(self.image_panel.imgtk)
            pil.save(fpath)
            messagebox.showinfo("Saved", f"Saved snapshot to {fpath}")
        else:
            # if camera running, take a quick capture
            ret, frame = self.cap.read()
            if not ret:
                messagebox.showerror("Error", "Cannot capture frame")
                return
            cv2.imwrite("snapshot.jpg", frame)
            messagebox.showinfo("Saved", "Saved snapshot to snapshot.jpg")

    def on_close(self):
        self.running = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        self.root.destroy()

def main():
    root = tk.Tk()
    style = ttk.Style()
    # use a clean theme; fallback to 'clam' if available
    try:
        style.theme_use('clam')
    except Exception:
        pass
    app = WasteApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
