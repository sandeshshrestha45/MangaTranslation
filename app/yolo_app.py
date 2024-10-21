import cv2
import tkinter as tk
import numpy as np
from tkinter import messagebox
from PIL import Image, ImageTk
from utils import test_fill_process, put_text_inside_bubble_pillow
class YOLOApp(tk.Tk):
    def __init__(self, bbox,cv_image,*args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.title("YOLO Image Viewer")
        self.attributes('-fullscreen', True)  # Set fullscreen mode
        
        self.image_frame = tk.Frame(self)
        self.image_frame.pack(side="left", fill="both", expand=True)
        
        self.side_frame = tk.Frame(self, height=200)
       
       
        self.side_frame.pack(side="right", fill="both", expand=False)

        self.canvas = tk.Canvas(self.image_frame)
        vbar=tk.Scrollbar(self.image_frame,orient=tk.VERTICAL)
        vbar.pack(side=tk.RIGHT,fill=tk.Y)
        
        vbar.config(command=self.canvas.yview)
        self.canvas.pack(side="top", fill="both", expand=True)
        self.canvas.config( yscrollcommand=vbar.set)
     

        self.side_label = tk.Label(self.side_frame, text="Selected Text:")
        self.side_label.pack(side="top")
        
        self.text_area = tk.Text(self.side_frame, wrap=tk.WORD, height=10)
        self.text_area.pack( expand=False)

        self.change_button = tk.Button(self.side_frame,text="change", command= self.change_text)
        self.change_button.pack()

        self.change_button = tk.Button(self.side_frame,text="Save", command= self.save_image)
        self.change_button.pack()

        self.image = Image.fromarray(cv_image)
        self.photo = ImageTk.PhotoImage(self.image)
        
        self.image_id = self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        
        self.canvas.bind("<Configure>", self.resize_image)
        self.bind("<Escape>", self.close_window)
        # self.canvas.bind_all("<MouseWheel>", self.on_motion)
        
        # Bounding boxes retrieved from YOLO model
        self.bboxes =  bbox
        
        self.draw_boxes()
        
        self.canvas.bind("<Button-1>", self.on_click)
    
    def resize_image(self, event):
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
        self.canvas.itemconfig(self.image_id, image=self.photo)
        
    def draw_boxes(self):
        for bbox in self.bboxes:
            box = bbox["bbox"]
            self.canvas.create_rectangle(box[0], box[1], box[2], box[3], outline="red")
    
    def on_click(self, event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        for bbox in self.bboxes:
            box = bbox["bbox"]
            if box[0] < x < box[2] and box[1] < y < box[3]:
                # if bbox["class"] == "text":
                self.text_area.delete(1.0, tk.END)
                self.text_area.insert(tk.END, bbox["text"])
                self.selected_box = bbox
                break
    
    def change_text(self):
        text = self.text_area.get("1.0",tk.END)
        # msg=messagebox.showinfo( text)
        x,y,w,h = self.selected_box["bbox"][0],self.selected_box["bbox"][1],self.selected_box["bbox"][2],self.selected_box["bbox"][3]
        crop = self.image.crop((x,y,w,h))
        opencv_image = np.array(crop)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
        changed_text, text_color = test_fill_process(opencv_image,text.strip(),self.selected_box["id"])
        changed_text = put_text_inside_bubble_pillow(changed_text,text.strip(),"1",text_color=text_color, box=[x,y,w,h] )
        changed_text = Image.fromarray(changed_text)
        self.image.paste(changed_text,(x,y))
        self.photo = ImageTk.PhotoImage(self.image)
      
        self.update_image()
        
        # crop.show()

    def update_image(self):
        # Clear canvas
        self.canvas.delete("all")
        
        # Redraw original image
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        
        # Draw bounding boxes with updated text
        self.draw_boxes()
    def save_image(self):
        self.image.save("img/detections/updated.png")
        self.image.show()
        self.destroy()
    def close_window(self, event=None):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.destroy()
    
    def on_motion(self, event):
        print("Scroll --- ")
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

if __name__ == "__main__":
    bbox = [{'id': '0_0', 'bbox': [1044, 3, 1362, 321], 'text': "Jinbei! They've found your successor!! That's great, isn't it?"}, {'id': '0_1', 'bbox': [548, 2, 807, 242], 'text': 'Are you saying that...?! Ace.'}, {'id': '0_2', 'bbox': [77, 43, 283, 295], 'text': 'At this moment...'}, {'id': '1_0', 'bbox': [1170, 565, 1353, 781], 'text': 'The former king...!!'}, {'id': '1_1', 'bbox': [885, 567, 1173, 826], 'text': 'Is there still a current king? The pirate world has changed.'}, {'id': '2_0', 'bbox': [611, 567, 812, 799], 'text': 'It has nothing to do with me!!'}, {'id': '2_1', 'bbox': [597, 799, 813, 1071], 'text': 'I will never become a part of the central core of life!!'}, {'id': '2_2', 'bbox': [74, 565, 343, 822], 'text': 'Nevertheless, I must gather information, even about the Snake Princess!!'}, {'id': '3_0', 'bbox': [1119, 1132, 1368, 1401], 'text': 'Kishishishi~~!!'}, {'id': '3_1', 'bbox': [652, 1771, 890, 2035], 'text': "But isn't Moria-sama one of the Seven Warlords of the Sea!?"}, {'id': '3_2', 'bbox': [651, 1130, 967, 1408], 'text': "He's known as the 'Paw Human.' What kind of ability is that!?"}, {'id': '4_0', 'bbox': [75, 1156, 259, 1377], 'text': '..........'}, {'id': '5_0', 'bbox': [347, 1444, 635, 1674], 'text': 'The Seven Warlords of the Sea must be directly under the government, right!?'}, {'id': '5_1', 'bbox': [115, 1443, 360, 1669], 'text': "And even Kumasan... I wouldn't expect that from him..."}]
    image = cv2.imread("img/test/one-piece-chapter-1100-ref-0.jpg")
    app = YOLOApp(bbox ,image)
    app.mainloop()
