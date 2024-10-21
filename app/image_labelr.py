from tkinter import filedialog, Label, Entry, Button, Tk, LEFT, RIGHT
from PIL import Image, ImageTk
import os
import csv
import pandas as pd

class ImageLabeler:
    def __init__(self, master, image_dir):
        self.master = master
        self.image_dir = image_dir
        self.images = [file for file in os.listdir(self.image_dir) if file.lower().endswith(('png', 'jpg', 'jpeg', 'gif'))]
        self.current_image_index = 0
        self.setup_ui()

    def setup_ui(self):
        self.image_label = Label(self.master)
        self.image_label.pack(side="left")

        self.image_label2= Label(self.master)
        self.image_label2.pack(side="right")

        self.label_entry = Entry(self.master)
        self.label_entry.pack(side="right")

        self.whole_img_entry = Entry(self.master)
        self.whole_img_entry.pack(side="right")

        self.next_button = Button(self.master, text="Next", command=self.next_image)
        self.next_button.pack(side="right")

        self.prev_button = Button(self.master, text="Previous", command=self.prev_image)
        self.prev_button.pack(side="left")

        self.save_button = Button(self.master, text="Save Label", command=self.save_label)
        self.save_button.pack(side="right")

        self.update_image()

    def update_image(self):
        if self.images:
            image_path = os.path.join(self.image_dir, self.images[self.current_image_index])
            df = pd.read_csv("labels.csv")
            print(image_path, " --------- ",df.iloc[:, 0].values )
            # breakpoint()
            if not  self.images[self.current_image_index] in df.iloc[:, 0].values:
                image = Image.open(image_path)


                image = image.resize((300,500), Image.LANCZOS)
                photo = ImageTk.PhotoImage(image)
                self.image_label.configure(image=photo)
                self.image_label.image = photo

                new =  self.images[self.current_image_index].split("_")
                # print(original_image, " ------------ ")
                # new = original_image.split("_")
                # path = new[2]+"/"+"_".join(new[3:])
                # if not "img/" in new:
                #     path =  "/media/mrcloud/New Volume/chulo/comitee-ml/img/"+ path
                # else:
                #     path = "/media/mrcloud/New Volume/chulo/comitee-ml/"+path
                path = os.path.join(image_directory, self.images[self.current_image_index])
                print(path,"  -----------------")
                
                # max_width = (self.master.winfo_width() // 2) - 10  # Half the window width minus some padding
                # max_height = self.master.winfo_height() - 100  
                og_image = Image.open(path)
                og_image = og_image.resize((600,600), Image.LANCZOS)
                og_photo = ImageTk.PhotoImage(og_image)
                self.image_label2.configure(image=og_photo)
                self.image_label2.image = og_photo
            else:
                print("Image already labled ")
                self.next_image()

    def next_image(self):
        if self.current_image_index < len(self.images) - 1:
            self.current_image_index += 1
            self.update_image()

    def prev_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.update_image()

    def save_label(self):
        if self.images:
            image_name = self.images[self.current_image_index]
            label = self.label_entry.get()
            with open('labels.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([image_name, label])
            self.label_entry.delete(0, "end")

if __name__ == "__main__":
    root = Tk()
    root.geometry("1024x768")
    # image_directory = filedialog.askdirectory(title='Select Image Directory')
    image_directory = "/home/mrcloud/Documents/projects/manga_download/mangas/panel_dataset"
    app = ImageLabeler(root, image_directory)
    root.mainloop()
