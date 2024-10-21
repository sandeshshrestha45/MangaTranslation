from tkinter import filedialog, Label, Entry, Button, Tk, Frame
from PIL import Image, ImageTk
import os
import csv
import pandas as pd

class ImageLabeler:
    def __init__(self, master, image_dir):
        self.master = master
        self.original_img_path = "/home/mrcloud/Documents/projects/manga_download/mangas/datasets"
        self.image_dir = image_dir
        self.images = [file for file in os.listdir(self.image_dir) if file.lower().endswith(('png', 'jpg', 'jpeg', 'gif'))]
        self.current_image_index = 0
        self.original_image= ""
        self.setup_ui()

    def setup_ui(self):
        # Frame for images
        self.image_frame = Frame(self.master)
        self.image_frame.pack(fill="both", expand=True)
        
        self.image_label = Label(self.image_frame)
        self.image_label.pack(side="left", fill="both", expand=True)

        self.image_label2 = Label(self.image_frame)
        self.image_label2.pack(side="left", fill="both", expand=True)

        # Frame for controls
        self.control_frame = Frame(self.master)
        self.control_frame.pack(fill="x")

        self.prev_button = Button(self.control_frame, text="Previous", command=self.prev_image)
        self.prev_button.pack(side="left")

        self.next_button = Button(self.control_frame, text="Next", command=self.next_image)
        self.next_button.pack(side="left")

        self.label_entry = Entry(self.control_frame)
        self.label_entry.pack(side="left")


        self.save_button = Button(self.control_frame, text="Save Label", command=self.save_label)
        self.save_button.pack(side="left")

        self.update_image()

    def update_image(self):
        if self.images:
            image_path = os.path.join(self.image_dir, self.images[self.current_image_index])
            self.original_image = os.path.join(self.original_img_path,f"{image_path.split("/")[-1].split("-id")[0]}.jpg")
            try:
                df = pd.read_csv("/home/mrcloud/Documents/projects/manga_download/mangas/labels.csv")
                if self.images[self.current_image_index] not in df.iloc[:, 0].values:
                    self.display_image(image_path, self.image_label, (300, 500))
                    self.display_image(self.original_image, self.image_label2, (600, 600))
                    df2 = pd.read_csv("/home/mrcloud/Documents/projects/manga_download/mangas/whole_image.csv")
                    if self.original_image.split("/")[-1] not in df2.iloc[:, 0].values:
                        print("NOt ins csv", "Label _creates", self.original_image.split("/")[-1])
                        self.whole_label = Entry(self.control_frame)
                        self.whole_label.pack(side="left")
                    else:
                        self.whole_label.destroy()

                else:
                    print("Image already labeled")
                    self.next_image()
                

            except FileNotFoundError:
                self.display_image(image_path, self.image_label, (300, 500))
                self.display_image(self.original_image, self.image_label2, (600, 600))
            except Exception as e:
                # raise
                print(e)
            # try :
            #     df2 = pd.read_csv("/home/mrcloud/Documents/projects/manga_download/mangas/whole_image.csv")
            #     if self.original_image.split("/")[-1] not in df2.iloc[:, 0].values:
            #         print("NOt ins csv", "Label _creates", self.original_image.split("/")[-1])
            #         self.whole_label = Entry(self.control_frame)
            #         self.whole_label.pack(side="left")
            #     else:
            #         self.whole_label.destroy()

            # except Exception as e:
            #     # self.whole_label = Entry(self.control_frame)
            #     # self.whole_label.pack(side="left")
            #     print(e)

    def display_image(self, path, label_widget, size):
        image = Image.open(path).resize(size, Image.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        label_widget.configure(image=photo)
        label_widget.image = photo

    def next_image(self):
        try:
            self.whole_label.destroy()
        except Exception as e:
            pass
        if self.current_image_index < len(self.images) - 1:
            self.current_image_index += 1
            self.update_image()

    def prev_image(self):
        self.whole_label.forget()
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.update_image()

    def save_label(self):
        if self.images:
            image_name = self.images[self.current_image_index]
            label = self.label_entry.get()
            with open('/home/mrcloud/Documents/projects/manga_download/mangas/labels.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([image_name, label])
            self.label_entry.delete(0, "end")
        try:
            if self.whole_label.get():
                wh_label = self.whole_label.get()
                with open('/home/mrcloud/Documents/projects/manga_download/mangas/whole_image.csv', 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([self.original_image.split("/")[-1], wh_label])
                self.whole_label.delete(0, "end")
                self.whole_label.forget()
        except Exception as e:
            print(e)


if __name__ == "__main__":
    root = Tk()
    root.geometry("1024x768")
    image_directory = "/home/mrcloud/Documents/projects/manga_download/mangas/panel_dataset"  # Change this to your image directory
    app = ImageLabeler(root, image_directory)
    root.mainloop()
