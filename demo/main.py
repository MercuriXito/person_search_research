from typing import Dict, List

import os
import tkinter as tk
from tkinter import ttk
import numpy as np

from tkinter import Text, Button, Frame, Label
from tkinter.messagebox import showinfo
import tkinter.filedialog as tkfiledialog
import tkinter.font as tkfont

from demo.widgets import BoxSelectImageCanvas, \
    BoxFixSizedImageCanvas, _box_area
from demo.search_tools import search


# predefined supported options
supported_gallery = ["CUHK-SYSU", "PRW"]
supported_methods = ["baseline", "cmm", "acae"]
image_extentions = ["jpg", "jpeg", "png", "bmp"]


# utils functions
def get_font_size():
    return tkfont.Font(font="TkDefaultFont").configure()["size"]


def is_supported_image(path):
    if not os.path.isfile(path):
        return False
    ext = path.split(".")[-1]
    if ext in image_extentions:
        return True
    return False


class MainFrame(ttk.Frame):
    def __init__(self, root, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        self._init_layout()
        self.bind_events()

    def _init_layout(self):
        w, h = 128, 256
        total_width = w * 6
        total_height = h * 2
        qframe_width = w * 2
        self.search_frame = SearchResPanel(self, pw=w, ph=h)
        self.query_frame = QueryPanel(self, qframe_width, total_height)
        self.query_frame.pack(anchor="n", side="left", fill="y")
        self.search_frame.pack(anchor="n", side="left", fill="y")

    def bind_events(self):

        # search
        def go_search():
            method = self.query_frame.methods_selector.get()
            gallery = self.query_frame.gallery_selector.get()
            print(method, gallery)
            images, boxes = self.query_frame.query_box.get_values()
            if images is None:
                print("Image is None, aborted.")
                return
            request_search_args = {
                "method": method,
                "gallery": gallery,
                "images": [images],
                "boxes": [np.asarray(boxes)],
            }
            # TODO: post to search.
            search_results = search(**request_search_args)
            self.search_frame.set_top10_result(search_results)
            showinfo(
                title="search info",
                message=f"Search {method} on {gallery} completed.")

        self.query_frame.search_button.config(command=go_search)


class QueryPanel(Frame):
    def __init__(self, master, w, h):
        super().__init__(master, height=h, width=w)
        self.w = w
        self.h = h
        print(self.w, self.h)
        # self.config(bg="green", width=self.w, height=self.h)
        self._init_layout()
        self.bind_events()

    def _init_layout(self):

        query_area_label_text = "Query Options"
        query_area_label_text += '-' * (self.w // get_font_size() - len(query_area_label_text) - 1)
        self.query_area_label = Label(
            self, text=query_area_label_text,
        )
        self.query_area_label.pack(side="top", anchor="n")

        # query selection area
        self.query_select_label = Label(self, text="select query file:")
        self.query_select_label.pack(side="top", anchor="w")

        self.query_select_area = Frame(self)
        self.query_select_area.pack(side="top", anchor="n")
        self.query_file_button = Button(self.query_select_area)
        self.query_file_show = Text(self.query_select_area)

        self.query_file_button.config(
            height=1,
            text="select",
        )
        self.query_file_show.config(
            width=int(self.w * 0.75 / get_font_size()),
            height=1,
        )
        self.query_file_show.grid(row=0, column=0)
        self.query_file_button.grid(row=0, column=1)
        self.query_select_area.pack(side="top", anchor="n")
        self.query_select_area.config(bg="black")

        # show area
        query_show_label_text = "Query Image"
        query_show_label_text += '-' * (self.w // get_font_size() - len(query_show_label_text) - 1)
        self.query_show_label = Label(
            self, text=query_show_label_text,
        )
        self.query_show_label.pack(side="top", anchor="n")

        self.query_show_area = Frame(self)
        self.query_show_area.pack(side="top", anchor="n")

        self.query_image_label = Label(self.query_show_area, text="Query Person")
        self.query_image_label.pack(side="top", anchor="w")

        self.query_box = BoxFixSizedImageCanvas(
            self.query_show_area, 128, 256,
            highlightthickness=2, highlightcolor="black",
            relief="ridge")
        self.query_box.pack(side="top", anchor="n")

        self.query_crop_button = Button(
            self.query_show_area, text="Select Person")
        self.query_crop_button.pack(side="top", anchor="n")

        # search area
        self.search_area = Frame(self)
        self.search_area.pack(side="top", anchor="n")

        # search area label
        search_area_label_text = "Search Options"
        search_area_label_text += '-' * (
            self.w // get_font_size() - len(search_area_label_text) - 1)
        self.search_area_label = Label(
            self.search_area, text=search_area_label_text,
        )
        self.search_area_label.pack(side="top", anchor="n")

        # search area options list
        self.search_area_opt_lists = Frame(self.search_area)
        self.search_area_opt_lists.pack(side="top", anchor="n")

        self.gallery_selector_label = Label(
            self.search_area_opt_lists,
            text="Gallery:")
        self.gallery_selector_label.grid(row=0, column=0)
        self.gallery_selector = ttk.Combobox(
            self.search_area_opt_lists, state="readonly")
        self.gallery_selector.config(
            values=supported_gallery,
            exportselection=0)
        self.gallery_selector.current(0)
        self.gallery_selector.grid(row=0, column=1)

        self.methods_label = Label(self.search_area_opt_lists, text="Methods:")
        self.methods_label.grid(row=1, column=0)
        self.methods_selector = ttk.Combobox(
            self.search_area_opt_lists, state="readonly")
        self.methods_selector.config(
            values=supported_methods,
            exportselection=0)
        self.methods_selector.current(0)
        self.methods_selector.grid(row=1, column=1)

        # go search !!!!!!!
        self.search_button = Button(self.search_area, text="Go Search!")
        self.search_button.pack(side="top", anchor="n")

    def bind_events(self):
        """ bind event of each button, and other events.
        """
        # open files.
        def select_images():
            filepath = tkfiledialog.askopenfilename(
                filetypes=[
                    (
                        "Images",
                        " ".join(["*.{}".format(x) for x in image_extentions])
                    )
                ]
            )
            self.query_file_show.delete('0.0', tk.END)
            self.query_file_show.insert('0.0', filepath)
            if len(filepath) == 0 or not os.path.exists(filepath) or \
                    not is_supported_image(filepath):
                print("Invalid filepath {}".format(filepath))
                return
            self.query_box.set_image_from_path(filepath)
            self.query_box.show_images()
        self.query_file_button.config(command=select_images)

        # select box in target image.
        def go_select_boxes():
            image, _ = self.query_box.get_values()
            if image is None:
                print("Image is None, please select query image first.")
                return
            
            top_level = tk.Toplevel()
            top_level.title("Complete Image")
            c_canvas = BoxSelectImageCanvas(top_level)
            c_canvas.set_image_from_ndarray(image.copy())
            c_canvas.show_images()
            c_canvas.pack()

            def _grab_box(event):
                box = c_canvas.get_selected_box()
                print(box)
                if _box_area(box) > 0:
                    self.query_box.set_box_of_image(box)
                else:
                    print("Invalid box {}".format(box))
                    print("box {}".format(self.query_box.box))
                self.query_box.show_images()

            c_canvas.bind("<Destroy>", _grab_box)
            top_level.grab_set()  # set dialog to modal.

        self.query_crop_button.config(command=go_select_boxes)


class SearchResPanel(Frame):
    def __init__(self, master, pw, ph, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.pw = pw
        self.ph = ph
        self._init_layout()

    def _init_layout(self):
        # hold-on item
        root_path = "demo/resources"
        images = [f"Top-{i+1}.png" for i in range(10)]

        all_canvas = []
        for i in range(10):
            canvas = BoxFixSizedImageCanvas(self, self.pw, self.ph)
            canvas.set_image_from_path(os.path.join(root_path, images[i]))
            canvas.show_images()
            canvas.grid(column=i%5, row=i//5, padx=0, pady=0, ipadx=0, ipady=0)
            all_canvas.append(canvas)
        self.res_canvas = all_canvas

    def set_top10_result(self, results: List[Dict]):
        for canvas, item in zip(self.res_canvas, results):
            img = item["image"]
            box = item["box"]
            canvas.set_image_from_ndarray(img, box)
            canvas.show_images()


def draw_main_framework():
    root = tk.Tk()
    frame = MainFrame(root, padding=0)
    # search options area.
    frame.pack()
    root.resizable(0, 0)
    root.mainloop()


if __name__ == '__main__':
    draw_main_framework()
