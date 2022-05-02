import os
import cv2
from copy import deepcopy
import numpy as np
from PIL import ImageTk
import PIL.Image as Image

import tkinter as tk
from tkinter import Canvas

# Constant
global_image_bank = dict()
CV2_COLOR_BLACK = (0, 0, 0)
CV2_COLOR_BLUE = (0, 0, 255)


def _box_area(box, map_int=False):
    if map_int:
        mapping = int
    else:
        mapping = lambda x: x
    x1, y1, x2, y2 = [mapping(x) for x in box]
    return (x2 - x1) * (y2 - y1)


class ScrollItem:
    """ Counter of Scroll to implement scroll event.
    """
    def __init__(self, sensitivity=60, ranges=[5, 2], int_val=0):
        self.int_eval = int_val
        self.counts = int_val
        self.sensitivity = sensitivity
        self.ranges = ranges

    def add_counts(self):
        if self.counts > self.ranges[0] * self.sensitivity:
            return
        self.counts += 10

    def delete_counts(self):
        if self.counts < self.ranges[1] * -1 * self.sensitivity:
            return
        self.counts -= 10

    def get_ratio(self):
        return self.counts / self.sensitivity

    def release(self):
        self.counts = self.int_eval

    def __str__(self) -> str:
        return f"{self.counts}"


class ImageCanvas(Canvas):
    """ Canvas widget providing an easy way for showing images.
    Image item are stored in np.ndarray format for better
    modification.
    """
    def __init__(self, master, img=None, *args, **kwargs):
        self.load_image(img)
        super().__init__(master, *args, **kwargs)
        self.bind_events()

    def set_image_from_ndarray(self, image: np.ndarray):
        assert image.ndim == 2 or image.ndim == 3
        self.image = image
        if not hasattr(self, "img_name"):
            self.img_name = ""
        if not hasattr(self, "img_ext"):
            self.img_ext = ""

    def set_image_from_path(self, path: str):
        # load from image path
        assert os.path.exists(path)
        self.image = np.asarray(Image.open(path))
        path_name = os.path.basename(path)
        path_ext = path_name.split(".")[-1]
        self.path_name = path_name
        self.path_ext = path_ext

    def load_image(self, img):
        if img is None:
            self.image = None
        if isinstance(img, str):
            self.set_image_from_path(img)
        elif isinstance(img, np.ndarray):
            self.set_image_from_ndarray(img)

    def get_img_height(self):
        if not hasattr(self, "image"):
            return -1
        return self.image.shape[0]

    def get_img_width(self):
        if not hasattr(self, "image"):
            return -1
        return self.image.shape[1]

    def show_images(self):
        assert hasattr(self, "image"), "Image of canvas is not assigned, aborted."
        self.config(
            width=self.get_img_width(),
            height=self.get_img_height(),
        )
        opened = ImageTk.PhotoImage(Image.fromarray(self.image))
        # To keep the buffer of opened images.
        global global_image_bank
        global_image_bank[self.__hash__()] = opened
        self.create_image(0, 0, anchor="nw", image=opened)

    def bind_events(self):
        """ ImageCanvas implements two events: MiddleWheel to scale up/down
        image; left mouse key to drag and view image.
        """
        # events related parameters.
        self.scroll_counts = ScrollItem()
        # dragging events related
        self._press_in = False
        self._move_start_x = 0
        self._move_start_y = 0
        self._moved_actual_dx = 0
        self._moved_actual_dy = 0

        # TODO: possible bugs if there is no image of Canvas.
        def _on_scroll_up(event):
            self.scroll_counts.add_counts()
            _change_show(event)

        def _on_scroll_down(event):
            self.scroll_counts.delete_counts()
            if self.scroll_counts.get_ratio() <= 0:
                self._move_start_x = 0
                self._move_start_y = 0
                self._moved_actual_dx = 0
                self._moved_actual_dy = 0
            _change_show(event)

        def _change_show(event):
            iw, ih = self.get_img_width(), self.get_img_height()
            self._show_transformed_images(
                self._calculate_showd_area(), (iw, ih),
                self.scroll_counts.get_ratio()
            )

        def _on_double_click(event):
            self.scroll_counts.release()
            self._move_start_x = 0
            self._move_start_y = 0
            self._moved_actual_dx = 0
            self._moved_actual_dy = 0
            self.show_images()

        # ScrollEvent is binded to scale up/down image.
        self.bind("<Button-4>", _on_scroll_up)
        self.bind("<Button-5>", _on_scroll_down)
        # DoubleClickEvent is binded to restore original image.
        self.bind("<Double-Button-1>", _on_double_click)

        # ----------- bind dragging events ----------------
        def _on_press_down(event):
            """ press down to start dragging
            """
            x, y = event.x, event.y
            self._move_start_x = x
            self._move_start_y = y
            self._press_in = True
            # print("Press Down: " + f"({x},{y})")

        def _on_press_up(event):
            """ press up to release dragging
            """
            x, y = event.x, event.y
            self._press_in = False
            # print("Press Up: " + f"({x},{y})")

        def _on_move(event):
            """ move when press down to move showed image.
            """
            if not self._press_in:
                return
            r = self.scroll_counts.get_ratio()
            if r < 0:
                return   # disable dragging when scale rate is lower than 1.
            r = abs(r) + 1
            x, y = event.x, event.y
            current_moved_x = (x - self._move_start_x) / r
            current_moved_y = (y - self._move_start_y) / r

            # dx, dy relative to center original places.
            dx = current_moved_x + self._moved_actual_dx
            dy = current_moved_y + self._moved_actual_dy

            # showed area.
            iw, ih = self.get_img_width(), self.get_img_height()
            new_iw, new_ih = iw/r, ih/r
            cx, cy = iw / 2, ih / 2
            x1, y1, x2, y2 = cx - new_iw / 2, cy - new_ih / 2, \
                cx + new_iw / 2, cy + new_ih / 2

            # constrain the present area.
            ava_left, ava_up, ava_right, ava_bottom = \
                x1, y1, iw - x2, ih - y2

            # actual moved distance.
            if dx > 0:
                ax = min(ava_right, dx)
            else:
                ax = -min(ava_left, abs(dx))
            if dy > 0:
                ay = min(ava_up, dy)
            else:
                ay = -min(ava_bottom, abs(dy))

            self._moved_actual_dx = ax
            self._moved_actual_dy = ay
            self._move_start_x = x
            self._move_start_y = y
            # print(self._moved_actual_dx, self._move_start_y)
            _change_show(event)

        self.bind("<ButtonPress-1>", _on_press_down)
        self.bind("<ButtonRelease-1>", _on_press_up)
        self.bind("<Motion>", _on_move)

    def _calculate_showd_area(self):
        r = self.scroll_counts.get_ratio()

        # center
        iw, ih = self.get_img_width(), self.get_img_height()
        t_cx, t_cy = iw / 2, ih / 2
        # mind the _moved_actual_dx is relative to scaled img.
        t_cx -= self._moved_actual_dx
        t_cy -= self._moved_actual_dy

        # area size
        ratio = abs(r) + 1
        new_iw, new_ih = iw / ratio, ih / ratio

        # bounding box
        x1, y1, x2, y2 = t_cx - new_iw / 2, t_cy - new_ih / 2, \
            t_cx + new_iw / 2, t_cy + new_ih / 2
        # overflow boundary process.
        if x1 < 0:
            mx = abs(x1)
            x1 += mx
            x2 += mx
        if y1 < 0:
            my = abs(y1)
            y1 += my
            y2 += my
        if x2 > iw:
            mx = abs(x2 - iw)
            x1 -= mx
            x2 -= mx
        if y2 > ih:
            my = abs(y2 - ih)
            y1 -= my
            y2 -= my
        # print(x1, y1, x2, y2)
        x2 -= 1
        y2 -= 1
        return (x1, y1, x2, y2)

    def _show_transformed_images(self, boxes, area_size, r):
        """ show image transformed from original image.
        args:
            - boxes: (list or tuple), coordinates in form of x1y1x2y2.
            - area_size: (list or tuple), (iw, ih)
            - r: float, scale ratio, r < 0 means scale down.
        """
        iw, ih = area_size
        x1, y1, x2, y2 = [round(x) for x in boxes]
        # print(x1, y1, x2, y2)
        # print("{:2.6f} {:2.6f}".format((x2-x1)/(y2-y1), (iw / ih)))

        assert hasattr(self, "image"), "Image of canvas is not assigned, aborted."
        vimage = self.image.copy()
        if r < 0:
            vimage = cv2.resize(vimage, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
            plain_image = np.zeros_like(self.image)
            plain_image[y1: y2, x1: x2, :] = vimage
            vimage = plain_image
        else:
            vimage = vimage[y1: y2, x1: x2, :]
            vimage = cv2.resize(vimage, (iw, ih), interpolation=cv2.INTER_LINEAR)

        opened = ImageTk.PhotoImage(Image.fromarray(vimage))
        # To keep the buffer of opened images.
        global global_image_bank
        global_image_bank[self.__hash__()] = opened
        self.create_image(0, 0, anchor="nw", image=opened)


class FixSizedImageCanvas(ImageCanvas):
    """ Fixed size canvas providing an easy way for showing images.
    """
    def __init__(self, master,
                 widget_width,
                 widget_height, img=None, *args, **kwargs):
        super().__init__(master, img)
        self.config(
            width=widget_width,
            height=widget_height,
            *args, **kwargs
        )
        self.wwidth = widget_width
        self.wheight = widget_height

    def bind_events(self):
        # TODO: currently disable events for FixSizedImageCanvas
        pass

    def show_images(self):
        """ FixSizedImageCanvas would resize image
        and show image in fixed size. (self.image is not changed.)
        """
        assert hasattr(self, "image"), "Image of canvas is not assigned, aborted."

        vis_img = self.image.copy()
        vis_img = cv2.resize(vis_img,
                             (self.wwidth, self.wheight),
                             interpolation=cv2.INTER_LINEAR)

        opened = ImageTk.PhotoImage(Image.fromarray(vis_img))
        # To keep the buffer of opened images.
        global global_image_bank
        global_image_bank[self.__hash__()] = opened
        self.create_image(0, 0, anchor="nw", image=opened)

    def _show_transformed_images(self, boxes, area_size, r):
        raise NotImplementedError("_show_transformed_images is not implemented.")


class BoxFixSizedImageCanvas(FixSizedImageCanvas):
    """ ImageCanvas with image and box annotation. This widget supports
    showing box cropped area of image and a popup modal dialog
    to view the whole image with drawed box.
    """
    def __init__(self, master,
                 widget_width,
                 widget_height,
                 img=None,
                 box=None,  # in x1y1x2y2 format.
                 *args, **kwargs):
        super().__init__(master, widget_width, widget_height, img, *args, **kwargs)
        if img is None:
            self.box = box
        else:
            self.set_box_of_image(box)
        self.bind_popup_show_complete_images_event()

    def set_image_from_ndarray(self, image: np.ndarray, box: list = None):
        super().set_image_from_ndarray(image)
        self.box = None  # when set new image, the box is also changed.
        self.set_box_of_image(box)

    def set_image_from_path(self, path: str, box: list = None):
        super().set_image_from_path(path)
        self.box = None  # when set new image, the box is also changed.
        self.set_box_of_image(box)

    def set_box_of_image(self, box=None):
        h, w = self.image.shape[:2]
        if box is None:
            self.box = [0, 0, w, h]
        else:
            self.box = self.clip_box(deepcopy(box), h, w)

    def clip_box(self, box, h, w):
        """ clip_box in [h, w] range of image.
        """
        x1, y1, x2, y2 = box
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(x2, w)
        y2 = min(y2, h)
        return box

    def show_images(self):
        """ BoxFixSizedImageCanvas would resize image
        and show image in fixed size. (self.image is not changed.)
        """
        assert hasattr(self, "image"), "Image of canvas is not assigned, aborted."

        if self.image is None or self.box is None:
            return

        x1, y1, x2, y2 = self.box
        cropped_image = self.image[y1:y2-1, x1:x2-1, :]
        vis_img = cropped_image.copy()
        vis_img = cv2.resize(vis_img,
                             (self.wwidth, self.wheight),
                             interpolation=cv2.INTER_LINEAR)

        opened = ImageTk.PhotoImage(Image.fromarray(vis_img))
        # To keep the buffer of opened images.
        global global_image_bank
        global_image_bank[self.__hash__()] = opened
        self.create_image(0, 0, anchor="nw", image=opened)

    def get_values(self):
        if self.image is None:
            return None, None
        return self.image.copy(), deepcopy(self.box)

    def bind_popup_show_complete_images_event(self):
        def _show(event):
            if self.image is None:
                return
            top_level = tk.Toplevel()
            top_level.title("Complete Image")
            c_canvas = ImageCanvas(top_level)
            # show the image with target person annotated.
            box_image = self.image.copy()
            x1, y1, x2, y2 = [int(x) for x in self.box]
            box_image = cv2.rectangle(
                box_image, (x1, y1), (x2, y2),
                color=CV2_COLOR_BLUE, thickness=2)
            c_canvas.set_image_from_ndarray(box_image)
            c_canvas.show_images()
            c_canvas.pack()

        self.bind("<Button-1>", _show)


class BoxSelectImageCanvas(ImageCanvas):
    """ ImageCanvas which supports draw box on image.
    """
    def bind_events(self):
        self._select_flag = False
        self._start_points = [0, 0]
        self._end_points = [0, 0]

        def _on_start_select_box(event):
            self._select_flag = not self._select_flag
            if self._select_flag:  # start flag
                self._start_points[0] = event.x
                self._start_points[1] = event.y
            else:  # Finish box selection.
                self._end_points[0] = event.x
                self._end_points[1] = event.y

        def _on_move_select_box(event):
            if not self._select_flag:
                return
            x, y = event.x, event.y
            tbox = [self._start_points[0], self._start_points[1], x, y]
            tbox = [int(x) for x in tbox]
            vis_img = self.image.copy()
            vis_img = cv2.rectangle(
                          vis_img,
                          (tbox[0], tbox[1]), (tbox[2], tbox[3]),
                          color=CV2_COLOR_BLACK, thickness=2)
            self._show_temp_images(vis_img)

        def _on_cancel_select_box(event):
            # cancel box selection
            if self._select_flag:
                self._select_flag = False
            self._start_points = [0, 0]
            self._end_points = [0, 0]
            self.show_images()

        self.bind("<Button-1>", _on_start_select_box)
        self.bind("<Motion>", _on_move_select_box)
        self.bind("<Button-3>", _on_cancel_select_box)

    def _show_temp_images(self, vimage):
        assert hasattr(self, "image"), "Image of canvas is not assigned, aborted."
        self.config(
            width=self.get_img_width(),
            height=self.get_img_height(),
        )
        opened = ImageTk.PhotoImage(Image.fromarray(vimage))
        # To keep the buffer of opened images.
        global global_image_bank
        global_image_bank[self.__hash__()] = opened
        self.create_image(0, 0, anchor="nw", image=opened)

    def get_selected_box(self):
        x1, y1 = self._start_points
        x2, y2 = self._end_points
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        return [x1, y1, x2, y2]


if __name__ == '__main__':
    root = tk.Tk()
    test_img_path = os.path.join("..", "exps/vis/test.png")
    frame = BoxSelectImageCanvas(root, img=test_img_path)
    frame.show_images()
    frame.pack()
    root.mainloop()
