import os
import cv2
import numpy as np
from PIL import ImageTk
import PIL.Image as Image
from copy import deepcopy

import tkinter as tk
from tkinter import Canvas

# Constant
global_image_bank = dict()
CV2_COLOR_BLACK = (0, 0, 0)
CV2_COLOR_BLUE = (0, 0, 255)


def clip_range(x, a, b):
    """ clip x in [a, b]
    """
    assert a <= b
    return min(max(x, a), b)


def inter_box(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    return (x1, y1, x2, y2)


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


# ---------------- new defined area ---------------------
class NewScrollItem(ScrollItem):
    def get_ratio(self):
        r = super().get_ratio()
        if r > 0:
            r = abs(r) + 1
        else:
            r = 1 / (abs(r) + 1)
        return r


class ScreenFitImageCanvas(ImageCanvas):
    def __init__(self, master, img=None, *args, **kwargs):
        super().__init__(master, img, *args, **kwargs)
        self.wwidth = 0
        self.wheight = 0
        self.config(background="black")

    def set_image_from_ndarray(self, image: np.ndarray):
        super().set_image_from_ndarray(image)
        if self.image is not None:
            nw, nh = self.get_img_width(), self.get_img_height()
            self.wwidth = nw
            self.wheight = nh

    def set_image_from_path(self, path: str):
        super().set_image_from_path(path)
        if self.image is not None:
            nw, nh = self.get_img_width(), self.get_img_height()
            self.wwidth = nw
            self.wheight = nh

    def show_images(self):
        assert hasattr(self, "image"), "Image of canvas is not assigned, aborted."
        if self.wwidth == 0 or self.wheight == 0:
            self.wwidth = self.get_img_width()
            self.wheight = self.get_img_height()

        self.configure(width=self.wwidth, height=self.wheight)
        self._show_transformed_images()

    def display_image(self, image, start):
        assert isinstance(image, np.ndarray)

        x1, y1 = int(start[0]), int(start[1])
        opened = ImageTk.PhotoImage(Image.fromarray(image))
        # To keep the buffer of opened images.
        global global_image_bank
        global_image_bank[self.__hash__()] = opened
        self.create_image(x1, y1, anchor="nw", image=opened)

    def center_display_image(self, image):
        """ display image in the center of canvas by
        accurately calculating the margin.
        """
        assert isinstance(image, np.ndarray)
        h, w = image.shape[:2]
        x1, y1 = (self.wwidth - w) / 2, (self.wheight - h) / 2
        self.display_image(image, (x1, y1))

    def bind_events(self):
        """ ImageCanvas implements two events: MiddleWheel to scale up/down
        image; left mouse key to drag and view image.
        """
        # events related parameters.
        self.scroll_counts = NewScrollItem()
        # dragging events related
        self._press_in = False
        self._move_start_x = 0
        self._move_start_y = 0
        self._moved_actual_dx = 0
        self._moved_actual_dy = 0

        # TODO: possible bugs if there is no image of Canvas.
        def _on_scroll_up(event):
            self.scroll_counts.add_counts()
            self._change_show(event)
            self.event_generate("<<ScaleChangeEvent>>")

        def _on_scroll_down(event):
            self.scroll_counts.delete_counts()
            if self.scroll_counts.get_ratio() <= 1:
                self._move_start_x = 0
                self._move_start_y = 0
                self._moved_actual_dx = 0
                self._moved_actual_dy = 0
            self.event_generate("<<ScaleChangeEvent>>")
            self._change_show(event)

        def _on_double_click(event):
            self.scroll_counts.release()
            self._move_start_x = 0
            self._move_start_y = 0
            self._moved_actual_dx = 0
            self._moved_actual_dy = 0
            self.event_generate("<<ScaleChangeEvent>>")
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
            if self.is_able_full_display():
                return   # disable dragging when the screen could fully display image.
            x, y = event.x, event.y
            current_moved_x = (x - self._move_start_x) / r
            current_moved_y = (y - self._move_start_y) / r
            # renew move start point
            self._move_start_x = x
            self._move_start_y = y

            # dx, dy relative to center original places.
            dx = current_moved_x + self._moved_actual_dx
            dy = current_moved_y + self._moved_actual_dy

            iw, ih = self.get_img_width(), self.get_img_height()
            cx, cy = iw / 2, ih / 2
            sw, sh = cx * r, cy * r

            # contrain move range
            column = max(sh - self.wheight / 2, 0)
            row = max(sw - self.wwidth / 2, 0)
            ax = clip_range(dx, -row/r, row/r)
            ay = clip_range(dy, -column/r, column/r)

            self._moved_actual_dx = ax
            self._moved_actual_dy = ay
            print(self._moved_actual_dx, self._move_start_y)
            self._change_show(event)

        self.bind("<ButtonPress-1>", _on_press_down)
        self.bind("<ButtonRelease-1>", _on_press_up)
        self.bind("<Motion>", _on_move)

    def _change_show(self, event):
        self._show_transformed_images()

    def _show_transformed_images(self):
        r = self.scroll_counts.get_ratio()
        move_x = self._moved_actual_dx
        move_y = self._moved_actual_dy
        iw, ih = self.get_img_width(), self.get_img_height()

        # constrain move here to comply
        # with change of scale ratio
        # 不是单独 constrain move 的范围，而是整体 box 的范围（保证 scale 也一样）。
        desired_image_box = [
            - iw * r / 2 + move_x * r,
            - ih * r / 2 + move_y * r,
            iw * r / 2 + move_x * r,
            ih * r / 2 + move_y * r,
        ]
        area_box = [
            -self.wwidth/2, -self.wheight/2, self.wwidth/2, self.wheight/2
        ]
        clip_box = list(inter_box(desired_image_box, area_box))
        image_box = deepcopy(desired_image_box)
        # rectified box
        if ih * r >= self.wheight:
            if clip_box[3] < area_box[3]:
                my = abs(clip_box[3] - area_box[3])
                image_box[1] += my
                image_box[3] += my
                self._moved_actual_dy += my / r
            if clip_box[1] > area_box[1]:
                my = abs(clip_box[1] - area_box[1])
                image_box[1] -= my
                image_box[3] -= my
                self._moved_actual_dy -= my / r
        if iw * r >= self.wwidth:
            if clip_box[2] < area_box[2]:
                mx = abs(clip_box[2] - area_box[2])
                image_box[0] += mx
                image_box[2] += mx
                self._moved_actual_dx += mx / r
            if clip_box[0] > area_box[0]:
                mx = abs(clip_box[0] - area_box[0])
                image_box[0] -= mx
                image_box[2] -= mx
                self._moved_actual_dx -= mx / r

        clip_box = list(inter_box(image_box, area_box))
        clip_width = clip_box[2] - clip_box[0]
        clip_height = clip_box[3] - clip_box[1]
        try:
            # TODO: accuracy problem here.
            assert abs(clip_width - min(self.wwidth, iw * r)) < 1e-2
            assert abs(clip_height - min(self.wheight, ih * r)) < 1e-2
        except Exception as ex:
            from IPython import embed
            embed()

        # to image coordinates
        clip_box[0] -= image_box[0]
        clip_box[1] -= image_box[1]
        clip_box[2] -= image_box[0]
        clip_box[3] -= image_box[1]
        x1, y1, x2, y2 = [int(x / r) for x in clip_box]

        vimage = self.image.copy()
        vimage = vimage[y1: y2, x1: x2, :]
        vimage = cv2.resize(
            vimage, (int(clip_width), int(clip_height)), interpolation=cv2.INTER_LINEAR)
        self.center_display_image(vimage)

    def is_able_full_display(self):
        r = self.scroll_counts.get_ratio()
        return self.wwidth >= self.get_img_width() * r and \
            self.wheight >= self.get_img_height() * r

    def register_window_event_handler(self, root_master):
        self.window_handler = ResizeHandler(root_master, self)
        self.window_handler.bind_events()
        self.window_handler.bind_ratio_change_event()

    def deregister_window_event_handler(self):
        self.window_handler.release()


class ResizeHandler:
    def __init__(self, toplevel, widget):
        self.toplevel = toplevel
        self.widget = widget
        assert isinstance(self.widget, ScreenFitImageCanvas)
        self._top_level_width = toplevel.winfo_width()
        self._top_level_height = toplevel.winfo_height()
        self.canvas_scale = 1

    def bind_events(self):
        self.toplevel.bind("<Configure>", self._event_dispatcher)

    def _event_dispatcher(self, event):
        if event.widget != self.toplevel:
            return
        if self._top_level_height == event.height and \
                self._top_level_width == event.width:
            return
        if self._top_level_height < event.height or \
                self._top_level_width < event.width:
            # window expand event.
            print("Emmit window expand events.")
        if self._top_level_height > event.height or \
                self._top_level_width > event.width:
            print("Emmit window shrink events.")

        self._top_level_height = event.height
        self._top_level_width = event.width

        eh, ew = event.height, event.width
        wh, ww = self.toplevel.winfo_height(), self.toplevel.winfo_width()
        assert eh == wh and ew == ww, f"{eh} =? {wh}; {ew} =? {ww}"
        # try:
        #     assert eh == wh and ew == ww, f"{eh} =? {wh}; {ew} =? {ww}"
        # except AssertionError:
        #     from IPython import embed
        #     embed()
        # self.widget.configure(height=wh, width=ww)
        self.widget.wheight = wh
        self.widget.wwidth = ww
        self.widget.show_images()

    def _none_events(self, event):
        pass

    def release(self):
        self.toplevel.bind("<Configure>", self._none_events)

    def bind_ratio_change_event(self):
        def _change_title(event):
            cscale = self.widget.scroll_counts.get_ratio()
            self.toplevel.title("{:5.2f}%".format(cscale * 100))
        self.widget.bind("<<ScaleChangeEvent>>", _change_title)


def main():
    """ Create a GUI program for showing target image.
    """
    import sys
    args = sys.argv
    if len(args) > 1:
        test_img_path = args[1]
        os.path.exists(test_img_path)
    else:
        test_img_path = os.path.join("..", "exps/vis/test.png")
    root = tk.Tk()
    root.geometry("800x600")  # needed!
    canvas = ScreenFitImageCanvas(root, img=test_img_path)
    canvas.set_image_from_path(test_img_path)
    canvas.show_images()
    canvas.pack()
    canvas.pack_propagate(False)
    canvas.register_window_event_handler(root)
    root.configure(background="black")
    root.mainloop()


if __name__ == '__main__':
    main()
