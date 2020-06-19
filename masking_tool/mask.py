import os
import tkinter as tk
import tkinter.filedialog as fd

import PIL.Image
import PIL.ImageDraw
import PIL.ImageTk
import gdal
import numpy as np
import tifffile as tiff

import tools.masking_tool.main as main

RED = 1
GREEN = 2
BLUE = 3
BLANK_DATA = "rgba(0,0,0,0)"
EXPECTED_SHAPE = (1, 256, 256, 1)
overlay_off_message = "Cannot add polygon while overlay is off."


class Application(tk.Frame):
    """
    Masking tool for cloud detector.
    """

    def __init__(self, master=None):
        super().__init__(master)
        self.scale = 2
        self.master = master
        self.pack()
        self.overlay_on = tk.BooleanVar()
        self.overlay_on.set(True)
        self.create_widgets()
        self.canvas_size = (512, 512)
        self.points = []
        self.sample_image = None
        self.create_canvas()
        try:
            self.directory = main.chip_dir
        except TypeError as t:
            if self.sample_image is None:
                pass
            else:
                raise t

    def create_widgets(self):
        self.buttons_frame = tk.Frame(self.master)
        self.files_frame = tk.Frame(self.buttons_frame)
        self.next_image_button = tk.Button(self.files_frame)
        self.next_image_button["text"] = "Next Image"
        self.next_image_button["command"] = self.next_image
        self.next_image_button.pack(side=tk.TOP)

        self.open_image_button = tk.Button(self.files_frame)
        self.open_image_button["text"] = "Open Image"
        self.open_image_button["command"] = self.open_file
        self.open_image_button.pack(side=tk.BOTTOM)

        self.save_image_button = tk.Button(self.files_frame)
        self.save_image_button["text"] = "Save Image"
        self.save_image_button["command"] = self.save_image_tiff
        self.save_image_button.pack(side=tk.BOTTOM)
        self.files_frame.pack(side=tk.LEFT)

        self.masks_frame = tk.Frame(self.buttons_frame)
        self.save_mask_button = tk.Button(self.masks_frame)
        self.save_mask_button["text"] = "Save Mask"
        self.save_mask_button["command"] = self.save_mask
        self.save_mask_button.pack(side=tk.TOP)

        self.clear_mask_button = tk.Button()
        self.clear_mask_button["text"] = "Clear Mask"
        self.clear_mask_button["command"] = self.clear_mask
        self.clear_mask_button.pack(side=tk.BOTTOM)

        self.predict_button = tk.Button(self.masks_frame)
        self.predict_button["text"] = "Predict"
        self.predict_button["command"] = self.predict
        self.predict_button.pack(side=tk.BOTTOM)
        self.masks_frame.pack(side=tk.RIGHT)
        self.buttons_frame.pack()

        self.option_buttons_frame = tk.Frame(self.master)
        self.pixel_frame = tk.Frame(self.option_buttons_frame)
        self.color_frame = tk.Frame(self.option_buttons_frame)
        self.overlay_button = tk.Checkbutton(self.option_buttons_frame,
                                             text="Overlay",
                                             variable=self.overlay_on,
                                             command=self.update_canvas).pack(anchor=tk.W)
        self.radius = tk.IntVar()
        self.radius.set(1 * self.scale)
        self.polygon = tk.Radiobutton(self.pixel_frame,
                                      text="Polygon",
                                      variable=self.radius,
                                      value=0,
                                      command=self.clear_polygon_vertices).pack(anchor=tk.W)
        self.fine = tk.Radiobutton(self.pixel_frame,
                                   text="Fine",
                                   variable=self.radius,
                                   value=1 * self.scale).pack(anchor=tk.W)
        self.coarse = tk.Radiobutton(self.pixel_frame,
                                     text="Coarse",
                                     variable=self.radius,
                                     value=4 * self.scale).pack(anchor=tk.W)

        self.fill = tk.StringVar()
        self.fill.set("red")
        self.eraser = tk.Radiobutton(self.color_frame, text="Eraser", variable=self.fill, value=BLANK_DATA).pack(
                anchor=tk.W)
        self.red = tk.Radiobutton(self.color_frame, text="Red", variable=self.fill, value="red").pack(anchor=tk.W)
        self.blue = tk.Radiobutton(self.color_frame, text="Blue", variable=self.fill, value="blue",
                                   state=tk.DISABLED).pack(anchor=tk.W)
        self.green = tk.Radiobutton(self.color_frame, text="green", variable=self.fill, value="green",
                                    state=tk.DISABLED).pack(anchor=tk.W)

        self.pixel_frame.pack(side=tk.LEFT)
        self.color_frame.pack(side=tk.RIGHT)
        self.option_buttons_frame.pack()

    def create_canvas(self):
        # Create the frame containing the frame containing the canvas
        self.frame = tk.Frame(self.master, width=500, height=400, bd=1)
        self.frame.pack(fill="both", expand=False)

        # Create the frame containing the canvas
        self.iframe5 = tk.Frame(self.frame, bd=2, relief=tk.RAISED)
        self.iframe5.pack(fill="both", expand=False, pady=10, padx=5)

        # Create the canvas on which drawing will occur
        self.canvas = tk.Canvas(self.iframe5, width=self.canvas_size[1], height=self.canvas_size[0])
        self.canvas.pack(fill="both", expand=False)

        # Bind mous functionality to canvas
        self.canvas.bind("<Button-1>", self.click)
        self.canvas.bind("<Double-Button-1>", self.double_click)
        self.canvas.bind("<B1-Motion>", self.drag)
        self.canvas.bind("<Motion>", self.change_cursor)
        self.canvas.bind("<Button-3>", self.right_click)

    def change_cursor(self, event):
        self.canvas.config(cursor="arrow")

    def next_image(self):
        pass

    def open_file(self):

        self.sample_image = fd.askopenfilename(initialdir=main.input_dir,
                                               title="Select file",
                                               filetypes=(("nitf files", "*.ntf"),
                                                          ("tiff files", "*.tif"),
                                                          ("all files", "*.*")))

        # Handle the case where user hits "cancel"
        if self.sample_image != "":
            self.canvas.delete("all")
            self.parse_bg_image()
            self.open_mask()

    def clear_polygon_vertices(self):
        """
        Function that wipes out the list of saved polygon vertices to start a new one.
        :return:
        :rtype:
        """
        log_msg = "Clearing polygon vertices"

        # Copy data to a temporary variable
        try:
            verts = self.polygon_vertices
        except AttributeError:
            log_msg = "Initializing polygon vertices data"
            verts = []

        print(log_msg)
        self.polygon_vertices = []

        # Erase selection data
        [self.draw(x, selection=BLANK_DATA) for x in verts]

    def add_polygon_vertex(self, vertex):
        """
        Function to add a vertex to the internally stored polygon list.
        The vertex order is preserved and the polygon will be drawn
        according to this order.

        :param vertex: tuple representing the x,y pixel coordinates for the
            polygon's vertex
        :type vertex: tuple(int,int)
        :return:
        :rtype:
        """
        print("Adding Vertex to Polygon")
        if self.polygon_vertices is not None:
            self.polygon_vertices.append(vertex)
        else:
            self.poplygon_vertices = [vertex]
        print("Polygon Vertices: %s" % str(self.polygon_vertices))

    def add_polygon(self):
        """
        Function that both adds the polygon, filled in, to the mask that
        will be saved out as well as draws the polygon on the canvas so
        that it can be observed by the user.
        :return:
        :rtype:
        """
        print("Finalizing Polygon")
        if self.polygon_vertices is not None:
            # Draw it on the mask
            _ = self.mask_draw.polygon(self.polygon_vertices, fill=self.fill.get()) if self.overlay_on.get() else print(
                    overlay_off_message)
            print("Polygon added to mask")
            self.clear_polygon_vertices()
            # Update display
            self.update_canvas()
        else:
            print("No Polygon Vertices Selected")

    def clear_mask(self):
        print("Clearing Mask")
        self.canvas.delete("all")
        self.parse_bg_image()
        self.mask_image = PIL.Image.new("RGBA",
                                        (self.canvas_size[0], self.canvas_size[1]),
                                        None)
        self.mask_draw = PIL.ImageDraw.Draw(self.mask_image)

    def parse_bg_image(self):
        print("Loading %s to canvas..." % self.sample_image)
        tif = gdal.Open(self.sample_image)
        if tif is None:
            print("Coud not open %s" % self.sample_image)

        # NITF is loaded as a (256, 256) size greyscale chip
        if ".ntf" in self.sample_image:
            print("Reading NITF")
            band = tif.GetRasterBand(1)
            # rescale pixel values to 0-255
            self.background = np.uint8(band.REadAsArray().T / np.max(band.ReadAsArray().T) * 255)
        # TIFF is loaded as a (256,256,3) chip
        else:
            print("Reading TIFF")
            self.background = tif.ReadAsArray().T
        print("Read TIF with shape: %s" % str(self.background.shape))

        # Calculate the scale
        print(np.divide(self.canvas_size, self.background.shape[0:2]))

        # Load the image onto the canvas
        self.photo = PIL.ImageTk.PhotoImage(
                image=PIL.Image.fromarray(self.background).rotate(270).transpose(PIL.Image.FLIP_LEFT_RIGHT).resize(
                        (self.canvas_size[0], self.canvas_size[1])))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        print("...loaded %s to canvas!" % self.sample_image)

    def parse_image_meta(self):
        directory = os.path.dirname(self.sample_image)
        base = os.path.basename(self.sample_image)
        file_name = os.path.splitext(base)[0]
        file_ext = os.path.splitext(base)[-1]
        return directory, file_name, file_ext

    def gen_mask_name(self):
        directory, file_name, file_ext = self.parse_image_meta()
        mask_name = os.path.join(main.masks_dir, "%s_masks.%s" % (file_name, "tif"))
        print("Mask name generated: %s" % mask_name)
        return mask_name

    def gen_image_name(self):
        directory, file_name, file_ext = self.parse_image_meta()
        image_name = os.path.join(main.chip_dir, "%s.%s" % (file_name, "tif"))
        print("Image name generated: %s" % image_name)
        return image_name

    def open_mask(self):
        try:
            del (self.mask_image)
            del (self.mask_draw)
            print("Deleting MASK from Memory")
        except Exception as e:
            print(str(e))

        mask_path = self.gen_mask_name()
        if os.path.exists(mask_path):
            print("Found existing mask! Loading...")
            self.mask_image = PIL.Image.open(mask_path).resize(size=self.canvas_size)
            print("Mask Opened: %s" % str(self.mask_image))

            # Decode masks
            x = np.array(self.mask_image)
            if len(x.shape) == 2:  # e.g. (512, 512) array full of 0's, 1's, 2's...etc
                new_shape = tuple([x.shape[0], x.shape[1], 4])
                y = np.zeros(new_shape)
                y[:, :, 0][x == RED] = 255  # Red Channel
                y[:, :, 1][x == GREEN] = 255  # Green Channel
                y[:, :, 2][x == BLUE] = 255  # Blue Channel
                y[:, :, 3][x == 0] = 0  # Alpha Channel: Make all the 0s in the mask transparent on the canvas
                y[:, :, 3][x != 0] = 255  # Alpha Channel: Make all the colors (1's, 2's, 3's) opaque.
                z = y.astype(np.uint8)
                print("2D: %s" % str(z.shape))
                print("2D: %s" % np.max(z))
                self.mask_image = PIL.Image.fromarray(z, "RGBA")
                y[:, :, 3][x != 0] = 0  # Make all values transparent
            else:  # e.g. (512, 512, 4) array full of 0's, 1's...255's...etc uint8s
                print("RGBA: %2" % str(np.array(self.mask_image).shape))
                print("RGBA: %s" % str(np.max(np.array(self.mask_image))))
                self.mask_array = PIL.Image.fromarray(np.array(self.mask_image))

            self.mask = PIL.ImageTk.PhotoImage(image=self.mask_image)
            self.empty_mask = PIL.Image.fromarray(y.astype(np.uint8), "RGBA") if "z" in dir() else self.mask_array
            self.canvas.create_image(0, 0, image=self.mask, anchor=tk.NW)
        else:
            print("Couldn't find %s" % mask_path)
            self.mask_image = PIL.Image.new("RGBA",
                                            (self.canvas_size[0], self.canvas_size[1]),
                                            None)
            self.empty_mask = self.mask_image.copy()

        print("Creating Mask Draw")
        self.mask_draw = PIL.ImageDraw.Draw(self.mask_image)

    def save_mask(self):
        """
        Function to reshape the mask to the desired size and convert it into a keras-readable/encoded TIF.
        :return:
        :rtype:
        """
        resize = ((int(self.canvas_size[0] / self.scale)), (int(self.canvas_size[1] / self.scale)))
        print("Resizing mask to width (%d), height(%d)" % (resize))
        print("Saving mask to %s" % self.gen_mask_name())

        x = np.array(self.mask_image)
        print("Image array is shape: %s" % str(x.shape))
        flat_shape = (x.shape[0], x.shape[1])
        y = np.zeros(flat_shape)
        y = (x[:, :, 0] == 255).astype(np.bool)
        encoded_image = PIL.Image.fromarray(x[:, :, 0]).convert("1")  # A little unsure if this is a one or L
        encoded_image = encoded_image.resize(size=resize)
        encoded_image.save(self.gen_mask_name())

    def save_image_tiff(self):
        if self.background is not None:
            print("Saving image %s" % self.sample_image)

            # Handle the case of a 2D greyscale image by copying the single image channel in to all three channels
            if len(self.background.T.shape) == 2:
                tiff_array = tt.copy_into_three_channels(self.background.T)
            else:
                tiff_array = self.background.T

            tiff.imsave("%s" % self.gen_image_name(), tiff_array)
        else:
            print("cannot save image, no image loaded.")

    def click(self, event):
        if self.radius.get() == 0:
            print("Polygon Click")
            self.add_polygon_vertex((event.x, event.y))
            self.draw((event.x, event.y), selection="black")
        else:
            print("Click")
            self.draw((event.x, event.y))

    def double_click(self, event):
        if self.radius.get() == 0:
            print("Polygon Double Click")
            self.add_polygon()
        else:
            print("Double Click")

    def right_click(self, event):
        r, g, b, a = self.mask_image.getpixel((event.x, event.y))

    def drag(self, event):
        if self.radius.get() == 0:
            print("Polygon Drag")
        else:
            print("DRag")
            self.draw((event.x, event.y))

    def draw(self, point, selection=None):
        """
        This function will update the mask information and draw it onto the image displayed to the user.
        The displayed image contains a background layer of the image and a foreground layer showing the mask information.

        :param point:
        :type point:
        :param selection:
        :type selection:
        :return:
        :rtype:
        """

        x, y = point
        # Grab the radius from the radio button
        radius = self.radius.get() if self.radius.get() != 0 else 1

        # Check whether we're drawing near the edges
        if x - radius < 0:
            left = 0
        else:
            left = x - radius

        if y - radius < 0:
            top = 0
        else:
            top = y - radius

        # Check for polygon selection flag
        fill = selection if selection else self.fill.get()

        # Draw on the mask
        _ = self.mask_draw.rectangle(xy=[(left, top), (x + radius, y + radius)],
                                     fill=fill) if self.overlay_on.get() else print(overlay_off_message)

        # Update the display
        self.update_canvas()

    def update_canvas(self):
        """
        This function wiwll update the display to the user to reflect changes made to the mask layer
        :return:
        :rtype:
        """
        # Generate tkk ready image
        try:
            self.mask = PIL.ImageTk.PhotoImage(image=self.mask_image if self.overlay_on.get() else self.empty_mask)
        except AttributeError:
            return print("Mask not initialized, please open an image to being.")

        # Draw mask information to image
        self.canvas.create_image(0, 0, image=self.mask, anchor=tk.NW)

    def predict(self):
        """
        This method is called with the predict button.  It will prompt the user for a model to use in
        evaluation.  The background image will be then evaluated by the model for detections.
        """

        # Prompt user for modele to evaluate
        model = fd.askopenfilename(initialdir=models_dir, title="Select Model to Evaluate",
                                   filetypes=(("hdf5", "*.h5"), ("all files", "*.*")))

        # Massage image data to fit model
        img = self.background
        if img.shape is not EXPECTED_SHAPE:
            img = np.expand_dims(img.T, (0, -1))

        # Run model evaluation
        self.prediction_layer = np.squeeze(qe.eval_real_time(img, model))

        # Massage model output to agree with display
        y = PIL.Image.fromarray(self.prediction_layer).resize(self.canvas_size)
        np_prediction = np.array(y)
        y = np.zeros(np_prediction.shape + (4,))
        y[:, :, 0][np_prediction != 0] = 255
        y[:, :, 3][np_prediction != 0] = 255
        self.mask_image = PIL.Image.fromarray(y.astype(np.uint8), "RGBA")
        self.mask_draw = PIL.ImageDraw.Draw(self.mask_image)

        # Update display with predictions
        print("Overlaying Predictions")
        self.update_canvas()


if __name__ == "__main__":
    root = tk.Tk()
    screen_width = root.winfo_width()
    screen_height = root.winfo_height()

    app = Application(master=root)
    app.mainloop()
