from tkinter import *
from tkinter.filedialog import askopenfilename, askdirectory

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import tkinter.simpledialog

from DataProcessing import Preprocessing
from constants import SIZE, TK_SIZE, TK_X_FACTOR, TK_Y_FACTOR, MODEL_PATH, THRESHOLD, EPS, MIN_SAMPLES
from Model import init_model, unet_make_prediction, get_coords_from_mask, dbscan_make_prediction, \
    get_points_from_dbscan, get_centres


class App:
    """Main application class for scatter plot point extraction GUI.

    This class provides a Tkinter-based interface for loading scatter plot images,
    calibrating coordinate systems, detecting points, and exporting results.
    """

    def __init__(self):
        """Initializes the application with preprocessing and model components."""
        self.preproc = Preprocessing()
        self.model = init_model(MODEL_PATH)
        self.initial_x = np.arange(0, SIZE[0])
        self.initial_y = np.arange(0, SIZE[1])
        self.xv, self.yv = np.meshgrid(self.initial_x, self.initial_y)

    def run_app(self):
        """Runs the main application GUI and starts the processing workflow."""
        self.root = Tk()
        self.w = Canvas(self.root, width=1000, height=1000)
        self.w.pack()

        self.File = askopenfilename(parent=self.root, initialdir="./", title='Select an image')
        q = self.File[::-1].find('/')
        p = self.File[::-1][:q].find('.')
        self.name = self.File[::-1][:q][p + 1:][::-1]

        self.image = self.preproc.prepare(self.File, SIZE)

        original = Image.open(self.File)
        original = original.resize(TK_SIZE)
        img = ImageTk.PhotoImage(original)
        self.w.create_image(0, 0, image=img, anchor="nw")

        self.x_tick_value = tkinter.simpledialog.askfloat("x-axis", "value at an arbitrary tick on the X-axis")
        self.y_tick_value = tkinter.simpledialog.askfloat("y-axis", "value at an arbitrary tick on the Y-axis")
        self.x_origin_value = tkinter.simpledialog.askfloat("x-axis", "X-coordinate of the origin")
        self.y_origin_value = tkinter.simpledialog.askfloat("y-axis", "Y-coordinate of the origin")

        tkinter.messagebox.showinfo("Instructions", "Click at: \n"
                                                    "1) Origin \n"
                                                    "2) Selected X-axis tick \n"
                                                    "3) Selected Y-axis tick")

        self.w.bind("<Button 1>", self.getorigin)
        self.root.mainloop()

    def getorigin(self, eventorigin):
        """Handles mouse click event for selecting origin point.

        Args:
            eventorigin: Mouse event containing click coordinates.
        """
        self.x_origin_coord = eventorigin.x
        self.x_origin_coord *= TK_X_FACTOR
        self.y_origin_coord = eventorigin.y
        self.y_origin_coord *= TK_Y_FACTOR
        self.w.bind("<Button 1>", self.getextentx)

    def getextentx(self, eventextentx):
        """Handles mouse click event for selecting X-axis reference point.

        Args:
            eventextentx: Mouse event containing click coordinates.
        """
        self.x_tick_coord = eventextentx.x
        self.x_tick_coord *= TK_X_FACTOR
        self.w.bind("<Button 1>", self.getextenty)

    def getextenty(self, eventextenty):
        """Handles mouse click event for selecting Y-axis reference point.

        Args:
            eventextenty: Mouse event containing click coordinates.
        """
        self.y_tick_coord = eventextenty.y
        self.y_tick_coord *= TK_Y_FACTOR
        tkinter.messagebox.showinfo("Configuration", "Choose path to save points")
        self.w.bind("<Button 1>", self.make_calculations())

    def make_calculations(self):
        """Performs the main processing pipeline including point detection and coordinate conversion."""
        unet_pred = unet_make_prediction(self.image, self.model, THRESHOLD)
        coords = get_coords_from_mask(unet_pred, self.xv, self.yv)
        dbscan_pred = dbscan_make_prediction(coords, EPS, MIN_SAMPLES)
        points = get_points_from_dbscan(coords, dbscan_pred)
        centres = get_centres(points)
        x_diff_coord = self.x_tick_coord - self.x_origin_coord
        x_factor = (self.x_tick_value - self.x_origin_value) / x_diff_coord
        y_diff_coord = self.y_tick_coord - self.y_origin_coord
        y_factor = -(self.y_tick_value - self.y_origin_value) / y_diff_coord

        self.origin_coord = np.array([self.x_origin_coord, self.y_origin_coord])
        self.factor = np.array([x_factor, -y_factor])
        self.origin_value = np.array([self.x_origin_value, self.y_origin_value])

        self.external_centres = (centres - self.origin_coord) * self.factor + self.origin_value

        self.Path_csv = askdirectory(parent=self.root, initialdir="./", title='Select image')

        np.savetxt(self.Path_csv + '/' + self.name + '.csv', self.external_centres, delimiter=',')
        tkinter.messagebox.showinfo("Configuration", "Choose path to save plot")
        self.w.bind("<Button 1>", self.save_plot())

    def save_plot(self):
        """Saves the extracted points as a new scatter plot image."""
        self.Path_plot = askdirectory(parent=self.root, initialdir="./", title='Select image')

        plt.scatter(*self.external_centres.T)

        min_lim = ((np.array([[0], [0]]) - self.origin_coord) * self.factor + self.origin_value).reshape(-1)
        max_lim = ((np.array([[SIZE[0]], [SIZE[1]]]) - self.origin_coord) * self.factor + self.origin_value).reshape(-1)
        plt.xlim((min_lim[0], max_lim[0]))
        plt.ylim((max_lim[1], min_lim[1]))
        plt.savefig(self.Path_plot + '/' + self.name + '_plot' + '.png')
        tkinter.messagebox.showinfo("Exit", "Press Ok to exit")
        self.w.bind("<Button 1>", self.exit_())

    def exit_(self):
        """Exits the application."""
        exit()
