import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import scipy.stats as sps
from random import uniform, choice, randint
import itertools
import string
import os


class DataProcessor:
    """A class for generating synthetic scatter plot images and processing images.

    This class provides functionality to create synthetic scatter plots with various
    configurations, convert images to grayscale, resize images, and apply sharpening.
    """

    def __init__(self):
        """Initializes the DataProcessor with default configuration parameters."""
        self.markers_list = ['.', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
        self.min_marker_size = 10
        self.max_marker_size = 150
        self.min_amount_points = 20
        self.max_amount_points = 60
        self.min_plot_size = 7
        self.max_plot_size = 18
        self.min_alpha = 0.3
        self.max_alpha = 0.99
        self.func_list = [lambda x: np.sin(x),
                          lambda x: np.sin(2 * x),
                          lambda x: np.arctan(x),
                          lambda x: np.arctan(2 * x),
                          lambda x: np.tan(x),
                          lambda x: np.tan(2 * x),
                          lambda x: x ** 2,
                          lambda x: np.sqrt(np.abs(x)),
                          lambda x: np.exp(-x ** 2),
                          lambda x: np.exp(-(2 * x) ** 2),
                          ]
        self.grid_loc_min = -5
        self.grid_loc_max = 5
        self.grid_scale_min = 2
        self.grid_scale_max = 10
        self.grid_style = ['minor', 'major', 'both']
        self.grid_axis = ['x', 'y', 'both']
        self.grid_type = list(itertools.product(self.grid_style, self.grid_axis))
        self.min_word_length = 3
        self.max_word_length = 15
        self.min_chain_length = int(self.min_amount_points / 4)
        self.max_chain_length = int(self.max_amount_points / 4)

    def randomword(self, length):
        """Generates a random word of specified length.

        Args:
            length (int): The length of the random word to generate.

        Returns:
            str: A random string of lowercase letters with the specified length.
        """
        letters = string.ascii_lowercase
        return ''.join(choice(letters) for i in range(length))

    def create_figures(self, figs_path, figs_amount):
        """Generates synthetic scatter plot figures and saves them to disk.

        Creates multiple scatter plots with random configurations including:
        - Mathematical functions for x and y coordinates
        - Random markers, sizes, and transparency
        - Grid styles and axes labels
        - Random titles and axis labels

        Args:
            figs_path (str): Directory path where figures will be saved.
            figs_amount (int): Number of figures to generate.
        """
        for i in range(figs_amount):
            loc = uniform(self.grid_loc_min, self.grid_loc_max)
            scale = uniform(self.grid_scale_min, self.grid_scale_max)
            x_func = choice(self.func_list)
            y_func = choice(self.func_list)
            amount = randint(self.min_amount_points, self.max_amount_points)
            grid = sps.uniform.rvs(loc=loc, scale=scale, size=amount)
            x = x_func(grid)
            x += sps.norm.rvs(loc=0, scale=(np.max(x) - np.min(x)) / 10, size=grid.size)
            y = y_func(grid)
            y += sps.norm.rvs(loc=0, scale=(np.max(y) - np.min(y)) / 10, size=grid.size)

            alphas = sps.uniform.rvs(loc=self.min_alpha, scale=self.max_alpha - self.min_alpha, size=len(grid))
            markers = list(np.random.choice(self.markers_list, len(grid)))
            sizes = sps.uniform.rvs(loc=self.min_marker_size, scale=self.max_marker_size - self.min_marker_size,
                                    size=len(grid))

            plt.figure(figsize=(np.random.randint(low=self.min_plot_size, high=self.max_plot_size),
                                np.random.randint(low=self.min_plot_size, high=self.max_plot_size)))

            for x_coord, y_coord, alpha, marker, s in zip(x, y, alphas, markers, sizes):
                plt.scatter(x_coord, y_coord, alpha=alpha, marker=marker, s=s, c='black')

            plt.minorticks_on()

            grid_style, grid_axis = choice(self.grid_type)
            plt.grid(True, axis=grid_axis, which=grid_style)

            x_label = self.randomword(np.random.randint(self.min_word_length, self.max_word_length))
            y_label = self.randomword(np.random.randint(self.min_word_length, self.max_word_length))
            title = self.randomword(np.random.randint(self.min_word_length, self.max_word_length))

            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(title)
            plt.savefig('./' + figs_path + '/fig' + f'{i}.png')
            plt.close()

    def make_grayscale(self, initial_path, final_path):
        """Converts all images in a directory to grayscale.

        Args:
            initial_path (str): Path to directory containing original images.
            final_path (str): Path to directory where grayscale images will be saved.
        """
        images = os.listdir(initial_path)
        for image_name in images:
            img = Image.open(initial_path + '/' + image_name).convert('L')
            img.save(final_path + '/' + image_name)

    def resize(self, initial_path, final_path, size):
        """Resizes all images in a directory to specified dimensions.

        Args:
            initial_path (str): Path to directory containing original images.
            final_path (str): Path to directory where resized images will be saved.
            size (tuple): Target size as (width, height) in pixels.
        """
        images = os.listdir(initial_path)
        for image_name in images:
            img = Image.open(initial_path + '/' + image_name)
            img = img.resize(size)
            img.save(final_path + '/' + image_name)

    def sharpening(self, initial_path, final_path):
        """Applies sharpening filter to all images in a directory.

        Converts images to binary format by thresholding and saves sharpened versions.

        Args:
            initial_path (str): Path to directory containing original images.
            final_path (str): Path to directory where sharpened images will be saved.
        """
        images = os.listdir(initial_path)
        for image_name in images:
            img = np.array(Image.open(initial_path + '/' + image_name))
            im = ((img > 1)*255).astype(np.uint8)
            im = Image.fromarray(im)
            im.save(final_path + '/' + image_name)

class Preprocessing:
    """A class for preprocessing images for model inference."""

    def prepare(self, image_name, size):
        """Prepares an image for model inference.

        Args:
            image_name (str): Path to the image file.
            size (tuple): Target size for resizing as (width, height).

        Returns:
            torch.Tensor: Preprocessed image as a PyTorch tensor with shape (1, H, W).
        """
        img = Image.open(image_name).convert('L')
        img = img.resize(size)
        return torch.tensor(np.array(img), dtype=torch.float32).unsqueeze(0)

