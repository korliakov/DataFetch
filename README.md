# DataFetch

A comprehensive computer vision application that automatically detects and extracts data points from scatter plot images using deep learning and clustering algorithms.

## Overview

This project provides a complete pipeline for:
- Generating synthetic scatter plot images for training
- Detecting data points using a U-Net CNN
- Clustering points with DBSCAN algorithm
- Converting pixel coordinates to actual data coordinates
- Exporting results as CSV files and visualization plots


## Installation

1. Clone this repository
2. Download "model.pt" from https://disk.360.yandex.com/d/uPpHenOBVsjeJg to the project's folder
3. Install required packages: `pip install -r requirements.txt`

## Usage

1. Run the application: `python run.py`
2. Follow the instructions:
   1. Select a screenshot of the plot
   2. Enter the value at an arbitrary tick on the X-axis
   3. Enter the value at an arbitrary tick on the Y-axis
   4. Enter the X-coordinate of the origin
   5. Enter the Y-coordinate of the origin
   6. Click on the origin point with your cursor
   7. Click on the tick selected in step (2) with your cursor
   8. Click on the tick selected in step (3) with your cursor
   9. Choose paths to save the .csv file with coordinates and the .png file with the scatter plot containing extracted points

## Interface and result

<img width="1920" height="999" alt="1" src="https://github.com/user-attachments/assets/43b73c8e-5e8c-44cf-8e60-35a61bc03b72" />

<img width="1920" height="999" alt="2_done" src="https://github.com/user-attachments/assets/40e29440-c97c-4952-82d5-c0a59b59f8e5" />


<img width="1920" height="999" alt="3_done" src="https://github.com/user-attachments/assets/87e7e62e-c03e-410b-b333-9981e198bb86" />


<img width="1920" height="999" alt="4_done" src="https://github.com/user-attachments/assets/78743bda-8ac4-4348-b852-e2f9f140c7a1" />


<img width="1920" height="999" alt="5_done" src="https://github.com/user-attachments/assets/90748a38-0675-4d6b-99cf-d11a64f76aec" />


<img width="1920" height="999" alt="6" src="https://github.com/user-attachments/assets/f48cc48e-2ba7-4e00-88e1-703610808fa4" />

<img width="1920" height="999" alt="7_done" src="https://github.com/user-attachments/assets/c77e5a52-afb2-4933-97c5-6228ff64d33c" />

<img width="1920" height="999" alt="8" src="https://github.com/user-attachments/assets/725095a2-51d8-4de7-a481-d41529cabe3c" />

<img width="1920" height="999" alt="9" src="https://github.com/user-attachments/assets/4621bc56-1ede-4dab-b5b2-688f6136d0d6" />

<img width="640" height="480" alt="fig8_plot" src="https://github.com/user-attachments/assets/aec0ebcf-81c0-400c-839d-1c0940bb6b84" />

<img width="1920" height="999" alt="10" src="https://github.com/user-attachments/assets/01122782-492d-49f6-8092-e9505f560531" />

<img width="1920" height="999" alt="11" src="https://github.com/user-attachments/assets/7e9a9848-d37e-4891-837d-89cec0e55a58" />

