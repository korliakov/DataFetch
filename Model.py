import numpy as np
import torch
from torch import nn
from sklearn.cluster import DBSCAN


def conv_bn_relu(in_channels, out_channels, kernel=3, stride=1, padding=1):
    """Creates a convolutional block with batch normalization and ReLU activation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel (int): Kernel size for convolution.
        stride (int): Stride for convolution.
        padding (int): Padding for convolution.

    Returns:
        nn.Sequential: Convolutional block module.
    """
    net = nn.Sequential(
        nn.Conv2d(in_channels, out_channels,
                  kernel_size=kernel, stride=stride, padding=padding),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True)
    )
    return net


class Stacked2ConvsBlock(nn.Module):
    """A block with two consecutive convolutional layers with batch norm and ReLU."""

    def __init__(self, in_channels, out_channels):
        """Initializes the stacked convolution block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super(Stacked2ConvsBlock, self).__init__()
        self.blocks = nn.Sequential(
            conv_bn_relu(in_channels, out_channels),
            conv_bn_relu(out_channels, out_channels)
        )

    def forward(self, net):
        net = self.blocks(net)
        return net


class UpSamplingBlock(nn.Module):
    """U-Net upsampling block with transpose convolution and skip connections."""

    def __init__(self, in_channels, out_channels):
        """Initializes the upsampling block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super(UpSamplingBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=2, stride=2)

        self.convolve = Stacked2ConvsBlock(2 * in_channels, out_channels)

    def forward(self, left_net, right_net):
        right_net = self.upsample(right_net)
        net = torch.cat((left_net, right_net), dim=(1))
        net = self.convolve(net)
        return net


class DownSamplingBlock(nn.Module):
    """U-Net downsampling block with max pooling and convolutions."""

    def __init__(self, in_channels, out_channels):
        """Initializes the downsampling block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super(DownSamplingBlock, self).__init__()
        self.blocks = nn.Sequential(
            nn.MaxPool2d(2, 2),
            Stacked2ConvsBlock(in_channels, out_channels)
        )

    def forward(self, net):
        return self.blocks(net)


class Unet(nn.Module):
    """U-Net architecture for semantic segmentation of scatter plot points."""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self):
        """Initializes the U-Net model with encoder-decoder architecture."""
        super(Unet, self).__init__()
        self.init_conv = Stacked2ConvsBlock(1, 64)

        self.downsample_1 = DownSamplingBlock(64, 128)
        self.downsample_2 = DownSamplingBlock(128, 256)
        self.downsample_3 = DownSamplingBlock(256, 512)
        self.downsample_4 = DownSamplingBlock(512, 1024)

        self.upconv = Stacked2ConvsBlock(1024, 512)

        self.upsample_1 = UpSamplingBlock(512, 256)
        self.upsample_2 = UpSamplingBlock(256, 128)
        self.upsample_3 = UpSamplingBlock(128, 64)
        self.upsample_4 = UpSamplingBlock(64, 64)

        self.agg_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        net0 = self.init_conv(x)  # 1 --> 64

        net1 = self.downsample_1(net0)  # 64 --> 128
        net2 = self.downsample_2(net1)  # 128 --> 256
        net3 = self.downsample_3(net2)  # 256 --> 512
        net = self.downsample_4(net3)  # 512 --> 1024

        net = self.upconv(net)  # 1024 --> 512

        net = self.upsample_1(net3, net)  # 512 --> 256
        net = self.upsample_2(net2, net)  # 256 --> 128
        net = self.upsample_3(net1, net)  # 128 --> 64
        net = self.upsample_4(net0, net)  # 64 --> 64

        net = self.agg_conv(net)  # 64 --> 1

        return net


def get_coords_from_mask(mask, xv, yv):
    """Extracts coordinates from binary mask.

    Args:
        mask (numpy.ndarray): Binary mask where values > 0 indicate points.
        xv (numpy.ndarray): X-coordinate meshgrid.
        yv (numpy.ndarray): Y-coordinate meshgrid.

    Returns:
        numpy.ndarray: Array of coordinates where mask is positive, shape (N, 2).
    """
    return np.vstack((xv[mask > 0], yv[mask > 0])).T


def unet_make_prediction(image, model, threshold):
    """Makes prediction using U-Net model and applies threshold.

    Args:
        image (torch.Tensor): Input image tensor.
        model (nn.Module): Trained U-Net model.
        threshold (float): Threshold for binary classification.

    Returns:
        torch.Tensor: Binary prediction mask.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        pred = model(image.unsqueeze(dim=0).to(device)).squeeze().cpu()
    return pred > torch.logit(torch.FloatTensor([threshold]))


def init_model(model_path):
    """Initializes and loads a pre-trained U-Net model.

    Args:
        model_path (str): Path to the saved model weights.

    Returns:
        Unet: Loaded and evaluated U-Net model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Unet().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def dbscan_make_prediction(coords, eps, min_samples):
    """Applies DBSCAN clustering to coordinates.

    Args:
        coords (numpy.ndarray): Array of coordinates, shape (N, 2).
        eps (float): DBSCAN epsilon parameter.
        min_samples (int): DBSCAN min_samples parameter.

    Returns:
        numpy.ndarray: Cluster labels for each coordinate.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    pred = dbscan.fit_predict(coords)
    return pred


def get_points_from_dbscan(coords, labels):
    """Groups coordinates by DBSCAN cluster labels, excluding outliers.

    Args:
        coords (numpy.ndarray): Array of coordinates, shape (N, 2).
        labels (numpy.ndarray): DBSCAN cluster labels.

    Returns:
        list: List of numpy arrays, each containing coordinates for one cluster.
    """
    outliers = (labels == -1)
    labels = labels[~outliers]
    coords = coords[~outliers]
    points = []
    for label in np.unique(labels):
        mask = labels == label
        points.append(coords[mask])
    return points


def get_centres(points):
    """Calculates centroid for each cluster of points.

    Args:
        points (list): List of numpy arrays, each containing cluster coordinates.

    Returns:
        numpy.ndarray: Array of cluster centroids, shape (M, 2).
    """
    centres = []
    for points_set in points:
        centres.append(points_set.mean(axis=0))
    return np.array(centres)
