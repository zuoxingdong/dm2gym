"""A simple OpenCV based viewer for dm_control images"""

import cv2
import uuid


class OpenCVImageViewer():
    """A simple OpenCV highgui based dm_control image viewer

    This class is meant to be a drop-in replacement for
    `gym.envs.classic_control.rendering.SimpleImageViewer`
    """

    def __init__(self, *, escape_to_exit=False):
        """Construct the viewing window"""
        self._escape_to_exit = escape_to_exit
        self._window_name = str(uuid.uuid4())
        cv2.namedWindow(self._window_name, cv2.WINDOW_AUTOSIZE)
        self._isopen = True

    def __del__(self):
        """Close the window"""
        cv2.destroyWindow(self._window_name)
        self._isopen = False

    def imshow(self, img):
        """Show an image"""
        # Convert image to BGR format
        cv2.imshow(self._window_name, img[:, :, [2, 1, 0]])
        # Listen for escape key, then exit if pressed
        if cv2.waitKey(1) in [27] and self._escape_to_exit:
            exit()

    @property
    def isopen(self):
        """Is the window open?"""
        return self._isopen

    def close(self):
        pass
