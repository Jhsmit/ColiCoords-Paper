from colicoords.gui.controller import GenerateBinaryController
from PyQt5 import QtGui
import tifffile
import os
import sys
import time
import numpy as np

"""
Annotation of binary images requires the packages PyQt5 and pyqtgraph

Controls:

a:  Previous image
d:  Next image
e:  Increase brush size
r:  Reduce brush size
f:  Switch between paint / zoom mode

Paint mode:
Mouse 1: Draw overlay
Mouse 2: Remove overlay

Zoom mode:
Mouse 1: Drag to pan
Mouse 2: Access menu
Mousewheel: Zoom in/out

"""

data_dir = r'.'

bf_path = os.path.join(data_dir, 'Bf_corrected.tif')
bf = tifffile.imread(bf_path)

# To continue with a previously annotated file uncomment the next two lines and update the binary path
#bin_path = 'BINARY_PATH'
#binary = (tifffile.imread(bin_path) > 0).astype(int)

# To annotate only the first 100 images (more images gives better segmentation)
binary = np.zeros_like(bf[:100])

app = QtGui.QApplication(sys.argv)
ctrl = GenerateBinaryController(bf[0:100], binary)

# Use try/finally to ensure the result if saved even if the GUI crashes.
try:
    ctrl.show()
    sys.exit(app.exec_())
finally:
    tifffile.imsave('binary_img_out_{}.tif'.format(int(time.time())), ctrl.output_binary)



