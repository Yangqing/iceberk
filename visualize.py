"""Functions that could be used to visualize patches
"""

import numpy as np
from matplotlib import cm, pyplot


class PatchVisualizer:
    '''PatchVisualizer visualizes patches.
    '''
    def __init__(self, gap = 1):
        self.gap = gap
    
    def show_single(self, patch):
        """Visualizes one single patch. The input patch could be a vector (in
        which case we try to inter the shape of the patch), a 2-D matrix, or a
        3-D matrix whose 3rd dimension has 3 channels.
        """
        if len(patch.shape) == 1:
            patch = patch.reshape(self.get_patch_shape(patch))
        elif len(patch.shape) > 2 and patch.shape[2] != 3:
            raise ValueError, "The input patch shape isn't correct."
        # determine color
        if len(patch.shape) == 2:
            cmap = cm.gray
        else:
            cmap = None
        pyplot.imshow(patch, cmap = cmap)
        return patch
    
    def show_multiple(self, patches, bg_func = np.mean):
        """Visualize multiple patches. In the passed in patches matrix, each row
        is a patch, in the shape of either n*n or n*n*3, either in a flattened
        format (so patches would be an 2-D array), or a multi-dimensional tensor
        (so patches will be higher dimensional). We will try our best to figure
        out automatically the patch size.
        """
        num_patches = patches.shape[0]
        num_patches_per_edge = int(np.ceil(np.sqrt(num_patches)))
        if len(patches.shape) == 2:
            patches = patches.reshape((patches.shape[0],) + 
                                  self.get_patch_shape(patches[0]))
        patch_size_expand = np.array(patches.shape[1:3]) + self.gap
        image_size = patch_size_expand * num_patches_per_edge - self.gap
        if len(patches.shape) == 4:
            if patches.shape[3] != 3:
                raise ValueError, "The input patch shape isn't correct."
            # color patches
            image_shape = tuple(image_size) + (3,)
            cmap = None
        else:
            image_shape = tuple(image_size)
            cmap = cm.gray
        image = np.ones(image_shape) * bg_func(patches)
        for pid in range(num_patches):
            row = pid / num_patches_per_edge * patch_size_expand[0]
            col = pid % num_patches_per_edge * patch_size_expand[1]
            image[row:row+patches.shape[1], col:col+patches.shape[2]] = \
                    patches[pid]
        # normalize the patches for better viewing results
        image -= np.min(image)
        image /= np.max(image) + np.finfo(np.float64).eps
        pyplot.imshow(image, cmap = cmap, interpolation='nearest')
        pyplot.axis('off')
        return image
    
    def show_channels(self, patch, bg_func = np.mean):
        """ This function shows the channels of a patch. The incoming patch
        should have shape [w, h, num_channels], and each channel will be
        visualized as a separate gray patch.
        """
        if len(patch.shape) != 3:
            raise ValueError, "The input patch shape isn't correct."
        patch_reordered = np.swapaxes(patch.T, 1, 2)
        return self.show_multiple(patch_reordered, bg_func = bg_func)
    
    def get_patch_shape(self, patch):
        """ gets the patch shape of a single patch. Basically it tries to
        interprete the patch as a square, and also check if it is in color (3
        channels)
        """
        edgeLen = np.sqrt(patch.size)
        if edgeLen != np.floor(edgeLen):
            # we are given color patches
            edgeLen = np.sqrt(patch.size / 3.)
            if edgeLen != np.floor(edgeLen):
                raise ValueError, "I can't figure out the patch shape."
            return (edgeLen,edgeLen,3)
        else:
            edgeLen = int(edgeLen)
            return (edgeLen,edgeLen)

