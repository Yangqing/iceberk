"""The Caltech-UCSD bird dataset
"""

from iceberk import datasets, mpi
import numpy as np
import os
from scipy import misc


class CUBDataset(datasets.ImageSet):
    """ The Caltech-UCSD bird dataset 2011
    """
    def __init__(self, root, is_training, crop = False, subset = None, 
                 prefetch = False, target_size = None):
        """Load the dataset.
        Input:
            root: the root folder of the CUB_200_2011 dataset.
            is_training: if true, load the training data. Otherwise, load the
                testing data.
            crop: if False, does not crop the bounding box. If a real value,
                crop is the ratio of the bounding box that gets cropped.
                e.g., if crop = 1.5, the resulting image will be 1.5 * the
                bounding box area.
            subset: if nonempty, we will only use the subset specified in the
                list. The content of the list should be class subfolder names, 
                like ['001.Black_footed_Albatross', ...]
            prefetch: if True, the images are prefetched to avoid disk read. If
                you have a large number of images, prefetch would require a lot
                of memory.
            target_size: if provided, all images are resized to the size 
                specified. Should be a list of two integers, like [640,480].
            
        Note that we will use the python indexing (labels start from 0).
        """
        super(CUBDataset, self).__init__()
        images = [line.split()[1] for line in
                    open(os.path.join(root, 'images.txt'), 'r')]
        boxes = [line.split()[1:] for line in
                    open(os.path.join(root, 'bounding_boxes.txt'),'r')]
        labels = [int(line.split()[1]) - 1 for line in
                    open(os.path.join(root, 'image_class_labels.txt'), 'r')]
        classnames = [line.split()[1] for line in
                    open(os.path.join(root, 'classes.txt'),'r')]
        class2id = dict(zip(classnames, range(len(classnames))))
        split = [int(line.split()[1]) for line in
                    open(os.path.join(root, 'train_test_split.txt'),'r')]
        # load parts
        rawparts = np.loadtxt(os.path.join(root, 'parts','part_locs.txt'))
        rawparts = rawparts[:,2:-1].reshape((len(images), 15, 2))
        if subset is not None:
            # create the subset mapping
            old2new = {}
            selected_ids = set()
            for new_id, name in enumerate(subset):
                old_id = class2id[name]
                selected_ids.add(old_id)
                old2new[old_id] = new_id
            # select the subset
            is_selected = [(label in selected_ids) for label in labels]
            images = [image for image, val in zip(images, is_selected) if val]
            boxes = [box for box, val in zip(boxes, is_selected) if val]
            labels = [old2new[label] for label, val in zip(labels, is_selected) \
                      if val]
            classnames = subset
            class2id = dict(zip(classnames, range(len(classnames))))
            split = [trte for trte, val in zip(split, is_selected) if val]
            rawparts = rawparts[np.asarray(is_selected, dtype=bool)]
        # now, do training testing split
        if is_training:
            target = 1
        else:
            target = 0
        images = [image for image, val in zip(images, split) if val == target]
        boxes = [box for box, val in zip(boxes, split) if val == target]
        labels = [label for label, val in zip(labels, split) if val == target]
        rawparts = rawparts[np.asarray(split)==target] - 1
        # store the necessary values
        self._root = root
        self._data = mpi.distribute_list(images)
        self._raw_name = self._data
        # for the boxes, we store them as a numpy array
        self._boxes = np.array(mpi.distribute_list(boxes)).astype(float)
        self._boxes -= 1
        self._parts = mpi.distribute(rawparts)
        self._prefetch = prefetch
        self._target_size = target_size
        self._crop = crop
        if target_size is not None:
            self._dim = tuple(target_size) + (3,)
        else:
            self._dim = False
        self._channels = 3
        # we store the raw dimensions for part location computation
        self._raw_dimension = np.zeros((len(self._data),2), dtype=int)
        if prefetch:
            self._data = [self._read(i) for i in range(len(self._data))]
        self._label = mpi.distribute_list(labels)
        self._classnames = mpi.COMM.bcast(classnames)

    def _read(self, idx):
        image = datasets.imread_rgb(os.path.join(self._root, 'images',\
                                                 self._raw_name[idx]))
        self._raw_dimension[idx] = image.shape[:2]
        xmin, ymin, xmax, ymax = \
                    self._get_cropped_coordinates(idx)
        image = image[ymin:ymax, xmin:xmax].copy()
        if self._target_size is not None:
            image = misc.imresize(image, self._target_size)
        return image
    
    def _get_cropped_coordinates(self, idx):
        box = self._boxes[idx]
        if self._raw_dimension[idx,0] == 0:
            self._read(idx)
        imheight, imwidth = self._raw_dimension[idx]
        if self._crop is not False:
            x, y, width, height = box
            centerx = x + width / 2.
            centery = y + height / 2.
            xoffset = width * self._crop / 2.
            yoffset = height * self._crop / 2.
            xmin = max(int(centerx - xoffset + 0.5), 0)
            ymin = max(int(centery - yoffset + 0.5), 0)
            xmax = min(int(centerx + xoffset + 0.5), imwidth - 1)
            ymax = min(int(centery + yoffset + 0.5), imheight - 1)
            if xmax - xmin <= 0 or ymax - ymin <= 0:
                    raise ValueError, "The cropped bounding box has size 0."
        else:
            xmin, ymin, xmax, ymax = 0, 0, imwidth, imheight
        return xmin, ymin, xmax, ymax
    
    def parts(self, idx):
        part = self._parts[idx].copy()
        invalid = (part[:,0] < 0)
        xmin, ymin, xmax, ymax = \
                self._get_cropped_coordinates(idx)
        # There are some errors in the training data (see e.g. image 5007)
        # so we manually check if things are valid
        invalid |= (part[:,0] < xmin)
        invalid |= (part[:,0] >= xmax)
        invalid |= (part[:,1] < ymin)
        invalid |= (part[:,1] >= ymax)
        part -= [xmin, ymin]
        if self._target_size is not None:
            # convert the location to the target size
            part *= np.asarray(self._target_size) / \
                    np.array([xmax-xmin, ymax-ymin], dtype = np.float)
        # for those parts that are not available, reset them to zero
        part[invalid] = -1
        return part
    
    @staticmethod
    def subset_farrell_iccv11():
        """Returns the list of 14 classes' names used in 
            Ryan Farrell, et al.
            Birdlets: Subordinate Categorization Using Volumetric Primitives 
                and Pose-Normalized Appearance.
            ICCV 2011.
        """
        return ['151.Black_capped_Vireo',
                '152.Blue_headed_Vireo',
                '153.Philadelphia_Vireo',
                '154.Red_eyed_Vireo',
                '155.Warbling_Vireo',
                '156.White_eyed_Vireo',
                '157.Yellow_throated_Vireo',
                '036.Northern_Flicker',
                '187.American_Three_toed_Woodpecker',
                '188.Pileated_Woodpecker',
                '189.Red_bellied_Woodpecker',
                '190.Red_cockaded_Woodpecker',
                '191.Red_headed_Woodpecker',
                '192.Downy_Woodpecker']
    
    def dump(self, target_folder):
        """Dump the current images to the target folder
        """
        mpi.mkdir(target_folder)
        for idx in range(self.size()):
            name = self._raw_name[idx]
            mpi.mkdir(os.path.join(target_folder, os.path.basename(name)))
            misc.imsave(os.path.join(target_folder, name),\
                        self._read(idx))

if __name__ == "__main__":
    from iceberk import classifier, pipeline, visualize, visiondata, mpi
    import numpy as np
    
    root = '/u/vis/farrell/datasets/CUB_200_2011'
    traindata = visiondata.CUBDataset(root, True, prefetch = False)