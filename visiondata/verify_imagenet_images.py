# this script allows you to verify if the imagenet ILSVRC images you downloaded
# are correct (i.e., images are not corrupted). You can run them in parallel if
# you have multiple machines.

from iceberk import mpi
import gflags, glob, logging, os, sys
from PIL import Image

gflags.DEFINE_string("train", "", "The root for the training data")
gflags.DEFINE_string("val", "", "The root for the validation data")
gflags.DEFINE_string("test", "", "The root for the testing data")
gflags.FLAGS(sys.argv)
FLAGS = gflags.FLAGS

mpi.log_level(logging.ERROR)
mpi.root_log_level(logging.INFO)

files = []
if mpi.is_root():
    if FLAGS.train != "":
        logging.info("Adding training images..")
        files += glob.glob(os.path.join(FLAGS.train, '*', '*.JPEG'))
    if FLAGS.val != "":
        logging.info("Adding validation images..")
        files += glob.glob(os.path.join(FLAGS.val, '*.JPEG'))
    if FLAGS.test != "":
        logging.info("Adding testing images..")
        files += glob.glob(os.path.join(FLAGS.test, '*.JPEG'))
    logging.info("A total of %d images to check" % (len(files)))
files = mpi.distribute_list(files)

logging.info('Validating...')
errornum = 0
for i, filename in enumerate(files):
    try:
        verify = Image.open(filename)
    except Exception, e:
        logging.error(filename)
        errornum += 1
errornum = mpi.COMM.allreduce(errornum)
if errornum == 0:
    logging.info("Done. No corrupted images found.")
else:
    logging.info("Done. %d corrupted images found." % (errornum,))

