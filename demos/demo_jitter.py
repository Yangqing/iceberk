from scipy import misc
import iceberk
from iceberk.visiondata import jitter
import os

def demo_jitter():
    filename = os.path.join(os.path.dirname(iceberk.__file__),
                            'test','data','lena.png')
    img = misc.imread(filename)
    img_out = jitter.randn_jitter(img, [10, 10, 0.2])
    misc.imsave('demo_jitter_lena.png', img_out)

if __name__ == "__main__":
    demo_jitter()