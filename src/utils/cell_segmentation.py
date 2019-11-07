import numpy as np

from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.segmentation import find_boundaries
import skimage

import scipy.ndimage
import multiprocessing

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def label_cells(image, make_cells_masks=False, 
                nucleus_channel=0, cytoplasm_channel=2, centroid_threshold=10):
    
    # Find the small dense regions in the image (nucleus)
    centroids = image[..., nucleus_channel] > centroid_threshold

    # Label each of the regions
    labels, _ = scipy.ndimage.label(centroids)

    # Ignore regions which are too small
    for colour in np.where(np.bincount(labels.flatten()) < 250)[0]:
        labels[labels == colour] = 0

    # Flatten the image in a grayscale and expand the labeled regions using
    # watershed (using only nucleus channel)
    img_grayscale = image.astype(np.float).sum(-1)
    centroids = skimage.morphology.watershed(-img_grayscale, labels, mask=image[..., nucleus_channel] > 1)
            
    if make_cells_masks:
        watershed = skimage.morphology.watershed(-img_grayscale, labels, mask=image[..., cytoplasm_channel] > 1)
        colours = np.unique(watershed)
        colours = colours[colours != 0]
        for i, colour in enumerate(colours):
            watershed[watershed == colour] = i + 1
        return centroids, watershed
        
    else:
        return centroids, None
        
    # Colorize the output
    colours = np.unique(watershed)
    colours = colours[colours != 0]
    for i, colour in enumerate(colours):
        watershed[watershed == colour] = i + 1
    return watershed

def segment_cells(paths, dataset_type='hpa', make_cells_masks=False):
    for path in tqdm(paths):
        
        if dataset_type == 'hpa':
            image = np.stack([ 
                cv2.imread(path.format(colour), -1) for colour in ['blue', 'red', 'green']
            ], axis=-1)
        elif dataset_type == 'rx':
            image = np.stack([
                cv2.imread(path.format('w'+str(ch))) for ch in range(1, 4)
            ], axis=-1)
        else:
            raise ValueError('Unknown dataset type ' + dataset_type)

        centroid_threshold = 10 if dataset_type == 'hpa' else 5
        centroids, watershed = label_cells(image, make_cells_masks=make_cells_masks,
                                           centroid_threshold=centroid_threshold)    
        
        cv2.imwrite(path.format('centroids_masks'), centroids.astype(np.uint16))
        if make_cells_masks:
            cv2.imwrite(path.format('cells_masks'), watershed.astype(np.uint16))
            
def segment_cells_parallel(paths, **kwargs):
    cpu_count = multiprocessing.cpu_count()
    N = len(paths)
    chunk_len = N // cpu_count
    if N % cpu_count > 0:
        chunk_len += 1
    
    processes = []
    
    for chunk in chunks(paths, chunk_len):
        p = multiprocessing.Process(target=segment_cells, args=(part,), kwargs=kwargs)
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    