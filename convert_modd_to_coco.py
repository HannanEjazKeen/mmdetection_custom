import os
import os.path as osp
import shutil
from xml.dom import minidom
import mmcv
from mmengine.fileio import dump, load
from mmengine.utils import track_iter_progress
import scipy
import numpy as np

def convert_modd_to_coco(out_file, dataset_path, category):
    mats_folder = dataset_path + 'mats/' + category;
    print("Annotation Folder = ", mats_folder)
    images_folder = dataset_path + 'images/' + category;
    print("Images Folder = ", images_folder)
    
    annotation_files = sorted(os.listdir(mats_folder));     # Read files from folder
    print( "Number of annotation files = ", len(annotation_files))
    
    annotations = []
    images = []
    obj_count = 0
    for idx, filename in enumerate(annotation_files):
        mat = scipy.io.loadmat(mats_folder+filename);

        large_objects = mat['largeobjects'];
        small_objects = mat['smallobjects'];

        if large_objects.size > 0:
            if small_objects.size > 0:
                objects = np.vstack((large_objects, small_objects));
            else:
                objects = large_objects;
        else:
            if small_objects.size > 0:
                objects = small_objects;
            else:
                objects = [];

        if len(objects) > 0:
            img_path = osp.join(images_folder, filename.replace('mat','jpg'))
            height, width = mmcv.imread(img_path).shape[:2]

            images.append(dict(id=idx, file_name=filename.replace('mat','jpg'), height=height, width=width))

            for obj in objects:
                x = obj[0];
                y = obj[1];
                w = obj[2];
                h = obj[3];

                data_anno = dict(image_id=idx,id=obj_count,category_id=0,bbox=[x, y, w, h],area=(w*h),segmentation=[],iscrowd=0)
                annotations.append(data_anno)
                obj_count += 1

    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=[{
            'id': 0,
            'name': 'object'
        }])
    dump(coco_format_json, out_file)

if __name__ == '__main__':
    convert_modd_to_coco(out_file='/home/hannan/Documents/mmdetection/datasets/MODD/images/train/annotation_coco.json',
                            dataset_path='/home/hannan/Documents/mmdetection/datasets/MODD/',category='train/')
    convert_modd_to_coco(out_file='/home/hannan/Documents/mmdetection/datasets/MODD/images/val/annotation_coco.json',
                            dataset_path='/home/hannan/Documents/mmdetection/datasets/MODD/',category='val/')
    convert_modd_to_coco(out_file='/home/hannan/Documents/mmdetection/datasets/MODD/images/test/annotation_coco.json',
                            dataset_path='/home/hannan/Documents/mmdetection/datasets/MODD/',category='test/')
