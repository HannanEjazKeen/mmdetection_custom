import os
import os.path as osp
from xml.dom import minidom
import mmcv
from mmengine.fileio import dump, load
from mmengine.utils import track_iter_progress
import scipy

def convert_modd2_to_coco(out_file, dataset_path, category):
    mats_folder = dataset_path + 'mats/' + category;
    print("Annotation Folder = ", mats_folder)
    images_folder = dataset_path + 'images/' + category;
    print("Images Folder = ", images_folder)
    
    annotation_files = sorted(os.listdir(mats_folder));		# Read files from folder
    print( "Number of annotation files = ", len(annotation_files))
    
    annotations = []
    images = []
    obj_count = 0
    for idx, filename in enumerate(annotation_files):
        mat = scipy.io.loadmat(mats_folder+filename);
        objects = mat['annotations']['obstacles'][0][0]

        if len(objects) > 0:
            img_path = osp.join(images_folder, filename.replace('mat','jpg'))
            height, width = mmcv.imread(img_path).shape[:2]

            images.append(dict(id=idx, file_name=filename.replace('mat','jpg'), height=height, width=width))

            for obj in objects:
                x_min = obj[0];
                y_min = obj[1];
                x_max = obj[0]+obj[2];
                y_max = obj[1]+obj[3];

                data_anno = dict(image_id=idx,id=obj_count,category_id=0,bbox=[x_min, y_min, x_max - x_min, y_max - y_min],area=(x_max - x_min) * (y_max - y_min),segmentation=[],iscrowd=0)
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
    convert_modd2_to_coco(out_file='/home/hannan/Documents/gitlab/mmdetection/dataset/MODD2/MODD2_COCO/images/train/annotation_coco.json',
                            dataset_path='/home/hannan/Documents/gitlab/mmdetection/dataset/MODD2/MODD2_COCO/',category='train/')
    convert_modd2_to_coco(out_file='/home/hannan/Documents/gitlab/mmdetection/dataset/MODD2/MODD2_COCO/images/val/annotation_coco.json',
                            dataset_path='/home/hannan/Documents/gitlab/mmdetection/dataset/MODD2/MODD2_COCO/',category='val/')
    convert_modd2_to_coco(out_file='/home/hannan/Documents/gitlab/mmdetection/dataset/MODD2/MODD2_COCO/images/test/annotation_coco.json',
                            dataset_path='/home/hannan/Documents/gitlab/mmdetection/dataset/MODD2/MODD2_COCO/',category='test/')
