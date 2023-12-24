import os
import os.path as osp
from xml.dom import minidom
import mmcv
from mmengine.fileio import dump, load
from mmengine.utils import track_iter_progress

def convert_marine_to_coco(out_file, dataset_path, category):
    annotations_folder = dataset_path + 'annotations/' + category;
    print("Annotation Folder = ", annotations_folder)
    images_folder = dataset_path + 'images/' + category;
    print("Images Folder = ", images_folder)
    
    annotation_files = sorted(os.listdir(annotations_folder));		# Read files from folder
    print( "Number of annotation files = ", len(annotation_files))
    
    annotations = []
    images = []
    obj_count = 0
    for idx, filename in enumerate(annotation_files):
        img_path = osp.join(images_folder, filename.replace('xml','png'))
        height, width = mmcv.imread(img_path).shape[:2]

        images.append(dict(id=idx, file_name=filename.replace('xml','png'), height=height, width=width))
        
        XML = minidom.parse(annotations_folder+filename);
        
        objects = XML.getElementsByTagName('object')
        
        for obj in objects:
        	x = int(obj.getElementsByTagName('x')[0].firstChild.data);
        	y = int(obj.getElementsByTagName('y')[0].firstChild.data);
        	w = int(obj.getElementsByTagName('w')[0].firstChild.data);
        	h = int(obj.getElementsByTagName('h')[0].firstChild.data);
        	
        	data_anno = dict(image_id=idx,id=obj_count,category_id=0,bbox=[x, y, w, h],area=(w * h),segmentation=[],iscrowd=0)
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
    convert_marine_to_coco(out_file='/home/hannan/Documents/mmdetection/datasets/MARINE_DEBRIS/images/train/annotation_coco.json',
                            dataset_path='/home/hannan/Documents/mmdetection/datasets/MARINE_DEBRIS/',category='train/')
    convert_marine_to_coco(out_file='/home/hannan/Documents/mmdetection/datasets/MARINE_DEBRIS/images/val/annotation_coco.json',
            dataset_path='/home/hannan/Documents/mmdetection/datasets/MARINE_DEBRIS/',category='val/')
    convert_marine_to_coco(out_file='/home/hannan/Documents/mmdetection/datasets/MARINE_DEBRIS/images/test/annotation_coco.json',
            dataset_path='/home/hannan/Documents/mmdetection/datasets/MARINE_DEBRIS/',category='test/')
