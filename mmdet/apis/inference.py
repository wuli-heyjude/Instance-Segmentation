import warnings

import matplotlib.pyplot as plt
import mmcv
import torch
from mmcv.ops import RoIAlign, RoIPool
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

import cv2
import numpy as np
from math import floor,ceil
from mmdet.core import get_classes
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector
import segmentation_refinement as refine

CLASS_NUM = 34

def init_detector(config, checkpoint=None, device='cuda:0'):
    """Initialize a detector from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    model = build_detector(config.model, test_cfg=config.test_cfg)
    if checkpoint is not None:
        map_loc = 'cpu' if device == 'cpu' else None
        checkpoint = load_checkpoint(model, checkpoint, map_location=map_loc)
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            warnings.simplefilter('once')
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use COCO classes by default.')
            model.CLASSES = get_classes('coco')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


class LoadImage(object):
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """
        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_fields'] = ['img']
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def inference_detector(model, img):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        # Use torchvision ops for CPU mode instead
        for m in model.modules():
            if isinstance(m, (RoIPool, RoIAlign)):
                if not m.aligned:
                    # aligned=False is not implemented on CPU
                    # set use_torchvision on-the-fly
                    m.use_torchvision = True
        warnings.warn('We set use_torchvision=True in CPU mode.')
        # just get the actual data from DataContainer
        data['img_metas'] = data['img_metas'][0].data

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result

def segmentation_refine(result,model_stage_2, img):
    refiner = refine.Refiner(device='cuda:0',model_folder = model_stage_2)

    img1 = cv2.imread(img)
    w,h,c = img1.shape

    img_background = np.zeros((w, h), dtype=np.uint8)
    print(img_background.shape)
    if isinstance(result, tuple):
        bbox_results, mask_results = result
        x1, x2, y1, y2 = 0.0,0.0,0.0,0.0

        for g1 in range(CLASS_NUM):
            # print('bbox:',bbox_results[g1])
            if not mask_results[g1] == None:
                for g2 in range(np.array(mask_results[g1]).shape[0]):
                    instance_background = np.zeros((w, h), dtype=np.uint8)
                    if (ceil(bbox_results[g1][g2][3]) - floor(bbox_results[g1][g2][1]) == 0) or (
                            ceil(bbox_results[g1][g2][2]) - floor(bbox_results[g1][g2][0]) == 0):
                        x1,x2,y1,y2 = (floor(bbox_results[g1][g2][1]) - 3),ceil(bbox_results[g1][g2][3]),(floor(bbox_results[g1][g2][0]) - 3),ceil(bbox_results[g1][g2][2])
                        #print(x1,x2,y1,y2)
                        cropped_bbox = img1[x1:x2,y1:y2]
                        cropped_mask = mask_results[g1][g2][x1:x2,y1:y2]
                        cropped_mask = np.uint8((cropped_mask + 0) * 255)
                        output = refiner.refine(cropped_bbox, cropped_mask, fast=False, L=900)
                        instance_background[x1:x2, y1:y2] = output
                        mask_results[g1][g2] = instance_background>0


                    else:
                        x1, x2, y1, y2 = (floor(bbox_results[g1][g2][1])), ceil(bbox_results[g1][g2][3]),(floor(bbox_results[g1][g2][0])), ceil(bbox_results[g1][g2][2])
                        #print(x1, x2, y1, y2)
                        cropped_bbox = img1[x1:x2, y1:y2]
                        cropped_mask = mask_results[g1][g2][x1:x2, y1:y2]
                        cropped_mask = np.uint8((cropped_mask + 0) * 255)
                        output = refiner.refine(cropped_bbox, cropped_mask, fast=False, L=900)
                        instance_background[x1:x2, y1:y2] = output
                        mask_results[g1][g2] = instance_background > 0


    result = (bbox_results, mask_results)

    return result

async def async_inference_detector(model, img):
    """Async inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        Awaitable detection results.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    data = scatter(collate([data], samples_per_gpu=1), [device])[0]

    # We don't restore `torch.is_grad_enabled()` value during concurrent
    # inference since execution can overlap
    torch.set_grad_enabled(False)
    result = await model.aforward_test(rescale=True, **data)
    return result


def show_result_pyplot(model, img, result, score_thr=0.3, fig_size=(15, 10)):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        fig_size (tuple): Figure size of the pyplot figure.
    """
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(img, result, score_thr=score_thr, show=False)
    #cv2.imwrite("demo_image/final_mask.jpg",img)
    plt.figure(figsize=fig_size)
    plt.imshow(mmcv.bgr2rgb(img))
    plt.show()