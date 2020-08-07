import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info

from skimage import io
import numpy as np
import os
from math import floor,ceil
from mmdet.core import encode_mask_results, tensor2imgs
CLASS_NUM = 34
INPUT_PATH = "/data/test"
OUTPUT_PATH = "data/test_dataset_stage_2"

def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3,
                    save_crop_image = True):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result,
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # encode mask results
        if isinstance(result, tuple):
            bbox_results, mask_results = result
            if save_crop_image:
                crop_image(result, INPUT_PATH, OUTPUT_PATH,i)
            encoded_mask_results = encode_mask_results(mask_results)
            result = bbox_results, encoded_mask_results
        results.append(result)

        batch_size = len(data['img_metas'][0].data)
        for _ in range(batch_size):
            prog_bar.update()
    return results

def crop_image(result,input_path,output_path,i):
    #import io
    # encode mask results
    #import glob
    if isinstance(result, tuple):
        #print("crop_image")
        bbox_results, mask_results = result
        #image_path_list = glob.glob(input_path + "/*.jpg")
        img1 = io.imread(input_path+'/{:07d}.jpg'.format(i))  # load the origin image
        for g1 in range(CLASS_NUM):
            if not mask_results[g1] == None:
                for g2 in range(np.array(mask_results[g1]).shape[0]):
                    bbox_saved = (output_path+'/{:d}_[{:.12f},{:.12f},{:.12f},{:.12f}]'
                                  '_{:d}_{:.12f}.jpg').format(i + 1,
                                                              bbox_results[g1][g2][0],
                                                              bbox_results[g1][g2][1],
                                                              bbox_results[g1][g2][2] - bbox_results[g1][g2][0],
                                                              bbox_results[g1][g2][3] - bbox_results[g1][g2][1],
                                                              g1 + 1,
                                                              bbox_results[g1][g2][4])
                    mask_saved = (output_path+'/{:d}_[{:.12f},{:.12f},{:.12f},{:.12f}]'
                                  '_{:d}_{:.12f}.png').format(i + 1,
                                                              bbox_results[g1][g2][0],
                                                              bbox_results[g1][g2][1],
                                                              bbox_results[g1][g2][2] - bbox_results[g1][g2][0],
                                                              bbox_results[g1][g2][3] - bbox_results[g1][g2][1],
                                                              g1 + 1,
                                                              bbox_results[g1][g2][4])
                    if (ceil(bbox_results[g1][g2][3]) - floor(bbox_results[g1][g2][1]) == 0) or \
                            (ceil(bbox_results[g1][g2][2]) - floor(bbox_results[g1][g2][0]) == 0):
                        cropped_bbox = img1[(floor(bbox_results[g1][g2][1]) - 3):ceil(bbox_results[g1][g2][3]),
                                       (floor(bbox_results[g1][g2][0]) - 3):ceil(bbox_results[g1][g2][2])]
                        cropped_mask = mask_results[g1][g2][
                                       (floor(bbox_results[g1][g2][1]) - 3):ceil(bbox_results[g1][g2][3]),
                                       (floor(bbox_results[g1][g2][0]) - 3):ceil(bbox_results[g1][g2][2])]
                    else:
                        cropped_bbox = img1[floor(bbox_results[g1][g2][1]):ceil(bbox_results[g1][g2][3]),
                                       floor(bbox_results[g1][g2][0]):ceil(bbox_results[g1][g2][2])]
                        cropped_mask = mask_results[g1][g2][
                                       floor(bbox_results[g1][g2][1]):ceil(bbox_results[g1][g2][3]),
                                       floor(bbox_results[g1][g2][0]):ceil(bbox_results[g1][g2][2])]
                    io.imsave(bbox_saved, cropped_bbox)
                    io.imsave(mask_saved, cropped_mask)

def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            # encode mask results
            if isinstance(result, tuple):
                bbox_results, mask_results = result
                encoded_mask_results = encode_mask_results(mask_results)
                result = bbox_results, encoded_mask_results
        results.append(result)

        if rank == 0:
            batch_size = (
                len(data['img_meta'].data)
                if 'img_meta' in data else len(data['img_metas'][0].data))
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
