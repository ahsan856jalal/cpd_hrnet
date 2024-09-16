import argparse
import os
import os.path as osp
import shutil
import tempfile
import json
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import load_checkpoint, get_dist_info
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from mmdet1.apis import init_dist
from mmdet1.core import results2json, coco_eval, wrap_fp16_model
from mmdet1.datasets import build_dataloader, build_dataset
from mmdet1.models import build_detector

from tools.cityPerson.eval_demo import validate
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from mmdet.datasets import CocoDataset
original_width, original_height = 2048, 1024
gt_width, gt_height = 704, 520

# Scaling factors
scale_x = gt_width / original_width
scale_y = gt_height / original_height
# Function to scale bbox
def scale_bbox(bbox, scale_x, scale_y):
    x, y, width, height = bbox
    x = x * scale_x
    y = y * scale_y
    width = width * scale_x
    height = height * scale_y
    return [x, y, width, height]

def evaluate_coco(gt_ann_file, results, data_loader):
    coco_gt = COCO(gt_ann_file)
    
    # Get the list of valid image IDs from the COCO ground truth
    gt_img_ids = set(coco_gt.getImgIds())
    
    # Print ground truth image IDs for debugging
    # print("Ground truth image IDs:", gt_img_ids)
    
    # Filter the results to only include entries with valid image IDs
    valid_results = [res for res in results if res['image_id'] in gt_img_ids]
    
    # Print results image IDs for debugging
    result_img_ids = set([res['image_id'] for res in valid_results])
    # print("Result image IDs:", result_img_ids)
    
    # Check if any result image IDs are not in ground truth
    missing_in_gt = result_img_ids - gt_img_ids
    if missing_in_gt:
        print(" Image IDs in results but not in ground truth:", missing_in_gt)
    
    # Load the filtered results
    coco_dt = coco_gt.loadRes(results)

    # Evaluate the results
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    # coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    return coco_eval.stats



def single_gpu_test(model, data_loader, show=False, save_img=False, save_img_dir=''):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
        results.append(result)
        print('show is ', show)
        # if show:
        #     model.module.show_result(data, result, dataset.img_norm_cfg, save_result=save_img, result_name=save_img_dir + '/' + str(i)+'.jpg')

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results

# def single_gpu_test(model, data_loader, save_img=False, save_img_dir=''):
#     model.eval()
#     results = []
#     dataset = data_loader.dataset
#     prog_bar = mmcv.ProgressBar(len(dataset))

#     for i, data in enumerate(data_loader):
#         with torch.no_grad():
#             result = model(return_loss=False, rescale=True, **data)
#         results.append(result)

#         if save_img:
#             print('data has',data.keys())
#             # Get image filename from img_metas
#             img_meta = data['img_meta'][0]
#             img_filename = img_meta['filename']
#             # img_filename = data['img_meta'][0]['filename']
#             img_name = os.path.basename(img_filename)
            
#             # Save result with original filename
#             model.module.show_result(
#                 data,
#                 result,
#                 dataset.img_norm_cfg,
#                 show=False,
#                 out_file=os.path.join(save_img_dir, img_name)
#             )

        # Update progress bar
    #     batch_size = data['img'][0].size(0)
    #     for _ in range(batch_size):
    #         prog_bar.update()

    # return results
def multi_gpu_test(model, data_loader, tmpdir=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, return_id=True, **data)
        results.append(result)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    results = collect_results(results, len(dataset), tmpdir)

    return results


def collect_results(result_part, size, tmpdir=None):
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
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
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


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('--config',default='/netscratch/jalal/F2DNet-hrnet/configs/f2dnet/cp/base_a172.py', help='test config file path')
    parser.add_argument('--checkpoint',default='/netscratch/jalal/F2DNet-hrnet/ahsan_results/a172_base/epoch_', help='checkpoint file')
    parser.add_argument('--checkpoint_start', type=int, default=15)#1
    parser.add_argument('--checkpoint_end', type=int, default=16)#100
    parser.add_argument('--out',default='/netscratch/jalal/F2DNet-hrnet/results_ahsan/result_a172_base.json', help='output result file')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--show',default=True, action='store_true', help='show results')
    parser.add_argument('--save_img', action='store_true', help='save result image')
    parser.add_argument('--save_img_dir', type=str, help='the dir for result image', default='/netscratch/jalal/F2DNet-hrnet/ahsan_results/EVALUATIONS/a172_base_results')
    parser.add_argument('--coco_json_path', type=str, help='the dir for result image', default='/netscratch/jalal/marunet/json_cells_data/yolo_A172_test.json')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--mean_teacher', action='store_true', help='test the mean teacher pth')

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args
def results_to_json(dataset, results):
    json_results = []
    for idx in range(len(dataset)):
        img_id = dataset.img_ids[idx]
        result = results[idx]
        if isinstance(result, tuple):
            bbox_result, _ = result
        else:
            bbox_result = result
        for label in range(len(bbox_result)):
            bboxes = bbox_result[label]
            for bbox in bboxes:
                data = dict()
                data['image_id'] = int(img_id)
                data['bbox'] = [float(bbox[0]), float(bbox[1]), float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])]
                data['score'] = float(bbox[4])
                data['category_id'] = int(dataset.cat_ids[label])
                json_results.append(data)
    return json_results

def main():
    args = parse_args()

    if args.out is not None and not args.out.endswith(('.json', '.pickle')):
        raise ValueError('The output file must be a pkl file.')
    for i in range(args.checkpoint_start, args.checkpoint_end):
        cfg = mmcv.Config.fromfile(args.config)
        # set cudnn_benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
        cfg.model.pretrained = None
        cfg.data.test.test_mode = True

        # init distributed env first, since logger depends on the dist info.
        if args.launcher == 'none':
            distributed = False
        else:
            distributed = True
            init_dist(args.launcher, **cfg.dist_params)

        # build the dataloader
        # TODO: support multiple images per gpu (only minor changes are needed)
        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)

        # build the model and load checkpoint
        model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        if not args.mean_teacher:
            while not osp.exists(args.checkpoint + str(i) + '.pth'):
                time.sleep(5)
            while i+1 != args.checkpoint_end and not osp.exists(args.checkpoint + str(i+1) + '.pth'):
                time.sleep(5)
            checkpoint = load_checkpoint(model, args.checkpoint + str(i) + '.pth', map_location='cpu')
            model.CLASSES = dataset.CLASSES
        else:
            while not osp.exists(args.checkpoint + str(i) + '.pth.stu'):
                time.sleep(5)
            while i+1 != args.checkpoint_end and not osp.exists(args.checkpoint + str(i+1) + '.pth.stu'):
                time.sleep(5)
            checkpoint = load_checkpoint(model, args.checkpoint + str(i) + '.pth.stu', map_location='cpu')
            checkpoint['meta'] = dict()
            if 'CLASSES' in checkpoint['meta']:
                model.CLASSES = checkpoint['meta']['CLASSES']
            else:
                model.CLASSES = dataset.CLASSES        # old versions did not save class info in checkpoints, this walkaround is
        # for backward compatibility
        

        if not distributed:
            model = MMDataParallel(model, device_ids=[0])
            os.makedirs(args.save_img_dir,exist_ok=True)
            # outputs = single_gpu_test(model, data_loader, args.show, args.save_img, args.save_img_dir)
            outputs = single_gpu_test(model, data_loader,show=args.show, save_img=True, save_img_dir=args.save_img_dir)
            json_results = results_to_json(data_loader.dataset, outputs)
            result_file = args.out
            with open(result_file, 'w') as f:
                json.dump(json_results, f)

        else:
            model = MMDistributedDataParallel(model.cuda())
            outputs = multi_gpu_test(model, data_loader, args.tmpdir)

        # res = []
        # for id, boxes in enumerate(outputs):
        #     if boxes:
        #         boxes=boxes[0]
        #         if type(boxes) == list:
        #             boxes = boxes[0]
        #         boxes[:, [2, 3]] -= boxes[:, [0, 1]]
        #     if len(boxes) > 0:
        #         for box in boxes:
        #             # box[:4] = box[:4] / 0.6
        #             temp = dict()
        #             temp['image_id'] = id+1
        #             temp['category_id'] = 1
        #             temp['bbox'] = box[:4].tolist()
        #             temp['score'] = float(box[4])
        #             res.append(temp)

        # with open(args.out, 'w') as f:
        #     json.dump(res, f)
        # Evaluate
        gt_ann_file = args.coco_json_path
        with open(args.out, 'r') as f:
            results = json.load(f)
        # for result in results:
        #     result['bbox'] = scale_bbox(result['bbox'], scale_x, scale_y)
        coco_eval_stats = evaluate_coco(gt_ann_file,results, data_loader)
        # MRs = validate('/netscratch/jalal/marunet/json_cells_data/yolo_A172_test.json', args.out)
        # print('Checkpoint %d: [Reasonable: %.2f%%], [Reasonable_Small: %.2f%%], [Heavy: %.2f%%], [All: %.2f%%]'
        #       % (i, MRs[0] * 100, MRs[1] * 100, MRs[2] * 100, MRs[3] * 100))


if __name__ == '__main__':
    main()
