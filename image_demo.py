from argparse import ArgumentParser
from mmdet.core import encode_mask_results, tensor2imgs
from mmdet.apis.inference import inference_detector, init_detector, show_result_pyplot,segmentation_refine


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file stage 1')
    parser.add_argument('--model_stage_2', help='Checkpoint file stage 2')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)

    # test a single image
    result = inference_detector(model,args.img)

    # segmentation_refine
    if args.model_stage_2 is not None:
        result = segmentation_refine(result,args.model_stage_2,args.img)

    result_img = show_result_pyplot(model, args.img, result, score_thr=args.score_thr)


if __name__ == '__main__':
    main()
