import os
import sys
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import pycocoevalcap.spice as spice


os.environ["CUDA_VISIBLE_DEVICES"] = '2'
os.environ['TORCH_HOME'] = '/mnt/sdc1/junhong/proj/text_guide_attack/cache'
# os.environ['SPICE_CACHE'] = '/new_data/yifei2/anaconda3/envs/lavis/lib/python3.8/site-packages/pycocoevalcap/spice'
import argparse

def eval_caption(gt_path,result_path):
    # print(sys.path)
    # for path in sys.path:
    #     package_path = os.path.join(path, 'pycocoevalcap')
    #     if os.path.exists(package_path):
    #         raise FileNotFoundError(f'pycocoevalcap was not installed')
    #
    # spice.SPICE_JAR=os.path.join(package_path,'spice')
    # print(spice.SPICE_JAR)
    coco = COCO(gt_path)
    coco_result = coco.loadRes(result_path)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # minigpt-4
    # parser.add_argument("--result_path", default="/new_data/yifei2/junhong/text_guide_attack/compared_methods/minigpt4/minigpt_temp.json", help="path to model caption "
    #                                                                                       "result file.")
    parser.add_argument("--result_path",
                        default="/new_data/yifei2/junhong/text_guide_attack/compared_methods/minigpt4/minigpt_temp.json",
                        help="path to model caption "
                             "result file.")
    # /new_data/yifei2/junhong/text_guide_attack/output/BLIP/Caption_coco/20240824203/test_epochbest.json
    parser.add_argument("--gt_path", default="/new_data/yifei2/junhong/dataset/coco_gt/coco_karpathy_test_gt.json", help="path to the ground truth file.")
    args = parser.parse_args()
    # print(sys.path)
    eval_caption(args.gt_path,args.result_path)
    # coco = COCO(args.gt_path)
    # coco_result = coco.loadRes(args.result_path)
    # coco_eval = COCOEvalCap(coco, coco_result)
    # coco_eval.evaluate()
    #
    # # print output evaluation scores
    # for metric, score in coco_eval.eval.items():
    #     print(f"{metric}: {score:.3f}")
