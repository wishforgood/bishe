import pickle
import sys
import torch
import os
if __name__ == "__main__":
    # input = '/home/pyw/SimSiam-main1/checkpoint/pixpro-ronghe-experiment_0323215124.pth'
    #
    # obj = torch.load(input, map_location="cpu")
    # obj = obj["state_dict"]
    #
    # new_model = {}
    # for k, v in obj.items():
    #     old_k = k
    #     k = k.replace("module.encoder.", "")
    #     if "backbone" not in k:
    #         k = "backbone." + k
    #     print(old_k, "->", k)
    #     new_model[k] = v
    # model_path = os.path.join('/home/pyw/SimSiam-main1/checkpoint/',f"pixpro-ronghe.pth")
    # torch.save ({
    #     "net": new_model,
    #     "__author__": "PixPro",
    #     },model_path)


    # input = '/home/pyw/mycode/resnet50-19c8e357.pth'
    #
    # obj = torch.load(input, map_location="cpu")
    # # obj = obj["model"]
    #
    # new_model = {}
    # for k, v in obj.items():
    #     old_k = k
    #     # k = k.replace("module.encoder.", "")
    #     if "backbone" not in k:
    #         k = "backbone." + k
    #     print(old_k, "->", k)
    #     new_model[k] = v
    # model_path = os.path.join('/home/pyw/mycode/checkpoint',f"only-resnet.pth")
    # torch.save ({
    #     "net": new_model,
    #     "__author__": "PixPro",
    #     },model_path)

    input = r'E:\code\lyf\mynet\mynet\unet6110moco-2023-0624_2033\checkpoints\latest.pth'

    obj = torch.load(input, map_location="cpu")
    obj = obj["net"]

    new_model = {}
    for k, v in obj.items():
        old_k = k
        k = k.replace("encoder_q.encoder", "")
        k = k.replace("decoder.", "")
        if "module" not in k:
            k = "module" + k
        print(old_k, "->", k)
        new_model[k] = v
    model_path = os.path.join('E:/code/lyf/mynet/mynet/unet6110moco-2023-0624_2033/checkpoints/',f"turn.pth")
    torch.save ({
        "net": new_model,
        },model_path)
