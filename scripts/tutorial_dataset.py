import json
import cv2
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, crop_size=(512, 512), top_crop=0, bottom_crop=200): 
        self.data = []
        self.crop_size = crop_size
        self.top_crop = top_crop
        self.bottom_crop = bottom_crop

        # 设置随机裁剪变换
        with open('./data/prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 1. 从JSON中读取4种条件的文件名
        source_filename = item['source']
        seg_filename = source_filename.replace("source","seg")
        depth_filename = source_filename.replace("source","depth")
        infrared_filename = source_filename.replace("source","infrared")
        rgb_filename = source_filename.replace("source","satellite")

        target_filename = source_filename.replace("source","persudo_perfect")
        mask_filename = source_filename.replace("source","persudo_mask").replace("jpg","npy")
        # prompt = item['prompt']

        def get_unified_crop_params(img_shape, crop_size):
            img_h, img_w = img_shape[:2]
            crop_h, crop_w = crop_size
            # 计算可裁剪的范围（避免超出图像边界）
            max_top = img_h - crop_h
            max_left = img_w - crop_w
            # 生成随机位置（固定seed可复现，不固定则每次不同）
            top = np.random.randint(0, max_top + 1)
            left = np.random.randint(0, max_left + 1)
            return top, left

        def load_and_process_mask(filename, unified_crop=None):
            mask_path = f'/mnt/mydisk/fkx/CVPR/data/s1/{filename}'
            mask = np.load(mask_path)
            crop_h, crop_w = self.crop_size
            h, w = mask.shape
            top, left = unified_crop
            # 确保裁剪不超出图像（兜底检查）
            if top + crop_h <= h and left + crop_w <= w:
                mask = mask[top:top + crop_h, left:left + crop_w]
            
            return mask  # 保持uint8类型，后续统一处理

        # 4. 加载图像函数（支持传入统一裁剪参数）
        def load_and_process_sate(filename):
            """
            filename: 图像文件名
            unified_crop: 统一裁剪参数 (top, left)，None则不做随机裁剪
            """
            img_path = f'/mnt/mydisk/fkx/CVPR/data/s1/{filename}'
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"图像不存在：{img_path}")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            h, w = img.shape[:2]
            img = img[0:h - 488, 244:w - 244] 
            
            return img.astype(np.float32) 
        

        def load_and_process_image(filename, unified_crop=None):
            img_path = f'/mnt/mydisk/fkx/CVPR/data/s1/{filename}'
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"图像不存在：{img_path}")
            
            # 转为RGB格式（与后续处理一致）
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 第一步：固定裁剪（移除顶部和底部）
            h, w = img.shape[:2]
            if h > (self.top_crop + self.bottom_crop):
                img = img[self.top_crop:h - self.bottom_crop, :]  # 所有图像先做相同的固定裁剪
                h, w = img.shape[:2]  # 更新裁剪后的尺寸
            
            # 第二步：统一随机裁剪（所有图像用同一组top/left）
            if unified_crop is not None:
                crop_h, crop_w = self.crop_size
                top, left = unified_crop
                # 确保裁剪不超出图像（兜底检查）
                if top + crop_h <= h and left + crop_w <= w:
                    img = img[top:top + crop_h, left:left + crop_w]
            
            return img.astype(np.float32)  # 保持uint8类型，后续统一处理
        
        temp_rgb = load_and_process_image(rgb_filename, unified_crop=None)  # 只做固定裁剪
        unified_crop_params = get_unified_crop_params(temp_rgb.shape, self.crop_size)  # 生成统一随机位置
        # 3. 加载所有图像（应用相同的裁剪参数）
        seg = load_and_process_image(seg_filename, unified_crop_params)
        depth = load_and_process_image(depth_filename, unified_crop_params)
        infrared = load_and_process_image(infrared_filename, unified_crop_params)
        # sate = load_and_process_sate(rgb_filename)
        target = load_and_process_image(target_filename, unified_crop_params)
        mask = load_and_process_mask(mask_filename, unified_crop_params)

        # 5. 归一化
        seg = seg / 255.0
        depth = depth / 255.0
        infrared = infrared / 255.0
        # sate = sate / 255.0
        target = (target / 127.5) - 1.0  # 目标归一化到[-1, 1]

        # 6. 将numpy数组转换为PyTorch张量，保持HWC格式
        target = torch.from_numpy(target).float()
        seg = torch.from_numpy(seg).float()
        depth = torch.from_numpy(depth).float()
        infrared = torch.from_numpy(infrared).float()
        # sate = torch.from_numpy(sate).float()
        mask = torch.from_numpy(mask).unsqueeze(-1).float()  # 掩码从HW转换为HWC格式

        return {
            "source": target,          # 目标图像
            "mask": mask,
            "prompt": "", # 使用CLIP生成的文本提示
            "seg": seg,             # 分割图
            "depth": depth,         # 深度图
            "infrared": infrared   # 红外图
            # "sate": sate              # RGB图
        }
