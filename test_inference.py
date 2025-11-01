import cv2
import sys
sys.path.append("./../")
import numpy as np
import torch
from cldm.model import create_model, load_state_dict
from ldm.models.diffusion.ddim import DDIMSampler


def load_and_process_sate(img):
    if img is None:
        raise FileNotFoundError(f"卫星图像不存在")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    img = img[0:h - 488, 244:w - 244] 
    
    return img.astype(np.float32) 

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
        
def load_and_process_image(img, top_crop, bottom_crop, crop_size, unified_crop=None):
    if img is None:
        raise FileNotFoundError(f"图像不存在")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    h, w = img.shape[:2]
    if h > (top_crop + bottom_crop):
        img = img[top_crop:h - bottom_crop, :]  # 所有图像先做相同的固定裁剪
        h, w = img.shape[:2]  # 更新裁剪后的尺寸
    
    if unified_crop is not None:
        crop_h, crop_w = crop_size
        top, left = unified_crop
        if top + crop_h <= h and left + crop_w <= w:
            img = img[top:top + crop_h, left:left + crop_w]
    
    return img.astype(np.float32)  # 保持uint8类型，后续统一处理


def test_process(controls, model, ddim_sampler, H, W):
    try:
        with torch.no_grad():

            # 构建条件
            cond = {
                "c_concat": controls,
                "c_crossattn": [model.get_learned_conditioning([""] * 1)]
            }
            un_cond = {
                "c_concat": controls,
                "c_crossattn": [model.get_learned_conditioning([""] * 1)]
            }
            
            # 计算潜在空间形状
            shape = (4, H // 8, W // 8)

            # 设置控制强度
            model.control_scales = [1.0] * 13
            
            # 采样生成图像
            samples, intermediates = ddim_sampler.sample(
                50, 1, shape, cond, verbose=False, eta=0.0,
                unconditional_guidance_scale=7.5,
                unconditional_conditioning=un_cond
            )

            # 解码生成RGB图像
            x_samples = model.decode_first_stage(samples)
            x_samples = (x_samples.permute(0, 2, 3, 1) * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            cv2.imwrite("/mnt/mydisk/fkx/CVPR/control_revised/test/6/test.jpg",x_samples[0])

            print(f"Generated image shape: {x_samples[0].shape}")
            print("Test completed successfully!")
            return True
            
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return False

# 运行测试
if __name__ == "__main__":
        # 处理输入图像
    # seg_test_image = cv2.imread("/mnt/mydisk/fkx/CVPR/data/s1/seg/1736938405733.jpg")
    # depth_test_image = cv2.imread("/mnt/mydisk/fkx/CVPR/data/s1/depth/1736938405733.jpg")
    # infrared_test_image = cv2.imread("/mnt/mydisk/fkx/CVPR/data/s1/infrared/1736938405733.jpg")
    # satellite_test_image = cv2.imread("/mnt/mydisk/fkx/CVPR/data/s1/satellite/1736938405733.jpg")

    # print(f"Test image shape: {seg_test_image.shape}")
    
    # crop_size = (512, 512)
    # top_crop=0
    # bottom_crop=200
    # temp_rgb = load_and_process_image(seg_test_image, top_crop, bottom_crop, crop_size, unified_crop=None)  # 只做固定裁剪


    # unified_crop_params = get_unified_crop_params(temp_rgb.shape, crop_size)  # 生成统一随机位置
    # # 3. 加载所有图像（应用相同的裁剪参数）
    # seg = load_and_process_image(seg_test_image, top_crop, bottom_crop, crop_size, unified_crop_params)
    # depth = load_and_process_image(depth_test_image, top_crop, bottom_crop, crop_size, unified_crop_params)
    # infrared = load_and_process_image(infrared_test_image, top_crop, bottom_crop, crop_size, unified_crop_params)
    # sate = load_and_process_sate(satellite_test_image)
    
    seg = cv2.imread("/mnt/mydisk/fkx/CVPR/baseline/fkx/seg/1726993057656.jpg")
    depth = cv2.imread("/mnt/mydisk/fkx/CVPR/baseline/fkx/depth/1726993057656.jpg")
    infrared = cv2.imread("/mnt/mydisk/fkx/CVPR/baseline/fkx/infrared/1726993057656.jpg")
    sate = cv2.imread("/mnt/mydisk/fkx/CVPR/baseline/fkx/sate/1726993057656.jpg")

    # infrared = cv2.imread("/mnt/mydisk/fkx/CVPR/baseline/fkx/1726992763773.jpg")

    target_height = 512  # 整除避免小数（如 1024→512，768→384）
    target_width = 512

    seg = cv2.resize(
        seg, 
        dsize=(target_width, target_height),
        interpolation=cv2.INTER_AREA  # 缩小图像优先用 INTER_AREA，抗锯齿效果更好
    )
    depth = cv2.resize(
        depth, 
        dsize=(target_width, target_height),
        interpolation=cv2.INTER_AREA  # 缩小图像优先用 INTER_AREA，抗锯齿效果更好
    )
    infrared = cv2.resize(
        infrared, 
        dsize=(target_width, target_height),
        interpolation=cv2.INTER_AREA  # 缩小图像优先用 INTER_AREA，抗锯齿效果更好
    )
    sate = cv2.resize(
        sate, 
        dsize=(target_width, target_height),
        interpolation=cv2.INTER_AREA  # 缩小图像优先用 INTER_AREA，抗锯齿效果更好
    )

    cv2.imwrite("/mnt/mydisk/fkx/CVPR/control_revised/test/6/test_seg.jpg",seg)
    cv2.imwrite("/mnt/mydisk/fkx/CVPR/control_revised/test/6/test_depth.jpg",depth)
    cv2.imwrite("/mnt/mydisk/fkx/CVPR/control_revised/test/6/test_infrared.jpg",infrared)
    cv2.imwrite("/mnt/mydisk/fkx/CVPR/control_revised/test/6/test_sate.jpg",sate)

    

    controls = []
    control = [seg, depth, infrared, sate]
    for c in control:
        co = cv2.cvtColor(c, cv2.COLOR_BGR2RGB).astype(np.float32)
        co = torch.from_numpy(co.copy()).float().cuda() / 255.0
        co = torch.stack([co], dim=0)
        co = co.permute(0, 3, 1, 2).clone()

        controls.append(co)
  
    # co = cv2.cvtColor(infrared, cv2.COLOR_BGR2RGB).astype(np.float32)
    # co = torch.from_numpy(co.copy()).float().cuda() / 255.0
    # co = torch.stack([co], dim=0)
    # co = co.permute(0, 3, 1, 2).clone()
    # c = torch.zeros_like(co)
    # controls = [c, c, co, c]

    print("Loading model...")
    model = create_model('./models/cldm_v21.yaml').cpu()
    model.load_state_dict(load_state_dict(
        '/mnt/mydisk/fkx/CVPR/control_revised/experiments/251029/checkpoints/last-epoch=epoch=79-step=step=042799.ckpt', 
        location='cpu'
    ))
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)
    print("Model loaded successfully!")

    success = test_process(controls, model, ddim_sampler, 512, 512)
    if success:
        print("✅ Inference test passed!")
    else:
        print("❌ Inference test failed!")
