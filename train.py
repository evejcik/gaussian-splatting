#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"



import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from PIL import Image
import numpy as np
import torchvision.transforms as T

# from transformers import AutoImageProcessor, DINOv2Model
# from transformers import AutoImageProcessor, CLIPProcessor, CLIPModel

import torchvision.models as models
import torch.nn as nn
from torchvision import transforms


import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


# args:
parser = ArgumentParser(description="Training script parameters")
lp = ModelParams(parser)
op = OptimizationParams(parser)
pp = PipelineParams(parser)
parser.add_argument('--ip', type=str, default="127.0.0.1")
parser.add_argument('--port', type=int, default=6009)
parser.add_argument('--debug_from', type=int, default=-1)
parser.add_argument('--detect_anomaly', action='store_true', default=False)
parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
parser.add_argument("--quiet", action="store_true")
parser.add_argument('--disable_viewer', action='store_true', default=False)
parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
parser.add_argument("--start_checkpoint", type=str, default = None)

parser.add_argument("--style_rgb_path", type=str)
parser.add_argument("--style_depth_path", type=str)
parser.add_argument("--style_rgb_weight", type=float, default=1.0)
parser.add_argument("--style_depth_weight", type=float, default=1.0)
parser.add_argument("--vgg_weight", type=float, default=0.0, help="Weight for VGG perceptual loss")


args = parser.parse_args(sys.argv[1:])



# ─── Configuration ────────────────────────────────────────────────────────────────
# Paths to your style inputs
style_rgb_path   = args.style_rgb_path #"gaussian_splatting/style/starrynight.jpg"
style_depth_path = args.style_depth_path #"style_depth.npy"  # the .npy you generated with ZoeDepth

assert os.path.exists(style_rgb_path), f"Style image not found at {style_rgb_path}"
assert os.path.exists(style_depth_path), f"Style depth map (.npy) not found at {style_depth_path}"

try:
    _ = Image.open(style_rgb_path)
except Exception as e:
    raise ValueError(f"Failed to open style image: {style_rgb_path}\nError: {e}")

try:
    _ = np.load(style_depth_path)
except Exception as e:
    raise ValueError(f"Failed to load style depth map: {style_depth_path}\nError: {e}")


# Your render / training resolution 
# Training resolution is set dynamically from viewpoint_cam resolution

# … any other configs (learning rates, checkpoints, etc.) …
resnet = models.resnet50(pretrained=True)
resnet.eval().cuda()

# Remove the final classification layer to get feature maps
feature_extractor = nn.Sequential(*list(resnet.children())[:-2]).cuda() #strips offthe average pooling and fully connected layers, because we only want the feature maps to compare visual textures and patterns, not class predictions

resnet_preprocess = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])
# ───────────────────────────────────────────────────────────────────────────────────


# --- VGG Class ---------------------------------------------------------------------
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features.eval()
        for param in vgg.parameters():
            param.requires_grad = False
        
        self.blocks = nn.ModuleList([
            vgg[:4].eval(),   # relu1_1
            vgg[4:9].eval(),  # relu2_1
            vgg[9:16].eval(), # relu3_1
            vgg[16:23].eval() # relu4_1
        ])
        
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    def encode(self, x):
        x = (x - self.mean) / self.std
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features

    def forward(self, input_tensor, target_features):
        input_features = self.encode(input_tensor)
        loss = 0.0
        for inp, tgt in zip(input_features, target_features):
            loss += F.l1_loss(inp, tgt)
        return loss

    
def extract_vgg_features(image_tensor, vgg_model):
    mean = torch.tensor([0.485, 0.456, 0.406], device=image_tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image_tensor.device).view(1, 3, 1, 1)
    image_tensor = (image_tensor - mean) / std

    features = []
    x = image_tensor
    for block in vgg_model.blocks:
        x = block(x)
        features.append(x)
    return features

    # -----------------------------------------------------------------------------------

def prepare_style_inputs(rgb_path, depth_npy_path, target_resolution=(800, 800), device="cuda"):
    """
    Load and resize the style image and depth map, then extract DINOv2 features.

    Args:
        rgb_path (str): Path to the style image.
        depth_npy_path (str): Path to the depth map (.npy).
        target_resolution (tuple): (width, height) to resize to.
        device (str): Device to load tensors on ("cuda" or "cpu").

    Returns:
        Tuple[Tensor, Tensor]: style_rgb_feats, style_depth_feats of shape [1, N, 768]
    """
    W, H = target_resolution

    # Load and resize style RGB
    style_rgb = Image.open(rgb_path).convert("RGB")
    style_rgb = style_rgb.resize((W, H), Image.BICUBIC)

    # Load and resize depth
    depth_array = np.load(depth_npy_path)
    depth_img = Image.fromarray((depth_array / depth_array.max() * 255).astype(np.uint8))
    depth_img = depth_img.resize((W, H), Image.BICUBIC)

   

    rgb_tensor = resnet_preprocess(style_rgb).unsqueeze(0).to(device)
    depth_img_rgb = depth_img.convert("RGB")  # ensures 3 channels
    depth_tensor = resnet_preprocess(depth_img_rgb).unsqueeze(0).to(device)


    # Extract features
    with torch.no_grad():
        style_rgb_feats = feature_extractor(rgb_tensor)
        style_depth_feats = feature_extractor(depth_tensor)

    return style_rgb_feats, style_depth_feats



def prepare_rendered_image_for_ResNet(rendered_tensor, resolution):
        """
    rendered_tensor: torch.Tensor of shape [3, H, W] (RGB) or [1, H, W] (grayscale/depth)
    resolution: tuple (W, H) — resolution to resize to before feeding into DINOv2

    Returns:
        torch.Tensor of shape [1, 3, 224, 224] — normalized for DINOv2
    """
        if rendered_tensor.shape[0] == 1:
            rendered_tensor = rendered_tensor.repeat(3, 1, 1)

        rendered_tensor = rendered_tensor.clamp(0, 1).unsqueeze(0)  # [1, 3, H, W]

        # Resize using F.interpolate for tensor input
        rendered_tensor = F.interpolate(rendered_tensor, size=(224, 224), mode='bilinear', align_corners=False)

        # Normalize manually
        mean = torch.tensor([0.485, 0.456, 0.406], device=rendered_tensor.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=rendered_tensor.device).view(1, 3, 1, 1)
        rendered_tensor = (rendered_tensor - mean) / std

        return rendered_tensor  # shape: [1, 3, 224, 224]


def extract_resnet_features(img: Image.Image) -> torch.Tensor:
    input_tensor = resnet_preprocess(img).unsqueeze(0).cuda()  # shape: [1, 3, 224, 224]
    with torch.no_grad():
        features = feature_extractor(input_tensor)  # shape: [1, 2048, 7, 7]
    return features

def flatten_features(feats):
    return feats.view(feats.size(0), feats.size(1), -1).permute(0, 2, 1)  # [1, N, C]

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    style_feature_cache = {}
    #############Emma's addition###############
    torch.autograd.set_detect_anomaly(True)

    





    feature_extractor.eval()
    # Cache for style features at different resolutions
    style_feature_cache = {}  # {(width, height): (style_rgb_feats, style_depth_feats)}




    # def cosine_patch_loss(A, B):
    #     """
    #     Compute cosine loss between two sets of feature tensors.
    #     A and B should have shape [B, C] or [B, C, H, W] (will be pooled if needed).
    #     """
    #     # If input has spatial dimensions, apply global average pooling
    #     if A.ndim == 4:
    #         A = F.adaptive_avg_pool2d(A, 1).squeeze(-1).squeeze(-1)  # [B, C]
    #     if B.ndim == 4:
    #         B = F.adaptive_avg_pool2d(B, 1).squeeze(-1).squeeze(-1)  # [B, C]

    #     A = F.normalize(A, dim=-1)
    #     B = F.normalize(B, dim=-1)
    #     return 1 - (A * B).sum(-1).mean()

    def style_patch_loss_l1(A, B):
        if A.ndim == 3:  # A is [1, 49, 2048]
            A = A.mean(dim=1)  # -> [1, 2048]
        if B.ndim == 3:
            B = B.mean(dim=1)
        return F.l1_loss(A, B)


    
##############Emma's addition end###############

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    
    if checkpoint:
          (model_params, first_iter) = torch.load(checkpoint)
          gaussians.restore(model_params, opt)
          print(f"[Checkpoint loaded] Resuming from iteration {first_iter}")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    vgg_loss_fn = VGGPerceptualLoss().to(device)


    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1): #training loop starts here
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)

        ###Emma's addition###
        # Match current training view resolution
        target_resolution = (viewpoint_cam.image_width, viewpoint_cam.image_height)

        # Check cache first
        if target_resolution in style_feature_cache:
            style_rgb_feats, style_depth_feats = style_feature_cache[target_resolution]
        else:
            # Resize + extract features once for this resolution
            style_rgb_feats, style_depth_feats = prepare_style_inputs(
                rgb_path=style_rgb_path,
                depth_npy_path=style_depth_path,
                target_resolution=target_resolution,
                device="cuda"
            )

            style_rgb_feats = flatten_features(style_rgb_feats)
            style_depth_feats = flatten_features(style_depth_feats)

        print("Extracting style features with shape:", style_rgb_feats.shape)


        ###Emm's addition end###

        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]


        ###Style Loss Block Start###

        # Match style resolution dynamically #check in output/.../cameras.json for camera resolutions
        target_resolution = (viewpoint_cam.image_width, viewpoint_cam.image_height)

        # Extract rendered outputs -> the rendered RGB and depth maps
        rendered_rgb = render_pkg["render"]
        rendered_depth = render_pkg["depth"]

        # Resize + normalize for ResNet
        rgb_input = prepare_rendered_image_for_ResNet(rendered_rgb, target_resolution)
        depth_input = prepare_rendered_image_for_ResNet(rendered_depth, target_resolution)

        # Normalize for VGG
        vgg_input = F.interpolate(rendered_rgb.unsqueeze(0), size=(224, 224), mode='bilinear')  # ensure shape is [1, 3, H, W]



        style_image = Image.open(style_rgb_path).convert("RGB")
        style_image_tensor = T.ToTensor()(style_image).unsqueeze(0).to(device)
        style_image_tensor = torch.nn.functional.interpolate(style_image_tensor, size=(224, 224), mode='bilinear')

        with torch.no_grad():
            vgg_target_features = extract_vgg_features(style_image_tensor, vgg_loss_fn)


        if depth_input.shape[1] == 1: #since resnet & clip expect 3 channels because they are trained on rgb, unlike dinov2 which can take 1 channel input
            depth_input = depth_input.repeat(1, 3, 1, 1)


        # Extract features
        #removing rgb feats and depth feats from inside the torch_no_grad() block, so that we can use them for style loss
        rgb_feats = feature_extractor(rgb_input)
        depth_feats = feature_extractor(depth_input)

                # Reshape to match style features: [1, 49, 2048]
        rendered_rgb_feats = flatten_features(rgb_feats)
        rendered_depth_feats = flatten_features(depth_feats)

        if tb_writer and iteration % 100 == 0:  # every 100 iters
            tb_writer.add_images("rendered_output", image.unsqueeze(0), iteration)




        # Compute cosine similarity loss

        print("Rendered feats shape:", rendered_rgb_feats.shape)
        print("Style feats shape:", style_rgb_feats.shape)
        style_loss_rgb = style_patch_loss_l1(rendered_rgb_feats, style_rgb_feats)
        style_loss_depth = style_patch_loss_l1(rendered_depth_feats, style_depth_feats)

        # Weight and add to total loss
        style_weight_rgb = getattr(opt, "style_rgb_weight", 1.0)
        style_weight_depth = getattr(opt, "style_depth_weight", 1.0)

        # VGG perceptual loss
        vgg_weight = getattr(opt, "vgg_weight", 1.0)
        vgg_style_loss = vgg_loss_fn(vgg_input, vgg_target_features)
        torch.cuda.empty_cache()


            

        


        if tb_writer:
            tb_writer.add_scalar("style_loss/rgb", style_loss_rgb.item(), iteration)
            tb_writer.add_scalar("style_loss/depth", style_loss_depth.item(), iteration)
            tb_writer.add_scalar("train_loss_patches/style_rgb", style_loss_rgb.item(), iteration)
            tb_writer.add_scalar("train_loss_patches/style_depth", style_loss_depth.item(), iteration)
            tb_writer.add_scalar("style_loss/vgg_rgb", vgg_style_loss.item(), iteration)


        



        #####Emma's addition end#####

        # if viewpoint_cam.alpha_mask is not None:
        #     alpha_mask = viewpoint_cam.alpha_mask.cuda()
        #     image *= alpha_mask

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image = image * alpha_mask  # NOT inplace anymore


        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        # loss += style_weight_rgb * style_loss_rgb + style_weight_depth * style_loss_depth
        # # Add to final loss
        # loss += vgg_weight * vgg_style_loss

        # loss = style_weight_rgb * style_loss_rgb + style_weight_depth * style_loss_depth + vgg_weight * vgg_style_loss #turning off l1 and dssim for now, since we are not training on a dataset with ground truth images
        lambda_dssim = opt.lambda_dssim  # e.g., 0.2

        reconstruction_loss = (1 - lambda_dssim) * Ll1 + lambda_dssim * (1 - ssim_value)

        loss = reconstruction_loss + style_weight_rgb * style_loss_rgb + style_weight_depth * style_loss_depth + vgg_weight * vgg_style_loss + Ll1depth #adding back in dssim to see if we can get a clearer image result

        # Depth regularization 
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]   # depth map in camera space # pulls the depth map from the render() call:


            #######Emma's addition end####
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        # Compute VGG features of the rendered image (no torch.no_grad)
        vgg_rendered_features = vgg_loss_fn.encode(vgg_input)
  

        print("Grad check (should be True):", rgb_feats.requires_grad, vgg_rendered_features[0].requires_grad)

        loss.backward()
        torch.cuda.empty_cache()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


         

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        # tb_writer.add_scalar('train_loss_patches/style_rgb', style_loss_rgb.item(), iteration)
        # tb_writer.add_scalar('train_loss_patches/style_depth', style_loss_depth.item(), iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
   
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
