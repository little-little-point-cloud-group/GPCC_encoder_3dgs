import numpy as np
import gsplat_rendering.read_write_model as model
from pathlib import Path
from PIL import Image
from gsplat_rendering.gs_read_write import read3DG_ply
import torch
from gsplat.rendering import rasterization

def render(cameras_path,ply_path,output_dir):
    factor = 1
    model_path=Path(cameras_path)
    frame_path=Path(ply_path)
    output_dir=Path(output_dir)

    if model_path==None or not model_path.exists():
        raise Exception("Model path does not exists")
    if frame_path==None or not frame_path.exists():
        raise Exception("Input PLY path does not exists")
    if output_dir==None:
        raise Exception("No output dir")

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = None

    def getCamParameters(camera):
        if camera.model == "SIMPLE_PINHOLE" or camera.model == "SIMPLE_RADIAL":
            fx = camera.params[0]
            fy = camera.params[0]
            cx = camera.params[1]
            cy = camera.params[2]
        elif camera.model == "PINHOLE" or camera.model == "RADIAL":
            fx = camera.params[0]
            fy = camera.params[1]
            cx = camera.params[2]
            cy = camera.params[3]
        else:
            raise Exception(f"Unkwon camera model {camera.model}")
        return float(fx), float(fy), float(cx), float(cy)

    def quaternion_to_rot(q):
        return np.eye(3) + 2 * np.array([
            [-q[2] * q[2] - q[3] * q[3], q[1] * q[2] - q[3] * q[0], q[1] * q[3] + q[2] * q[0]],
            [q[1] * q[2] + q[3] * q[0], -q[1] * q[1] - q[3] * q[3], q[2] * q[3] - q[1] * q[0]],
            [q[1] * q[3] - q[2] * q[0], q[2] * q[3] + q[1] * q[0], -q[1] * q[1] - q[2] * q[2]]])

    def render(path_ply, width, height, focals, principal_points, viewmats):
        if focals.ndim == 1:
            assert viewmats.ndim==2
            assert focals.shape[0]==2
            assert principal_points.shape[0]==2
            assert viewmats.shape[0]==4 and viewmats.shape[1]==4
            Ks = np.array([[[focals[0],0,principal_points[0]],[0,focals[1],principal_points[0]],[0,0,1]]])
            viewmats = viewmats.reshape(1,4,4)
            numberOfCams = 1
        else:
            assert focals.ndim==2
            assert viewmats.ndim==3
            assert focals.shape[0]==viewmats.shape[0]
            assert focals.shape[1]==2
            assert principal_points.shape[0]==viewmats.shape[0]
            assert principal_points.shape[1]==2
            assert viewmats.shape[1]==4 and viewmats.shape[2]==4
            numberOfCams = focals.shape[0]
            Ks = np.zeros((numberOfCams,3,3))
            for i in range(numberOfCams):
                Ks[i,0,0] = focals[i,0]
                Ks[i,1,1] = focals[i,1]
                Ks[i,0,2] = principal_points[i,0]
                Ks[i,1,2] = principal_points[i,1]
                Ks[i,2,2] = 1

        means3d, coeffs, opacities, scales, quats = read3DG_ply(path_ply)

        # CONVERT TO TENSOR
        viewmats = torch.tensor(viewmats, dtype=torch.float, device=device)
        opacities = torch.sigmoid(torch.tensor(opacities.reshape(opacities.shape[0]), dtype=torch.float, device=device))
        means3d = torch.tensor(means3d, dtype=torch.float, device=device)
        scales = torch.exp(torch.tensor(scales, dtype=torch.float, device=device))
        quats = torch.tensor(quats, dtype=torch.float, device=device)
        Ks = torch.tensor(Ks, dtype=torch.float, device=device)

        quats = quats / quats.norm(dim=-1, keepdim=True)

        colors = torch.tensor(coeffs, dtype=torch.float, device=device)
        render_colors, _, _ = rasterization(
          means=means3d,
          quats=quats,
          scales=scales,
          opacities=opacities,
          colors=colors,
          viewmats=viewmats,  # [C, 4, 4]
          Ks=Ks, # [C, 3, 3]
          width=width,
          height=height,
          sh_degree=int(np.sqrt(coeffs.shape[1])-1),
        )

        return (torch.clamp(render_colors,0.0,1.0)*255).to(torch.uint8).cpu().numpy()

    # Getting camera parameters
    if Path(model_path, "cameras.txt").exists() and Path(model_path, "images.txt").exists():
        cameras = model.read_cameras_text(Path(model_path, "cameras.txt"))
        images = model.read_images_text(Path(model_path, "images.txt"))
    elif Path(model_path, "cameras.bin").exists() and Path(model_path, "images.bin").exists():
        cameras = model.read_cameras_binary(Path(model_path, "cameras.bin"))
        images = model.read_images_binary(Path(model_path, "images.bin"))
    else:
        raise ValueError("Unable to find cameras and images files")
    if len(images) == 0:
        raise ValueError("No images found in COLMAP.")

    width = 0
    height = 0
    qvecs = []
    tvecs = []
    fvecs = []
    cvecs = []
    for k in images:
        im = images[k]
        qvecs.append(im.qvec)
        tvecs.append(im.tvec)

        cam = cameras[im.camera_id]
        fx, fy, cx, cy = getCamParameters(cam)
        fvecs.append([fx/factor,fy/factor])
        cvecs.append([cx/factor,cy/factor])
        width = max(width, cam.width // factor)
        height = max(height, cam.height // factor)

    qvecs = np.array(qvecs)
    tvecs = np.array(tvecs)
    focals = np.array(fvecs)
    principal_points = np.array(cvecs)

    # Converting to camtoworlds matrix
    w2c_mats = []
    bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
    for j in range(len(qvecs)):
        rot = quaternion_to_rot(qvecs[j])
        trans = tvecs[j].reshape(3, 1)
        w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
        w2c_mats.append(w2c)
    w2c_mats = np.stack(w2c_mats, axis=0)

    # Rendering model
    frames = render(frame_path, width, height, focals, principal_points, w2c_mats)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Writting output images
    for j, image in enumerate(images.values()):
        im = Image.fromarray(frames[j], 'RGB')
        im.save(Path(output_dir, Path(image.name).stem + ".png"))



