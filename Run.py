import time

import colour
import rawpy
import numpy as np
import torch  # not needed, but fixes an error when onnxruntime creates the CUDAExecutionProvider in case of an incompatible installation of CUDA/CUDNN
import onnxruntime as ort
import imageio.v3 as iio

patch_size = 512
stride = patch_size - 16


def load_raw_image(raw_path):
    with rawpy.imread(raw_path) as raw:
        raw_data = raw.raw_image_visible.astype(np.float32)
        raw_data -= raw.black_level_per_channel[0]  # black level
        raw_data /= raw.camera_white_level_per_channel[0]  # white level
        cwb = np.array(raw.camera_whitebalance[:3]) / 1024
        dwb = np.array(raw.daylight_whitebalance[:3])
        raw_data[::2, ::2] *= cwb[0]
        raw_data[::2, 1::2] *= cwb[1]
        raw_data[1::2, ::2] *= cwb[1]
        raw_data[1::2, 1::2] *= cwb[2]
        ccm = raw.rgb_xyz_matrix[:3, :]
        # reverse daylight white balance (is baked into the matrix)
        ccm[0, :] *= dwb[0]
        ccm[1, :] *= dwb[1]
        ccm[2, :] *= dwb[2]
        return raw_data.clip(0, 2), ccm


def rawpy_process(in_path, out_path, demosaic_algorithm):
    # initially convert to xyz, otherwise there seems to be some contrast adjustment for sRGB
    # seems to have some problems with highlights when configured like that, but good enough as a test
    with rawpy.imread(in_path) as raw:
        rgb_image = raw.postprocess(demosaic_algorithm=demosaic_algorithm,
                                    use_camera_wb=True,
                                    no_auto_bright=True,
                                    output_color=rawpy.ColorSpace.XYZ,
                                    gamma=(1, 1),
                                    output_bps=16)
    rgb_image = colour.XYZ_to_sRGB((np.array(rgb_image) / 2 ** 16))
    iio.imwrite(out_path, (rgb_image.clip(0, 1) * 256).astype(np.uint8))


def ml_process(in_path, out_path):
    session = ort.InferenceSession('model.onnx', providers=['CUDAExecutionProvider'])

    raw_image, ccm = load_raw_image(in_path)

    H, W = raw_image.shape
    result = np.zeros((3, H, W))
    elapsed_list = []
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            xoffset = min(0, W - x - patch_size)
            yoffset = min(0, H - y - patch_size)
            patch = raw_image[np.newaxis, np.newaxis, y + yoffset:y + yoffset + patch_size, x + xoffset:x + xoffset + patch_size]
            start = time.time()
            outputs = session.run(None, {'in': patch})
            elapsed_list.append(time.time() - start)
            pad = (patch_size - stride) // 2
            result[:, y + pad:y + yoffset + patch_size - pad, x + pad:x + xoffset + patch_size - pad] \
                = outputs[0][0, :, pad - yoffset:-pad, pad - xoffset:-pad]

    # excludes first inference as this normally does some initializations
    print(f'Perf: {np.mean(np.array(elapsed_list)[1:]) / (512 ** 2) * (1000 ** 2):.6f}s/MP')

    # convert camera rgb to xyz (and multiply with some constant factor to match the rawpy result)
    result = (np.linalg.inv(ccm) @ (result.reshape(3, -1))).reshape(3, H, W).clip(0, 1) * 1.15

    result = colour.XYZ_to_sRGB(np.transpose(result, (1, 2, 0)))
    iio.imwrite(out_path, (result.clip(0, 1) * 255).astype(np.uint8))


if __name__ == '__main__':
    image_path = 'examples/091A7296.CR3'
    rawpy_process(image_path, 'examples/output_rp_dcb.png', rawpy.DemosaicAlgorithm.DCB)
    rawpy_process(image_path, 'examples/output_rp_ahd.png', rawpy.DemosaicAlgorithm.AHD)
    ml_process(image_path, 'examples/output_ml.png')
