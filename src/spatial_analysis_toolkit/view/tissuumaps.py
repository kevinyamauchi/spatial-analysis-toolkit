import numpy as np
import pyvips
import tissuumaps.jupyter as tmap
from anndata import AnnData
from skimage.util import img_as_ubyte
from squidpy.im import ImageContainer


def _convert_numpy_to_vips(image: np.ndarray) -> pyvips.Image:
    im_byte = img_as_ubyte(image)

    WIDTH = im_byte.shape[1]
    HEIGHT = im_byte.shape[0]

    im_shape = im_byte.shape
    if len(im_shape) == 3:
        # RGB image
        assert im_shape[2] == 3, "3D image must be YXC and be RGB"
        # reshape into a huge linear array
        linear = im_byte.reshape(WIDTH * HEIGHT * 3)

        img_vips = pyvips.Image.new_from_memory(
            linear.data, WIDTH, HEIGHT, bands=3, format="uchar"
        )
    elif len(im_shape) == 2:
        # grayscale image
        linear = im_byte.reshape(-1)
        img_vips = pyvips.Image.new_from_memory(
            linear.data, WIDTH, HEIGHT, bands=1, format="uchar"
        )
    else:
        raise ValueError("Invalid image shape. Must be YX, or YXC (RGB)")

    return img_vips


def _rescale_intensity_vips(
    img_vips: pyvips.Image,
    min_percentile: float = 0.5,
    max_percentile: float = 99.5,
) -> pyvips.Image:
    min_val = img_vips.percent(min_percentile)
    max_val = img_vips.percent(max_percentile)
    if min_val == max_val:
        min_val = 0
        max_val = 255
    if img_vips.percent(1) < 0 or img_vips.percent(99) > 255:
        img_vips = (255.0 * (img_vips - min_val)) / (max_val - min_val)
        img_vips = (img_vips < 0).ifthenelse(0, img_vips)
        img_vips = (img_vips > 255).ifthenelse(255, img_vips)
        img_vips = img_vips.scaleimage()
    return img_vips


def _write_image_container(
    file_name: str, image_container, spatial_scale_factor: float = 0.5
):
    """Write a squidpy image container image to disk"""
    im = np.squeeze(image_container.data["image"].to_numpy())

    img_vips = _convert_numpy_to_vips(im)
    img_vips = _rescale_intensity_vips(
        img_vips, min_percentile=0.5, max_percentile=99.5
    )
    img_vips = img_vips.resize(spatial_scale_factor)

    img_vips.tiffsave(
        file_name,
        pyramid=True,
        tile=True,
        tile_width=256,
        tile_height=256,
        compression="jpeg",
        Q=95,
        properties=True,
    )


def _write_anndata_to_tissuumaps(
    file_name: str, adata: AnnData, spatial_scale_factor: float = 0.5
):
    """Write an AnnData obs table to disk, formatted for TissUUmaps"""
    obs = adata.obs.copy()
    if "X_umap" in adata.obsm:
        obs["umap_0"] = adata.obsm["X_umap"][:, 0]
        obs["umap_1"] = adata.obsm["X_umap"][:, 1]
    if "spatial" in adata.obsm:
        obs["x"] = adata.obsm["spatial"][:, 0] * spatial_scale_factor
        obs["y"] = adata.obsm["spatial"][:, 1] * spatial_scale_factor
    else:
        raise KeyError("adata.obsm must contain coordinates in 'spatial' key")
    obs.to_csv(file_name)


def tissuumaps_notebook_viewer(
    adata: AnnData,
    image_container: ImageContainer,
    spatial_scale_factor: float = 0.5,
    point_scale_factor: float = 10,
    host: str = "localhost",
    port: float = 5101,
    image_path: str = "img.tif",
    csv_path: str = "markers_table.csv",
    plugins=("Feature_Space"),
):
    _write_image_container(
        file_name=image_path,
        image_container=image_container,
        spatial_scale_factor=spatial_scale_factor,
    )

    _write_anndata_to_tissuumaps(
        file_name=csv_path,
        adata=adata,
        spatial_scale_factor=spatial_scale_factor,
    )

    tmap.loaddata(
        images=[image_path],
        csvFiles=[csv_path],
        xSelector="x",
        ySelector="y",
        scaleFactor=point_scale_factor,
        host=host,
        port=port,
        plugins=["Feature_Space"],
    )
