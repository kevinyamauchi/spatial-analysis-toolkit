import numpy as np
import tissuumaps.jupyter as tmap
from anndata import AnnData
from skimage.io import imsave
from squidpy.im import ImageContainer


def _write_image_container(file_name: str, image_container):
    """Write a squidpy image container image to disk"""
    imsave(file_name, np.squeeze(image_container.data.to_array()))


def _write_anndata_to_tissuumaps(file_name: str, adata: AnnData):
    """Write an AnnData obs table to disk, formatted for TissUUmaps"""
    obs = adata.obs.copy()
    if "X_umap" in adata.obsm:
        obs["umap_0"] = adata.obsm["X_umap"][:, 0]
        obs["umap_1"] = adata.obsm["X_umap"][:, 1]
    if "spatial" in adata.obsm:
        obs["x"] = adata.obsm["spatial"][:, 0]
        obs["y"] = adata.obsm["spatial"][:, 1]
    else:
        raise KeyError("adata.obsm must contain coordinates in 'spatial' key")
    obs.to_csv(file_name)


def tissuumaps_notebook_viewer(
    adata: AnnData,
    image_container: ImageContainer,
    point_scale_factor: float = 10,
    port: float = 5101,
    image_path: str = "img.tif",
    csv_path: str = "markers_table.csv",
    plugins=("Feature_Space"),
):
    _write_image_container(
        file_name=image_path, image_container=image_container
    )

    _write_anndata_to_tissuumaps(file_name=csv_path, adata=adata)

    tmap.loaddata(
        images=[image_path],
        csvFiles=[csv_path],
        xSelector="x",
        ySelector="y",
        scaleFactor=point_scale_factor,
        port=port,
        plugins=["Feature_Space"],
    )
