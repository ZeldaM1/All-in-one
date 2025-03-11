# All-in-One Image Compression and Restoration  
We provide results and checkpoints of typical image restoration methods ([WGWSNet](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhu_Learning_Weather-General_and_Weather-Specific_Features_for_Image_Restoration_Under_Multiple_CVPR_2023_paper.pdf)
, [AirNet](https://openaccess.thecvf.com/content/CVPR2022/html/Li_All-in-One_Image_Restoration_for_Unknown_Corruption_CVPR_2022_paper.html)
, [Restormer](https://openaccess.thecvf.com/content/CVPR2022/html/Zamir_Restormer_Efficient_Transformer_for_High-Resolution_Image_Restoration_CVPR_2022_paper.html)
, [SwinIR](https://openaccess.thecvf.com/content/ICCV2021W/AIM/html/Liang_SwinIR_Image_Restoration_Using_Swin_Transformer_ICCVW_2021_paper.html)) on **all-in-one image restoration** this this doc.
## All-in-one Image Restoration
### Dataset
| Setting        | Degradation   | Train           | Test            |
|----------------|---------------|-----------------|-----------------|
| Weather        | Haze-Snow-Rain         | [RESIDE](https://sites.google.com/site/boyilics/website-builder/reside)-[CSD](https://github.com/weitingchen83/ICCV2021-Single-Image-Desnowing-HDCWNet?tab=readme-ov-file)-[Rain1400](https://xueyangfu.github.io/projects/cvpr2017.html)      | [RESIDE](https://sites.google.com/site/boyilics/website-builder/reside)-[CSD](https://github.com/weitingchen83/ICCV2021-Single-Image-Desnowing-HDCWNet?tab=readme-ov-file)-[Rain1400](https://xueyangfu.github.io/projects/cvpr2017.html)       |
| Gaussian Noise | σ = 15-25-50        | [Open Images](https://storage.googleapis.com/openimages/web/index.html)| [Kodak](https://r0k.us/graphics/kodak/)     |
### Benchmark
The metrics are shown as `PSNR/SSIM/MS-SSIM`

* Weather Degradation 


| Method        | Venue   |      RESIDE        |   CSD       |Rain1400  |
|----------------|---------------|-----------------|-----------------|-----------------|
[Restormer](https://openaccess.thecvf.com/content/CVPR2022/html/Zamir_Restormer_Efficient_Transformer_for_High-Resolution_Image_Restoration_CVPR_2022_paper.html) | CVPR 2022 | 33.72 / 0.9840 / 0.9946  |  32.25 / 0.9589  / 0.9779 | 31.54 / 0.9177 / 0.9698  |
[SwinIR](https://openaccess.thecvf.com/content/ICCV2021W/AIM/html/Liang_SwinIR_Image_Restoration_Using_Swin_Transformer_ICCVW_2021_paper.html) | ICCVW 2021 |32.60 / 0.9832/  0.9900	| 31.84 /  0.9613 /  0.9613	| 31.13 /   0.9146 /   0.9678 
[AirNet](https://openaccess.thecvf.com/content/CVPR2022/html/Li_All-in-One_Image_Restoration_for_Unknown_Corruption_CVPR_2022_paper.html) |CVPR 2022 |29.85 /  0.9750 / 0.9886 |	31.28	/  0.9593 / 0.9769 | 30.64/ 0.9105 /  0.9656 
[WGWSNet](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhu_Learning_Weather-General_and_Weather-Specific_Features_for_Image_Restoration_Under_Multiple_CVPR_2023_paper.pdf) | CVPR 2023 |21.43 / 	0.8973 /   0.9583 | 19.47 /   0.7688 /  0.7716 |	29.78 /   0.8952 /  0.9643 
[WGWSNet*](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhu_Learning_Weather-General_and_Weather-Specific_Features_for_Image_Restoration_Under_Multiple_CVPR_2023_paper.pdf) | CVPR 2023 | 30.22	/ 0.9586 / 0.9883 | 23.03 / 0.8459 / 0.8554 |	 	30.32 	/	0.9000		/	0.9691 |

**Note:**
In our paper, we train the above all-in-one restoration methods with both degraded and clean images. For [WGWSNet](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhu_Learning_Weather-General_and_Weather-Specific_Features_for_Image_Restoration_Under_Multiple_CVPR_2023_paper.pdf), we notice significant performance drop when including clean images, therefore we also provide the results trained w/o clean images (denoted as WGWSNet*).

* Gaussian Noise Degradation


| Method        | Venue   |      σ = 15        |   σ = 25        |σ = 50    |
|----------------|---------------|-----------------|-----------------|-----------------|
[Restormer](https://openaccess.thecvf.com/content/CVPR2022/html/Zamir_Restormer_Efficient_Transformer_for_High-Resolution_Image_Restoration_CVPR_2022_paper.html) | CVPR 2022 |   |   |   |
[SwinIR](https://openaccess.thecvf.com/content/ICCV2021W/AIM/html/Liang_SwinIR_Image_Restoration_Using_Swin_Transformer_ICCVW_2021_paper.html) | ICCVW 2021 |   |   |   |
[AirNet](https://openaccess.thecvf.com/content/CVPR2022/html/Li_All-in-One_Image_Restoration_for_Unknown_Corruption_CVPR_2022_paper.html) |CVPR 2022|   |   |   |
[WGWSNet](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhu_Learning_Weather-General_and_Weather-Specific_Features_for_Image_Restoration_Under_Multiple_CVPR_2023_paper.pdf) | CVPR 2023 |   |   |   |
[WGWSNet*](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhu_Learning_Weather-General_and_Weather-Specific_Features_for_Image_Restoration_Under_Multiple_CVPR_2023_paper.pdf) | CVPR 2023 |   |   |   |


### Checkpoint
Checkpoint of the above restoration methods under all-in-one setting can be found on [GoogleDrive](https://drive.google.com/drive/folders/1uQNUGUWtibMX1EB9Y33Zh-Ss6fK7tC9C?usp=sharing).

## TODO
* noise performance
* inference code for restoration models
