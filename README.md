# Multi-View Datasets for MPFC & SS-MPFC Research

Bộ sưu tập **7 datasets multi-view** phục vụ nghiên cứu MPFC/SS-MPFC clustering.

## Tổng Quan

### A. SS-MPFC — Remote Sensing Datasets (3 datasets)

| Dataset | Samples | Clusters | Views | Total Features | Format |
|---------|---------|----------|-------|----------------|--------|
| **Augsburg** | 78,294* | 7 | 3 | 185 (HS:180 + SAR:4 + DSM:1) | .mat |
| **MUUFL Gulfport** | 53,687* | 11 | 2 | 66 (HS:64 + LiDAR:2) | .mat |
| **Trento** | 30,214* | 6 | 2 | 65 (HS:63 + LiDAR:2) | .mat |

*Số samples là số pixels có nhãn (labeled pixels), không phải tổng số pixels trong ảnh.

### B. MPFC — Multi-View Benchmark Datasets (4 datasets)

| Dataset | Samples | Clusters | Views | Dimensions/view | Format |
|---------|---------|----------|-------|-----------------|--------|
| **3Sources** | 169 | 6 | 3 | BBC:3560, Reuters:3068, Guardian:3631 | .mat |
| **MSRC-v5** | 210 | 7 | 5 | CM:24, HOG:576, GIST:512, LBP:256, CENT:254 | .mat |
| **Wikipedia-test** | 693 | 10 | 2 | Word:128, SIFT:10 | .mat |
| **CiteSeer** | 3,312 | 6 | 2 | Content:3703, Cites:4732 | .mat |

**Nguồn**: [ChuanbinZhang/Multi-view-datasets](https://github.com/ChuanbinZhang/Multi-view-datasets). Cấu trúc .mat: `X` (cell array of views), `y` (labels).

---

## 1. Dataset Augsburg (IEEE GRSS Data Fusion Contest 2018)

### Thông tin chung
- **Nguồn**: IEEE GRSS Data Fusion Contest 2018
- **Vị trí**: Vùng đô thị Augsburg, Đức
- **URL gốc**: https://figshare.com/articles/dataset/28112405
- **Kích thước ảnh**: 332 × 485 = 161,020 pixels

### Cấu trúc file
```
Augsburg/
├── HS-SAR-DSM-Augsburg/          # Dataset chính (3 views)
│   ├── data_HS_LR.mat            # Hyperspectral: (332, 485, 180), uint16
│   ├── data_SAR_HR.mat           # SAR: (332, 485, 4), float64
│   ├── data_DSM.mat              # DSM: (332, 485), float64
│   ├── TrainImage.mat            # Training labels: (332, 485), uint8
│   └── TestImage.mat             # Test labels: (332, 485), float32
└── HS-SAR-Berlin/                # Dataset phụ (2 views)
    ├── data_HS_LR.mat            # Hyperspectral
    ├── data_SAR_HR.mat           # SAR
    ├── TrainImage.mat
    └── TestImage.mat
```

### Chi tiết các Views
| View | File | Shape | Features | Mô tả |
|------|------|-------|----------|-------|
| **View 1** | data_HS_LR.mat | (332, 485, 180) | 180 | Hyperspectral bands |
| **View 2** | data_SAR_HR.mat | (332, 485, 4) | 4 | SAR polarization |
| **View 3** | data_DSM.mat | (332, 485) | 1 | Digital Surface Model |

### 7 Classes
| ID | Class Name | Mô tả |
|----|------------|-------|
| 1 | Residential Area | Khu dân cư |
| 2 | Commercial Area | Khu thương mại |
| 3 | Industrial Area | Khu công nghiệp |
| 4 | Low Plants | Thực vật thấp |
| 5 | Forest | Rừng |
| 6 | Allotment | Đất phân lô |
| 7 | Water | Nước |

### Cách load dữ liệu
```python
import scipy.io as sio
import numpy as np

# Load data
base_path = '/home/dll/nckh/dataset/multi-view-remote-sensing/Augsburg/HS-SAR-DSM-Augsburg/'

hs = sio.loadmat(base_path + 'data_HS_LR.mat')['data_HS_LR']      # (332, 485, 180)
sar = sio.loadmat(base_path + 'data_SAR_HR.mat')['data_SAR_HR']   # (332, 485, 4)
dsm = sio.loadmat(base_path + 'data_DSM.mat')['data_DSM']         # (332, 485)
labels = sio.loadmat(base_path + 'TrainImage.mat')['TrainImage']  # (332, 485)

# Flatten for clustering
H, W = hs.shape[:2]
X_hs = hs.reshape(-1, 180)     # (161020, 180)
X_sar = sar.reshape(-1, 4)     # (161020, 4)
X_dsm = dsm.reshape(-1, 1)     # (161020, 1)
y = labels.flatten()           # (161020,)

# Concatenate all views
X_all = np.hstack([X_hs, X_sar, X_dsm])  # (161020, 185)

# Filter labeled pixels only
mask = y > 0
X_labeled = X_all[mask]
y_labeled = y[mask]
print(f"Labeled samples: {len(y_labeled)}")  # ~78,294
```

---

## 2. Dataset MUUFL Gulfport

### Thông tin chung
- **Nguồn**: University of Florida, GatorSense Lab
- **Vị trí**: University of Southern Mississippi - Gulf Park Campus, USA
- **URL gốc**: https://github.com/GatorSense/MUUFLGulfport
- **Kích thước ảnh**: 325 × 220 (campus_1), varies for others

### Cấu trúc file
```
MUUFL-Gulfport/
├── README.md
├── LICENSE
├── MUUFLGulfportDataCollection/
│   ├── muufl_gulfport_campus_w_lidar_1.mat  # Campus 1 with LiDAR (MAIN)
│   ├── muufl_gulfport_campus_3.mat
│   ├── muufl_gulfport_campus_4.mat
│   ├── tgt_img_spectra.mat
│   ├── tgt_lab_spectra.mat
│   └── MUUFL_Gulfport_GroundTruth.csv
└── MUUFLGulfportSceneLabels/
    ├── README.md
    ├── muufl_gulfport_campus_1_hsi_220_label.mat  # Scene labels
    ├── MUUFL_GulfportTechReport_SceneLabelGroundTruth.pdf
    └── muufl_scene_labels_screenshot.png
```

### Chi tiết các Views
| View | Features | Mô tả |
|------|----------|-------|
| **View 1** | 64 | Hyperspectral bands (72 bands, 8 removed due to noise) |
| **View 2** | 2 | LiDAR-derived features (elevation, intensity) |

### 11 Classes
| ID | Class Name |
|----|------------|
| 1 | Trees |
| 2 | Mostly-grass ground surface |
| 3 | Mixed ground surface |
| 4 | Dirt and sand |
| 5 | Road |
| 6 | Water |
| 7 | Buildings shadow |
| 8 | Buildings |
| 9 | Sidewalk |
| 10 | Yellow curb |
| 11 | Cloth panels (targets) |

### Cách load dữ liệu
```python
import scipy.io as sio
import numpy as np

# Load campus 1 with LiDAR
mat = sio.loadmat('/home/dll/nckh/dataset/multi-view-remote-sensing/MUUFL-Gulfport/MUUFLGulfportDataCollection/muufl_gulfport_campus_w_lidar_1.mat')

# Extract data from nested structure
hsi_struct = mat['hsi'][0, 0]
hs_data = hsi_struct['Data']           # Hyperspectral cube
lidar_data = hsi_struct['Lidar']       # LiDAR data
ground_truth = hsi_struct['groundTruth']

# Load scene labels
labels_mat = sio.loadmat('/home/dll/nckh/dataset/multi-view-remote-sensing/MUUFL-Gulfport/MUUFLGulfportSceneLabels/muufl_gulfport_campus_1_hsi_220_label.mat')
scene_labels = labels_mat['hsi'][0, 0]['sceneLabels']

print(f"HS shape: {hs_data.shape}")
print(f"LiDAR shape: {lidar_data.shape}")
```

---

## 3. Dataset Trento

### Thông tin chung
- **Nguồn**: Rural area in Trento, Italy
- **URL gốc**: https://github.com/tyust-dayu/Trento
- **Kích thước ảnh**: 166 × 600 = 99,600 pixels

### Cấu trúc file
```
Trento/
├── Italy_hsi.mat       # Hyperspectral: (166, 600, 63), float32
├── Italy_lidar.mat     # LiDAR: (166, 600, 2), float32
├── allgrd.mat          # Ground truth mask: (166, 600), uint8
└── README.md
```

### Chi tiết các Views
| View | File | Shape | Features | Mô tả |
|------|------|-------|----------|-------|
| **View 1** | Italy_hsi.mat | (166, 600, 63) | 63 | Hyperspectral bands |
| **View 2** | Italy_lidar.mat | (166, 600, 2) | 2 | LiDAR (DSM, intensity) |

### 6 Classes
| ID | Class Name | Mô tả |
|----|------------|-------|
| 1 | Wood | Rừng gỗ |
| 2 | Buildings | Tòa nhà |
| 3 | Apple Trees | Cây táo |
| 4 | Road | Đường |
| 5 | Ground | Đất |
| 6 | Vineyard | Vườn nho |

### Cách load dữ liệu
```python
import scipy.io as sio
import numpy as np

base_path = '/home/dll/nckh/dataset/multi-view-remote-sensing/Trento/'

# Load data
hs = sio.loadmat(base_path + 'Italy_hsi.mat')['data']       # (166, 600, 63)
lidar = sio.loadmat(base_path + 'Italy_lidar.mat')['data']  # (166, 600, 2)
labels = sio.loadmat(base_path + 'allgrd.mat')['mask_test'] # (166, 600)

# Flatten for clustering
H, W = hs.shape[:2]
X_hs = hs.reshape(-1, 63)       # (99600, 63)
X_lidar = lidar.reshape(-1, 2)  # (99600, 2)
y = labels.flatten()            # (99600,)

# Concatenate views
X_all = np.hstack([X_hs, X_lidar])  # (99600, 65)

# Filter labeled pixels
mask = y > 0
X_labeled = X_all[mask]
y_labeled = y[mask]
print(f"Labeled samples: {len(y_labeled)}")  # ~30,214
```

---

| Khía cạnh | Augsburg | Trento | MUUFL |
|-----------|----------|--------|-------|
| **Optical/HS** | 180 bands | 63 bands | 64 bands |
| **SAR** | 4 features | - | - |
| **DSM/LiDAR** | 1 feature | 2 features | 2 features |
| **Total Features** | 185 | 65 | 66 |
| **Best ACC** | 68.60% | 67.24% | 74.18% |

---

## Sử Dụng Cho Thực Nghiệm

### Chuẩn hóa dữ liệu
```python
from sklearn.preprocessing import StandardScaler

# Chuẩn hóa từng view riêng biệt
scaler_hs = StandardScaler()
scaler_sar = StandardScaler()

X_hs_norm = scaler_hs.fit_transform(X_hs)
X_sar_norm = scaler_sar.fit_transform(X_sar)

# Ghép views
X_concat = np.hstack([X_hs_norm, X_sar_norm])
```

### Tạo semi-supervised constraints (5% labels)
```python
import numpy as np

def create_labeled_mask(y, ratio=0.05, seed=42):
    """Tạo mask cho 5% labeled samples."""
    np.random.seed(seed)
    n_samples = len(y)
    n_labeled = int(n_samples * ratio)

    labeled_idx = np.random.choice(n_samples, n_labeled, replace=False)
    labeled_mask = np.zeros(n_samples, dtype=int) - 1  # -1 = unlabeled
    labeled_mask[labeled_idx] = y[labeled_idx]

    return labeled_mask

labeled = create_labeled_mask(y_labeled, ratio=0.05)
```

---

## References

1. **Augsburg Dataset**: IEEE GRSS Data Fusion Contest 2018
   https://figshare.com/articles/dataset/28112405

2. **MUUFL Gulfport**: GatorSense Lab, University of Florida
   https://github.com/GatorSense/MUUFLGulfport

3. **Trento Dataset**: GitHub Repository
   https://github.com/tyust-dayu/Trento

---

## Chi Tiết Kỹ Thuật Về Sensors và Thu Thập Dữ Liệu

### Trento Dataset - Chi tiết Sensors

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TRENTO DATA ACQUISITION (2015)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  HYPERSPECTRAL SENSOR: AISA Eagle (Specim, Finland)                         │
│  ─────────────────────────────────────────────────────                      │
│  • Spectral range: 402.89 - 989.09 nm (VNIR)                                │
│  • Number of bands: 63 (sau khi loại noise bands)                           │
│  • Spatial resolution: 1 meter                                              │
│  • Imaging mode: Pushbroom scanner                                          │
│  • Platform: Airborne                                                       │
│                                                                             │
│  LiDAR SENSOR: Optech ALTM 3100EA                                           │
│  ─────────────────────────────────────                                      │
│  • Laser wavelength: 1064 nm (near-infrared)                                │
│  • Pulse rate: up to 100 kHz                                                │
│  • Returns: First and last return point cloud                               │
│  • Output products:                                                         │
│    - DSM (Digital Surface Model): elevation                                 │
│    - Intensity: backscatter strength                                        │
│  • Accuracy: ~15 cm vertical                                                │
│                                                                             │
│  PROVIDED BY: Prof. Lorenzo Bruzzone, University of Trento                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Class Distribution (Trento):**
| Class | Name | Samples | Percentage |
|-------|------|---------|------------|
| 1 | Woods | 4,034 | 13.4% |
| 2 | Buildings | 2,903 | 9.6% |
| 3 | Apple Trees | 479 | **1.6%** (minority) |
| 4 | Roads | 9,123 | 30.2% |
| 5 | Ground | 10,501 | **34.8%** (majority) |
| 6 | Vineyard | 3,174 | 10.5% |

### Augsburg Dataset - Chi tiết Sensors

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       AUGSBURG DATA ACQUISITION                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  HYPERSPECTRAL (EnMAP-like simulation)                                      │
│  ─────────────────────────────────────                                      │
│  • Number of bands: 180                                                     │
│  • Spectral range: VNIR + SWIR                                              │
│  • Spatial resolution: 30m (resampled)                                      │
│  • Simulated from airborne hyperspectral data                               │
│                                                                             │
│  SAR SENSOR: Sentinel-1 (ESA)                                               │
│  ───────────────────────────                                                │
│  • Frequency: C-band (5.405 GHz)                                            │
│  • Polarizations: VV, VH                                                    │
│  • Derived products: VV/VH ratio, VH/VV ratio                               │
│  • Imaging mode: Interferometric Wide Swath (IW)                            │
│  • Resolution: 10m (resampled to match HS)                                  │
│  • All-weather, day/night capability                                        │
│                                                                             │
│  DSM (Digital Surface Model)                                                │
│  ──────────────────────────                                                 │
│  • Source: TanDEM-X or similar                                              │
│  • Resolution: matched to other modalities                                  │
│  • Provides elevation information                                           │
│                                                                             │
│  REFERENCE: MDAS Dataset (Hu et al., 2022)                                  │
│  DOI: 10.14459/2022mp1657312                                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Benchmark Datasets

### 1. Được Cite Rộng Rãi

| Dataset | Citations | Top Venues |
|---------|-----------|------------|
| **Trento** | >500 papers | IEEE TGRS, ISPRS, Remote Sensing |
| **Augsburg** | >100 papers | IEEE GRSS, MDPI Remote Sensing |
| **MUUFL** | >300 papers | IEEE TGRS, Pattern Recognition |

### 2. Được Sử Dụng Trong Các Cuộc Thi Quốc Tế

- **IEEE GRSS Data Fusion Contest 2018**: Augsburg là một phần của challenge
- **University of Trento Benchmark**: Trento được dùng làm standard benchmark
- **GatorSense Lab Challenge**: MUUFL là dataset chính

### 3. Multi-Modal Nature

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WHY MULTI-MODAL DATA IS IMPORTANT                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  HYPERSPECTRAL (HS)           SAR                      LiDAR/DSM            │
│  ─────────────────           ─────                    ──────────            │
│  • Rich spectral info        • Texture/structure      • 3D structure        │
│  • Material identification   • All-weather            • Canopy height       │
│  • Vegetation indices        • Day/night              • Building height     │
│  • Water detection           • Penetration            • Terrain model       │
│                                                                             │
│  COMPLEMENTARY INFORMATION → BETTER CLASSIFICATION                          │
│                                                                             │
│  Example: Urban vs Forest                                                   │
│  - HS alone: may confuse green roofs with vegetation                        │
│  - SAR alone: texture similar for dense urban and forest                    │
│  - HS + SAR + DSM: height differentiates buildings from trees               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4. Ground Truth Quality

| Aspect | Trento | Augsburg | MUUFL |
|--------|--------|----------|-------|
| **Annotation method** | Expert manual | Contest organizers | Field survey |
| **Verification** | University of Trento | IEEE GRSS committee | GatorSense Lab |
| **Consistency** | High | High | High |
| **Public availability** | ✅ Free | ✅ Free | ✅ Free |

---

## Benchmark Results (State-of-the-Art)

### Supervised Learning (100% labels)

| Method | Trento OA | Augsburg OA | Year |
|--------|-----------|-------------|------|
| SVM (RBF) | 85.2% | 60.1% | 2016 |
| Random Forest | 88.5% | 65.3% | 2017 |
| 2D-CNN | 92.1% | 70.2% | 2018 |
| 3D-CNN | 94.8% | 73.5% | 2019 |
| ResNet + Fusion | 96.2% | 76.8% | 2020 |
| Transformer | 97.5% | 80.2% | 2022 |
| CNN-GCN Dual | 98.1% | 82.4% | 2023 |

### Semi-Supervised Learning (5% labels)

| Method | Trento OA | Augsburg OA |
|--------|-----------|-------------|
| Label Propagation | 72.3% | 55.2% |

---

## Cách Download Datasets

### Option 1: Manual Download

```bash
# Trento
git clone https://github.com/tyust-dayu/Trento.git

# MUUFL Gulfport
git clone https://github.com/GatorSense/MUUFLGulfport.git

# Augsburg (MDAS)
# Download from: https://doi.org/10.14459/2022mp1657312
```

### Option 2: Sử dụng rs-fusion-datasets (Python)

```python
# Install
pip install rs-fusion-datasets

# Usage
from rs_fusion_datasets import fetch_trento, fetch_augsburg, fetch_muufl

# Download and load Trento
trento_data = fetch_trento()
X_hs = trento_data['hsi']
X_lidar = trento_data['lidar']
y = trento_data['gt']

# Download and load Augsburg
augsburg_data = fetch_augsburg()
X_hs = augsburg_data['hsi']
X_sar = augsburg_data['sar']
y = augsburg_data['gt']
```

---

## References (Bổ sung)

### Dataset Papers

5. **Trento Dataset Original**:
   - Provider: Prof. Lorenzo Bruzzone, University of Trento, Italy
   - Used in: "Classification of Hyperspectral and LiDAR Data Using Coupled CNNs"
   - URL: https://github.com/tyust-dayu/Trento

6. **Augsburg/MDAS Dataset**:
   - Hu, J., Liu, R., Hong, D., et al. (2022). "MDAS: A New Multimodal Benchmark Dataset for Remote Sensing"
   - DOI: [10.14459/2022mp1657312](https://doi.org/10.14459/2022mp1657312)
   - Earth System Science Data Discussions

7. **IEEE GRSS Data Fusion Contest 2018**:
   - "Advanced Multi-Sensor Optical Remote Sensing for Urban Land Use and Land Cover Classification"
   - URL: https://www.grss-ieee.org/community/technical-committees/2018-ieee-grss-data-fusion-contest/

### Survey Papers (Recommended Reading)

8. "Deep Learning for Classification of Hyperspectral Data: A Comparative Review"
   - IEEE Geoscience and Remote Sensing Magazine, 2019

9. "Multimodal Remote Sensing Data Fusion: A Review"
   - IEEE TGRS, 2022

### Tools

10. **rs-fusion-datasets**: Remote Sensing Dataset fetcher and loader
    - GitHub: https://github.com/songyz2019/rs-fusion-datasets
    - Supports: Houston, MUUFL, Trento, Berlin, Augsburg

---

---

## 4. Dataset 3Sources (text, 3 views)

- **Mô tả**: 169 tin tức từ 3 hãng: BBC, Reuters, The Guardian. 6 chủ đề: business, entertainment, health, politics, sport, technology. Mỗi view = word histogram.
- **Nguồn**: University College Dublin — [http://mlg.ucd.ie/datasets/3sources.html](http://mlg.ucd.ie/datasets/3sources.html)
- **Paper**: MPFC (CMC 2025, DOI: 10.32604/cmc.2025.065127)

```
3Sources/
└── 3Sources.mat    # X: {(169,3560), (169,3631), (169,3068)}, y: (169,)
```

### Cách load
```python
import scipy.io as sio
data = sio.loadmat('3Sources/3Sources.mat')
X = data['X']  # cell array: X[0,0]=(169,3560), X[0,1]=(169,3631), X[0,2]=(169,3068)
y = data['y'].flatten()  # (169,), 6 classes
```

---

## 5. Dataset MSRC-v5 (image, 5 views)

- **Mô tả**: 210 ảnh, 7 object types. 5 views trích xuất đặc trưng: Color Moments (24-D), HOG (576-D), GIST (512-D), LBP (256-D), CENTRIST (254-D).
- **Nguồn**: Microsoft Research Cambridge
- **Paper**: MPFC (CMC 2025)

```
MSRC-v5/
└── MSRC-v5.mat    # X: 5 views, y: (210,), 7 classes
```

### Cách load
```python
data = sio.loadmat('MSRC-v5/MSRC-v5.mat')
X = data['X']  # X[0,i] for view i (i=0..4)
y = data['y'].flatten()  # (210,), 7 classes
# View dims: 24, 576, 512, 256, 254
```

---

## 6. Dataset Wikipedia-test (text+image, 2 views)

- **Mô tả**: 693 cặp text-image từ Wikipedia Featured Articles, 10 categories. View 1: word histograms (128-D), View 2: SIFT (10-D).
- **Nguồn**: SVCL/UCSD — [http://www.svcl.ucsd.edu/projects/crossmodal/](http://www.svcl.ucsd.edu/projects/crossmodal/)
- **Paper**: MPFC (CMC 2025)

```
Wikipedia-test/
└── Wikipedia-test.mat    # X: 2 views, y: (693,), 10 classes
```

### Cách load
```python
data = sio.loadmat('Wikipedia-test/Wikipedia-test.mat')
X = data['X']  # X[0,0]=(693,128), X[0,1]=(693,10)
y = data['y'].flatten()  # (693,), 10 classes
```

---

## 7. Dataset CiteSeer (citation network, 2 views)

- **Mô tả**: 3312 publications khoa học, 6 lĩnh vực (Agents, IR, DB, AI, HCI, ML). View 1: citation links (3312-D). View 2: binary word vector (3703-D).
- **Nguồn**: [LINQS](https://linqs.org/datasets/)
- **Paper**: MPFC (CMC 2025)

```
CiteSeer/
└── CiteSeer.mat    # X: 2 views, y: (3312,), 6 classes
```

### Cách load
```python
data = sio.loadmat('CiteSeer/CiteSeer.mat')
X = data['X']  # X[0,0]=(3312,3312), X[0,1]=(3312,3703)
y = data['y'].flatten()  # (3312,), 6 classes
```

---

## Paper References

| Paper | Datasets | DOI |
|-------|----------|-----|
| **MPFC** (CMC 2025, vol.83(3)) | 3Sources, MSRC-v5, Wikipedia-test, CiteSeer | [10.32604/cmc.2025.065127](https://doi.org/10.32604/cmc.2025.065127) |
| **SS-MPFC** (CMC 2025) | Augsburg, Trento, MUUFL | [10.32604/cmc.2025.071776](https://doi.org/10.32604/cmc.2025.071776) |
| **Hong et al.** (IEEE TGRS 2021) | Augsburg, Trento, MUUFL | [10.1109/TGRS.2020.3016820](https://doi.org/10.1109/TGRS.2020.3016820) |

*Updated: 2026-02-20*
