# 2D Material Optical Contrast Fitting Tool (2D材料光学对比度拟合工具)

## 项目简介 (Introduction)

本项目是一个用于分析和拟合二维材料（如 $\text{MoS}_2$, $\text{WS}_2$, $\text{WSe}_2$ 等）光学反射对比度（Optical Contrast）谱的专业工具。它结合了**传输矩阵法 (TMM)** 和 **Faddeeva-Voigt / Lorentz 振子模型**，用于提取激子峰位、振子强度及展宽。

本项目提供两种使用方式，满足不同场景需求：
1.  **Web 版 (Streamlit)**: 无需安装复杂环境，单文件 (`index.html`) 即可运行，支持离线使用，界面现代友好。
2.  **桌面版 (PyQt6)**: 功能最全，适合需要批量处理和深度定制的高级用户。

---

## 核心功能 (Key Features)

### 1. 数据处理与可视化
*   **多格式支持**: 兼容 `.csv`, `.txt` 等常见光谱数据格式。
*   **智能单位识别**: 自动识别波长单位（nm）或能量单位（eV），并进行标准化处理。
*   **自动插值对齐**: 自动将样品光谱插值到衬底光谱的波长格点上，确保对比度计算精确。
*   **实时交互绘图**: 拟合过程中实时更新曲线，支持局部放大、数据点查看及高清图片导出。

### 2. 精确的物理模型
*   **多层膜结构 TMM 计算**: 支持任意层厚的堆叠结构计算：
    *   Substrate: Si (支持温度修正), Quartz, Sapphire, TiO2 等。
    *   Dielectric: SiO2, hBN (Top/Bottom 封装层)。
    *   2D Material: 单层或少层样品。
*   **高级光学默认值**: 温度、有限 NA、Si 光学数据源和背景介电常数保留为代码级参数，界面使用稳定默认值。
*   **受控材料插值**: Si 光学常数在 400--1305 nm 内使用保形 PCHIP，区间外使用端点切线线性延拓并保持被动性。
*   **复 Voigt 介电函数（推荐）**: 使用 Faddeeva 函数分别拟合 Lorentz 均匀展宽 $w_L$ 和 Gaussian 非均匀展宽 $w_G$；经典 Lorentz 模型仍可选择。
    
  $$
  \epsilon(E) = \epsilon_\infty + \sum_j \frac{f_j}{E_{0,j}^2 - E^2 - i E \Gamma_j}
    $$

### 3. 先进的拟合算法
*   **量测级多阶段优化**:
    *   **Robust LM (推荐)**: 先用有边界 `soft_l1` Trust Region Reflective 抑制坏点并定位，再通过有界参数变换使用真正的 Levenberg-Marquardt 精修。
    *   **Global + Robust LM**: 先进行差分进化全局初始化，再执行 Robust TRF + LM，适用于初值不确定或多峰耦合情况。
    *   **Derivative + LM**: 先用原始谱预热，再联合拟合原始谱与 Savitzky-Golay 平滑的 $dC/dE$ 或 $d^2C/dE^2$，避免噪声主导。
*   **结构参数联合拟合**: 可联合拟合 SiO2 以及已启用的上/下 hBN 厚度。
*   **任意层堆栈表**: 按入射侧到基底侧逐行设置 `Sample`、hBN、Graphene、SiO2、Quartz、Sapphire 或 TiO2。每层可独立设置厚度、参考区域、是否拟合及拟合上下界，并提供常见封装结构预设。
*   **拟合控制**: 界面保留优化预算和分阶段进度；E0 搜索半宽及背景阶数使用代码默认值。
*   **弱峰保护**: 每个初始共振区域按自身去趋势幅度平衡残差，并报告逐峰局部 $R^2$ 和振幅恢复率，避免高全谱 GOF 掩盖弱峰漏拟合。
*   **引导式操作**: 界面按数据、层结构、拟合设置、共振和结果分步组织；无效操作自动禁用，修改模型或共振后旧拟合自动失效。
*   **可分离慢变背景**: 每次非线性迭代中用线性最小二乘消去三阶慢变漂移，避免将光源/探测器基线误归因于介电函数。
*   **Auto-Guess (自动猜峰)**: 使用 Savitzky-Golay 平滑、低阶背景消除和鲁棒噪声阈值生成初值，并合并同一色散共振产生的相邻峰谷。
*   **参数约束与锁定**: 支持设置参数范围 (Bounds) 和锁定特定参数 (Lock) 不参与拟合。
*   **拟合诊断**: 输出参数标准误差、Jacobian 条件数、RMSE、约化卡方和 Durbin-Watson 残差指标；条件数过大时提示参数不可辨识。

### Example 全谱验收
仓库 WS2 示例在 `1.908--3.187 eV` 全谱上自动识别约 `2.10、2.50、3.06 eV` 三个振子。联合拟合 SiO2 厚度并使用三阶慢变基线后，鲁棒拟合 GOF（$R^2$）回归门槛为 `0.99`；当前三套 Si 数据的基准约为 `0.9944`，4--5 振子模型可进一步达到约 `0.996--0.997`。

### 4. 结果导出
*   **全数据导出**: 将实验对比度、拟合对比度、波长/能量对应数据导出为 CSV。
*   **参数导出**: 将提取的物理参数 ($\epsilon_\infty, f, E_0, \Gamma$) 导出为 CSV 表格。

---

## 快速开始 (Quick Start)

### 方式一：Web 版 (推荐)
**无需安装任何 Python 环境**。
1.  **在线访问**: 点击 [https://reflectance.streamlit.app/](https://reflectance.streamlit.app/) 直接使用。
2.  **或者本地运行**: 直接用浏览器（Chrome/Edge）打开项目根目录下的 `index.html` 文件。

### 方式二：桌面版 (PyQt6)
适合开发人员或需要本地高性能计算的用户。
1.  **环境配置**:
    ```bash
    pip install numpy pandas scipy matplotlib PyQt6
    ```
2.  **运行程序**:
    ```bash
    python gui_app.py
    ```

---

## 使用指南 (Usage Guide)

### 1. 数据准备
你需要两组实验数据：
*   **Substrate Spectrum (Ref)**: 空白衬底位置的反射谱（Intensity vs Wavelength/Energy）。
*   **Sample Spectrum**: 长有二维材料位置的反射谱。

### 2. 载入与设置
1.  **Upload Files**: 分别上传衬底和样品光谱文件。程序会自动计算实验对比度：

   $$
    C_{exp} = C_{model} = \frac{R_{sample} - R_{sub}}{R_{sub}}
    $$
    
    *(注: 本工具采用相对于衬底反射率的差分定义。)*
3.  **Structure Config**: 设置实验样品的物理结构（如 SiO2 厚度 285nm，是否覆盖 hBN 等）。
4.  **Material Data**: 默认使用内置的 Si (n,k) 数据。如有特殊需求，可上传自定义的 Si 折射率文件。

### 3. 设置激子 (Excitons)
*   **Auto-Guess**: 点击 "Auto Guess" 按钮，程序会自动在 ROI 范围内寻找峰位。
*   **Manual Add**: 也可以手动点击 "Add Exciton" 添加振子，并调节初始值。
*   **Lock**: 如果你确定某个参数（例如已知 A 激子峰位），可以勾选 "🔒" 将其锁定。

### 4. 拟合 (Fitting)
1.  **ROI Range**: 设置感兴趣的能量范围 (ROI Min/Max)，例如 1.5 eV - 3.0 eV。
2.  **Method**: 优先使用 **Robust LM**；初值不可靠时使用 **Global + Robust LM**，弱峰或重叠峰可尝试导数模式。
3.  **Start Fitting**: 点击开始拟合。等待进度条完成。

### 5. 结果分析与导出
*   拟合完成后，右侧绘图区会显示红色拟合曲线。
*   点击 **Download Fitted Spectrum** 导出光谱数据。
*   点击 **Download Fit Parameters** 导出拟合得到的物理参数。

---
## 文件结构说明
*   `index.html`: Web 版主程序（包含前端和嵌入的 Python 逻辑），单文件部署。
*   `gui_app.py`: 桌面版主程序入口 (PyQt6)。
*   `streamlit_app.py`: Web 版的 Python 源码（开发用，已编译进 index.html）。
*   `materials.py`: 核心材料折射率库和处理逻辑。
*   `Si_data.csv`: 默认的硅折射率数据源。
*   `Schinke.csv`, `Green-2008.csv`: 可选的分段 `wl,n` / `wl,k` Si 光学常数。
*   `example_benchmark.py`: example 全谱模型与 Si 数据源基准工具。

# English Version

## Introduction

This project fits 2D-material optical-contrast spectra with a transfer-matrix model and selectable complex Faddeeva-Voigt or Lorentz oscillators.

The tool offers two modes of operation:
1.  **Web Version (Streamlit)**: Runs directly in a browser via a single HTML file (`index.html`) without complex installation. Includes a modern, user-friendly interface.
2.  **Desktop Version (PyQt6)**: A full-featured desktop application suitable for batch processing and advanced customization.

## Key Features

### 1. Data Processing & Visualization
*   **Multi-Format Support**: Compatible with common formats like `.csv` and `.txt`.
*   **Smart Unit Recognition**: Automatically detects and standardizes wavelength (nm) or energy (eV) units.
*   **Auto-Interpolation**: Automatically interpolates sample spectra onto the substrate's wavelength grid for precise contrast calculation.
*   **Interactive Plotting**: Real-time updates during fitting, with support for zooming, data inspection, and high-quality image export.

### 2. Precise Physical Modeling
*   **Multi-Layer TMM Calculation**: Supports arbitrary stack structures:
    *   Substrate: Si (with temp correction), Quartz, Sapphire, TiO2, etc.
    *   Dielectric: SiO2, hBN (Top/Bottom encapsulation).
    *   2D Material: Monolayer or few-layer samples.
*   **Advanced Optical Defaults**: temperature, finite NA, Si optical source, and background epsilon remain code-level parameters while the UI uses stable defaults.
*   **Controlled Interpolation**: Si uses shape-preserving PCHIP in 400--1305 nm and passive endpoint-tangent linear extrapolation outside that range.
*   **Complex Voigt Dielectric Function (recommended)**: Faddeeva oscillators separate Lorentzian homogeneous width $w_L$ from Gaussian inhomogeneous width $w_G$; the classical Lorentz model remains available.
  
  $$  
    \epsilon(E) = \epsilon_\infty + \sum_j \frac{f_j}{E_{0,j}^2 - E^2 - i E \Gamma_j}
    $$

### 3. Advanced Fitting Algorithms
*   **Metrology-oriented staged solver**:
    *   **Robust LM**: bounded soft-L1 TRF localization followed by true Levenberg-Marquardt refinement through bounded parameter transforms.
    *   **Global + Robust LM**: Differential Evolution initialization followed by robust TRF and LM.
    *   **Derivative + LM**: warm-starts on the original spectrum, then jointly fits the spectrum and smoothed first or second energy derivative.
*   **Joint Structure Fit**: optionally fits SiO2 and enabled top/bottom hBN thicknesses.
*   **Arbitrary Layer Table**: order Sample, hBN, Graphene, SiO2, Quartz, Sapphire, and TiO2 layers from the incident side to the substrate. Each row controls thickness, reference-region inclusion, fit state, and bounds.
*   **Fit Controls**: the UI exposes the optimization budget and staged progress; E0 search width and baseline order use code defaults.
*   **Weak-feature preservation**: resonance neighborhoods are balanced by their own detrended amplitudes and receive per-peak local $R^2$ and amplitude-recovery diagnostics.
*   **Guided workflow**: data, layer stack, fit setup, resonances, and results are organized as explicit steps; unavailable actions are disabled and stale fits are invalidated automatically.
*   **Variable-projection baseline**: cubic slow measurement drift is separated from nonlinear dielectric parameters.
*   **Auto-Guess**: smoothed, detrended, noise-adaptive initialization that merges the peak/dip pair of one dispersive resonance.
*   **Constraints & Locking**: Supports parameter bounds and locking specific parameters (e.g., fixing a known peak position) during fitting.
*   **Fit Diagnostics**: parameter standard errors, Jacobian condition number, RMSE, reduced chi-square, and Durbin-Watson residual statistics.

### Full-spectrum example acceptance
The WS2 example automatically initializes oscillators near `2.10`, `2.50`, and `3.06 eV`. Joint SiO2-thickness fitting with a cubic slow baseline has a full-range GOF ($R^2$) regression threshold of `0.99`; all three bundled Si datasets benchmark near `0.9944`, while 4--5 oscillators reach approximately `0.996--0.997`.

### 4. Export Results
*   **Data Export**: Download experimental vs. fitted contrast data alongside wavelength/energy as CSV.
*   **Parameter Export**: Download extracted physical parameters ($\epsilon_\infty, f, E_0, \Gamma$) as CSV.

---

## Quick Start

### Option 1: Web Version (Recommended)
**No Python environment required.**
1.  **Online**: Visit [https://reflectance.streamlit.app/](https://reflectance.streamlit.app/).
2.  **Local Run**: Simply open the `index.html` file in your browser (Chrome/Edge).

### Option 2: Desktop Version (PyQt6)
For developers or high-performance local computing.
1.  **Setup Environment**:
    ```bash
    pip install numpy pandas scipy matplotlib PyQt6
    ```
2.  **Run Application**:
    ```bash
    python gui_app.py
    ```

---

## Usage Guide

### 1. Data Preparation
You need two sets of experimental data:
*   **Substrate Spectrum (Ref)**: Reflection spectrum from a bare substrate region.
*   **Sample Spectrum**: Reflection spectrum from the region with the 2D material.

### 2. Load & Setup
1.  **Upload Files**: Upload both spectra. The program calculates contrast:

$$
    C_{exp} = C_{model} = \frac{R_{sample} - R_{sub}}{R_{sub}}
    $$
    
3.  **Structure Config**: Define the physical structure (e.g., SiO2 thickness, hBN layers).
4.  **Material Data**: Uses built-in Si (n,k) data by default. Custom data can be uploaded if needed.

### 3. Exciton Setup
*   **Auto-Guess**: Click to automatically find peaks within the ROI.
*   **Manual Add**: Manually add oscillators and adjust initial guesses.
*   **Lock**: Use the "🔒" checkbox to fix known parameters.

### 4. Fitting
1.  **ROI Range**: Set the energy range of interest (e.g., 1.5 eV - 3.0 eV).
2.  **Method**: Start with **Robust LM**; use **Global + Robust LM** for uncertain initialization, or derivative modes for weak/overlapping peaks.
3.  **Start Fitting**: Run the fit and wait for completion.

### 5. Analysis
*   View the red fitted curve overlaid on experimental data.
*   Use the **Download** buttons to save spectra and parameters.

---

## File Structure
*   `index.html`: Web version (contains frontend + embedded Python), single-file deployment.
*   `gui_app.py`: Desktop application entry point (PyQt6).
*   `streamlit_app.py`: Source code for the Web version (compiled into index.html).
*   `optical_model.py`: TMM, Faddeeva-Voigt/Lorentz dielectric functions, finite-NA averaging, and contrast definition.
*   `fitting_engine.py`: Robust TRF, bounded LM, baseline projection, uncertainty, and diagnostics.
*   `sync_index.py`: Synchronizes Python sources into the standalone stlite HTML file.
*   `materials.py`: Core material library and physics logic.
*   `Si_data.csv`: Default Silicon refractive index data.
*   `Schinke.csv`, `Green-2008.csv`: Optional segmented `wl,n` / `wl,k` Si optical constants.
*   `example_benchmark.py`: Full-spectrum example and Si-source benchmark utility.
