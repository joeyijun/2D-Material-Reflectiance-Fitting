# 2D Material Optical Contrast Fitting Tool (2D材料光学对比度拟合工具)

## 项目简介 (Introduction)

本项目是一个用于分析和拟合二维材料（如 $\text{MoS}_2$, $\text{WS}_2$, $\text{WSe}_2$ 等）光学反射对比度（Optical Contrast）谱的专业工具。它结合了**传输矩阵法 (Transfer Matrix Method, TMM)** 和 **洛伦兹振子模型 (Lorentz Oscillator Model)**，能够从实验测量的反射光谱中精确提取材料的关键光学参数（如激子峰位、振子强度、展宽等）。

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
*   **温度相关性**: 内置 Si 折射率的温度修正模型 (10K - 300K)，适用于低温光谱拟合。
*   **Lorentz 介电函数**: 使用经典洛伦兹振子模型描述激子吸收：
    
  $$
  \epsilon(E) = \epsilon_\infty + \sum_j \frac{f_j}{E_{0,j}^2 - E^2 - i E \Gamma_j}
    $$

### 3. 先进的拟合算法
*   **多种拟合策略**:
    *   **Standard**: 传统的最小二乘法拟合，速度快。
    *   **High Precision (Global)**: 使用差分进化算法 (Differential Evolution) 进行全局寻优，避免陷入局部极小值。
    *   **Multi-Stage**: 多阶段拟合策略，按粗略 $\rightarrow$ 中等 $\rightarrow$ 高精度的顺序逐步优化，提高收敛稳定性。
    *   **MCMC (Basin Hopping)**: 基于蒙特卡洛马尔可夫链的全局优化算法，有效克服多维参数空间的局部极值问题，适合复杂谱线。
    *   **Derivative (导数拟合)**: 拟合光谱的一阶导数 ($dC/dE$)，极大地提高了对微弱激子峰和峰位的敏感度，消除背景干扰。
    *   **2nd Derivative**: 二阶导数拟合，进一步增强对精细结构的解析能力。
*   **Auto-Guess (自动猜峰)**: 基于峰值检测算法，自动从实验数据中识别潜在的激子峰位，减少手动设置参数的繁琐。
*   **参数约束与锁定**: 支持设置参数范围 (Bounds) 和锁定特定参数 (Lock) 不参与拟合。

### 4. 结果导出
*   **全数据导出**: 将实验对比度、拟合对比度、波长/能量对应数据导出为 CSV。
*   **参数导出**: 将提取的物理参数 ($\epsilon_\infty, f, E_0, \Gamma$) 导出为 CSV 表格。

---

## 快速开始 (Quick Start)

### 方式一：Web 版 (推荐)
**无需安装任何 Python 环境**。
1.  **在线访问**: 点击 [https://reflectiance.streamlit.app/](https://reflectiance.streamlit.app/) 直接使用。
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
    C_{exp} = \frac{R_{sample} - R_{sub}}{R_{sample} + R_{sub}}
    $$
    
    *(注: 本工具采用归一化差异定义，数值范围通常在 -1 到 1 之间*
3.  **Structure Config**: 设置实验样品的物理结构（如 SiO2 厚度 285nm，是否覆盖 hBN 等）。
4.  **Material Data**: 默认使用内置的 Si (n,k) 数据。如有特殊需求，可上传自定义的 Si 折射率文件。

### 3. 设置激子 (Excitons)
*   **Auto-Guess**: 点击 "Auto Guess" 按钮，程序会自动在 ROI 范围内寻找峰位。
*   **Manual Add**: 也可以手动点击 "Add Exciton" 添加振子，并调节初始值。
*   **Lock**: 如果你确定某个参数（例如已知 A 激子峰位），可以勾选 "🔒" 将其锁定。

### 4. 拟合 (Fitting)
1.  **ROI Range**: 设置感兴趣的能量范围 (ROI Min/Max)，例如 1.5 eV - 3.0 eV。
2.  **Method**: 选择拟合方法。推荐优先尝试 **Standard**，如果效果不佳或需要更高精度，尝试 **High Precision** 或 **Derivative**。
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

# English Version

## Introduction

This project is a professional tool designed for analyzing and fitting the **Optical Contrast** spectra of 2D materials (e.g., $\text{MoS}_2$, $\text{WS}_2$, $\text{WSe}_2$). By combining the **Transfer Matrix Method (TMM)** with the **Lorentz Oscillator Model**, it accurately extracts key optical properties (such as exciton peak positions, oscillator strengths, and broadening) from experimental reflection spectra.

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
*   **Temperature Dependence**: Built-in temperature-dependent refractive index model for Silicon (10K - 300K), ideal for low-temperature spectroscopy.
*   **Lorentz Dielectric Function**: Describes exciton absorption using the classical Lorentz oscillator model:
  
  $$  
    \epsilon(E) = \epsilon_\infty + \sum_j \frac{f_j}{E_{0,j}^2 - E^2 - i E \Gamma_j}
    $$

### 3. Advanced Fitting Algorithms
*   **Multiple Strategies**:
    *   **Standard**: Traditional Least Squares fitting for speed.
    *   **High Precision (Global)**: Uses Differential Evolution for global optimization to avoid local minima.
    *   **Multi-Stage**: Sequential optimization strategy (Coarse $\rightarrow$ Medium $\rightarrow$ Fine) to improve convergence stability.
    *   **MCMC (Basin Hopping)**: Monte Carlo Markov Chain based global optimization, effective for complex multidimensional parameter spaces.
    *   **Derivative**: Fits the first derivative of the spectrum ($dC/dE$), significantly enhancing sensitivity to weak exciton peaks and eliminating background offsets.
    *   **2nd Derivative**: Second-order derivative fitting for resolving fine spectral structures.
*   **Auto-Guess**: Automatically identifies potential exciton peaks from experimental data using peak detection algorithms.
*   **Constraints & Locking**: Supports parameter bounds and locking specific parameters (e.g., fixing a known peak position) during fitting.

### 4. Export Results
*   **Data Export**: Download experimental vs. fitted contrast data alongside wavelength/energy as CSV.
*   **Parameter Export**: Download extracted physical parameters ($\epsilon_\infty, f, E_0, \Gamma$) as CSV.

---

## Quick Start

### Option 1: Web Version (Recommended)
**No Python environment required.**
1.  **Online**: Visit [https://reflectiance.streamlit.app/](https://reflectiance.streamlit.app/).
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
    C_{exp} = \frac{R_{sample} - R_{sub}}{R_{sample} + R_{sub}}
    $$
    
3.  **Structure Config**: Define the physical structure (e.g., SiO2 thickness, hBN layers).
4.  **Material Data**: Uses built-in Si (n,k) data by default. Custom data can be uploaded if needed.

### 3. Exciton Setup
*   **Auto-Guess**: Click to automatically find peaks within the ROI.
*   **Manual Add**: Manually add oscillators and adjust initial guesses.
*   **Lock**: Use the "🔒" checkbox to fix known parameters.

### 4. Fitting
1.  **ROI Range**: Set the energy range of interest (e.g., 1.5 eV - 3.0 eV).
2.  **Method**: Choose a fitting strategy. Start with **Standard**, then try **High Precision** or **Derivatives** if needed.
3.  **Start Fitting**: Run the fit and wait for completion.

### 5. Analysis
*   View the red fitted curve overlaid on experimental data.
*   Use the **Download** buttons to save spectra and parameters.

---

## File Structure
*   `index.html`: Web version (contains frontend + embedded Python), single-file deployment.
*   `gui_app.py`: Desktop application entry point (PyQt6).
*   `streamlit_app.py`: Source code for the Web version (compiled into index.html).
*   `materials.py`: Core material library and physics logic.
*   `Si_data.csv`: Default Silicon refractive index data.
