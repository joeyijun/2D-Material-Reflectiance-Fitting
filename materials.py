import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

class MaterialLoader:
    def __init__(self, si_csv_path):
        self._init_si_interpolation(si_csv_path)

    def _init_si_interpolation(self, csv_path):
        """
        读取 Si 的 CSV 文件并创建插值函数
        假设 CSV 列顺序: Wavelength(um), n, k
        """
        # 读取数据 (根据你的csv是否有表头header，可能需要调整 header=None 或 header=0)
        # 假设你的csv第一行是数据，没有标题，则 header=None
        df = pd.read_csv(csv_path, header=None, names=['lam_um', 'n', 'k'])
        
        # 将微米转换为纳米，统一单位
        lam_nm = df['lam_um'].values * 1000.0
        n_vals = df['n'].values
        k_vals = df['k'].values
        
        # 创建插值函数 (处理 TMM 计算时波长点和 CSV 不对齐的问题)
        # fill_value="extrapolate" 防止波长稍微超出一点点导致报错
        self.si_n_interp = interp1d(lam_nm, n_vals, kind='cubic', fill_value="extrapolate")
        self.si_k_interp = interp1d(lam_nm, k_vals, kind='cubic', fill_value="extrapolate")

    def get_si_n(self, lam_nm):
        """
        获取 Si 的复折射率 (298 K)
        输入: 波长 (nm), 支持标量或 numpy 数组
        """
        n = self.si_n_interp(lam_nm)
        k = self.si_k_interp(lam_nm)
        return n + 1j * k

    def get_si_n_with_temp(self, lam_nm, temp_k):
        """
        获取给定温度下的 Si 折射率
        基于 25 C (298 K) 和 10 K 的差值模型进行线性插值
        Delta n formula provided by user.
        """
        # 1. Start with base (298 K)
        n_base = self.get_si_n(lam_nm)
        
        # 2. Calculate 10K difference (n)
        # Delta n (10K vs 298K) = 0.02514 + 0.00850 / (lam^2 - 0.10165)
        # Assuming lam in microns
        lam_um = lam_nm / 1000.0
        delta_n_abs = 0.02514 + 0.00850 / (lam_um**2 - 0.10165)
        
        # 3. Calculate 10K difference (k)
        # Delta k (10K vs 298K) = A * exp(-B * lam) + C
        # A = 270.16, B = 18.91, C = 0.0029
        delta_k_abs = 270.16 * np.exp(-18.91 * lam_um) + 0.0029
        
        # 4. Interpolate
        # If T = 298, correction = 0
        # If T = 10, correction = -delta_n_abs (Assuming params decrease with T)
        ratio = (298.0 - temp_k) / (298.0 - 10.0)
        
        correction_n = ratio * delta_n_abs
        correction_k = ratio * delta_k_abs
        
        n_real = np.real(n_base) - correction_n
        n_imag = np.imag(n_base) - correction_k
        
        # Ensure k is not negative (physically impossible for passive material)
        if np.isscalar(n_imag):
            if n_imag < 0: n_imag = 0
        else:
            n_imag[n_imag < 0] = 0
        
        return n_real + 1j * n_imag

    def get_sio2_n(self, lam_nm):
        """
        利用 Sellmeier 方程计算 SiO2 折射率
        输入: 波长 (nm)
        """
        # 1. 单位转换: nm -> um (因为 Sellmeier 公式里的 x 通常是微米)
        x = lam_nm / 1000.0
        
        # 2. 你的公式
        # 注意：公式里的项是 (C/x)**2，确保分母不为0
        term1 = 0.9310 / (1 - (0.079 / x)**2)
        term2 = 0.1735 / (1 - (0.130 / x)**2)
        term3 = 2.1121 / (1 - (14.918 / x)**2)
        
        n_squared = 1 + term1 + term2 + term3
        n = np.sqrt(n_squared)
        
        # 3. 虚部 k 设为 0
        return n + 0j

    def get_hbn_n(self, lam_nm):
        """
        计算 hBN 折射率 (Ordinary ray, in-plane)
        Source: S.-Y. Lee et al., Phys. Status Solidi B 256, 1800417 (2018).
        适用范围: 可见光到近红外
        """
        lam_um = lam_nm / 1000.0  # 必须转换为微米
        
        # Lee et al. (2018) 实验拟合参数:
        # A = 3.263 (也有文献给 3.336)
        # B = 0.1644^2 (也有文献给 0.162^2)
        
        # 避免分母为0或负数 (虽然在可见光段不会发生)
        denom = lam_um**2 - 0.1644**2
        
        # Sellmeier Equation: n^2 = 1 + A * lam^2 / (lam^2 - B^2)
        n_squared = 1 + (3.263 * lam_um**2) / denom
        
        return np.sqrt(n_squared) + 0j

    def get_quartz_n(self, lam_nm):
        """
        Quartz (Fused Silica) 折射率
        直接复用 SiO2 Sellmeier 方程
        """
        return self.get_sio2_n(lam_nm)

    def get_sapphire_n(self, lam_nm):
        """
        Sapphire (Al2O3) Refractive Index (Ordinary Ray)
        Formula:
        n^2 = 1 + 1.4313493/(1-(0.0726631/x)^2) + 0.65054713/(1-(0.1193242/x)^2) + 5.3414021/(1-(18.028251/x)^2)
        (x in microns)
        """
        x = lam_nm / 1000.0 # Convert to microns
        
        # Sellmeier Equation Terms
        # Using the provided coefficients
        # A1=1.4313493, B1=0.0726631
        # A2=0.65054713, B2=0.1193242
        # A3=5.3414021, B3=18.028251
        
        term1 = 1.4313493 / (1 - (0.0726631 / x)**2)
        term2 = 0.65054713/ (1 - (0.1193242 / x)**2)
        term3 = 5.3414021 / (1 - (18.028251 / x)**2)
        
        n_squared = 1 + term1 + term2 + term3
        n = np.sqrt(n_squared)
        
        return n + 0j

    def get_tio2_n(self, lam_nm):
        """
        TiO2 Refractive Index
        Formula: n = sqrt(5.913 + 0.2441 / (x^2 - 0.0803))
        (x in microns)
        """
        x = lam_nm / 1000.0 # Convert to microns
        
        # Formula provided by user
        n_squared = 5.913 + 0.2441 / (x**2 - 0.0803)
        n = np.sqrt(n_squared)
        
        return n + 0j

# === 使用示例 ===
if __name__ == "__main__":
    # 假设你的 csv 文件名为 'Si_data.csv'
    # 请确保你的 csv 路径正确
    try:
        loader = MaterialLoader('Si_data.csv')
        
        # 测试 532 nm (绿光) 下的折射率
        test_wav = 532.0
        n_si = loader.get_si_n(test_wav)
        n_sio2 = loader.get_sio2_n(test_wav)
        
        print(f"波长: {test_wav} nm")
        print(f"Si   折射率: {n_si:.6f}")  # 应该会有较大的虚部
        print(f"SiO2 折射率: {n_sio2:.6f}") # 虚部应该是 0j
        
    except Exception as e:
        print(f"加载失败，请检查 CSV 路径。错误: {e}")