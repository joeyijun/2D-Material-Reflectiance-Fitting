import numpy as np
import csv
import io
from pathlib import Path
from scipy.interpolate import PchipInterpolator


def read_si_optical_constants(source):
    """Read numeric wavelength,n,k data or separate wl,n and wl,k sections."""
    if hasattr(source, "read"):
        if hasattr(source, "seek"):
            source.seek(0)
        content = source.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8-sig")
    else:
        content = Path(source).read_text(encoding="utf-8-sig")

    legacy = []
    sections = {"n": [], "k": []}
    current_section = None
    for row in csv.reader(io.StringIO(content)):
        row = [cell.strip() for cell in row]
        if len(row) < 2 or not row[0]:
            continue
        first = row[0].lower()
        second = row[1].lower()
        if first in {"wl", "wavelength", "lambda", "wavelength_um", "wavelength_nm"}:
            third_header = row[2].lower() if len(row) >= 3 else ""
            current_section = "legacy" if second == "n" and third_header == "k" else (
                second if second in sections else None
            )
            continue
        try:
            wavelength = float(row[0])
            value = float(row[1])
            third = float(row[2]) if len(row) >= 3 and row[2] else None
        except ValueError:
            continue
        if third is not None and current_section in {None, "legacy"}:
            legacy.append((wavelength, value, third))
        elif current_section is not None:
            sections[current_section].append((wavelength, value))

    if legacy:
        wavelength, n_values, k_values = np.asarray(legacy, dtype=float).T
    elif sections["n"] and sections["k"]:
        n_data = np.asarray(sections["n"], dtype=float)
        k_data = np.asarray(sections["k"], dtype=float)
        n_data = n_data[np.argsort(n_data[:, 0])]
        k_data = k_data[np.argsort(k_data[:, 0])]
        wavelength = n_data[:, 0]
        n_values = n_data[:, 1]
        k_values = np.interp(wavelength, k_data[:, 0], k_data[:, 1])
    else:
        raise ValueError("Si data require wavelength,n,k columns or separate wl,n / wl,k sections")

    if np.nanmedian(wavelength) < 10.0:
        wavelength = wavelength * 1000.0
    return wavelength, n_values, k_values

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
        lam_nm, n_vals, k_vals = read_si_optical_constants(csv_path)
        
        # 创建插值函数 (处理 TMM 计算时波长点和 CSV 不对齐的问题)
        # fill_value="extrapolate" 防止波长稍微超出一点点导致报错
        order = np.argsort(lam_nm)
        lam_nm = lam_nm[order]
        self.si_wavelength_range = (float(lam_nm[0]), float(lam_nm[-1]))
        self.si_n_interp = PchipInterpolator(lam_nm, n_vals[order], extrapolate=False)
        self.si_k_interp = PchipInterpolator(lam_nm, k_vals[order], extrapolate=False)
        self.si_n_edge_slopes = tuple(
            float(self.si_n_interp.derivative()(edge)) for edge in self.si_wavelength_range
        )
        self.si_k_edge_slopes = tuple(
            float(self.si_k_interp.derivative()(edge)) for edge in self.si_wavelength_range
        )

    def _evaluate_si_component(self, wavelengths_nm, interpolation, edge_slopes):
        lower, upper = self.si_wavelength_range
        clipped = np.clip(wavelengths_nm, lower, upper)
        values = interpolation(clipped)
        lower_value = float(interpolation(lower))
        upper_value = float(interpolation(upper))
        values = np.where(
            wavelengths_nm < lower,
            lower_value + edge_slopes[0] * (wavelengths_nm - lower),
            values,
        )
        return np.where(
            wavelengths_nm > upper,
            upper_value + edge_slopes[1] * (wavelengths_nm - upper),
            values,
        )

    def get_si_n(self, lam_nm):
        """
        获取 Si 的复折射率 (298 K)
        输入: 波长 (nm), 支持标量或 numpy 数组
        """
        lam_nm = np.asarray(lam_nm, dtype=float)
        if np.any(~np.isfinite(lam_nm)) or np.any(lam_nm <= 0):
            raise ValueError("Si wavelengths must be finite and positive")
        n = self._evaluate_si_component(lam_nm, self.si_n_interp, self.si_n_edge_slopes)
        k = self._evaluate_si_component(lam_nm, self.si_k_interp, self.si_k_edge_slopes)
        n = np.maximum(n, np.finfo(float).eps)
        return n + 1j * np.maximum(k, 0.0)

    def get_si_n_with_temp(self, lam_nm, temp_k):
        """
        获取给定温度下的 Si 折射率
        基于 25 C (298 K) 和 10 K 的差值模型进行线性插值
        Delta n formula provided by user.
        """
        if not 10.0 <= temp_k <= 300.0:
            raise ValueError("Si temperature correction is calibrated only for 10-300 K")
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
        n_imag = np.maximum(np.imag(n_base) - correction_k, 0.0)
        
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
