import numpy as np
import math
from scipy.optimize import fsolve

class Engine:
    def __init__(self, p: float, pwr: float):
        """
        涡轮轴发动机模型
        Args:
            p:发动机功率,通常指最大起飞功率[kW]
            pwr:功率重量比[kW/kg]
        """
        self.p = p
        self.pwr = pwr
        self.mass_kg = self.p / self.pwr # 发动机重量
        self.constant = {
            'R_air': 287.0, # 空气气体常数[J/(kg·K)]
            'gamma_air': 1.4, # 比热比
            'cp_air': 1005.0, # 空气定压比热[J/(kg·K)]
            'cp_gas': 1150.0, # 燃气定压比热[J/(kg·K)]
            'gamma_gas': 1.33, # 燃气比热比
        } # 常量
        self.engine_config = {
            'pi_c': 12.0, # 压气机压比
            'eta_c': 0.85, # 压气机效率
            'eta_tg': 0.88, # 燃气发生器涡轮效率
            'eta_pt': 0.90, # 动力涡轮效率
            'TIT': 1200.0, # 涡轮前温度[K]
        } # 发动机参数
        self.fuel_config = {
            'LHV': 43e6, # 航空煤油低热值[J/kg]
            'eta_comb': 0.99 # 燃烧效率
        } # 燃料相关参数


    def get_mass(self):
        return self.mass_kg


    def turboshaft_sfc_full(self, P_shaft_kW: float, altitude_m: float, v_tas: float, tol: float=1e-6) -> float:
        """
        计算涡轮轴发动机在给定轴功率下的单位燃油消耗率
        Args:
            P_shaft_kW:所需轴功率[kW]
            altitude_m:飞行高度
            v_tas:飞行速度[m/s]
            tol:迭代精度
        Returns:
            单位燃油消耗率[kg/(kW·h)]
        """
        # 内部函数定义
        def isa_atmosphere(altitude_m):
            """返回标准大气下的静温(K)和静压(Pa)"""
            if altitude_m < 0:
                raise ValueError("高度不能为负")
            T0_sea = 288.15      # K
            P0_sea = 101325.0    # Pa
            L = 0.0065           # K/m (对流层温度递减率)
            R = 287.0            # J/(kg·K)
            g = 9.80665          # m/s²

            if altitude_m <= 11000:
                # 对流层
                T = T0_sea - L * altitude_m
                P = P0_sea * (T / T0_sea) ** (g / (L * R))
            elif altitude_m <= 20000:
                # 平流层下层（11–20 km）：等温
                T = 216.65
                P11 = 22632.1  # 11 km 处的精确压力
                h = altitude_m
                P = P11 * math.exp(-g * (h - 11000) / (R * T))
            else:
                raise NotImplementedError("高度超过 20 km 暂未实现")
            
            return T, P

        def power_residual(mdot_air_in):
            """
            功率平衡残差函数
            """
            mdot_air = float(mdot_air_in[0])
            if mdot_air <= 0:
                return 1e6

            # --- 1. 压气机 ---
            T2s = T0 * (pi_c ** ((gamma_air - 1) / gamma_air))
            T2 = T0 + (T2s - T0) / eta_c
            P2 = P0 * pi_c
            W_comp = mdot_air * cp_air * (T2 - T0) # 压气机耗功

            # --- 2. 燃烧室 (计算一致的燃油流量) ---
            numerator_f = cp_gas * TIT - cp_air * T2
            denominator_f = LHV * eta_comb - cp_gas * TIT
            if denominator_f <= 0 or numerator_f <= 0:
                return 1e6 # 物理上不可能的情况
            f = numerator_f / denominator_f
            mdot_fuel = f * mdot_air
            mdot_gas = mdot_air + mdot_fuel

            # 3燃气发生器涡轮
            delta_T_tg = W_comp / (mdot_gas * cp_gas)
            T4 = TIT - delta_T_tg / eta_tg # 燃气涡轮出口实际温度
            
            if T4 <= 0:
                return 1e6

            # 4压力计算
            P3 = P2 # 假设燃烧室无压力损失
            T4s = TIT - delta_T_tg # 燃气涡轮出口等熵温度
            P4 = P3 * (T4s / TIT) ** (gamma_gas / (gamma_gas - 1))
            
            if P4 <= P_static: # 如果压力已经低于环境压力，无法做功
                return 1e6

            # 5动力涡轮
            T5s = T4 * (P_static / P4) ** ((gamma_gas - 1) / gamma_gas)
            W_pt = mdot_gas * cp_gas * (T4 - T5s) * eta_pt

            # 6残差
            return W_pt - P_shaft

        # 常数定义
        R_air = self.constant['R_air']
        gamma_air = self.constant['gamma_air']
        cp_air = self.constant['cp_air']
        cp_gas = self.constant['cp_gas']
        gamma_gas = self.constant['gamma_gas']
        pi_c = self.engine_config['pi_c']
        eta_c = self.engine_config['eta_c']
        eta_tg = self.engine_config['eta_tg']
        eta_pt = self.engine_config['eta_pt']
        TIT = self.engine_config['TIT']
        LHV = self.fuel_config['LHV']
        eta_comb = self.fuel_config['eta_comb']

        # 参数计算
        P_shaft = P_shaft_kW * 1000.0  # 转换为 [W]
        T_static, P_static = isa_atmosphere(altitude_m)
        a = math.sqrt(gamma_air * R_air * T_static)
        mach_number = v_tas / a
        T0 = T_static * (1 + (gamma_air - 1) / 2 * mach_number**2)
        P0 = P_static * (1 + (gamma_air - 1) / 2 * mach_number**2)**(gamma_air / (gamma_air - 1))
        
        # 初始猜测
        mdot_guess = P_shaft_kW / 250.0

        # 迭代求解mdot_air
        try:
            mdot_air_solution, info, ier, mesg = fsolve(
                power_residual, [mdot_guess], full_output=True, xtol=tol
            )
            if ier != 1:
                raise RuntimeError(f"迭代未收敛: {mesg}")
            mdot_air = float(mdot_air_solution[0])
        except Exception as e:
            raise RuntimeError(f"求解失败: {e}")

        # 用收敛的mdot_air重新计算所有参数以获得最终的燃油流量
        T2s = T0 * (pi_c ** ((gamma_air - 1) / gamma_air))
        T2 = T0 + (T2s - T0) / eta_c
        
        # 计算最终的 mdot_fuel
        numerator_f = cp_gas * TIT - cp_air * T2
        denominator_f = LHV * eta_comb - cp_gas * TIT
        f = numerator_f / denominator_f
        mdot_fuel = f * mdot_air
        
        if mdot_fuel <= 0:
            raise ValueError(f"燃油流量非正: {mdot_fuel}, 检查热平衡")

        # 计算 SFC
        sfc = (mdot_fuel * 3600.0) / P_shaft_kW
        print(f"mdot_air={mdot_air:.2f} kg/s, mdot_fuel={mdot_fuel:.4f} kg/s")

        return sfc  


class Generator:
    def __init__(self, p: float, pwr: float = 8.0, eta_peak: float = 0.98, peak_eta_load_ratio: float = 0.8):
        """
        发电机模型
        Args:
            p:额定功率[kW]
            pwr:功率重量比[kW/kg]
            eta_peak:峰值效率
            peak_eta_load_ratio:达到峰值效率时的负载率
        """
        self.p = p
        self.pwr = pwr
        self.mass_kg = self.p / self.pwr # 电动机重量

        # 确定损耗系数
        self._eta_peak = eta_peak
        self._peak_load_ratio = peak_eta_load_ratio
        p_peak = self.p * self._peak_load_ratio
        p_loss_peak = p_peak * (1 - self._eta_peak) / self._eta_peak
        self._p_loss_fixed = p_loss_peak / 2.0
        self._c1 = (p_loss_peak / 2.0) / (p_peak**2) if p_peak > 0 else 0


    def _get_efficiency(self, p_electric_out_kw: float) -> float:
        """
        根据输出功率计算当前工况的效率
        Args:
            p_electric_out_kw:输出功率[kW]
        Returns:
            当前工况效率
        """
        if p_electric_out_kw <= 0:
            return 0.0
        
        load_ratio = p_electric_out_kw / self.p
        if load_ratio > 1.2: # 假设最多超载20%
             p_electric_out_kw = self.p * 1.2

        # 计算损耗
        p_loss_variable = self._c1 * p_electric_out_kw**2
        p_loss_total = self._p_loss_fixed + p_loss_variable
        
        # 计算效率
        p_mech_in_kw = p_electric_out_kw + p_loss_total
        efficiency = p_electric_out_kw / p_mech_in_kw if p_mech_in_kw > 0 else 0.0
        
        return efficiency


    def calculate_performance(self, p_electric_out_kw: float):
        """
        计算所需发动机功率和热功率
        Args:
            p_electric_out_kw:输出功率[kW]
        Returns:
            所需发动机功率, 热功率
        """
        if p_electric_out_kw > self.p_rated_kw * 1.2:
             print(f"警告: 请求功率 {p_electric_out_kw} kW 超过最大允许功率")
             p_electric_out_kw = self.p_rated_kw * 1.2

        current_eta = self.get_efficiency(p_electric_out_kw)
        
        if current_eta == 0:
            p_mech_in_kw = 0
            q_rejected_kw = self._p_loss_fixed # 即使无输出，只要转动就有固定损耗
        else:
            p_mech_in_kw = p_electric_out_kw / current_eta
            q_rejected_kw = p_mech_in_kw - p_electric_out_kw
        
        return p_mech_in_kw, q_rejected_kw


    def get_mass(self):
        return self.mass_kg


class Battery:
    def __init__(self, 
                 e_rated_kwh: float, 
                 spec_energy_wh_kg: float = 250.0, 
                 spec_power_w_kg: float = 1500.0,
                 pack_mass_fraction: float = 0.75,
                 dod_max: float = 0.8,
                 soc_initial: float = 0.95,
                 eta_discharge: float = 0.97):
        """
        锂电池模型
        Args:
            e_rated_kwh:电池包额定能量容量[kWh]
            spec_energy_wh_kg:电芯比能量[Wh/kg]
            spec_power_w_kg:电芯比功率[W/kg]
            pack_mass_fraction:电芯质量在电池包总重中的占比
            dod_max:最大允许放电深度
            soc_initial:初始荷电状态
            eta_discharge:恒定放电效率
        """
        if not (0 < pack_mass_fraction <= 1.0):
            raise ValueError("打包系数必须在 (0, 1] 之间")

        self.e_rated_kwh = e_rated_kwh
        self.spec_energy_wh_kg = spec_energy_wh_kg
        self.spec_power_w_kg = spec_power_w_kg
        self.pack_mass_fraction = pack_mass_fraction
        
        # 重量计算
        mass_cell_kg = self.e_rated_kwh * 1000 / self.spec_energy_wh_kg # 电芯重量
        self.mass_pack_kg = mass_cell_kg / self.pack_mass_fraction

        # 功率限制计算
        self.p_max_cont_kw = (mass_cell_kg * self.spec_power_w_kg) / 1000.0 # 这里的比功率是基于电芯重量还是电池包总重，需要根据数据来源确定，这里假设基于电芯重量的

        # 状态变量
        self.dod_max = dod_max
        self.soc_min = 1.0 - self.dod_max
        self.soc = soc_initial  # 当前SoC
        self.eta_discharge = eta_discharge


    def update_state(self, p_draw_req_kw: float, dt_seconds: float) -> float:
        """
        根据功率需求和时间步长，更新电池状态并计算热功率
        Args:
            p_draw_req_kw:需求电功率[kW]
            dt_seconds:时间步长[s]
        Returns:
            实际电功率[kW], 电池热功率[kW]
        """
        # 输入检查
        if p_draw_req_kw < 0:
            raise ValueError("请求功率不能为负")

        # 功率限制检查
        if p_draw_req_kw > self.p_max_cont_kw:
            print(f"警告: 请求功率 {p_draw_req_kw:.1f} kW 超过电池最大持续功率 {self.p_max_cont_kw:.1f} kW")
            p_out_kw = self.p_max_cont_kw
        else:
            p_out_kw = p_draw_req_kw

        # 能量限制检查
        e_available_kwh = (self.soc - self.soc_min) * self.e_rated_kwh
        if e_available_kwh <= 0:
            print("警告: 电池电量耗尽！")
            return 0, 0

        dt_hr = dt_seconds / 3600.0
        
        # 计算实际提供的能量和功率
        energy_drawn_from_storage_kwh = (p_out_kw / self.eta_discharge) * dt_hr
        
        if energy_drawn_from_storage_kwh > e_available_kwh:
            # 如果请求的能量超过剩余可用能量，则只能提供所有剩余能量
            print("警告: 电池电量不足！")
            energy_drawn_from_storage_kwh = e_available_kwh
            p_out_kw = (energy_drawn_from_storage_kwh / dt_hr) * self.eta_discharge if dt_hr > 0 else 0
            
        # 更新SoC
        delta_soc = energy_drawn_from_storage_kwh / self.e_rated_kwh
        self.soc -= delta_soc
        
        # 计算热损失
        q_rejected_kw = p_out_kw * (1.0 - self.eta_discharge) / self.eta_discharge

        return p_out_kw, q_rejected_kw


    def get_mass(self):
        return self.mass_pack_kg



