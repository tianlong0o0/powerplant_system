import numpy as np
import math
import scipy.optimize


def isa_atmosphere(altitude_m: float) -> float:
    """
    返回标准大气下的气体参数
    Args:
        altitude_m: 海拔高度[m]
    Returns:
        标准大气下的静温[K], 标准大气下的静压[Pa], 空气密度[kg/m^3]
    """
    # 物理常数
    T0 = 288.15  # 海平面标准温度 [K]
    P0 = 101325.0 # 海平面标准压力 [Pa]
    R = 287.058   # 理想气体常数 [J/(kg*K)]
    g0 = 9.80665  # 标准重力加速度 [m/s^2]
    L = -0.0065   # 对流层温度梯度 [K/m]
    h_tropo = 11000.0 # 对流层顶高度 [m]

    altitude_m = max(0, altitude_m) # 确保高度不为负

    if altitude_m <= h_tropo:
        # 对流层
        temperature = T0 + L * altitude_m
        pressure = P0 * (temperature / T0)**(-g0 / (L * R))
    elif altitude_m <= 20000:
        # 平流层下部
        T_tropo = T0 + L * h_tropo
        P_tropo = P0 * (T_tropo / T0)**(-g0 / (L * R))
        temperature = T_tropo
        h_in_strato = altitude_m - h_tropo
        pressure = P_tropo * np.exp(-g0 * h_in_strato / (R * temperature))
    else:
        raise NotImplementedError("高度超过 20 km 暂未实现")
        
    density = pressure / (R * temperature)

    return temperature, pressure, density


class Engine:
    def __init__(self, p: float, pwr: float):
        """
        涡轮轴发动机模型
        Args:
            p: 发动机功率,通常指最大起飞功率[kW]
            pwr: 功率重量比[kW/kg]
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
            P_shaft_kW: 所需轴功率[kW]
            altitude_m: 飞行高度
            v_tas: 飞行速度[m/s]
            tol: 迭代精度
        Returns:
            单位燃油消耗率[kg/(kW·h)]
        """
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
        T_static, P_static, _ = isa_atmosphere(altitude_m)
        a = math.sqrt(gamma_air * R_air * T_static)
        mach_number = v_tas / a
        T0 = T_static * (1 + (gamma_air - 1) / 2 * mach_number**2)
        P0 = P_static * (1 + (gamma_air - 1) / 2 * mach_number**2)**(gamma_air / (gamma_air - 1))
        
        # 初始猜测
        mdot_guess = P_shaft_kW / 250.0

        # 迭代求解mdot_air
        try:
            mdot_air_solution, info, ier, mesg = scipy.optimize.fsolve(
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
            p: 额定功率[kW]
            pwr: 功率重量比[kW/kg]
            eta_peak: 峰值效率
            peak_eta_load_ratio: 达到峰值效率时的负载率
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
            p_electric_out_kw: 输出功率[kW]
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


    def calculate_performance_inverse(self, p_electric_out_kw: float):
        """
        计算所需发动机输入功率和热功率
        Args:
            p_electric_out_kw: 输出功率[kW]
        Returns:
            所需发动机输入功率[kW], 热功率[kW]
        """
        if p_electric_out_kw > self.p * 1.2:
             print(f"警告: 请求功率 {p_electric_out_kw} kW 超过最大允许功率")
             p_electric_out_kw = self.p * 1.2

        current_eta = self.get_efficiency(p_electric_out_kw)
        
        if current_eta == 0:
            p_mech_in_kw = 0
            q_rejected_kw = self._p_loss_fixed # 即使无输出，只要转动就有固定损耗
        else:
            p_mech_in_kw = p_electric_out_kw / current_eta
            q_rejected_kw = p_mech_in_kw - p_electric_out_kw
        
        return p_mech_in_kw, q_rejected_kw


    def calculate_performance_forward(self, p_mech_in_kw: float):
        """
        计算输出功率和热功率
        Args:
            p_mech_in_kw: 发动机输入功率[kW]
        Returns:
            输出功率[kW], 热功率[kW]
        """
        if p_mech_in_kw <= 0:
            return 0, 0

        # 定义误差函数
        def error_function(p_electric_out_kw_guess):
            required_mech_in, _ = self.calculate_performance_inverse(p_electric_out_kw_guess)
            return required_mech_in - p_mech_in_kw

        try:
            # 使用求解器寻找使误差为0的解
            p_solution = scipy.optimize.brentq(error_function, 0, self.p * 1.2)
            _, q_rejected_kw = self.calculate_performance_inverse(p_solution)
            return p_solution, q_rejected_kw

        except ValueError:
            print(f"警告: 无法为输入功率 {p_mech_in_kw} kW 找到精确解，可能超出工作范围。")


    def get_mass(self):
        return self.mass_kg


class Battery:
    def __init__(self, 
                 e_rated_kwh: float, 
                 spec_energy_wh_kg: float = 250.0, 
                 spec_power_w_kg: float = 1500.0,
                 pack_mass_fraction: float = 0.75,
                 dod_max: float = 0.8,
                 soc_max: float = 0.95,
                 soc_initial: float = 0.95,
                 eta_discharge: float = 0.97,
                 eta_charge: float = 0.96,
                 c_rate_charge_max: float = 1.0):
        """
        锂电池模型
        Args:
            e_rated_kwh: 电池包额定能量容量[kWh]
            spec_energy_wh_kg: 电芯比能量[Wh/kg]
            spec_power_w_kg: 电芯比功率[W/kg]
            pack_mass_fraction: 电芯质量在电池包总重中的占比
            dod_max: 最大允许放电深度
            soc_max: 最大允许充电状态
            soc_initial: 初始荷电状态
            eta_discharge: 恒定放电效率
            eta_charge: 恒定充电效率
            c_rate_charge_max: 最大允许充电倍率(C-rate)
        """
        if not (0 < pack_mass_fraction <= 1.0):
            raise ValueError("打包系数必须在 (0, 1] 之间")

        self.e_rated_kwh = e_rated_kwh
        
        # 重量计算
        mass_cell_kg = self.e_rated_kwh * 1000 / spec_energy_wh_kg
        self.mass_pack_kg = mass_cell_kg / pack_mass_fraction

        # 功率限制计算
        self.p_max_discharge_kw = (mass_cell_kg * spec_power_w_kg) / 1000.0
        self.p_max_charge_kw = c_rate_charge_max * self.e_rated_kwh

        # 状态变量和限制
        self.soc_min = 1.0 - dod_max
        self.soc_max = soc_max
        self.soc = soc_initial
        self.eta_discharge = eta_discharge
        self.eta_charge = eta_charge


    def update_state(self, p_req_kw: float, dt_seconds: float):
        """
        根据功率请求更新电池状态:正功率表示放电，负功率表示充电。
        Args:
            p_req_kw: 请求的交换功率[kW]
            dt_seconds: 时间步长[s]
        Returns:
            实际与总线交换的功率[kW], 产生的热功率[kW]
        """
        dt_hr = dt_seconds / 3600.0

        # 放电(Discharging)
        if p_req_kw > 0:
            p_out_req_kw = p_req_kw
            
            # 功率限制检查
            if p_out_req_kw > self.p_max_discharge_kw:
                print(f"警告: 请求放电功率 {p_out_req_kw:.1f} kW 超过最大值 {self.p_max_discharge_kw:.1f} kW")
                p_out_kw = self.p_max_discharge_kw
            else:
                p_out_kw = p_out_req_kw

            # 能量限制检查
            e_available_kwh = (self.soc - self.soc_min) * self.e_rated_kwh
            if e_available_kwh <= 0:
                print("警告: 电池电量耗尽，无法放电！")
                return 0, 0

            # 计算此时间步需要从内部取出的能量
            energy_drawn_from_storage_kwh = (p_out_kw / self.eta_discharge) * dt_hr
            
            # 如果请求能量超过可用能量，则只能提供所有剩余能量
            if energy_drawn_from_storage_kwh > e_available_kwh:
                energy_drawn_from_storage_kwh = e_available_kwh
                p_out_kw = (energy_drawn_from_storage_kwh * self.eta_discharge) / dt_hr if dt_hr > 0 else 0
            
            # 更新SoC
            delta_soc = energy_drawn_from_storage_kwh / self.e_rated_kwh
            self.soc -= delta_soc
            
            # 计算热损失
            q_rejected_kw = p_out_kw * (1.0 - self.eta_discharge) / self.eta_discharge
            p_exchanged_kw = p_out_kw

        # 充电(Charging)
        elif p_req_kw < 0:
            p_in_req_kw = abs(p_req_kw)

            # 功率限制检查
            if p_in_req_kw > self.p_max_charge_kw:
                print(f"警告: 请求充电功率 {p_in_req_kw:.1f} kW 超过最大值 {self.p_max_charge_kw:.1f} kW")
                p_in_kw = self.p_max_charge_kw
            else:
                p_in_kw = p_in_req_kw

            # 能量限制检查(防止过充)
            e_absorbable_kwh = (self.soc_max - self.soc) * self.e_rated_kwh
            if e_absorbable_kwh <= 0:
                print("信息: 电池已达充电上限，无法充电！")
                return 0, 0
            
            # 计算此时间步能存入电池的能量
            energy_stored_kwh = (p_in_kw * self.eta_charge) * dt_hr

            # 如果要存入的能量超过剩余容量，则只能充满到上限
            if energy_stored_kwh > e_absorbable_kwh:
                energy_stored_kwh = e_absorbable_kwh
                p_in_kw = (energy_stored_kwh / self.eta_charge) / dt_hr if dt_hr > 0 else 0

            # 更新SoC
            delta_soc = energy_stored_kwh / self.e_rated_kwh
            self.soc += delta_soc

            # 计算热损失
            q_rejected_kw = p_in_kw * (1.0 - self.eta_charge)
            p_exchanged_kw = -p_in_kw # 充电功率为负值

        # 无操作
        else:
            p_exchanged_kw = 0
            q_rejected_kw = 0

        return p_exchanged_kw, q_rejected_kw


    def get_mass(self):
        return self.mass_pack_kg


class Motor:
    def __init__(self, 
                 p: float, 
                 pwr: float = 10.0, 
                 eta_peak: float = 0.97, 
                 peak_eta_load_ratio: float = 0.85):
        """
        电动机模型
        Args:
            p: 额定机械输出功率[kW]
            pwr: 功率重量比[kW/kg]
            eta_peak: 峰值效率
            peak_eta_load_ratio: 达到峰值效率时的负载率(相对于额定功率)
        """
        self.p = p
        self.power_density = pwr
        self.mass_kg = self.p / self.power_density
        
        # 确定损耗系数
        p_peak = self.p * peak_eta_load_ratio
        p_loss_peak = p_peak * (1 - eta_peak) / eta_peak
        self._p_loss_fixed = p_loss_peak / 2.0
        self._k_loss = (p_loss_peak / 2.0) / (p_peak**2) if p_peak > 0 else 0


    def get_efficiency(self, p_shaft_out_kw: float) -> float:
        """
        根据输出轴功率计算当前工况的效率
        Args:
            p_shaft_out_kw: 输出轴功率[kW]
        """
        if p_shaft_out_kw <= 0:
            return 0.0
        
        # 计算损耗
        p_loss_variable = self._k_loss * p_shaft_out_kw**2
        p_loss_total = self._p_loss_fixed + p_loss_variable
        
        # 计算效率
        p_elec_in_kw = p_shaft_out_kw + p_loss_total
        efficiency = p_shaft_out_kw / p_elec_in_kw if p_elec_in_kw > 0 else 0.0
        
        return efficiency


    def calculate_performance_inverse(self, p_shaft_out_req_kw: float):
        """
        计算所需输入电功率和热功率
        Args:
            p_shaft_out_req_kw: 当前需要的输出轴功率[kW]
        Returns:
            所需输入电功率, 热功率
        """
        if p_shaft_out_req_kw > self.p * 1.1: # 假设允许10%短时超载
             print(f"警告: 请求轴功率 {p_shaft_out_req_kw} kW 超过最大允许功率")
             p_shaft_out_kw = self.p * 1.1
        else:
             p_shaft_out_kw = p_shaft_out_req_kw
        
        if p_shaft_out_kw <= 0:
            return 0, 0

        current_eta = self.get_efficiency(p_shaft_out_kw)       
        p_elec_in_kw = p_shaft_out_kw / current_eta
        q_rejected_kw = p_elec_in_kw - p_shaft_out_kw

        return p_elec_in_kw, q_rejected_kw
    

    def calculate_performance_forward(self, p_elec_in_kw: float):
        """
        计算输出轴功率和热功率
        Args:
            p_elec_in_kw: 输入电功率[kW]
        Returns:
            输出轴功率[kW], 热功率[kW]
        """
        if p_elec_in_kw <= 0:
            return 0, 0

        # 定义误差函数
        def error_function(p_mech_out_kw_guess):
            required_elec_in, _ = self.calculate_performance_inverse(p_mech_out_kw_guess)
            return required_elec_in - p_elec_in_kw

        try:
            # 使用求解器寻找使误差为0的解
            p_solution = scipy.optimize.brentq(error_function, 0, self.p * 1.1)
            _, q_rejected_kw = self.calculate_performance_inverse(p_solution)
            return p_solution, q_rejected_kw

        except ValueError:
            print(f"警告: 无法为输入功率 {p_elec_in_kw} kW 找到精确解，可能超出工作范围。")


    def get_mass(self):
        return self.mass_kg


class DuctedFan:
    def __init__(self, 
                 p: float, 
                 diameter_m: float,
                 static_thrust_per_kw: float = 80.0,
                 pwr: float = 8.0):
        """
        基于经验数据的涵道风扇宏观模型
        Args:
            p:风扇设计的额定输入轴功率[kW]
            diameter_m:风扇直径 [m]
            static_thrust_per_kw:单位功率静推力[N/kW]
            pwr:功率重量比[kW/kg]
        """
        self.p_rated_kw = p
        self.diameter = diameter_m
        self.static_thrust_per_kw = static_thrust_per_kw
        
        self.mass_kg = self.p_rated_kw / pwr # 估算重量
        self.v_max_effective_ms = 220.0 # 估算最大有效速度
        self.rho_sea_level = 1.225 # 海平面标准空气密度 [kg/m^3]


    def calculate_performance_inverse(self, thrust_req_N: float, velocity_ms: float, altitude_m: float) -> float:
        """
        计算所需的轴功率
        Args:
            thrust_req_N: 当前需要的推力[N]
            velocity_ms: 飞行速度[m/s]
            altitude_m: 飞行海拔高度[m]
        Returns:
            所需轴功率[kW]
        """
        # 处理无效的推力请求
        if thrust_req_N <= 0:
            return 0.0
            
        # 检查物理限制
        if velocity_ms >= self.v_max_effective_ms:
            print("飞行速度超过有效速度，无法产生正推力")
            return 0.0

        # 计算当前工况下的空气密度
        _, _, air_density_kg_m3 = isa_atmosphere(altitude_m)

        # 求解功率
        density_ratio = air_density_kg_m3 / self.rho_sea_level
        velocity_effect = 1 - (velocity_ms / self.v_max_effective_ms)**2
        denominator = self.static_thrust_per_kw * density_ratio * velocity_effect
        
        if denominator <= 1e-6: # 避免除以零
            print("无法产生需求推力")
            return 0.0

        p_shaft_req_kw = thrust_req_N / denominator

        # 检查计算出的功率是否超过额定功率
        if p_shaft_req_kw > self.p_rated_kw:
            print("超过额定功率")
            p_shaft_req_kw = self.p_rated_kw
            thrust_req_N = p_shaft_req_kw * denominator
            print(f"实际推力:{thrust_req_N}N")

        return p_shaft_req_kw
    

    def calculate_performance_forward(self, p_shaft_kw: float, velocity_ms: float, altitude_m: float) -> float:
        """
        计算产生的推力
        Args:
            p_shaft_kw: 输入轴功率[kW]
            velocity_ms: 飞行速度[m/s]
            altitude_m: 飞行海拔高度[m]
        Returns:
            产生的推力[N]
        """
        if p_shaft_kw <= 0:
            return 0, 0

        # 定义误差函数
        def error_function(thrust_N_guess):
            required_p_shaft = self.calculate_performance_inverse(thrust_N_guess, velocity_ms, altitude_m)
            return required_p_shaft - p_shaft_kw

        try:
            # 使用求解器寻找使误差为0的解
            p_solution = scipy.optimize.brentq(error_function, 0, self.p_rated_kw)
            return p_solution

        except ValueError:
            print(f"警告: 无法为输入轴功率 {p_shaft_kw} kW 找到精确解，可能超出工作范围。")


    def get_mass(self):
        return self.mass_kg


class ThermalManagementSystem:
    def __init__(self,
                 q_design_kw: float,
                 spec_dissipation_kw_m2: float = 20.0,
                 rad_area_density_kg_m2: float = 10.0,
                 rad_drag_coeff: float = 0.1,
                 pump_power_fraction: float = 0.02,
                 pump_mass_per_kw_kg: float = 1.0,
                 fan_power_fraction: float = 0.05,
                 fan_mass_per_kw_kg: float = 1.5,
                 fan_activation_speed_ms: float = 40.0,
                 plumbing_mass_fraction: float = 0.3):
        """
        热管理系统模型
        Args:
            q_design_kw: 系统需要能散掉的最大设计热功率[kW]
            spec_dissipation_kw_m2: 比耗散率[kW/(m^2)](每平方米的散热器迎风面积能够散发掉多少千瓦的热量)
            rad_area_density_kg_m2: 散热器面密度[kg/(m^2)](每平方米的散热器核心有多重)
            rad_drag_coeff: 散热器阻力系数(将散热器的迎风面积转化为等效的阻力)
            pump_power_fraction: 泵功率分数(用于驱动冷却液循环泵所需的电功率占比)
            pump_mass_per_kw_kg: 泵单位功率质量[kg/kW]
            fan_power_fraction: 风扇功率分数(冷却风扇所需的电功率占比)
            fan_mass_per_kw_kg: 风扇单位功率质量[kg/kW]
            fan_activation_speed_ms: 风扇激活速度阈值[m/s]
            plumbing_mass_fraction: 管路与冷却液质量分数(所有干组件总重量的一个固定比例)
        """
        # 散热器选型和称重
        self.radiator_area_m2 = q_design_kw / spec_dissipation_kw_m2
        self.mass_radiator_kg = self.radiator_area_m2 * rad_area_density_kg_m2
        self.rad_drag_coeff = rad_drag_coeff
        
        # 泵和风扇选型和称重
        p_pump_rated_kw = q_design_kw * pump_power_fraction
        self.mass_pump_kg = p_pump_rated_kw * pump_mass_per_kw_kg
        self.pump_power_fraction = pump_power_fraction

        p_fan_rated_kw = q_design_kw * fan_power_fraction
        self.mass_fan_kg = p_fan_rated_kw * fan_mass_per_kw_kg
        self.fan_power_fraction = fan_power_fraction
        self.fan_activation_speed_ms = fan_activation_speed_ms
        
        # 管路和冷却液称重
        mass_dry_components = self.mass_radiator_kg + self.mass_pump_kg + self.mass_fan_kg
        self.mass_plumbing_fluid_kg = mass_dry_components * plumbing_mass_fraction
        
        # 计算总重量
        self.total_mass_kg = mass_dry_components + self.mass_plumbing_fluid_kg


    def calculate_performance(self, q_current_kw: float, velocity_ms: float, altitude_m: float):
        """
        计算当前工况下的TMS性能 (耗电和阻力)
        Args:
            q_current_kw: 当前时刻需要散发的热量[kW]
            velocity_ms: 当前飞行速度[m/s]
            altitude_m: 当前海拔高度[m]
        Returns:
            热管理系统总耗电功率[kW], 热管理系统产生的阻力[N]
        """
        # 计算泵的耗电
        power_draw_pump_kw = q_current_kw * self.pump_power_fraction
        
        # 风扇根据速度决定是否工作
        if velocity_ms < self.fan_activation_speed_ms:
            power_draw_fan_kw = q_current_kw * self.fan_power_fraction
        else:
            power_draw_fan_kw = 0.0
            
        total_power_draw_kw = power_draw_pump_kw + power_draw_fan_kw
        
        # 计算阻力
        _, _, air_density_kg_m3 = isa_atmosphere(altitude_m)
        drag_radiator_N = 0.5 * air_density_kg_m3 * velocity_ms**2 * self.radiator_area_m2 * self.rad_drag_coeff

        return total_power_draw_kw, drag_radiator_N


    def get_mass(self):
        return self.total_mass_kg


class Rectifier:
    def __init__(self, p_rated_kw: float, power_density_kw_per_kg: float = 15.0, efficiency: float = 0.98):
        """
        整流器模型(AC to DC)
        Args:
            p_rated_kw: 额定输入功率[kW], 通常匹配发电机额定功率
            power_density_kw_per_kg: 功率密度[kW/kg]
            efficiency: 转换效率
        """
        self.p_rated_kw = p_rated_kw
        self.power_density = power_density_kw_per_kg
        self.eta = efficiency
        self.mass_kg = self.p_rated_kw / self.power_density


    def calculate_performance_forward(self, p_ac_in_kw: float):
        """
        根据输入的AC功率,计算输出的DC功率和热损失
        Args:
            p_ac_in_kw: 输入的AC功率[kW]
        Returns:
            输出的DC功率[kW], 热功率[kW]
        """
        p_dc_out_kw = p_ac_in_kw * self.eta
        q_rejected_kw = p_ac_in_kw - p_dc_out_kw
        
        return p_dc_out_kw, q_rejected_kw
    
    
    def calculate_performance_inverse(self, p_dc_out_kw: float):
        """
        根据输出的DC功率,反向计算需要的AC输入功率和热损失
        Args:
            p_dc_out_kw: 输出DC功率[kW]
        Returns:
            输入AC功率[kW], 热功率[kW]
        """
        p_ac_in_kw = p_dc_out_kw / self.eta
        q_rejected_kw = p_ac_in_kw - p_dc_out_kw
        
        return p_ac_in_kw, q_rejected_kw


    def get_mass(self):
        return self.mass_kg


class Inverter:
    def __init__(self, p_rated_kw: float, power_density_kw_per_kg: float = 15.0, efficiency: float = 0.98):
        """
        逆变器模型(DC to AC)
        Args:
            p_rated_kw: 额定输出功率[kW], 通常匹配电机额定功率
            power_density_kw_per_kg: 功率密度[kW/kg]
            efficiency: 转换效率
        """
        self.p_rated_kw = p_rated_kw
        self.power_density = power_density_kw_per_kg
        self.eta = efficiency
        self.mass_kg = self.p_rated_kw / self.power_density


    def calculate_performance_inverse(self, p_ac_out_req_kw: float):
        """
        根据电机所需的AC功率,反向计算需要的DC输入功率和热损失
        Args:
            p_ac_out_req_kw: 电机所需的AC功率[kW]
        Returns:
            输入的DC功率[kW], 热功率[kW]
        """
        p_dc_in_kw = p_ac_out_req_kw / self.eta
        q_rejected_kw = p_dc_in_kw - p_ac_out_req_kw
        
        return p_dc_in_kw, q_rejected_kw
    

    def calculate_performance_forward(self, p_dc_in_kw: float):
        """
        计算输出AC功率和热功率
        Args:
            p_dc_in_kw: DC输入功率[kW]
        Returns:
            输出AC功率[kW], 热功率[kW]
        """
        p_ac_out_req_kw = p_dc_in_kw * self.eta
        q_rejected_kw = p_dc_in_kw - p_ac_out_req_kw

        return p_ac_out_req_kw, q_rejected_kw


    def get_mass(self):
        return self.mass_kg


class DCDCConverter:
    def __init__(self, p_rated_kw: float, power_density_kw_per_kg: float = 18.0, 
                 eta_boost: float = 0.985, eta_buck: float = 0.975):
        """
        双向DC-DC转换器模型
        Args:
            p_rated_kw: 额定处理功率[kW](充电或放电的最大值)
            power_density_kw_per_kg: 功率密度[kW/kg]
            eta_boost: 升压(放电)效率
            eta_buck: 降压(充电)效率
        """
        self.p_rated_kw = p_rated_kw
        self.power_density = power_density_kw_per_kg
        self.eta_boost = eta_boost
        self.eta_buck = eta_buck
        self.mass_kg = self.p_rated_kw / self.power_density


    def calculate_performance(self, p_req_kw: float, direction: str):
        """
        计算性能
        Args:
            p_req_kw: 输入功率[kW]
            direction: 'discharge'(从电池到母线)或'charge'(从母线到电池)
        Returns:
            输出功率[kW], 热功率[kW]
        """
        if direction.lower() == 'discharge':
            # 电池提供p_req_kw,计算母线得到多少
            p_from_batt_kw = p_req_kw
            p_to_bus_kw = p_from_batt_kw * self.eta_boost
            q_rejected_kw = p_from_batt_kw * (1 - self.eta_boost)
            return p_to_bus_kw, q_rejected_kw
        
        elif direction.lower() == 'charge':
            # 母线提供p_req_kw,计算电池充入多少
            p_from_bus_kw = p_req_kw
            p_to_batt_kw = p_from_bus_kw * self.eta_buck
            q_rejected_kw = p_from_bus_kw * (1 - self.eta_buck)
            return p_to_batt_kw, q_rejected_kw
        
        else:
            raise ValueError("Direction must be 'discharge' or 'charge'")


    def get_mass(self):
        return self.mass_kg


class WiringSystem:
    # 材料常数 (示例: 铝)
    ALUMINUM_RESISTIVITY = 2.82e-8  # [Ω·m]
    ALUMINUM_DENSITY = 2700      # [kg/m^3]
    ALUMINUM_MAX_CURRENT_DENSITY = 4e6 # [A/m^2]

    def __init__(self, p_rated_kw: float, length_m: float, voltage_v: float):
        """
        输电线缆模型
        Args:
            p_rated_kw: 线缆设计的额定输送功率[kW]
            length_m: 线缆长度[m](估算值)
            voltage_v: 系统直流电压[V]
        """
        self.p_rated_kw = p_rated_kw
        self.length_m = length_m
        self.voltage_v = voltage_v

        # 线缆选型和称重
        if self.voltage_v <= 0:
            raise ValueError("Voltage must be positive.")
        
        i_max_a = (self.p_rated_kw * 1000) / self.voltage_v
        self.cross_section_area_m2 = i_max_a / self.ALUMINUM_MAX_CURRENT_DENSITY
        
        # 计算电阻和重量
        self.resistance_ohm = self.ALUMINUM_RESISTIVITY * self.length_m / self.cross_section_area_m2
        self.mass_kg = self.ALUMINUM_DENSITY * self.length_m * self.cross_section_area_m2


    def calculate_performance(self, p_transmitted_kw: float):
        """
        计算当前工况下的功率损失
        Args:
            p_transmitted_kw: 输送功率[kW]
        Returns:
            热功率[kW]
        """
        if p_transmitted_kw < 0: p_transmitted_kw = 0
            
        current_a = (p_transmitted_kw * 1000) / self.voltage_v
        power_loss_kw = (current_a**2 * self.resistance_ohm) / 1000
        
        return power_loss_kw


    def get_mass(self):
        return self.mass_kg
    