from system_components import *


class PowerSystem:
    def __init__(self):
        """
        分布式电推进飞机动力系统模型
        """
        # 瞬时飞行状态
        self.altitude = 0.0 # 飞行高度[m]
        self.velocity = 0.0 # 飞行速度[m/s]
        self.time_step = 1.0 # 时间步长 [s]
        self.total_thrust_req_N = 0.0 # 飞机所需的总推力[N]
        self.hr = 0.0 # 功率混合度(dc-dc输出功率与rectifiers输出功率的比值,hr>0表示电池放电,hr<0表示电池充电,hr=0表示纯燃油模式)

        # 动力系统组件实例化
        # 核心发电机组(Gas Power Units - GPUs)
        self.gpu_a = {
            'engine': Engine(p=1600, pwr=7),
            'generators': [Generator(p=800), Generator(p=800)],
            'rectifiers': [Rectifier(p_rated_kw=800), Rectifier(p_rated_kw=800)]
        }
        self.gpus = [self.gpu_a]

        # 储能系统(Energy Storage System - ESS)
        self.ess = {
            'battery': Battery(e_rated_kwh=80),
            'dc_dc': DCDCConverter(p_rated_kw=400)
        }

        # 分布式推进单元(Propulsor Pods)
        self.num_propulsors = 4
        self.propulsors = []
        for i in range(self.num_propulsors):
            pod = {
                'fan': DuctedFan(p=400, diameter_m=2),
                'motor': Motor(p=400),
                'inverter': Inverter(p_rated_kw=400)
            }
            self.propulsors.append(pod)

        # 电网和热管理系统(Common Systems)
        self.wiring = WiringSystem(p_rated_kw=1600, length_m=50, voltage_v=1000)
        self.tms = ThermalManagementSystem(q_design_kw=500)

        # 动力系统参数
        self._calculate_total_mass() # 计算动力系统的总重量[kg]
        self.battery_soc = self.ess['battery'].soc
        self.fuel_consumption = 0.0 # [kg]
        self.fuel_flow_rate = 0.0 # [kg/s]
        self.pp_power_in = 0.0 # [kW]
        self.tms_power_in = 0.0 # [kW]
        self.ess_power_out = 0.0 # [kW]
        self.gpus_power_out = 0.0 # [kW]
        self.total_q = 0.0 # [kW]
        self.tms_drag = 0.0 # 热管理系统产生的阻力[N]

    
    def update(self, altitude_m: float, velocity_ms: float, total_thrust_req_N: float, hr: float, dt_s: float = 1.0):
        """
        根据外部飞行状态计算并更新动力系统内部所有状态
        """
        # 更新输入状态
        self.altitude = altitude_m
        self.velocity = velocity_ms
        self.total_thrust_req_N = total_thrust_req_N
        self.hr = hr
        self.time_step = dt_s
        
        # 重置瞬时热量
        self.total_q = 0.0

        # 计算功率流(这是一个逻辑性很强的顺序)
        self._calculate_pp_power_flow()
        self._calculate_power_split()
        self._calculate_gpus_power_flow()
        self._calculate_ess_power_flow()
        self._calculate_wiring_power_flow()
        self._calculate_tms_power_flow()
        
        # 更新电池SOC和总油耗
        self.battery_soc = self.ess['battery'].soc
        self.fuel_consumption += self.fuel_flow_rate * self.time_step


    def get_mass(self):
        return self.mass


    def get_summary(self) -> dict:
        """
        汇总并返回外部程序需要获取的所有参数
        Returns:
            包含动力系统关键参数的字典
        """
        summary = {
            # 核心状态与资源
            'total_fuel_consumed_kg': self.fuel_consumption,
            'fuel_flow_rate_kg_s': self.fuel_flow_rate,
            'battery_soc': self.battery_soc,

            # 内部功率流详情
            'tms_drag_N': self.tms_drag,
            'total_electrical_load_kw': self.pp_power_in + self.tms_power_in,
            'power_from_gpus_kw': self.gpus_power_out,
            'power_from_ess_kw': self.ess_power_out, # 正为放电, 负为充电
            'power_to_propulsion_kw': self.pp_power_in,
            'power_to_tms_kw': self.tms_power_in,
            'total_heat_generated_kw': self.total_q
        }

        return summary


    def _calculate_total_mass(self) -> float:
        """计算整个动力系统的总质量"""
        mass = 0
        for gpu in self.gpus:
            mass += gpu['engine'].get_mass()
            for gen in gpu['generators']: mass += gen.get_mass()
            for rec in gpu['rectifiers']: mass += rec.get_mass()
        
        mass += self.ess['battery'].get_mass()
        mass += self.ess['dc_dc'].get_mass()
        
        for pod in self.propulsors:
            mass += pod['fan'].get_mass()
            mass += pod['motor'].get_mass()
            mass += pod['inverter'].get_mass()
            
        mass += self.wiring.get_mass()
        mass += self.tms.get_mass()
        
        self.mass = mass


    def _calculate_pp_power_flow(self):
        """
        从推力需求反向计算分布式推进单元的功率需求和总热负载
        """
        # 推进器功率需求
        thrust_per_fan = self.total_thrust_req_N / self.num_propulsors
        
        total_shaft_power_kw = 0
        total_motor_elec_power_kw = 0
        total_inverter_dc_power_kw = 0
        
        q_motors = 0
        q_inverters = 0
        q_propulsion_total = 0

        for pod in self.propulsors:
            # 风扇 -> 电机轴
            p_shaft_kw = pod['fan'].calculate_performance_inverse(thrust_per_fan, self.velocity, self.altitude)
            total_shaft_power_kw += p_shaft_kw
            
            # 电机轴 -> 电机电输入 (AC)
            p_motor_in_kw, q_motor = pod['motor'].calculate_performance_inverse(p_shaft_kw)
            total_motor_elec_power_kw += p_motor_in_kw
            q_motors += q_motor
            
            # 电机电输入 -> 逆变器电输入 (DC)
            p_inverter_in_kw, q_inverter = pod['inverter'].calculate_performance_inverse(p_motor_in_kw)
            total_inverter_dc_power_kw += p_inverter_in_kw
            q_inverters += q_inverter

        # 更新功率需求和热功率
        self.pp_power_in = total_inverter_dc_power_kw
        q_propulsion_total = q_motors + q_inverters
        self.total_q += q_propulsion_total


    def _calculate_power_split(self):
        """根据混合度(hr)分配总功率需求到GPU和ESS"""
        total_bus_demand_kw = self.pp_power_in + self.tms_power_in
        # 功率分配
        self.ess_power_out = total_bus_demand_kw * self.hr
        self.gpus_power_out = total_bus_demand_kw * (1 - self.hr)


    def _calculate_gpus_power_flow(self):
        """
        反向计算核心发电机组的单位燃油消耗率和总热负载
        """
        p_required_per_gpu = self.gpus_power_out / len(self.gpus)
        
        total_fuel_flow_kg_s = 0
        q_gpus_total = 0

        for gpu in self.gpus:
            p_per_rectifier = p_required_per_gpu / len(gpu['rectifiers'])
            p_shaft_for_gens = 0

            # 整流器 -> 发电机
            for i in range(len(gpu['rectifiers'])):
                p_gen_out_ac, q_rec = gpu['rectifiers'][i].calculate_performance_inverse(p_per_rectifier)
                q_gpus_total += q_rec

                # 发电机 -> 发动机轴
                p_shaft_in, q_gen = gpu['generators'][i].calculate_performance_inverse(p_gen_out_ac)
                q_gpus_total += q_gen
                p_shaft_for_gens += p_shaft_in

            # 发动机轴 -> 燃油消耗
            sfc = gpu['engine'].turboshaft_sfc_full(p_shaft_for_gens, self.altitude, self.velocity)
            fuel_flow_kg_s = (sfc * p_shaft_for_gens) / 3600.0
            total_fuel_flow_kg_s += fuel_flow_kg_s

        self.fuel_flow_rate = total_fuel_flow_kg_s
        self.total_q += q_gpus_total


    def _calculate_ess_power_flow(self):
        """
        更新储能系统的状态并计算总热负载
        """
        p_bus_req_kw = self.ess_power_out
        q_ess_total = 0

        if p_bus_req_kw > 0: # 放电: 功率从ESS流向母线
            p_batt_out_kw, q_dcdc = self.ess['dc_dc'].calculate_performance_inverse(p_bus_req_kw, 'discharge')
            
            # 更新电池状态
            _, q_batt = self.ess['battery'].update_state(p_batt_out_kw, self.time_step)
            
        elif p_bus_req_kw < 0: # 充电: 功率从母线流向ESS
            p_bus_in_kw = abs(p_bus_req_kw)
            p_to_batt_kw, q_dcdc = self.ess['dc_dc'].calculate_performance_forward(p_bus_in_kw, 'charge')
            
            # 更新电池状态 (充电功率为负)
            _, q_batt = self.ess['battery'].update_state(-p_to_batt_kw, self.time_step)
        else:
            q_dcdc, q_batt = 0, 0
        
        q_ess_total = q_dcdc + q_batt
        self.total_q += q_ess_total


    def _calculate_wiring_power_flow(self):
        """
        计算输电系统总热负载
        """
        p_transmitted_kw = self.pp_power_in + self.tms_power_in
        q_wiring = self.wiring.calculate_performance(p_transmitted_kw)
        self.total_q += q_wiring


    def _calculate_tms_power_flow(self):
        """
        依据总热负载计算热管理系统的功率需求和产生阻力
        """
        # 此函数计算出的 tms_power_in 和 tms_drag 将在下一个时间步被使用
        p_draw_kw, drag_N = self.tms.calculate_performance(self.total_q, self.velocity, self.altitude)
        self.tms_power_in = p_draw_kw
        self.tms_drag = drag_N

