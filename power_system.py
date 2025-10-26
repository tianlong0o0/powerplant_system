from system_components import *


class PowerSystem:
    def __init__(self):
        """
        分布式电推进飞机动力系统模型
        """
        # 瞬时飞行状态
        self.altitude = 0.0 # 飞行高度[m]
        self.velocity = 0.0 # 飞行速度[m/s]
        self.time_step = 0.1 # 时间步长 [s]
        self.total_thrust_req_N = 0.0 # 飞机所需的总推力[N]
        self.hr = 0.0 # 功率混合度(dc-dc输出功率与rectifiers输出功率的比值)

        # 动力系统组件实例化
        # 核心发电机组(Gas Power Units - GPUs)
        self.gpu_a = {
            'engine': Engine(p=1600, pwr=7),
            'generators': [Generator(p=800), Generator(p=800)],
            'rectifiers': [Rectifier(p=800), Rectifier(p=800)]
        }
        self.gpu_b = {
            'engine': Engine(p=1600, pwr=7),
            'generators': [Generator(p=800), Generator(p=800)],
            'rectifiers': [Rectifier(p=800), Rectifier(p=800)]
        }
        self.gpus = [self.gpu_a, self.gpu_b]

        # 储能系统(Energy Storage System - ESS)
        self.ess = {
            'battery': Battery(e_rated_kwh=80),
            'dc_dc': DCDCConverter(p_rated_kw=400)
        }

        # 分布式推进单元(Propulsor Pods)
        self.num_propulsors = 6
        self.propulsors = []
        for i in range(self.num_propulsors):
            pod = {
                'fan': DuctedFan(p=600, diameter_m=2),
                'motor': Motor(p=600),
                'inverter': Inverter(p=600)
            }
            self.propulsors.append(pod)

        # 电网和热管理系统(Common Systems)
        self.wiring = WiringSystem(p_rated_kw=3600, length_m=50, voltage_v=1000)
        self.tms = ThermalManagementSystem(q_design_kw=500)

        # 动力系统参数
        self.mass = self._calculate_total_mass() # 动力系统的总重量[kg]
        self.battery_soc = self.ess['battery'].soc
        self.fuel_consumption = 0.0 # [kg]
        self.pp_power_in = 0.0 # [kW]
        self.tms_power_in = 0.0 # [kW]
        self.ess_power_out = 0.0 # [kW]
        self.gpus_power_out = 0.0 # [kW]
        self.total_q = 0.0 # [kW]
        self.tms_resistance = 0.0 # 热管理系统产生的阻力[N]


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
        self.total_q += (q_motors + q_inverters)

