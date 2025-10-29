import math


from utils import isa_atmosphere
from power_system import PowerSystem


class Aircraft:
    def __init__(self, m0, dt = 1.0):
        """一个简化的飞机模型，用于飞行性能计算"""
        self.m0 = m0
        self.mass = m0
        self.t = 0 # 初始化时间
        self.dt = dt
        self.power_system = PowerSystem()
        self.total_fuel_burned = 0


    def _update_mass(self, fuel_burned_kg: float):
        """
        更新飞机重量
        Args:
            fuel_burned_kg: 燃油消耗量[kg]
        """
        self.mass -= fuel_burned_kg


    def _simulate(self):
        """
        飞行约束
        Returns:
            所需推力, 飞行高度, 飞行速度
        """
        # 常数
        s_G = 1200 # 起飞滑跑距离
        rou = 1.225 # 空气密度（这里是海平面标准密度）
        C_Ltakeoff = 2.3 # 起飞阶段的升力系数
        C_Lland = 2.5 # 降落阶段的升力系数
        C_D0 = 0.015 # 零升力阻力系数
        Span = 16 # 翼展
        S = 45 # 机翼面积
        A = Span**2 / S # 展弦比
        A_0 = 5 / 180 * math.pi
        e_A = 4.61 * (1 - 0.045 * A**0.68) * math.cos(A_0)**0.15 - 3.1 # Oswald效率因子
        K1 = 1 / math.pi / e_A / A # 诱导阻力系数
        Ma_cruise = 0.4 # 巡航马赫数
        v_cruise = Ma_cruise * 312 # 巡航速度
        tzb = 0.32 # 推重比
        g = 9.80665  # 标准重力加速度 [m/s^2]

        t_total = 0
        h_total = 0

        # 起飞阶段
        v_lof = math.sqrt(self.mass * g / (0.5 * rou * C_Ltakeoff * S))
        total_thrust_req_N = self.mass * g * tzb
        t_to = 2 * s_G / v_lof

        t = self.t - t_total
        t_total += t_to
        if self.t <= t_total:
            return total_thrust_req_N, 0, v_lof * t / t_to


        # 匀速爬升阶段
        h1 = 457.2
        dht_1 = 6
        t_pa1 = h1 / dht_1
        v_pa1 = math.sqrt(dht_1**2 + v_lof**2)
        alpha_pa1 = math.atan(dht_1 / v_lof)
        q_pa1 = 0.5 * rou * (v_pa1**2)
        T_pa1 = q_pa1 * S * C_D0 + K1 * (self.mass * g)**2 / (q_pa1 * S) + self.mass * g * math.sin(alpha_pa1)

        t = self.t - t_total
        t_total += t_pa1
        if self.t <= t_total:
            return T_pa1, dht_1 * t, v_pa1
        h_total += dht_1 * t_pa1


        # 加速爬升阶段
        h2 = 3048 - h1
        dht_2 = 6
        t_pa2 = h2 / dht_2
        v_pa2_mo = 65
        a_pa2 = (v_pa2_mo - v_lof) / t_pa2

        t = self.t - t_total
        v_pa2 = v_lof + a_pa2 * t
        rou_pa2 = -0.0871 * ((h1 + dht_2 * t) / 1000) + 1.1912
        q_pa2 = 0.5 * rou_pa2 * (v_pa2**2)
        alpha_pa2 = math.atan(dht_2 / v_pa2)
        T_pa2 = q_pa2 * S * C_D0 + K1 * (self.mass * g)**2 / (q_pa2 * S) + self.mass * g * math.sin(alpha_pa2) + self.mass * a_pa2
        
        t_total += t_pa2
        if self.t <= t_total:
            return T_pa2, h_total + dht_2 * t, math.sqrt(v_pa2**2 + dht_2**2)
        h_total += dht_2 * t_pa2


        # 加速阶段
        a_pa3 = 0.4
        v_pa3_mo = 80
        t_pa3 = (v_pa3_mo - v_pa2_mo) / a_pa3
        rou_pa3 = -0.0871 * 3.048 + 1.1912

        t = self.t - t_total
        v_pa3 = v_pa2_mo + a_pa3 * t
        q_pa3 = 0.5 * rou_pa3 * (v_pa3 ** 2)
        T_pa3 = 0.98 * self.mass * a_pa3 + q_pa3 * S* C_D0 + K1 * (0.98 * self.mass * g)**2 / (q_pa3 * S)

        t_total += t_pa3
        if self.t <= t_total:
            return T_pa3, h_total, v_pa3


        # 匀速爬升阶段
        h4=7000-3048
        dht_4=6
        t_pa4=h4/dht_4
        v_pa4=math.sqrt(dht_4**2+v_pa3_mo**2)
        alpha_pa4=math.asin(dht_4/v_pa4)

        t = self.t - t_total
        rou_pa4=-0.0871*(3.048+dht_4*t/1000)+1.1912
        q_pa4=0.5*rou_pa4*(v_pa4**2)
        T_pa4=q_pa4*S*C_D0+K1*(0.95*self.mass*g)**2/(q_pa4*S)+self.mass*g*math.sin(alpha_pa4)

        t_total += t_pa4
        if self.t <= t_total:
            return T_pa4, h_total + dht_4 * t, v_pa4
        h_total += dht_4 * t_pa4


        # 加速至巡航阶段
        a_pa5=0.3
        t_pa5=(v_cruise-v_pa4)/a_pa5
        rou_pa5=-0.0871*7+1.1912

        t = self.t - t_total
        v_pa5=v_pa4+a_pa5*t
        q_pa5=0.5*rou_pa5*(v_pa5**2)
        T_pa5=0.95*self.mass*a_pa5+q_pa5*S*C_D0+K1*(0.95*self.mass*g)**2/(q_pa5*S)

        t_total += t_pa5
        if self.t <= t_total:
            return T_pa5, h_total, v_pa5


        # 巡航阶段
        range_cruise=1411.1
        t_cruise=range_cruise*1000/v_cruise
        rou_cruise=0.59
        C_L_cruise=0.95*self.mass*g/(0.5*rou_cruise*v_cruise**2*S)
        C_D_cruise=C_D0+K1*(C_L_cruise)**2
        f_cruise=0.5*rou_cruise*v_cruise**2*S*C_D_cruise

        t_total += t_cruise
        if self.t <= t_total:
            return f_cruise, h_total, v_cruise


        # 水平减速阶段
        a_down1=0.3
        v_down1_mo=80
        t_down1=(v_cruise-v_down1_mo)/a_down1

        t = self.t - t_total
        v_down1=v_cruise-a_down1*t
        q_down1=0.5*rou_pa5*(v_down1**2)
        T_down1=-0.9*self.mass*a_down1+q_down1*S*C_D0+K1*(0.9*self.mass*g)**2/(q_down1*S)

        t_total += t_down1
        if self.t <= t_total:
            return T_down1, h_total, v_down1


        # 匀速下降阶段
        dht_down2=6
        h_down2=7000-3048
        t_down2=h_down2/dht_down2
        v_down2=v_down1_mo

        t = self.t - t_total
        rou_down2=-0.0871*(7-dht_down2/1000*t)+1.1912
        q_down2=0.5*rou_down2*(v_down2**2)
        T_down2=q_down2*S*C_D0+K1*(0.9*self.mass*g)**2/(q_down2*S)

        t_total += t_down2
        if self.t <= t_total:
            return T_down2, h_total - dht_down2 * t, v_down2
        h_total -= dht_down2 * t_down2


        # 水平减速阶段
        a_down3=0.4
        v_down3_mo=60
        t_down3=(v_down1_mo-v_down3_mo)/a_down3
        rou_down3=-0.0871*3.048+1.1912

        t = self.t - t_total
        v_down3=v_down1_mo-a_down3*t
        q_down3=0.5*rou_down3*(v_down3**2)
        T_down3=-0.88*self.mass*a_down3+q_down3*S*C_D0+K1*(0.88*self.mass*g)**2/(q_down3*S)

        t_total += t_down3
        if self.t <= t_total:
            return T_down3, h_total, v_down3


        # 减速下降阶段
        dht_down4=6
        h_down4=3048-457.2
        t_down4=h_down4/dht_down4
        v_down4_mo=40
        a_down4=(v_down3_mo-v_down4_mo)/t_down4

        t = self.t - t_total
        rou_down4=-0.0871*(3.048-dht_down4/1000*t)+1.1912
        v_down4=v_down3_mo-a_down4*t
        q_down4=0.5*rou_down4*(v_down4**2)
        T_down4=-0.88*self.mass*a_down4+q_down4*S*C_D0+K1*(0.88*self.mass*g)**2./(q_down4*S)

        t_total += t_down4
        if self.t <= t_total:
            return T_down4, h_total - dht_down4 * t, v_down4
        h_total -= dht_down4 * t_down4


        # 降落阶段
        v_land=math.sqrt(0.87*self.mass*g/(0.5*rou*S*C_Lland))
        dht_down5=3
        h_down5=457.2
        t_down5=h_down5/dht_down5
        a_down5=(v_down4_mo-v_land)/t_down5; alpha_land=15/180*math.pi

        t = self.t - t_total
        v_down5=v_down4_mo-a_down5*t
        q_down5=0.5*rou*(v_down5**2)
        T_down5=-0.88*self.mass*a_down5+q_down5*S*C_D0+K1*(0.88*self.mass*g)**2/(q_down5*S)+0.88*self.mass*g*math.sin(alpha_land)

        t_total += t_down5
        if self.t <= t_total:
            return T_down5, h_total - dht_down5 * t, v_down5
        

        return 0, 0, 0


    def run(self):
        """飞行运行"""
        f, h, v = 1, 0, 0
        while [f, h, v] != [0, 0, 0]:
            self.t += self.dt
            f, h, v = self._simulate()
            self.power_system.update(h, v, f, 0, self.dt)
            fuel_burned_kg = self.power_system.get_summary()['fuel_flow_rate_kg_s']
            self.total_fuel_burned += fuel_burned_kg
            self._update_mass(fuel_burned_kg)

        return 3500 + math.pow(self.m0, 0.94) * 0.743 + self.total_fuel_burned / 0.94

