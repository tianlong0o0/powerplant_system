

class Aircraft:
    def __init__(self, initial_mass):
        self.mass = initial_mass # 总质量
        self.altitude = 0 # 飞行高度
        self.velocity = 0 # 飞行速度
        self.distance = 0 # 已飞行距离
        self.time = 0 # 飞行时间
        self.battery_soc = 0.95 # 电池SOC