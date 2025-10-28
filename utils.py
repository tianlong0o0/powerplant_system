import math


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
        pressure = P_tropo * math.exp(-g0 * h_in_strato / (R * temperature))
    else:
        raise NotImplementedError("高度超过 20 km 暂未实现")
        
    density = pressure / (R * temperature)

    return temperature, pressure, density