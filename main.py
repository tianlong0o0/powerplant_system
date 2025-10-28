from mission_profile import Aircraft

if __name__ == "__main__":
    m1 = 0
    m2 = 10000
    tms = []
    while abs(m1-m2) > 0.001:
        m1 = m2
        air = Aircraft(m1)
        m2 = air.run()

    print(f"{m2}")