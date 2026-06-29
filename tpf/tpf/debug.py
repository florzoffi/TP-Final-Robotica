import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCANS_CSV = "src/TP-Final-Robotica/tpf/scans.csv"
POSES_CSV = "src/TP-Final-Robotica/tpf/poses_optimized_keyframes.csv"

RESOLUTION = 0.05
MAP_SIZE = 30.0
GRID_N = int(MAP_SIZE / RESOLUTION)
ORIGIN_X = MAP_SIZE / 2.0
ORIGIN_Y = MAP_SIZE / 2.0

BEAM_STEP = 20
MAX_RANGE = 2.0

MIN_OCC_HITS = 4
MIN_FREE_HITS = 2

LIDAR_ANGLE_OFFSET = np.pi / 2
LIDAR_X_OFFSET = -0.04
LIDAR_Y_OFFSET = 0.0

def normalize_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def world_to_grid(x, y):
    gx = int((x + ORIGIN_X) / RESOLUTION)
    gy = int((y + ORIGIN_Y) / RESOLUTION)
    return gx, gy


def bresenham(x0, y0, x1, y1):
    points = []

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)

    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    err = dx - dy

    x, y = x0, y0

    while True:
        points.append((x, y))

        if x == x1 and y == y1:
            break

        e2 = 2 * err

        if e2 > -dy:
            err -= dy
            x += sx

        if e2 < dx:
            err += dx
            y += sy

    return points


def interp_pose(t, poses):
    times = poses["time"].to_numpy()

    if t < times[0] or t > times[-1]:
        return None

    idx = np.searchsorted(times, t)

    if idx == 0:
        row = poses.iloc[0]
        return row["x"], row["y"], row["theta"]

    if idx >= len(times):
        row = poses.iloc[-1]
        return row["x"], row["y"], row["theta"]

    p0 = poses.iloc[idx - 1]
    p1 = poses.iloc[idx]

    t0 = p0["time"]
    t1 = p1["time"]

    if t1 == t0:
        return p0["x"], p0["y"], p0["theta"]

    alpha = (t - t0) / (t1 - t0)

    x = p0["x"] + alpha * (p1["x"] - p0["x"])
    y = p0["y"] + alpha * (p1["y"] - p0["y"])

    dtheta = normalize_angle(p1["theta"] - p0["theta"])
    theta = normalize_angle(p0["theta"] + alpha * dtheta)

    return x, y, theta


def parse_ranges(ranges_str):
    values = []
    for s in str(ranges_str).split(";"):
        if s == "" or s.lower() == "nan":
            values.append(np.nan)
        else:
            values.append(float(s))
    return np.array(values)


def main():
    scans = pd.read_csv(SCANS_CSV)
    poses = pd.read_csv(POSES_CSV)

    occ_count = np.zeros((GRID_N, GRID_N), dtype=np.int32)
    free_count = np.zeros((GRID_N, GRID_N), dtype=np.int32)

    used_scans = 0

    for _, scan in scans.iterrows():
        t = scan["time"]
        pose = interp_pose(t, poses)

        if pose is None:
            continue

        rx, ry, rtheta = pose
        lidar_x = rx + np.cos(rtheta) * LIDAR_X_OFFSET - np.sin(rtheta) * LIDAR_Y_OFFSET
        lidar_y = ry + np.sin(rtheta) * LIDAR_X_OFFSET + np.cos(rtheta) * LIDAR_Y_OFFSET

        ranges = parse_ranges(scan["ranges"])
        angle_min = scan["angle_min"]
        angle_increment = scan["angle_increment"]
        range_min = scan["range_min"]
        range_max = min(scan["range_max"], MAX_RANGE)

        rgx, rgy = world_to_grid(lidar_x, lidar_y)

        if not (0 <= rgx < GRID_N and 0 <= rgy < GRID_N):
            continue

        used_scans += 1

        for i in range(0, len(ranges), BEAM_STEP):
            r = ranges[i]

            if np.isnan(r):
                continue

            if r < range_min or r > range_max:
                continue

            angle_lidar = angle_min + i * angle_increment
            angle_world = rtheta + LIDAR_ANGLE_OFFSET + angle_lidar

            ex = lidar_x + r * np.cos(angle_world)
            ey = lidar_y + r * np.sin(angle_world)

            egx, egy = world_to_grid(ex, ey)

            if not (0 <= egx < GRID_N and 0 <= egy < GRID_N):
                continue

            cells = bresenham(rgx, rgy, egx, egy)

            for cx, cy in cells[:-1]:
                if 0 <= cx < GRID_N and 0 <= cy < GRID_N:
                    free_count[cy, cx] += 1

            occ_count[egy, egx] += 1

    grid = np.full((GRID_N, GRID_N), -1, dtype=np.int8)

    free = free_count >= MIN_FREE_HITS
    occ = (occ_count >= MIN_OCC_HITS) & (occ_count > free_count * 0.25)

    grid[free] = 0
    grid[occ] = 100

    print("Scans usados:", used_scans)
    print("Celdas libres:", np.sum(grid == 0))
    print("Celdas ocupadas:", np.sum(grid == 100))

    plt.figure(figsize=(8, 8))
    plt.imshow(
        grid,
        origin="lower",
        cmap="gray_r",
        extent=[-ORIGIN_X, MAP_SIZE - ORIGIN_X, -ORIGIN_Y, MAP_SIZE - ORIGIN_Y],
    )
    plt.plot(poses["x"], poses["y"], linewidth=1)
    plt.axis("equal")
    plt.title(f"Mapa offline con LIDAR + poses optimizadas | offset={LIDAR_ANGLE_OFFSET:.2f}")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.grid(True)
    plt.show()

    plt.imsave("src/TP-Final-Robotica/tpf/final_map_offline.png", grid, cmap="gray_r")
    print("Guardado src/TP-Final-Robotica/tpf/final_map_offline.png")


if __name__ == "__main__":
    main()