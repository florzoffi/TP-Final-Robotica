import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


ODOM_CSV = "src/tpf/odom.csv"
ARUCO_CSV = "src/tpf/aruco_observations.csv"


def normalize_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def main():
    odom = pd.read_csv(ODOM_CSV)
    obs = pd.read_csv(ARUCO_CSV)

    odom_times = odom["time"].values

    landmark_points = {}

    for _, row in obs.iterrows():
        t = row["time"]
        tag_id = int(row["tag_id"])
        r = row["distance"]
        bearing = row["bearing"]

        # buscar pose de odometría más cercana en tiempo
        idx = np.argmin(np.abs(odom_times - t))

        robot_x = odom.iloc[idx]["x"]
        robot_y = odom.iloc[idx]["y"]
        robot_theta = odom.iloc[idx]["theta"]

        # pasar observación polar robot->landmark a coordenadas globales
        global_angle = normalize_angle(robot_theta + bearing)

        lx = robot_x + r * np.cos(global_angle)
        ly = robot_y + r * np.sin(global_angle)

        if tag_id not in landmark_points:
            landmark_points[tag_id] = []

        landmark_points[tag_id].append((lx, ly))

    landmark_estimates = {}

    for tag_id, points in landmark_points.items():
        pts = np.array(points)
        mean = pts.mean(axis=0)
        landmark_estimates[tag_id] = mean

    print("\n=== LANDMARKS ESTIMADOS ===")
    for tag_id in sorted(landmark_estimates.keys()):
        lx, ly = landmark_estimates[tag_id]
        print(f"Tag {tag_id}: x={lx:.3f}, y={ly:.3f}, n={len(landmark_points[tag_id])}")

    # gráfico
    plt.figure()
    plt.plot(odom["x"], odom["y"], label="Odometría")

    for tag_id, (lx, ly) in landmark_estimates.items():
        plt.scatter(lx, ly)
        plt.text(lx, ly, str(tag_id), fontsize=8)

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Estimación inicial de landmarks ArUco")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()