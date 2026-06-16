import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


ODOM_CSV = "src/tpf/odom.csv"
ARUCO_CSV = "src/tpf/aruco_observations.csv"

START_POSE = 0
MAX_POSES = None
MIN_TRANSLATION = 0.25
MIN_ROTATION = 0.45
MAX_TIME_GAP = 1.5
OBS_STRIDE = 1

def normalize_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def relative_motion(p1, p2):
    dx = p2["x"] - p1["x"]
    dy = p2["y"] - p1["y"]
    dtheta = normalize_angle(p2["theta"] - p1["theta"])

    theta = p1["theta"]

    dx_local = np.cos(theta) * dx + np.sin(theta) * dy
    dy_local = -np.sin(theta) * dx + np.cos(theta) * dy

    return dx_local, dy_local, dtheta


def residuals(state, n_poses, tag_to_idx, odom_factors, landmark_factors, poses_prior):
    poses = state[: n_poses * 3].reshape((n_poses, 3))
    landmarks = state[n_poses * 3:].reshape((len(tag_to_idx), 2))

    res = []

    res.append((poses[0, 0] - poses_prior[0, 0]) * 100.0)
    res.append((poses[0, 1] - poses_prior[0, 1]) * 100.0)
    res.append(normalize_angle(poses[0, 2] - poses_prior[0, 2]) * 100.0)

    for i in range(n_poses):
        res.append((poses[i, 0] - poses_prior[i, 0]) * 20.0)
        res.append((poses[i, 1] - poses_prior[i, 1]) * 20.0)
        res.append(normalize_angle(poses[i, 2] - poses_prior[i, 2]) * 10.0)

    for f in odom_factors:
        i = f["from"]
        j = f["to"]

        pred_dx, pred_dy, pred_dtheta = relative_motion(
            {"x": poses[i, 0], "y": poses[i, 1], "theta": poses[i, 2]},
            {"x": poses[j, 0], "y": poses[j, 1], "theta": poses[j, 2]},
        )

        res.append((pred_dx - f["dx"]) * 50.0)
        res.append((pred_dy - f["dy"]) * 50.0)
        res.append(normalize_angle(pred_dtheta - f["dtheta"]) * 20.0)

    for f in landmark_factors:
        pose_idx = f["pose"]
        tag_idx = tag_to_idx[f["tag"]]

        px, py, ptheta = poses[pose_idx]
        lx, ly = landmarks[tag_idx]

        dx = lx - px
        dy = ly - py

        pred_dist = np.sqrt(dx**2 + dy**2)
        pred_bearing = normalize_angle(np.arctan2(dy, dx) - ptheta)

        res.append((pred_dist - f["distance"]) * 1.0)
        res.append(normalize_angle(pred_bearing - f["bearing"]) * 0.5)

    return np.array(res)


def select_keyframes(odom):
    selected_rows = [odom.iloc[0]]

    last_x = odom.iloc[0]["x"]
    last_y = odom.iloc[0]["y"]
    last_theta = odom.iloc[0]["theta"]
    last_t = odom.iloc[0]["time"]

    for _, row in odom.iterrows():
        dx = row["x"] - last_x
        dy = row["y"] - last_y
        dtrans = np.sqrt(dx**2 + dy**2)

        drot = abs(normalize_angle(row["theta"] - last_theta))
        dt = row["time"] - last_t

        if dtrans >= MIN_TRANSLATION or drot >= MIN_ROTATION or dt >= MAX_TIME_GAP:
            selected_rows.append(row)

            last_x = row["x"]
            last_y = row["y"]
            last_theta = row["theta"]
            last_t = row["time"]

    return pd.DataFrame(selected_rows).reset_index(drop=True)


def main():
    odom = pd.read_csv(ODOM_CSV)
    obs = pd.read_csv(ARUCO_CSV)

    odom_sub = select_keyframes(odom)

    print(f"Total keyframes seleccionados: {len(odom_sub)}")

    if MAX_POSES is None:
        odom_sub = odom_sub.iloc[START_POSE:].reset_index(drop=True)
    else:
        odom_sub = odom_sub.iloc[START_POSE:START_POSE + MAX_POSES].reset_index(drop=True)

    if len(odom_sub) == 0:
        print(f"No hay poses para START_POSE={START_POSE}. Fin del recorrido.")
        return

    odom_times = odom_sub["time"].to_numpy()

    best_observations = {}

    for _, row in obs.iterrows():
        t = row["time"]

        if t < odom_sub["time"].iloc[0] or t > odom_sub["time"].iloc[-1]:
            continue

        pose_idx = np.argmin(np.abs(odom_times - t))
        tag_id = int(row["tag_id"])
        distance = float(row["distance"])
        bearing = float(row["bearing"])

        key = (pose_idx, tag_id)

        if key not in best_observations:
            best_observations[key] = {
                "pose": pose_idx,
                "tag": tag_id,
                "distance": distance,
                "bearing": bearing,
            }
        elif distance < best_observations[key]["distance"]:
            best_observations[key] = {
                "pose": pose_idx,
                "tag": tag_id,
                "distance": distance,
                "bearing": bearing,
            }

    landmark_factors = list(best_observations.values())
    landmark_factors = landmark_factors[::OBS_STRIDE]

    odom_factors = []

    for i in range(len(odom_sub) - 1):
        p1 = odom_sub.iloc[i]
        p2 = odom_sub.iloc[i + 1]

        dx, dy, dtheta = relative_motion(p1, p2)

        odom_factors.append({
            "from": i,
            "to": i + 1,
            "dx": dx,
            "dy": dy,
            "dtheta": dtheta,
        })

    tags = sorted(set(f["tag"] for f in landmark_factors))
    tag_to_idx = {tag: i for i, tag in enumerate(tags)}

    n_poses = len(odom_sub)
    n_landmarks = len(tags)

    print(f"Poses usadas: {n_poses}")
    print(f"Landmarks: {n_landmarks}")
    print(f"Factores odometría: {len(odom_factors)}")
    print(f"Factores ArUco: {len(landmark_factors)}")

    if n_landmarks == 0:
        print("No hay factores ArUco en este tramo.")
        return

    poses0 = odom_sub[["x", "y", "theta"]].to_numpy()

    landmark_points = {tag: [] for tag in tags}

    for f in landmark_factors:
        p = poses0[f["pose"]]
        px, py, ptheta = p

        angle = ptheta + f["bearing"]

        lx = px + f["distance"] * np.cos(angle)
        ly = py + f["distance"] * np.sin(angle)

        landmark_points[f["tag"]].append((lx, ly))

    landmarks0 = []

    for tag in tags:
        pts = np.array(landmark_points[tag])
        landmarks0.append(pts.mean(axis=0))

    landmarks0 = np.array(landmarks0)

    x0 = np.concatenate([poses0.flatten(), landmarks0.flatten()])

    print("Optimizando...")

    result = least_squares(
        residuals,
        x0,
        args=(n_poses, tag_to_idx, odom_factors, landmark_factors, poses0),
        max_nfev=8,
        loss="huber",
        f_scale=1.0,
        verbose=1
    )

    x_opt = result.x
    poses_opt = x_opt[: n_poses * 3].reshape((n_poses, 3))
    landmarks_opt = x_opt[n_poses * 3:].reshape((n_landmarks, 2))

    poses_df = pd.DataFrame(poses_opt, columns=["x", "y", "theta"])
    poses_df["time"] = odom_sub["time"].values
    poses_df.to_csv("src/tpf/poses_optimized_keyframes.csv", index=False)

    landmarks_rows = []

    for tag, idx in tag_to_idx.items():
        lx, ly = landmarks_opt[idx]
        landmarks_rows.append({
            "tag_id": tag,
            "x": lx,
            "y": ly
        })

    landmarks_df = pd.DataFrame(landmarks_rows)
    landmarks_df.to_csv("src/tpf/landmarks_optimized_keyframes.csv", index=False)

    print("Guardado src/tpf/poses_optimized_keyframes.csv")
    print("Guardado src/tpf/landmarks_optimized_keyframes.csv")

    print("Optimización terminada")
    print(f"Costo final: {result.cost:.3f}")

    plt.figure()
    plt.plot(poses0[:, 0], poses0[:, 1], label="Odometría inicial")
    plt.plot(poses_opt[:, 0], poses_opt[:, 1], label="Trayectoria optimizada")

    for tag, idx in tag_to_idx.items():
        lx, ly = landmarks_opt[idx]
        plt.scatter(lx, ly)
        plt.text(lx, ly, str(tag), fontsize=8)

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Graph SLAM global con keyframes")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()