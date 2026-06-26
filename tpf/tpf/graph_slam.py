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

W_PRIOR0_XY = 100.0
W_PRIOR0_TH = 100.0

W_ODOM_XY = 60.0
W_ODOM_TH = 30.0

W_ARUCO_DIST = 5.0
W_ARUCO_BEARING = 2.0
TAGS_TO_EXCLUDE = [41]

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

    res.append((poses[0, 0] - poses_prior[0, 0]) * W_PRIOR0_XY)
    res.append((poses[0, 1] - poses_prior[0, 1]) * W_PRIOR0_XY)
    res.append(normalize_angle(poses[0, 2] - poses_prior[0, 2]) * W_PRIOR0_TH)

    for f in odom_factors:
        i = f["from"]
        j = f["to"]

        pred_dx, pred_dy, pred_dtheta = relative_motion(
            {"x": poses[i, 0], "y": poses[i, 1], "theta": poses[i, 2]},
            {"x": poses[j, 0], "y": poses[j, 1], "theta": poses[j, 2]},
        )

        res.append((pred_dx - f["dx"]) * W_ODOM_XY)
        res.append((pred_dy - f["dy"]) * W_ODOM_XY)
        res.append(normalize_angle(pred_dtheta - f["dtheta"]) * W_ODOM_TH)

    for f in landmark_factors:
        pose_idx = f["pose"]
        tag_idx = tag_to_idx[f["tag"]]

        px, py, ptheta = poses[pose_idx]
        lx, ly = landmarks[tag_idx]

        dx = lx - px
        dy = ly - py

        pred_dist = np.sqrt(dx**2 + dy**2)
        pred_bearing = normalize_angle(np.arctan2(dy, dx) - ptheta)

        res.append((pred_dist - f["distance"]) * W_ARUCO_DIST)
        res.append(normalize_angle(pred_bearing - f["bearing"]) * W_ARUCO_BEARING)

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
    #obs["bearing"] = -obs["bearing"]

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

    observations_by_key = {}

    for _, row in obs.iterrows():
        t = row["time"]

        if t < odom_sub["time"].iloc[0] or t > odom_sub["time"].iloc[-1]:
            continue

        pose_idx = np.argmin(np.abs(odom_times - t))
        time_error = abs(odom_times[pose_idx] - t)

        if time_error > 0.15:
            continue

        tag_id = int(row["tag_id"])
        distance = float(row["distance"])
        bearing = float(row["bearing"])

        if distance < 0.2 or distance > 1.5:
            continue

        key = (pose_idx, tag_id)

        if key not in observations_by_key:
            observations_by_key[key] = []

        observations_by_key[key].append((distance, bearing))

    print(f"Observaciones crudas usadas: {sum(len(v) for v in observations_by_key.values())}")
    print(f"Pose-tag agrupados: {len(observations_by_key)}")


    # Diagnóstico: ¿qué tags se re-observan en tramos lejanos (loop closure)?
    tag_pose_indices = {}
    for (pose_idx, tag_id) in observations_by_key.keys():
        tag_pose_indices.setdefault(tag_id, []).append(pose_idx)

    print("\n--- Diagnóstico de loop closure ---")
    for tag_id, pose_idxs in sorted(tag_pose_indices.items()):
        pose_idxs = sorted(pose_idxs)
        span = pose_idxs[-1] - pose_idxs[0]
        print(f"Tag {tag_id}: visto en {len(pose_idxs)} poses, "
              f"rango de poses {pose_idxs[0]}–{pose_idxs[-1]} (span={span})")
    print("-----------------------------------\n")

    
    landmark_factors = []

    for (pose_idx, tag_id), values in observations_by_key.items():
        if tag_id in TAGS_TO_EXCLUDE:
            continue
        values = np.array(values)

        if len(values) < 6:
            continue

        distance = np.median(values[:, 0])
        bearing = np.median(values[:, 1])

        landmark_factors.append({
            "pose": pose_idx,
            "tag": tag_id,
            "distance": distance,
            "bearing": bearing,
        })
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
        landmarks0.append(np.median(pts, axis=0))

    landmarks0 = np.array(landmarks0)
    x0 = np.concatenate([poses0.flatten(), landmarks0.flatten()])

    landmarks0_by_tag = {
        tag: landmarks0[idx]
        for tag, idx in tag_to_idx.items()
    }

    
    plt.figure()

    plt.plot(poses0[:,0], poses0[:,1])

    for tag, idx in tag_to_idx.items():
        lx, ly = landmarks0[idx]
        plt.scatter(lx, ly)
        plt.text(lx, ly, str(tag))

    plt.axis("equal")
    plt.show()

    print("Optimizando...")

    result = least_squares(
        residuals,
        x0,
        args=(n_poses, tag_to_idx, odom_factors, landmark_factors, poses0),
        max_nfev=80,
        loss="huber",
        f_scale=0.5,
        verbose=2
    )

    """
    Pegar este bloque en graph_slam.py, justo despues de la linea:
        result = least_squares(...)
    y antes de armar los DataFrames de salida (poses_df / landmarks_df).

    Localiza en que tramo de la trayectoria se concentra el mayor error
    residual despues de optimizar, separando por tipo de factor
    (odometria vs ArUco), para encontrar la causa del "quiebre".
    """

    # --- Diagnostico de residuos por factor ---

    x_opt_diag = result.x
    poses_opt_diag = x_opt_diag[: n_poses * 3].reshape((n_poses, 3))
    landmarks_opt_diag = x_opt_diag[n_poses * 3:].reshape((len(tag_to_idx), 2))

    print("\n--- Residuos de factores de odometria (ordenados, peores primero) ---")
    odom_residuals = []
    for f in odom_factors:
        i, j = f["from"], f["to"]
        pred_dx, pred_dy, pred_dtheta = relative_motion(
            {"x": poses_opt_diag[i, 0], "y": poses_opt_diag[i, 1], "theta": poses_opt_diag[i, 2]},
            {"x": poses_opt_diag[j, 0], "y": poses_opt_diag[j, 1], "theta": poses_opt_diag[j, 2]},
        )
        err_xy = np.hypot(pred_dx - f["dx"], pred_dy - f["dy"])
        err_th = abs(normalize_angle(pred_dtheta - f["dtheta"]))
        odom_residuals.append((i, j, err_xy, err_th))

    odom_residuals.sort(key=lambda r: -r[2])
    for i, j, err_xy, err_th in odom_residuals[:15]:
        print(f"poses {i}->{j}: error_xy={err_xy:.3f} m, error_theta={np.degrees(err_th):.1f} deg")

    print("\n--- Residuos de factores ArUco (ordenados, peores primero) ---")
    aruco_residuals = []
    for f in landmark_factors:
        pose_idx = f["pose"]
        tag_idx = tag_to_idx[f["tag"]]
        px, py, ptheta = poses_opt_diag[pose_idx]
        lx, ly = landmarks_opt_diag[tag_idx]
        dx = lx - px
        dy = ly - py
        pred_dist = np.sqrt(dx**2 + dy**2)
        pred_bearing = normalize_angle(np.arctan2(dy, dx) - ptheta)
        err_dist = abs(pred_dist - f["distance"])
        err_bear = abs(normalize_angle(pred_bearing - f["bearing"]))
        aruco_residuals.append((pose_idx, f["tag"], err_dist, err_bear))

    aruco_residuals.sort(key=lambda r: -r[2])
    for pose_idx, tag, err_dist, err_bear in aruco_residuals[:15]:
        print(f"pose {pose_idx}, tag {tag}: error_dist={err_dist:.3f} m, "
              f"error_bearing={np.degrees(err_bear):.1f} deg")

    # Tambien: identificar tramos de poses con pocas o ninguna observacion
    # ArUco cercana (zonas "ciegas" donde la odometria no se corrige).
    poses_with_obs = sorted(set(f["pose"] for f in landmark_factors))
    print("\n--- Gaps mas grandes sin observaciones ArUco (indices de pose) ---")
    gaps = []
    for k in range(len(poses_with_obs) - 1):
        gap = poses_with_obs[k + 1] - poses_with_obs[k]
        gaps.append((poses_with_obs[k], poses_with_obs[k + 1], gap))
    gaps.sort(key=lambda g: -g[2])
    for start, end, gap in gaps[:10]:
        print(f"sin observaciones entre pose {start} y {end} (gap={gap} poses)")

    x_opt = result.x
    poses_opt = x_opt[: n_poses * 3].reshape((n_poses, 3))
    landmarks_opt = x_opt[n_poses * 3:].reshape((n_landmarks, 2))

    plt.figure(figsize=(10, 8))

    plt.plot(poses0[:, 0], poses0[:, 1], label="Trayectoria inicial", alpha=0.5)
    plt.plot(poses_opt[:, 0], poses_opt[:, 1], label="Trayectoria optimizada", alpha=0.8)

    for tag, idx in tag_to_idx.items():
        x_init, y_init = landmarks0_by_tag[tag]
        x_opt_l, y_opt_l = landmarks_opt[idx]

        plt.scatter(x_init, y_init, marker="x", s=60)
        plt.scatter(x_opt_l, y_opt_l, marker="o", s=60)

        plt.plot([x_init, x_opt_l], [y_init, y_opt_l], linewidth=1)
        plt.text(x_opt_l, y_opt_l, str(tag), fontsize=8)

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Landmarks iniciales vs landmarks optimizados")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()

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