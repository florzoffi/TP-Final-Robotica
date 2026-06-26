"""
Para cada tag sospechoso (alto sesgo ArUco-LIDAR y muchas observaciones),
grafica la nube de puntos triangulados crudos (sin promediar) en el mundo.

Si todos los puntos triangulados de un tag forman UN solo cluster compacto,
el problema es de calibracion/oclusion puntual de esa medicion.

Si forman DOS (o mas) clusters separados y distantes entre si, es porque
el mismo tag_id esta colocado en mas de una ubicacion fisica del laberinto,
y el codigo los esta tratando como si fueran el mismo landmark.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ODOM_CSV = "src/tpf/odom.csv"
ARUCO_CSV = "src/tpf/aruco_observations.csv"

# Tags a inspeccionar: completar con los que salieron con sesgo alto
# en la comparacion ArUco-vs-LIDAR (ej: 41, 26, 25, 34, 0, 16, 43, 18)
TAGS_TO_CHECK = [41, 26, 25, 34, 0, 16, 43, 18]

MAX_TIME_DIFF_ODOM = 0.1


def main():
    odom = pd.read_csv(ODOM_CSV)
    obs = pd.read_csv(ARUCO_CSV)

    odom_times = odom["time"].to_numpy()

    points_by_tag = {tag: [] for tag in TAGS_TO_CHECK}

    for _, row in obs.iterrows():
        tag_id = int(row["tag_id"])
        if tag_id not in points_by_tag:
            continue

        t = row["time"]
        idx = np.argmin(np.abs(odom_times - t))
        if abs(odom_times[idx] - t) > MAX_TIME_DIFF_ODOM:
            continue

        px = odom.iloc[idx]["x"]
        py = odom.iloc[idx]["y"]
        ptheta = odom.iloc[idx]["theta"]

        angle = ptheta + row["bearing"]
        lx = px + row["distance"] * np.cos(angle)
        ly = py + row["distance"] * np.sin(angle)

        # Guardamos tambien el tiempo, para poder colorear por orden
        # temporal y distinguir "smear" gradual (deriva de odometria)
        # de un salto discreto (posible tag duplicado).
        points_by_tag[tag_id].append((lx, ly, t))

    n = len(TAGS_TO_CHECK)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.array(axes).reshape(-1)

    for i, tag in enumerate(TAGS_TO_CHECK):
        pts = np.array(points_by_tag[tag])
        ax = axes[i]

        if len(pts) == 0:
            ax.set_title(f"Tag {tag}: sin puntos validos")
            continue

        # Color = orden temporal (azul = primeras observaciones,
        # amarillo = ultimas). Si el desplazamiento es gradual (deriva
        # de odometria) el color deberia variar suavemente a lo largo
        # de la nube. Si hay un salto brusco entre dos grupos de color
        # muy distintos sin transicion, es mas sospechoso de tag
        # duplicado en otra ubicacion fisica.
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2],
                         cmap="viridis", s=10, alpha=0.6)
        ax.set_title(f"Tag {tag}: {len(pts)} obs, "
                     f"std=({pts[:,0].std():.2f},{pts[:,1].std():.2f})")
        ax.axis("equal")
        ax.grid(True)
        plt.colorbar(sc, ax=ax, label="tiempo")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()