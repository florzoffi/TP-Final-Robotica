import pandas as pd
import matplotlib.pyplot as plt


ARUCO_CSV = "src/tpf/aruco_observations.csv"
ODOM_CSV = "src/tpf/odom.csv"


def main():
    aruco = pd.read_csv(ARUCO_CSV)
    odom = pd.read_csv(ODOM_CSV)

    print("\n=== RESUMEN ARUCO ===")
    print(f"Cantidad total de observaciones: {len(aruco)}")

    print("\nTags detectados:")
    print(sorted(aruco["tag_id"].unique()))

    print("\nCantidad de observaciones por tag:")
    print(aruco["tag_id"].value_counts().sort_index())

    print("\n=== RESUMEN ODOM ===")
    print(f"Cantidad total de poses: {len(odom)}")

    t0 = odom["time"].iloc[0]
    tf = odom["time"].iloc[-1]
    duration = tf - t0

    print(f"Tiempo inicial: {t0}")
    print(f"Tiempo final: {tf}")
    print(f"Duración del recorrido: {duration:.2f} segundos")

    print("\nPose inicial:")
    print(odom.iloc[0])

    print("\nPose final:")
    print(odom.iloc[-1])

    # Trayectoria odométrica
    plt.figure()
    plt.plot(odom["x"], odom["y"])
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Trayectoria por odometría")
    plt.axis("equal")
    plt.grid(True)
    plt.show()

    # Histograma de observaciones por tag
    plt.figure()
    aruco["tag_id"].value_counts().sort_index().plot(kind="bar")
    plt.xlabel("ID ArUco")
    plt.ylabel("Cantidad de observaciones")
    plt.title("Observaciones por tag")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()