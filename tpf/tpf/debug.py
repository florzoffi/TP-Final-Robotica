plt.figure(figsize=(8, 8))
plt.plot(poses["x"], poses["y"], "b-", alpha=0.3)

scan = scans.iloc[1000]  # o cualquier índice
pose_idx = np.argmin(np.abs(pose_times - scan["time"]))

x = poses.iloc[pose_idx]["x"]
y = poses.iloc[pose_idx]["y"]
theta = poses.iloc[pose_idx]["theta"]

plt.scatter(x, y, c="red", label="Robot")

tokens = str(scan["ranges"]).split(";")
for k, t in enumerate(tokens):
    if t == "":
        continue
    r = float(t)
    angle = theta + scan["angle_min"] + k * scan["angle_increment"]

    wx = x + r * np.cos(angle)
    wy = y + r * np.sin(angle)
    plt.plot([x, wx], [y, wy], "k-", linewidth=0.3)

plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()