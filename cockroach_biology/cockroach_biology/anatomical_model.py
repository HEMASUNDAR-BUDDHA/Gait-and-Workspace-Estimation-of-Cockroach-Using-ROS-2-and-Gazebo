import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cm

# --- 1. Set Up 3D Plot ---
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')

u = np.linspace(0, 2 * np.pi, 60)
v = np.linspace(0, np.pi, 60)

# --- 2. UPDATED: Central Body Core Truncated by 20% (Cut 8mm off front) ---
body_length = 32.0    # Reduced from 40mm to 32mm
body_width = 10.5   
body_thickness = 1.0  
body_center_x = -13.0 # Shifted back so the front cut edge hits exactly at X = 3.0
body_center_z = -0.6

x_body = body_center_x + (body_length / 2) * np.outer(np.cos(u), np.sin(v))
y_body = (body_width / 2) * np.outer(np.sin(u), np.sin(v))
z_body = body_center_z + (body_thickness / 2) * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x_body, y_body, z_body, color='maroon', alpha=0.01, edgecolor='none')

# --- 3. Slender Elongated Elliptic Wings ---
x_wlines = np.linspace(-32.0, 3.5, 85)
y_wlines = np.linspace(-6.5, 6.5, 85)
X_w, Y_w = np.meshgrid(x_wlines, y_wlines)

Z_w_base = 0.1 - 0.0008 * (X_w - 3.5)**2 - 0.01 * Y_w**2

w_center_x = -14.25
w_half_len = 17.75
wing_ellipse_bound = 6.2 * np.sqrt(np.clip(1.0 - ((X_w - w_center_x) / w_half_len)**2, 0, 1))
wing_mask = np.abs(Y_w) <= wing_ellipse_bound

Z_wl = np.where(wing_mask & (Y_w >= -0.1), Z_w_base, np.nan)
Z_wr = np.where(wing_mask & (Y_w <= 0.1), Z_w_base + 0.02, np.nan)

ax.plot_surface(X_w, Y_w, Z_wl, color='sienna', alpha=0.92, edgecolor='saddlebrown', linewidth=0.005)
ax.plot_surface(X_w, Y_w, Z_wr, color='saddlebrown', alpha=0.95, edgecolor='maroon', linewidth=0.005)

# Wing texture lines
for y_offset in np.linspace(-5.5, 5.5, 12):
    line_x = np.linspace(-31.0, 2.5, 65)
    line_y = y_offset * np.sqrt(np.clip(1.0 - ((line_x - w_center_x) / w_half_len)**2, 0, 1))
    line_z = 0.12 - 0.0008 * (line_x - 3.5)**2 - 0.01 * line_y**2
    ax.plot(line_x, line_y, line_z, color='maroon', alpha=0.18, linewidth=0.6, zorder=4)

# --- 4. UPDATED: Thorax Shield Inserted Into the Cut Slot ---
# Positioned seamlessly from X=1.5 to X=12.5, perfectly covering the trunk junction
pr_x = np.linspace(1.5, 12.5, 80)
pr_y = np.linspace(-4.5, 4.5, 80)
PR_X, PR_Y = np.meshgrid(pr_x, pr_y)

pr_center_x = 7.0
pr_half_len = 5.5
pr_half_wid = 4.35   

pronotum_mask = (((PR_X - pr_center_x) / pr_half_len)**2 + (PR_Y / pr_half_wid)**2) <= 1.0
PR_Z = np.where(pronotum_mask, 0.03 + 0.03 * np.cos((PR_X - 6.5)/4.5) - 0.0015 * PR_Y**2, np.nan)

color_intensity = np.zeros_like(PR_X)
for i in range(PR_X.shape[0]):
    for j in range(PR_X.shape[1]):
        if pronotum_mask[i, j]:
            dist = np.sqrt(((PR_X[i, j] - 7.5)/2.4)**2 + (PR_Y[i, j]/1.5)**2)
            color_intensity[i, j] = np.clip(dist, 0, 1)
        else:
            color_intensity[i, j] = np.nan

colors = cm.YlOrBr(0.18 + 0.72 * color_intensity)
colors[:, :, 3] = 0.96
ax.plot_surface(PR_X, PR_Y, PR_Z, facecolors=colors, edgecolor='saddlebrown', linewidth=0.01, shade=True)

# --- 5. Forward Projected Head ---
head_length = 4.6
head_width = 3.0   
head_thickness = 1.4
head_center_x = 13.2     
head_center_y = 0.0
head_center_z = -0.3     

x_head = head_center_x + (head_length / 2) * np.outer(np.cos(u), np.sin(v))
y_head = head_center_y + (head_width / 2) * np.outer(np.sin(u), np.sin(v))
z_head = head_center_z + (head_thickness / 2) * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x_head, y_head, z_head, color='maroon', alpha=0.95, edgecolor='darkred', linewidth=0.01)

eye_radius = 0.45
eye_l_x, eye_l_y, eye_l_z = head_center_x + 0.6, head_center_y + 1.1, head_center_z + 0.2
eye_r_x, eye_r_y, eye_r_z = head_center_x + 0.6, head_center_y - 1.1, head_center_z + 0.2

x_eye, y_eye, z_eye = np.outer(np.cos(u), np.sin(v)), np.outer(np.sin(u), np.sin(v)), np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(eye_l_x + eye_radius * x_eye, eye_l_y + eye_radius * y_eye, eye_l_z + eye_radius * z_eye, color='black', zorder=10)
ax.plot_surface(eye_r_x + eye_radius * x_eye, eye_r_y + eye_radius * y_eye, eye_r_z + eye_radius * z_eye, color='black', zorder=10)

# --- 6. Sweeping Antennae ---
t = np.linspace(0, 1, 60)
ant_left_x = eye_l_x + 1.0 + 42.0 * t
ant_left_y = eye_l_y + 24.0 * np.sin(t * np.pi / 2.1)
ant_left_z = eye_l_z + 0.2 * t
ax.plot(ant_left_x, ant_left_y, ant_left_z, color='black', linewidth=1.2)

ant_right_x = eye_r_x + 1.0 + 42.0 * t
ant_right_y = eye_r_y - 24.0 * np.sin(t * np.pi / 2.1)
ant_right_z = eye_r_z + 0.2 * t
ax.plot(ant_right_x, ant_right_y, ant_right_z, color='black', linewidth=1.2)

# --- 7. Sensory Tail Cerci ---
ax.plot([-30.0, -34.5], [1.2, 3.2], [-0.4, -0.6], color='maroon', linewidth=2.0)
ax.plot([-30.0, -34.5], [-1.2, -3.2], [-0.4, -0.6], color='maroon', linewidth=2.0)

# --- 8. Trailing Legs & Spines ---
legs_data = {
    "Foreleg": [3.5, 1.0, 4.0, 3.8, 2.5],
    "Midleg":  [4.5, 1.2, 7.5, 8.0, 4.5],
    "Hindleg": [6.0, 1.5, 13.0, 14.5, 7.5]
}
attachment_x = {"Foreleg": 2.0, "Midleg": -2.3, "Hindleg": -12.65}

def plot_accurate_leg(side, leg_type, segments):
    start_x = attachment_x[leg_type]
    start_y = side * 1.3  
    start_z = -0.6        
    
    c_len, tr_len, f_len, tb_len, ta_len = segments
    coxa_x = start_x - (c_len * 0.5)
    coxa_y = start_y + (side * c_len * 0.7) 
    coxa_z = start_z - (c_len * 0.05)
    
    troch_x = coxa_x - (tr_len * 0.4)
    troch_y = coxa_y + (side * tr_len * 0.7)
    troch_z = coxa_z - (tr_len * 0.02)
    
    if leg_type == "Foreleg":
        fem_x = troch_x + (f_len * 0.2)
        fem_y = troch_y + (side * f_len * 0.8)
    elif leg_type == "Midleg":
        fem_x = troch_x - (f_len * 0.5)
        fem_y = troch_y + (side * f_len * 0.8)
    else:
        fem_x = troch_x - (f_len * 0.85)
        fem_y = troch_y + (side * f_len * 0.45)
    fem_z = troch_z + (f_len * 0.05)
    
    if leg_type == "Foreleg":
        tib_x = fem_x + (tb_len * 0.4)
        tib_y = fem_y + (side * tb_len * 0.6)
    elif leg_type == "Midleg":
        tib_x = fem_x - (tb_len * 0.65)
        tib_y = fem_y + (side * tb_len * 0.7)
    else:
        tib_x = fem_x - (tb_len * 0.92)
        tib_y = fem_y + (side * tb_len * 0.25)
    tib_z = fem_z - (tb_len * 0.2)
    
    if leg_type == "Foreleg":
        tars_x = tib_x + (ta_len * 0.2)
        tars_y = tib_y + (side * ta_len * 0.8)
    else:
        tars_x = tib_x - (ta_len * 0.85)
        tars_y = tib_y + (side * ta_len * 0.45)
    tars_z = tib_z - 0.2 
    
    x_coords = [start_x, coxa_x, troch_x, fem_x, tib_x, tars_x]
    y_coords = [start_y, coxa_y, troch_y, fem_y, tib_y, tars_y]
    z_coords = [start_z, coxa_z, troch_z, fem_z, tib_z, tars_z]
    
    color = 'crimson' if leg_type == "Foreleg" else ('orange' if leg_type == "Midleg" else 'darkgreen')
    ax.plot(x_coords, y_coords, z_coords, color=color, linewidth=2.5)
    ax.scatter(x_coords, y_coords, z_coords, color='black', s=6, zorder=5)

    # Fine spines
    for frac in [0.2, 0.4, 0.6, 0.8]:
        sx = fem_x + frac * (tib_x - fem_x)
        sy = fem_y + frac * (tib_y - fem_y)
        sz = fem_z + frac * (tib_z - fem_z)
        ax.plot([sx, sx - 0.15], [sy, sy + (side * 0.6)], [sz, sz - 0.05], color='black', linewidth=0.7)

# Generate Sweeping Legs
for leg_type, segments in legs_data.items():
    plot_accurate_leg(side=1, leg_type=leg_type, segments=segments)  
    plot_accurate_leg(side=-1, leg_type=leg_type, segments=segments) 

# --- 9. Vector Graphic Formatting & Top-Down View ---
ax.set_box_aspect((1, 1, 0.15))  
ax.set_xlim(-35, 35)
ax.set_ylim(-25, 25)
ax.set_zlim(-5, 5)

ax.view_init(elev=90, azim=-90)

ax.axis('off')
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

output_path = Path(__file__).resolve().parent.parent / "results" / "cockroach_anatomy.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Anatomical model saved to: {output_path}")
plt.show()
