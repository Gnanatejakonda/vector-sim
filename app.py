import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 1. Page Config
st.set_page_config(page_title="Basis Transformation Sim", page_icon="📐")
st.title("📐 Basis Transformation")

st.markdown("""
**Theory Note:** 1. **Linear Rotation:** Only applies if Origin is (0,0) and axes are perpendicular.
2. **Affine Transform:** If Origin is shifted, we must subtract the offset first.
""")

# 2. Sidebar Inputs
with st.sidebar:
    st.header("1. Input Point (Global Coords)")
    vx = st.number_input("Point X", value=3.0, step=0.5)
    vy = st.number_input("Point Y", value=2.0, step=0.5)
    
    st.header("2. New Basis Vectors (Green)")
    st.write("Define the direction of the new axes:")
    b1x = st.number_input("Axis 1 (x) coeff", value=1.0, step=0.1)
    b1y = st.number_input("Axis 1 (y) coeff", value=0.5, step=0.1)
    
    b2x = st.number_input("Axis 2 (x) coeff", value=-0.5, step=0.1)
    b2y = st.number_input("Axis 2 (y) coeff", value=1.0, step=0.1)
    
    st.header("3. Origin Location")
    c1 = st.number_input("Origin X", value=0.0, step=0.5)
    c2 = st.number_input("Origin Y", value=0.0, step=0.5)

# 3. Calculations
v_global = np.array([vx, vy])       
origin_pos = np.array([c1, c2])     
basis_matrix = np.array([[b1x, b2x], 
                         [b1y, b2y]])

# CONDITIONS CHECK
det = np.linalg.det(basis_matrix)

# A) Is it Perpendicular? (Dot product close to 0)
dot_prod = (b1x * b2x) + (b1y * b2y)
is_orthogonal = abs(dot_prod) < 1e-5

# B) Is the Origin at (0,0)?
is_centered = np.linalg.norm(origin_pos) < 1e-5

if abs(det) < 1e-10:
    st.error("⚠️ LINEAR DEPENDENCE ERROR: The basis vectors are parallel.")
else:
    # --- Transformation Logic ---
    v_relative = v_global - origin_pos
    v_new_coords = np.linalg.solve(basis_matrix, v_relative)

    # 4. Display Results
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Global Point (Blue):**\n\n ({vx}, {vy})")
        st.write(f"**Shifted Vector (relative to Green Origin):**\n\n ({v_relative[0]:.2f}, {v_relative[1]:.2f})")
        
    with col2:
        st.success(f"**New Basis Coords (Green):**\n\n [{v_new_coords[0]:.2f}, {v_new_coords[1]:.2f}]")
        
        # --- CONDITIONAL ROTATION MATRIX DISPLAY ---
        if is_orthogonal and is_centered:
            st.write("✅ **Pure Rotation detected**")
            
            # Calculate angle
            angle_rad = np.arctan2(b1y, b1x)
            angle_deg = np.degrees(angle_rad)
            
            rot_matrix = np.array([
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad),  np.cos(angle_rad)]
            ])
            
            st.write(f"**Rotation Matrix ($R$) for {angle_deg:.1f}°:**")
            st.latex(r"R = \begin{bmatrix} " + 
                     f"{rot_matrix[0,0]:.2f} & {rot_matrix[0,1]:.2f} \\\\ " + 
                     f"{rot_matrix[1,0]:.2f} & {rot_matrix[1,1]:.2f}" + 
                     r" \end{bmatrix}")
                     
        elif not is_centered:
            st.warning("⚠️ **Affine Shift Detected**")
            st.write("Origin is NOT (0,0). This is a translation + rotation.")
            st.write("A pure Rotation Matrix only applies to fixed origins.")
            
        elif not is_orthogonal:
            st.warning("⚠️ **Skewed Axes**")
            st.write("Axes are not perpendicular.")

    # 5. Visualization
    fig, ax = plt.subplots(figsize=(8, 8))
    
    limit = max(np.linalg.norm(v_global), np.linalg.norm(origin_pos), 4.0) * 1.5
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect('equal')
    
    ax.axhline(0, color='lightgray', linewidth=1)
    ax.axvline(0, color='lightgray', linewidth=1)
    ax.grid(True, linestyle=':', alpha=0.3)

    # Extended Axes Lines
    scale = 100
    ax.plot([c1 - scale*b1x, c1 + scale*b1x], 
            [c2 - scale*b1y, c2 + scale*b1y], 
            color='green', alpha=0.4, linewidth=1, linestyle='--')
    ax.plot([c1 - scale*b2x, c1 + scale*b2x], 
            [c2 - scale*b2y, c2 + scale*b2y], 
            color='green', alpha=0.4, linewidth=1, linestyle='--')

    def draw_arrow(vec, start, color, label, width=0.006):
        ax.quiver(start[0], start[1], vec[0], vec[1], 
                  angles='xy', scale_units='xy', scale=1, 
                  color=color, label=label, width=width)

    if not is_centered:
        draw_arrow(origin_pos, [0,0], 'gray', 'Origin Shift', width=0.004)
        ax.text(c1, c2, " New Origin", color='green', fontsize=8)

    draw_arrow([b1x, b1y], origin_pos, 'green', 'Basis 1', width=0.01)
    draw_arrow([b2x, b2y], origin_pos, 'green', 'Basis 2', width=0.01)
    draw_arrow(v_global, [0,0], 'blue', 'Global Vector', width=0.015)

    ax.text(vx, vy, f" P({vx},{vy})", color='blue', fontweight='bold')

    plt.legend(loc='lower right')
    plt.title("Affine Transformation Simulation")
    st.pyplot(fig)

