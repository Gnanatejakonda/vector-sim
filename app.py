import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 1. Page Config
st.set_page_config(page_title="Basis Transformation Sim", page_icon="📐")
st.title("📐 Affine & Basis Transformation")

st.markdown("""
**Theory Note:** 1. **Origin Shift:** This makes the system *Affine*, not just Linear. We calculate $V_{relative} = V_{global} - Origin$.
2. **Skewed Axes:** If Green axes are not 90°, this is a *General Linear Transformation*, not a pure rotation.
""")

# 2. Sidebar Inputs
with st.sidebar:
    st.header("1. Input Point (Global Coords)")
    # We treat this as a point in space
    vx = st.number_input("Point X", value=3.0, step=0.5)
    vy = st.number_input("Point Y", value=2.0, step=0.5)
    
    st.header("2. New Basis Vectors (Green)")
    st.write("Define the direction of the new axes:")
    b1x = st.number_input("Axis 1 (x) coeff", value=1.0, step=0.1)
    b1y = st.number_input("Axis 1 (y) coeff", value=0.5, step=0.1)
    
    b2x = st.number_input("Axis 2 (x) coeff", value=-0.5, step=0.1)
    b2y = st.number_input("Axis 2 (y) coeff", value=1.0, step=0.1)
    
    st.header("3. Origin Location")
    st.write("Where is the (0,0) of the new system located?")
    c1 = st.number_input("Origin X", value=1.0, step=0.5)
    c2 = st.number_input("Origin Y", value=1.0, step=0.5)

# 3. Calculations
v_global = np.array([vx, vy])       # The point in global space
origin_pos = np.array([c1, c2])     # The location of the new origin
basis_matrix = np.array([[b1x, b2x], 
                         [b1y, b2y]])

# LINEARITY CHECK: Calculate Determinant
det = np.linalg.det(basis_matrix)

# ORTHOGONALITY CHECK: Are axes perpendicular?
# Dot product should be 0
dot_prod = np.dot([b1x, b1y], [b2x, b2y])
is_orthogonal = abs(dot_prod) < 1e-5

if abs(det) < 1e-10:
    st.error("⚠️ LINEAR DEPENDENCE ERROR: The basis vectors are parallel. They form a line, not a 2D plane.")
else:
    # --- MATH CORRECTION FOR ORIGIN SHIFT ---
    # Step 1: Translate (Affine part)
    # We find the vector relative to the new origin
    v_relative = v_global - origin_pos
    
    # Step 2: Change Basis (Linear part)
    # Solve: Basis * v_new = v_relative
    v_new_coords = np.linalg.solve(basis_matrix, v_relative)

    # 4. Display Results
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Global Point (Blue):**\n\n ({vx}, {vy})")
        st.write(f"**Shifted Vector (relative to Green Origin):**\n\n ({v_relative[0]:.2f}, {v_relative[1]:.2f})")
        
    with col2:
        st.success(f"**New Basis Coords (Green):**\n\n [{v_new_coords[0]:.2f}, {v_new_coords[1]:.2f}]")
        
        if is_orthogonal:
            st.write("✅ **Axes are Orthogonal (90°)**")
        else:
            st.warning("⚠️ **Axes are SKEWED (Not 90°)**")
            st.write("Standard Rotation Matrix does not apply.")

        st.write("**Basis Matrix B:**")
        st.latex(r"B = \begin{bmatrix} " + f"{b1x:.2f} & {b2x:.2f} \\\\ {b1y:.2f} & {b2y:.2f}" + r" \end{bmatrix}")

    # 5. Visualization
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Dynamic zoom limits
    limit = max(np.linalg.norm(v_global), np.linalg.norm(origin_pos), 4.0) * 1.5
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect('equal')
    
    # Global Grid (Standard) - Faint Gray
    ax.axhline(0, color='lightgray', linewidth=1)
    ax.axvline(0, color='lightgray', linewidth=1)
    ax.grid(True, linestyle=':', alpha=0.3)

    # --- DRAWING NEW AXES EXTENDED ---
    # We draw lines passing through the New Origin (c1, c2)
    scale = 100
    # Axis 1 Line (Green)
    ax.plot([c1 - scale*b1x, c1 + scale*b1x], 
            [c2 - scale*b1y, c2 + scale*b1y], 
            color='green', alpha=0.4, linewidth=1, linestyle='--')
    # Axis 2 Line (Green)
    ax.plot([c1 - scale*b2x, c1 + scale*b2x], 
            [c2 - scale*b2y, c2 + scale*b2y], 
            color='green', alpha=0.4, linewidth=1, linestyle='--')

    # --- ARROW PLOTTING ---
    def draw_arrow(vec, start, color, label, width=0.006):
        ax.quiver(start[0], start[1], vec[0], vec[1], 
                  angles='xy', scale_units='xy', scale=1, 
                  color=color, label=label, width=width)

    # 1. Draw Origin Shift Vector (Gray)
    # Shows the distance from Global (0,0) to New Origin
    if np.linalg.norm(origin_pos) > 0.1:
        draw_arrow(origin_pos, [0,0], 'gray', 'Origin Shift', width=0.004)
        ax.text(c1, c2, " New Origin", color='green', fontsize=8)

    # 2. Draw New Basis Vectors (Short, starting at New Origin)
    draw_arrow([b1x, b1y], origin_pos, 'green', 'Basis 1', width=0.01)
    draw_arrow([b2x, b2y], origin_pos, 'green', 'Basis 2', width=0.01)

    # 3. Draw The Vector Point
    # A) The Blue Vector from Global Origin (0,0) to Point
    draw_arrow(v_global, [0,0], 'blue', 'Global Vector', width=0.015)
    
    # B) The Relative Vector (from New Origin to Point) - Dashed or distinct
    # We draw this effectively by overlapping, but visually it connects Green Origin to Tip
    # ax.plot([c1, vx], [c2, vy], 'k:', alpha=0.5, label='Relative Component')

    # Annotations
    ax.text(vx, vy, f" P({vx},{vy})", color='blue', fontweight='bold')

    plt.legend(loc='lower right')
    plt.title("Affine Transformation Simulation")
    st.pyplot(fig)
