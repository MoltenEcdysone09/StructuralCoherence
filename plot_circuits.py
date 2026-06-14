import os
import glob
import math
import pandas as pd
import subprocess

# --- Configuration & Styling ---
INPUT_DIR = "./AllUniqueNets/Topologies"
OUTPUT_DIR_SVG = "./AllUniqueNets/Plots_SVG"
OUTPUT_DIR_PNG = "./AllUniqueNets/Plots_PNG"  # Pass this variable to generate_svg() below to enable PNGs

COLOR_ACT = "#5C82B6"
COLOR_INH = "#D05A5A"
COLOR_ZERO = "#999999"
COLOR_NODE_BORDER = "#363942"
NODE_FILL = "white"

NODE_RADIUS = 22
STROKE_WIDTH = 3.5

# SVG Marker Definitions
MARKER_DEFS = f"""
  <defs>
    <marker id="arrow_act" viewBox="0 0 12 12" refX="8" refY="6" 
            markerWidth="5" markerHeight="5" orient="auto-start-reverse">
      <path d="M 2 2 L 9 6 L 2 10 z" fill="{COLOR_ACT}" stroke="{COLOR_ACT}" stroke-width="2" stroke-linejoin="round" />
    </marker>
    
    <marker id="arrow_zero" viewBox="0 0 12 12" refX="8" refY="6" 
            markerWidth="5" markerHeight="5" orient="auto-start-reverse">
      <path d="M 2 2 L 9 6 L 2 10 z" fill="{COLOR_ZERO}" stroke="{COLOR_ZERO}" stroke-width="2" stroke-linejoin="round" />
    </marker>

    <marker id="bar_inh" viewBox="0 0 12 24" refX="5" refY="12" 
            markerWidth="4.5" markerHeight="6.5" orient="auto">
      <line x1="5" y1="5" x2="5" y2="19" stroke="{COLOR_INH}" stroke-width="4" stroke-linecap="round" />
    </marker>
  </defs>
"""


def get_node_positions(nodes):
    if len(nodes) == 2:
        return {
            nodes[0]: {"x": 100, "y": 150, "angle": 180},
            nodes[1]: {"x": 250, "y": 150, "angle": 0},
        }
    elif len(nodes) == 3:
        return {
            nodes[0]: {"x": 175, "y": 80, "angle": 270},
            nodes[1]: {"x": 100, "y": 210, "angle": 150},
            nodes[2]: {"x": 250, "y": 210, "angle": 30},
        }
    return {}


def calculate_parallel_edge_points(x1, y1, x2, y2, radius_offset=27, parallel_shift=10):
    dx = x2 - x1
    dy = y2 - y1
    dist = math.hypot(dx, dy)

    if dist == 0:
        return x1, y1, x2, y2

    ux = dx / dist
    uy = dy / dist

    px = -uy
    py = ux

    cx1 = x1 + px * parallel_shift
    cy1 = y1 + py * parallel_shift
    cx2 = x2 + px * parallel_shift
    cy2 = y2 + py * parallel_shift

    start_x = cx1 + ux * radius_offset
    start_y = cy1 + uy * radius_offset
    end_x = cx2 - ux * radius_offset
    end_y = cy2 - uy * radius_offset

    return start_x, start_y, end_x, end_y


def generate_svg(filepath, output_dir_svg, output_dir_png=None):
    df = pd.read_csv(filepath, sep=r"\s+")
    df.columns = [col.strip() for col in df.columns]

    nodes = sorted(list(set(df["Source"].tolist() + df["Target"].tolist())))
    pos = get_node_positions(nodes)

    svg_elements = []

    for _, row in df.iterrows():
        source = row["Source"]
        target = row["Target"]
        edge_type = row["Type"]

        if edge_type == 1:
            color, dash, marker = COLOR_ACT, "none", "url(#arrow_act)"
        elif edge_type == -1:
            color, dash, marker = COLOR_INH, "none", "url(#bar_inh)"
        else:
            color, dash, marker = COLOR_ZERO, "1, 6", "url(#arrow_zero)"

        if source == target:
            nx = pos[source]["x"]
            ny = pos[source]["y"]
            angle_deg = pos[source]["angle"]

            rad_center = math.radians(angle_deg)
            rad_offset = math.radians(35)

            r_inset = NODE_RADIUS + 7
            x_start = nx + r_inset * math.cos(rad_center - rad_offset)
            y_start = ny + r_inset * math.sin(rad_center - rad_offset)

            x_end = nx + r_inset * math.cos(rad_center + rad_offset)
            y_end = ny + r_inset * math.sin(rad_center + rad_offset)

            arc_radius = 24
            path = f"M {x_start} {y_start} A {arc_radius} {arc_radius} 0 1 1 {x_end} {y_end}"

            svg_elements.append(
                f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{STROKE_WIDTH}" '
                f'stroke-dasharray="{dash}" stroke-linecap="round" stroke-linejoin="round" marker-end="{marker}" />'
            )

        else:
            x1, y1 = pos[source]["x"], pos[source]["y"]
            x2, y2 = pos[target]["x"], pos[target]["y"]

            sx, sy, ex, ey = calculate_parallel_edge_points(x1, y1, x2, y2)

            path = f"M {sx} {sy} L {ex} {ey}"

            svg_elements.append(
                f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{STROKE_WIDTH}" '
                f'stroke-dasharray="{dash}" stroke-linecap="round" stroke-linejoin="round" marker-end="{marker}" />'
            )

    for n in nodes:
        nx = pos[n]["x"]
        ny = pos[n]["y"]
        svg_elements.append(
            f'<circle cx="{nx}" cy="{ny}" r="{NODE_RADIUS}" fill="{NODE_FILL}" '
            f'stroke="{COLOR_NODE_BORDER}" stroke-width="{STROKE_WIDTH}" />'
        )

    svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 350 300" width="350" height="300">
{MARKER_DEFS}
{"".join([f"  {el}\n" for el in svg_elements])}
</svg>"""

    base_filename = os.path.basename(filepath).replace(".topo", "")
    svg_out_path = os.path.join(output_dir_svg, f"{base_filename}.svg")

    with open(svg_out_path, "w") as f:
        f.write(svg_content)

    if output_dir_png:
        png_out_path = os.path.join(output_dir_png, f"{base_filename}.png")
        try:
            subprocess.run(
                [
                    "magick",
                    "-density",
                    "300",
                    "-background",
                    "none",
                    svg_out_path,
                    png_out_path,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            print(
                "Error: The 'magick' command was not found. Please ensure ImageMagick is installed."
            )
        except subprocess.CalledProcessError as e:
            print(f"Error converting {svg_out_path} to PNG: {e}")


if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR_SVG):
        os.makedirs(OUTPUT_DIR_SVG)

    # If generating PNGs, uncomment the directory creation logic
    if not os.path.exists(OUTPUT_DIR_PNG):
        os.makedirs(OUTPUT_DIR_PNG)

    topo_files = glob.glob(os.path.join(INPUT_DIR, "*.topo"))

    for file in topo_files:
        # Pass OUTPUT_DIR_PNG as the third argument to enable PNG generation
        generate_svg(file, OUTPUT_DIR_SVG, OUTPUT_DIR_PNG)
