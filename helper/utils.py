import cv2
import numpy as np

def draw_analytics_panel(frame, lean_angle, touches, gate_speed=None, turn_efficiency=None):
    # Ensure all values are numbers
    lean_angle = 0.0 if lean_angle is None else lean_angle
    touches = 0 if touches is None else touches
    gate_speed = 0.0 if gate_speed is None else gate_speed
    turn_efficiency = 0.0 if turn_efficiency is None else turn_efficiency

    # Panel position and size
    panel_x1, panel_y1 = 10, 10
    panel_width, panel_height = 220, 120
    panel_x2 = panel_x1 + panel_width
    panel_y2 = panel_y1 + panel_height

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (50, 50, 50), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Panel border
    cv2.rectangle(frame, (panel_x1, panel_y1), (panel_x2, panel_y2), (0, 255, 255), 1)

    # Text parameters
    x_text = panel_x1 + 10
    y_text = panel_y1 + 25
    line_gap = 25
    font_scale = 0.55
    font_thick = 1

    # Body lean color
    lean_color = (0, 255, 255) if lean_angle > 15 else (0, 255, 0)

    # Draw all metrics (without speed)
    cv2.putText(frame, f"Lean: {lean_angle:.1f} deg", (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, font_scale, lean_color, font_thick)
    y_text += line_gap
    cv2.putText(frame, f"Touches: {touches}", (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thick)
    y_text += line_gap
    cv2.putText(frame, f"Gate Speed: {gate_speed:.1f} m/s", (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), font_thick)
    y_text += line_gap
    cv2.putText(frame, f"Turn Efficiency: {turn_efficiency:.2f} s", (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 255), font_thick)

    return frame




def draw_player_heatmap(player_positions, stadium_path, output_size=(960, 640) , blur_size=(31, 31), alpha=0.6):
    """
    Draws a heatmap of player positions on a stadium image.

    Args:
        
        player_positions (list of tuples): List of (x, y) positions of the player on the video frame.
        stadium_path : The Staduim image Path.
        output_size : the output image Size.
        blur_size (tuple): Kernel size for Gaussian blur.
        alpha (float): Transparency of stadium image overlay.

    Returns:
        heatmap_overlay (np.array): Stadium image with heatmap overlay.
    """
    stadium_img = cv2.imread(stadium_path)
    if stadium_img is None:
        raise FileNotFoundError(f"Could not load stadium image from {stadium_path}")

    # Resize stadium
    stadium_img = cv2.resize(stadium_img, output_size)
    stadium_h, stadium_w, _ = stadium_img.shape
   
    

   
    heatmap = np.zeros((stadium_h, stadium_w), dtype=np.float32)

    
    for x, y in player_positions:
       
        sx = int(x / stadium_w * stadium_w)
        sy = int(y / stadium_h * stadium_h)
        if 0 <= sx < stadium_w and 0 <= sy < stadium_h:
            heatmap[sy, sx] += 1

   
    heatmap = cv2.GaussianBlur(heatmap, blur_size, 0)

    
    heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

    heatmap_overlay = cv2.addWeighted(stadium_img, alpha, heatmap_color, 1 - alpha, 0)

    return heatmap_overlay
