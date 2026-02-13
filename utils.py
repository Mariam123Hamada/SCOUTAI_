import cv2
import numpy as np

def draw_analytics_panel(frame, current_speed, lean_angle, touches, gate_speed=None, turn_efficiency=None):
    # Ensure all values are numbers
    current_speed = 0.0 if current_speed is None else current_speed
    lean_angle = 0.0 if lean_angle is None else lean_angle
    touches = 0 if touches is None else touches
    gate_speed = 0.0 if gate_speed is None else gate_speed
    turn_efficiency = 0.0 if turn_efficiency is None else turn_efficiency

    # Panel position and size
    panel_x1, panel_y1 = 10, 10
    panel_width, panel_height = 220, 140
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

    # Draw all metrics
    cv2.putText(frame, f"Speed: {current_speed:.2f} m/s", (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick)
    y_text += line_gap
    cv2.putText(frame, f"Lean: {lean_angle:.1f} deg", (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, font_scale, lean_color, font_thick)
    y_text += line_gap
    cv2.putText(frame, f"Touches: {touches}", (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thick)
    y_text += line_gap
    cv2.putText(frame, f"Gate Speed: {gate_speed:.1f} m/s", (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), font_thick)
    y_text += line_gap
    cv2.putText(frame, f"Turn Efficiency: {turn_efficiency:.2f} s", (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 255), font_thick)

    return frame
