import cv2
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="OpenCV Pixel Inspector")
    parser.add_argument("image_path", help="Path to image file")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"[Error] File not found: {args.image_path}")
        return

    # Load image in BGR
    img = cv2.imread(args.image_path)
    if img is None:
        print("[Error] Failed to load image.")
        return

    window_name = "Pixel Inspector"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    img_display = img.copy()
    info = {"x": 0, "y": 0, "r": 0, "g": 0, "b": 0}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            b, g, r = img[y, x]
            info["x"], info["y"] = x, y
            info["r"], info["g"], info["b"] = r, g, b

    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        img_display = img.copy()
        text = f"Pixel Coordinate({info['x']}, {info['y']}) RGB: ({info['r']}, {info['g']}, {info['b']})"
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

        # Position of text
        x_text = img.shape[1] - text_size[0] - 30
        y_text = 30

        # Draw the color dot (circle)
        cv2.circle(img_display, (x_text - 10, y_text - 5), 6,
                (int(info['b']), int(info['g']), int(info['r'])), -1)

        # Draw the text
        cv2.putText(img_display, text,
                    (x_text, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(window_name, img_display)
        key = cv2.waitKey(10)
        if key == 27:  # ESC to quit
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
