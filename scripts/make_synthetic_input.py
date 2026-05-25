from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    out_dir = Path("examples")
    out_dir.mkdir(exist_ok=True)
    image = np.zeros((360, 520, 3), dtype=np.uint8)

    cv2.putText(
        image,
        "upper artifact photo placeholder",
        (40, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (70, 70, 70),
        2,
        cv2.LINE_AA,
    )
    cv2.rectangle(image, (20, 190), (500, 340), (18, 18, 18), -1)
    cv2.line(image, (70, 270), (450, 270), (230, 230, 230), 4, cv2.LINE_AA)
    cv2.line(image, (130, 230), (130, 315), (210, 210, 210), 3, cv2.LINE_AA)
    cv2.ellipse(image, (300, 270), (70, 35), 0, 15, 330, (190, 190, 190), 3, cv2.LINE_AA)
    cv2.polylines(
        image,
        [np.array([(90, 305), (150, 286), (230, 310), (345, 295), (420, 320)])],
        False,
        (235, 235, 235),
        3,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(out_dir / "synthetic_panel.png"), image)


if __name__ == "__main__":
    main()
