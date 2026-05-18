import { ImageResponse } from "next/og";

export const size = { width: 512, height: 512 };
export const contentType = "image/png";

export default function Icon() {
  return new ImageResponse(
    (
      <svg
        width="512"
        height="512"
        viewBox="0 0 512 512"
        role="img"
        aria-label="Logo ChantiFlow"
      >
        <rect width="512" height="512" fill="#161512" />
        <rect
          x="80"
          y="80"
          width="352"
          height="352"
          fill="#161512"
          stroke="#F1EBDF"
          strokeWidth="16"
          transform="rotate(-3 256 256)"
        />
        <text
          x="256"
          y="340"
          textAnchor="middle"
          fontFamily="monospace"
          fontSize="240"
          fontWeight="500"
          fill="#F1EBDF"
        >
          C
        </text>
      </svg>
    ),
  );
}
