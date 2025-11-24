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
        <defs>
          <pattern
            id="diag"
            patternUnits="userSpaceOnUse"
            width="32"
            height="32"
            patternTransform="skewX(-20)"
          >
            <rect width="32" height="32" fill="#050505" />
            <rect width="32" height="16" fill="#0f0f0f" opacity="0.5" />
          </pattern>
          <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow dx="0" dy="12" stdDeviation="16" floodColor="#000" floodOpacity="0.4" />
          </filter>
        </defs>
        <rect width="512" height="512" fill="#050505" />
        <rect width="512" height="512" fill="url(#diag)" opacity="0.5" />
        <g filter="url(#shadow)">
          <circle cx="256" cy="256" r="220" fill="#050505" />
        </g>
        <g filter="url(#shadow)">
          <rect
            x="146"
            y="146"
            width="220"
            height="220"
            rx="56"
            fill="#ffffff"
            stroke="#050505"
            strokeWidth="10"
          />
        </g>
        <path
          d="M230 150 L336 150 L274 268 L348 268 L206 362 L256 246 L186 246 Z"
          fill="#050505"
        />
      </svg>
    ),
  );
}

