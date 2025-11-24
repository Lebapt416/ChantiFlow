import { ImageResponse } from "next/og";

export const size = {
  width: 256,
  height: 256,
};

export const contentType = "image/png";

export default function Icon() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: "#050505",
        }}
      >
        <div
          style={{
            width: 180,
            height: 180,
            borderRadius: 48,
            background: "#fff",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <div
            style={{
              fontSize: 140,
              fontWeight: 700,
              color: "#050505",
              fontFamily: "SF Pro Display, Inter, sans-serif",
            }}
          >
            ⚡️
          </div>
        </div>
      </div>
    ),
  );
}

