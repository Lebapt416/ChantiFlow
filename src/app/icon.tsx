import { ImageResponse } from "next/og";

export const size = { width: 512, height: 512 };
export const contentType = "image/png";

/**
 * App icon — logo mark Industrial Editorial
 * Carré ink avec lettre C en paper.
 * Next.js ImageResponse ne supporte pas <text> SVG — on utilise
 * un div avec le caractère C via la syntaxe JSX/CSS d'ImageResponse.
 */
export default function Icon() {
  return new ImageResponse(
    (
      <div
        style={{
          width: 512,
          height: 512,
          backgroundColor: "#161512",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          transform: "rotate(-3deg)",
        }}
      >
        <div
          style={{
            color: "#F1EBDF",
            fontSize: 300,
            fontWeight: 500,
            fontFamily: "monospace",
            lineHeight: 1,
            marginTop: 20, // optical centering
          }}
        >
          C
        </div>
      </div>
    ),
    { ...size },
  );
}
