import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RetinaScope — A clearer vision of what's ahead",
  description: "Explainable AI for diabetic retinopathy screening. A research and demonstration prototype.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
