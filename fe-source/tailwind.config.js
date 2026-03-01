/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#0B0F14",
        fog: "#EEF2F6",
        tide: "#0F6FFF",
        ember: "#FF6A3D",
        pine: "#0F8B6B",
        haze: "#9DB3C8",
      },
      fontFamily: {
        display: ["'Space Grotesk'", "system-ui", "sans-serif"],
        serif: ["'Newsreader'", "Georgia", "serif"],
      },
      boxShadow: {
        soft: "0 10px 40px rgba(11, 15, 20, 0.15)",
      },
      backgroundImage: {
        "mesh": "radial-gradient(circle at 20% 20%, rgba(15, 111, 255, 0.35), transparent 55%), radial-gradient(circle at 80% 0%, rgba(255, 106, 61, 0.35), transparent 60%), radial-gradient(circle at 30% 80%, rgba(15, 139, 107, 0.35), transparent 60%)",
      },
    },
  },
  plugins: [],
};
