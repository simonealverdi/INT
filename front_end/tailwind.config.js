/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./interview.html",
    "./*.js",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ["SF Pro", "-apple-system", "BlinkMacSystemFont", "Helvetica Neue", "Helvetica", "Arial", "sans-serif"],
      },
      colors: {
        primary: "#0071e3",
        background: "#f5f5f7",
        text: "#1d1d1f",
      },
      borderRadius: {
        xl: "12px",
      },
    },
  },
  plugins: [],
};
