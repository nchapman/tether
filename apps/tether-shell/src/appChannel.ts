const APP_CHANNEL = import.meta.env.VITE_TETHER_CHANNEL === "dev" ? "dev" : "release";

const APP_TITLE = APP_CHANNEL === "dev" ? "Tether Dev" : "Tether";
const DEFAULT_PORT = APP_CHANNEL === "dev" ? "7384" : "7374";

export { APP_CHANNEL, APP_TITLE, DEFAULT_PORT };
