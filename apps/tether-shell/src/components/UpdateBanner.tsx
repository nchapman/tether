import { useEffect, useState } from "react";
import { checkForUpdates, installUpdate } from "../ipc";

// Asks the backend for an available update on mount; if there is one, shows a
// banner whose button downloads + installs the signed bundle and restarts into
// it — so a successful install never returns here.
export function UpdateBanner() {
  const [version, setVersion] = useState<string | null>(null);
  const [installing, setInstalling] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    checkForUpdates()
      .then(setVersion)
      .catch((e) => console.warn("update check failed", e));
  }, []);

  if (!version) return null;

  async function install() {
    setError(null);
    setInstalling(true);
    try {
      await installUpdate();
    } catch (e) {
      setError(String(e));
      setInstalling(false);
    }
  }

  return (
    <div className="update-banner">
      <span>
        Update <strong>{version}</strong> available
      </span>
      <button className="btn sm" onClick={install} disabled={installing}>
        {installing ? "Installing…" : "Install & restart"}
      </button>
      {error && <span className="form-error">{error}</span>}
    </div>
  );
}
