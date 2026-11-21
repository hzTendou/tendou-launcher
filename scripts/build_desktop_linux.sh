#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "============================================================================"
echo "  TENDOU STUDIO v0.16.3 — LINUX TAURI v2 DESKTOP BUILD"
echo "============================================================================"

# 1. Verify GTK / WebKit2GTK / AppIndicator system libraries
for pkg in webkit2gtk-4.1 ayatana-appindicator3-0.1 librsvg-2.0; do
    if ! pkg-config --exists "$pkg"; then
        echo "[ERROR] Missing Linux system library: $pkg"
        echo "        Install with: sudo apt install libwebkit2gtk-4.1-dev libayatana-appindicator3-dev librsvg2-dev"
        exit 1
    fi
done

# 2. Build Tendou Studio Web UI bundle (ui/dist)
echo "[1/3] Building Tendou Studio React+TS frontend (ui/dist)..."
npm --prefix ui install --no-audit --no-fund
npm --prefix ui run build

# 3. Compile Tauri v2 desktop binary and package .deb bundle
echo "[2/3] Building Rust Tauri v2 desktop application (.deb bundle)..."
if command -v cargo-tauri >/dev/null 2>&1 || cargo tauri --version >/dev/null 2>&1; then
    (cd desktop/src-tauri && cargo tauri build --bundles deb)
elif command -v npx >/dev/null 2>&1 && (cd desktop && npx --yes @tauri-apps/cli@2 --version >/dev/null 2>&1); then
    echo "[INFO] cargo-tauri not found; packaging .deb via npx --yes @tauri-apps/cli@2..."
    (cd desktop && npx --yes @tauri-apps/cli@2 build --bundles deb)
else
    echo "[INFO] Neither cargo-tauri nor reachable @tauri-apps/cli@2 found; building optimized release binary via cargo build --release..."
    (cd desktop/src-tauri && cargo build --release)
fi

echo "[3/3] Build complete!"
ls -lh desktop/src-tauri/target/release/tendou-studio 2>/dev/null || true
ls -lh desktop/src-tauri/target/release/bundle/deb/*.deb 2>/dev/null || true
