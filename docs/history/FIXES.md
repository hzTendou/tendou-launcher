# Tattoo Studio fixes

- Tattoo material no longer switches into wireframe when selected; the debug edge overlay is only shown while Debug mode is enabled.
- Surface and limb wrap raycasts no longer fabricate fallback vertices when a ray misses. Invalid samples are excluded from triangle construction, preventing long spikes and warped geometry at silhouettes.
- Limb wrapping now uses the actual hit surface normal for the skin offset, instead of the original anchor normal, so the decal stays outside the skin around the whole circumference.
- Wrap angle is clamped just below 360° to avoid self-overlap at the seam.
- Chest/torso regions use the surface-conformal projection rather than the bone-cylinder wrapper, which is more stable on broad torso surfaces.
- Tattoo dimensions preserve the source image aspect ratio by default while still allowing intentional non-uniform X/Y scaling.
- Clicking empty canvas space or the floor deselects the active tattoo.

Validation: the changed TypeScript/TSX files pass TypeScript transpilation/parsing. Full `npm run build` could not be completed in this environment because dependency installation timed out and left type-definition packages incomplete; direct TypeScript transpile/parse checks passed for the changed source files.


### v3 artwork visibility fix
- Tattoo artwork is now loaded into a real transparent `CanvasTexture` and the white/background pixels are keyed out at render load time as a final safety net.
- The renderer no longer relies on `MultiplyBlending` for the tattoo itself. Clean alpha + `NormalBlending` keeps the artwork visible while transparent pixels reveal the skin underneath.
- The initial texture is transparent instead of a white/black placeholder, so a slow or failed image load cannot create a black rectangle.
- High-quality mipmaps/anisotropy are enabled on the final artwork texture.
- Full-limb wraps now travel around at least 92% of the limb circumference (with a tiny seam gap), instead of collapsing to a small arc when the source image is narrow.
