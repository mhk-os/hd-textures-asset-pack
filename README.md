# hd-textures-asset-pack

Asset pack for Moorhuhn Kart: Open Source Edition with sharper, more detailed track textures for Moorhuhn Kart 2, Moorhuhn Kart 3, Moorhuhn Kart Extra, Moorhuhn Kart Thunder and TINcan Race. Install it from the Asset Packs section of the Mods screen.

## Textures

The pack replaces the textures of every track (road, ground, walls, props, signs and skies) and the shared track textures of each game. Textures of 256 px and less are four times as large, 512 px textures twice; textures of 1024 px and more are already sharp and stay as they are. Each texture keeps the colours and layout of the original, so it lines up with the meshes and the lighting of the tracks. Where a track repeats a texture whose original showed a seam at the repeat, the seam is healed.

`Tools/manifest.json` lists every texture with its source size, its size in the pack, the axes along which the tracks repeat it and how it was made.

## Rebuilding

The textures are built from the mods' bundles and loose files with the scripts in `Tools/` (Python 3.12, [uv](https://docs.astral.sh/uv/)):

```sh
cd Tools
uv run build.py                                   # every core mod, from the local builds in ../../mhk-os-<mod>-mod
uv run build.py --mods mhk2 --mod mhk2=/path/to/Mods/mhk2 --only 'mhk2/Tracks/mhk2_egypt/*'
```

A mod folder is what a mod build, an installed mod or a mod release zip holds: `<mod>.mhkmod`, `<mod>.scenes.mhkmod` and `Assets/`. Per texture the build:

1. reads from the meshes and materials in the bundles along which axes the tracks repeat the texture (`uvs.py`),
2. heals a seam at a repeat by patching it with the stretch of the texture that continues both sides best (`seamless.py`), where the seam is the hardest edge of the texture; drawn edges such as outlines, window frames or beams stay, and so do the seams of the textures in `keep-seams.txt`, whose border content is part of the picture,
3. upscales the texture 4x with [4xNomos8kSC](https://github.com/Phhofm/models/releases/tag/4xNomos8kSC) (`upscale.py`, the weights download on first use), wrapping around the repeated axes so the result still repeats, and keeps the broad colours of the original,
4. writes a JPG, or a PNG where the texture uses alpha, to `Assets/<virtual path>` and records it in the manifest. Unchanged textures are skipped.

A remaster from an image model replaces the upscale: put it at `Tools/generated/<virtual path>.png` (for example `Tools/generated/mhk2/Tracks/mhk2_egypt/Textures/sandroad.png`, any size). The build takes over the colours of the original and joins its ends along the repeated axes, so a generated texture that does not tile on its own tiles in the game.

`seamless.py` also works on its own. It joins the ends of a texture along the cheapest cut through an overlap of the image with itself, heals the seam in place, or crossfades like most online seamless texture makers:

```sh
uv run seamless.py in.png out.png --axes uv --overlap 0.2 --size 1024 --preview tiled.png    # cut, for generated textures
uv run seamless.py in.png out.png --axes v --overlap 0.12 --method heal                     # same size and scale
```

## License

MIT. Game assets belong to their respective owners. If there is a problem with any of them, message me and I will remove it. The upscaling model 4xNomos8kSC is by Philip Hofmann, CC-BY-4.0.
