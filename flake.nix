{
  description = "ren — offline-first computational cytometry & cell pathology engine";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true; # rocmPackages pulls in some unfree bits
        };

        # The whole point of this project's inference stack: llama.cpp built
        # against Vulkan, not ROCm. No gfx override dance needed for this half.
        llama-cpp-vulkan = pkgs.llama-cpp.override { vulkanSupport = true; };

        commonPkgs = with pkgs; [
          git
          git-lfs
          cmake
          ninja
          pkg-config
          sqlite
          just
          curl

          llama-cpp-vulkan
          vulkan-tools # vulkaninfo — sanity check the RX 6600 shows up via RADV
          vulkan-loader
          vulkan-validation-layers

          # uv owns the Python venv + deps (pyproject.toml/uv.lock); Nix only
          # supplies the interpreter and system-level shared libs below.
          python312
          uv

          openslide # C lib backing openslide-python, for whole-slide tissue images
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          packages = commonPkgs;

          # uv downloads its own prebuilt Python + wheels (torch, opencv, ...),
          # which are linked against a generic glibc that doesn't exist on
          # NixOS. nix-ld is the shim that makes them runnable; these two vars
          # only work if `programs.nix-ld.enable = true;` is set in your
          # system configuration.nix — the flake can't create the shim itself.
          NIX_LD = "${pkgs.stdenv.cc.libc}/lib/ld-linux-x86-64.so.2";
          NIX_LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath commonPkgs;

          shellHook = ''
            echo "ren dev shell — llama.cpp (Vulkan) + uv-managed Python"
            echo "first time: run 'uv sync', then 'just --list'"
            [ -d .venv ] || uv venv
          '';
        };

        # Only for the vision-engine path: extracting features from a frozen
        # ViT (Path Foundation/Virchow) and training the small classification
        # head on top.
        # PyTorch has no real Vulkan backend, so this is where ROCm comes back.
        devShells.rocm = pkgs.mkShell {
          packages = commonPkgs ++ (with pkgs.rocmPackages; [
            rocminfo
            clr
            hipblas
            rocblas
          ]);

          # RX 6600 is gfx1032 — not in ROCm's officially supported list.
          # Standard spoof: report as gfx1030 (RX 6800/6900, first-class
          # support). If ops still segfault, gfx1031 (10.3.1) is the other
          # value worth trying before giving up and falling back to CPU —
          # the head you're training is small enough that CPU is genuinely
          # viable if ROCm keeps fighting you on this card.
          HSA_OVERRIDE_GFX_VERSION = "10.3.0";

          NIX_LD = "${pkgs.stdenv.cc.libc}/lib/ld-linux-x86-64.so.2";
          NIX_LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (commonPkgs ++ (with pkgs.rocmPackages; [
            clr
            hipblas
            rocblas
          ]));

          shellHook = ''
            echo "ren ROCm shell — gfx1032 spoofed as gfx1030 (HSA_OVERRIDE_GFX_VERSION=10.3.0)"
            echo "only needed for CV feature extraction/training — llama.cpp stays on Vulkan"
            rocminfo 2>/dev/null | grep -i gfx || echo "rocminfo found nothing — check the override / driver"
          '';
        };
      });
}
