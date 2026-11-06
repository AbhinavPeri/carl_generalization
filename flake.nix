{
  description = "carl_generalization: reproducible environment for the CARL clones notebook";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});

      # An FHS shell, so the manylinux wheels uv installs (jax, CUDA libraries, ...) find libstdc++, zlib, ...
      # at the usual paths.
      fhs = pkgs: pkgs.buildFHSEnv {
        name = "carl-generalization";
        targetPkgs = p: with p; [ uv python312 git stdenv.cc.cc.lib zlib ];
        profile = ''
          export LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH   # NixOS: libcuda of the NVIDIA driver
          export UV_PYTHON_DOWNLOADS=never
          export UV_PYTHON=python3.12
        '';
        runScript = "bash";
      };
    in {
      packages = forAllSystems (pkgs: { default = fhs pkgs; });
      devShells = forAllSystems (pkgs: { default = pkgs.mkShell { packages = [ (fhs pkgs) ]; }; });
    };
}
