#!/nix/store/90nk33c4fkyg4x4dfk5cykqiryf2nlqq-bash-interactive-5.3p15/bin/bash
set -euo pipefail
BASE="/nix/store/0vqb1mcas5j8dv6bhbrshinlgsg6bvgi-gcc-15.3.0-lib/lib:/nix/store/wi6kaycsa2479qrxv9xy6yg3q5ggfs6j-gcc-15.3.0-libgcc/lib:/nix/store/zks9mfsn4rqr6z9g6pcj2xqzcsplj0nb-zlib-1.3.2/lib"
EXTRA="/nix/store/nayywgvfsgzi1pma0a7c64pqz3zrz7vb-libxcb-1.17.0/lib:/nix/store/skiv9g443q6b5zanvnih7xsgnhwlc0v2-libxau-1.0.12/lib:/nix/store/w2dvvgma5r0iyhh53iq0yxxxgjikwdhb-libxdmcp-1.1.5/lib:/nix/store/dwc1r464zf5379jr69vv9gl84h28bzc0-libglvnd-1.7.0/lib:/nix/store/a3hr0l5skscvbkcr7kz3nhi4linz1p71-glib-2.88.3/lib"
export LD_PRELOAD="/nix/store/wi6kaycsa2479qrxv9xy6yg3q5ggfs6j-gcc-15.3.0-libgcc/lib/libgcc_s.so.1"
exec /nix/store/n51dhmdbik1kfrsm62j5knavmigwrl1a-glibc-2.42-84/lib/ld-linux-x86-64.so.2 --library-path "$BASE:$EXTRA" /home/yuzy/ren/.venv/bin/python "$@"
