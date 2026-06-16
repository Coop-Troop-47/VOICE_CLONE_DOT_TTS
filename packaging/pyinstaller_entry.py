from __future__ import annotations

import os

from voice_clone_dot_tts.main import main


if __name__ == "__main__":
    if os.environ.get("VOICE_CLONE_DOT_TTS_IMPORT_CHECK") == "1":
        from dots_tts.runtime import DotsTtsRuntime

        print(f"dots_tts.runtime import ok: {DotsTtsRuntime!r}")
        raise SystemExit(0)
    if os.environ.get("VOICE_CLONE_DOT_TTS_MLX_IMPORT_CHECK") == "1":
        import mlx.core as mx
        from dots_tts_mlx.loader import from_pretrained

        print(f"dots_tts_mlx.loader import ok: {from_pretrained!r}; mlx default dtype: {mx.bfloat16}")
        raise SystemExit(0)
    raise SystemExit(main())
