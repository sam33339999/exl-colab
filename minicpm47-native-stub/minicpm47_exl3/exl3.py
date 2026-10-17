def register() -> None:
    """Satisfy TabbyAPI's pre-load import.

    TabbyAPI imports ``minicpm47_exl3.exl3.register`` before
    ``Config.from_directory`` whenever ``config.json`` says
    ``MiniCPMV4_7ForConditionalGeneration``. The local checkout
    ``/home/sam/exllamav3-origin`` (branch ``feat/minicpm-v4.7``) already
    registers that class. This function intentionally does nothing.

    It is not the BlackwellBoy adapter. There is no ``VisionBridge`` here.
    Keep ``vision: false``. Turning vision on makes Tabby import
    ``minicpm47_exl3.bridge``, which this package does not provide.
    Images stay on ``examples/multimodal.py``.
    """
    return
