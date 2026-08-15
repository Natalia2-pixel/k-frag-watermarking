"""Neural models for the clean K-FRAG watermarking baseline."""

from .clean_encoder import CleanWatermarkEncoder
from .clean_system import CleanWatermarkSystem
from .message_projector import MessageProjector
from .regional_decoder import RegionalDecoder
from .regional_carrier import RegionalCarrierBank, StructuredChannelSystem, StructuredRegionalDecoder

__all__ = [
    "MessageProjector",
    "CleanWatermarkEncoder",
    "RegionalDecoder",
    "CleanWatermarkSystem",
    "RegionalCarrierBank",
    "StructuredRegionalDecoder",
    "StructuredChannelSystem",
]
