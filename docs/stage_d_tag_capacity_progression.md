# Stage-D P0-P4 tag-capacity progression

This isolated child stage loads only the validated real-COCO 12-bit checkpoint with SHA-256 `37B312A8FCCB93F23D5A519BB51EDFCA105A962BD9C4ECA24C395826C91BCC0A`. P0 reproduces its index and RS pathway exactly. P1, P2, P3, and P4 add the first 8, 16, 24, and 32 canonical HMAC tag bits. Future tag bits are excluded from carrier routing, loss, gradients, and metrics.

All parent parameters remain frozen. A shared regional tag carrier and blind tag head are trained with field-aware BCE and parent-logit distillation. The process stops at the first failed capacity, always saves `last.pt`, conditionally saves `best.pt`, and always keeps Stage E disabled.
