# Voice Samples

Audio samples demonstrating omnivoice-server output quality.

## Download

Voice samples are stored as GitHub Release assets to keep the repository size small:

- [test_english.wav](https://github.com/maemreyo/omnivoice-server/releases/download/v0.1.0/test_english.wav) - English (Female, American accent) - 199KB
- [test_vietnamese.wav](https://github.com/maemreyo/omnivoice-server/releases/download/v0.1.0/test_vietnamese.wav) - Vietnamese (Female) - 203KB

## Details

Both samples were generated on CPU with:
- Device: cpu
- num_step: 32
- Quality: Clear, natural speech
- RTF: 4.92 (slower than real-time on CPU, expected)

For production deployments, use CUDA GPU for 20-25x faster synthesis (RTF ~0.2).
