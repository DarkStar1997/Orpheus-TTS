from flask import Flask, Response, request, stream_with_context
import struct, itertools
from orpheus_tts import OrpheusModel

app = Flask(__name__)
engine = OrpheusModel(model_name="canopylabs/orpheus-tts-0.1-finetune-prod")

SAMPLE_RATE = 24000
BITS = 16
CHANNELS = 1
FRAME_BYTES = CHANNELS * (BITS // 8)

def wav_header(sample_rate=SAMPLE_RATE, bits=BITS, channels=CHANNELS):
    """
    Minimal RIFF/WAVE header with unknown data size.
    Many browsers/players will begin playback with data_size=0.
    """
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    data_size = 0  # unknown; streaming
    riff_size = 36 + data_size
    return struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        riff_size,          # 36 + data_size
        b'WAVE',
        b'fmt ',
        16,                 # PCM fmt chunk size
        1,                  # PCM format
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits,
        b'data',
        data_size
    )

def chunker(iterable, n):
    """Coalesce unpredictable generator pieces into ~n-byte chunks to keep players fed."""
    buf = bytearray()
    for piece in iterable:
        if not piece:
            continue
        buf.extend(piece)
        while len(buf) >= n:
            out = bytes(buf[:n])
            del buf[:n]
            yield out
    if buf:
        yield bytes(buf)

@app.route('/tts', methods=['GET'])
def tts():
    prompt = request.args.get('prompt') or 'Hey there, looks like you forgot to provide a prompt!'

    @stream_with_context
    def generate():
        # 1) header first so playback can start ASAP
        yield wav_header()

        # 2) generate speech tokens/bytes and drip-feed them
        syn_tokens = engine.generate_speech(
            prompt=prompt,
            voice="tara",
            repetition_penalty=1.1,
            stop_token_ids=[128258],
            max_tokens=2000,
            temperature=0.4,
            top_p=0.9,
        )

        # 3) coalesce to ~20ms per chunk (tweak for your player)
        # 20 ms of 16-bit mono @ 24kHz ≈ 0.02 * 24000 * 2 = 960 bytes
        for pcm in chunker(syn_tokens, n=960):
            # guard: ensure chunk size is a multiple of frame size
            keep = (len(pcm) // FRAME_BYTES) * FRAME_BYTES
            if keep:
                yield pcm[:keep]

    headers = {
        # Make it “streamy”
        "Content-Type": "audio/wav",
        "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Connection": "keep-alive",
        # Nginx/Cloudflare compatible hint to not buffer
        "X-Accel-Buffering": "no",
        # Helpful when calling from a browser app on a different origin
        "Access-Control-Allow-Origin": "*",
    }
    return Response(generate(), headers=headers, direct_passthrough=True)

if __name__ == '__main__':
    # threaded=True lets one request stream while others are served
    app.run(host='0.0.0.0', port=8080, threaded=True)

