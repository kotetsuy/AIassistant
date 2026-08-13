"""Cache-aware streaming session for the NeMo backend.

One `StreamSession` per WebSocket connection. The encoder cache (`cache_last_*`)
and the partial hypotheses are per-session state; only the model's forward pass
is shared, and the backend's lock guards that.

Audio arrives as raw 16kHz mono float32 PCM. It is appended to NeMo's
`CacheAwareStreamingAudioBuffer`, which turns it into mel frames, and consumed
one encoder chunk at a time. A chunk is only processed once a full one is
available — feeding a short chunk mid-stream would desynchronise the cache.
"""

from __future__ import annotations

import logging

import numpy as np

from .nemo import LANG_TAG_RE

logger = logging.getLogger("uvicorn.error")

SAMPLE_RATE = 16000


class StreamSession:
    """Incremental decoding state for one utterance."""

    def __init__(self, model, lock, tail_pad_sec: float = 0.5) -> None:
        from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

        self._model = model
        self._lock = lock
        self._tail_pad_sec = tail_pad_sec

        # online_normalization=True makes the mel features themselves un-normalized
        # (normalization happens per encoder chunk inside the buffer's iterator).
        # That is what keeps already-computed frames stable as more audio arrives,
        # which _append relies on. With offline normalization the statistics span
        # the whole utterance, so every earlier frame would shift on each append —
        # inherently incompatible with streaming.
        self._buffer = CacheAwareStreamingAudioBuffer(model, online_normalization=True)
        self._pcm = np.zeros(0, dtype=np.float32)
        (
            self._cache_last_channel,
            self._cache_last_time,
            self._cache_last_channel_len,
        ) = model.encoder.get_initial_cache_state(batch_size=1)

        self._previous_hypotheses = None
        self._pred_out_stream = None
        self._step_num = 0
        self._stream_id = -1  # -1 creates the stream on the first append
        self._text = ""
        self._closed = False

    @property
    def text(self) -> str:
        return self._text

    def add_audio(self, pcm: np.ndarray) -> str:
        """Append PCM and decode every complete chunk it made available.

        Returns the transcript so far (cumulative, not a delta)."""
        if self._closed:
            raise RuntimeError("session already finished")
        if pcm.size:
            self._append(pcm)
            self._drain(final=False)
        return self._text

    def finish(self) -> str:
        """Flush with trailing silence and return the final transcript.

        Cache-aware streaming cannot emit the last token without right context,
        and push-to-talk recordings stop the instant the button is released, so
        the padding is what keeps the sentence-final particle from being lost.
        """
        if self._closed:
            return self._text
        pad = np.zeros(int(SAMPLE_RATE * self._tail_pad_sec), dtype=np.float32)
        self._append(pad)
        self._drain(final=True)
        self._closed = True
        return self._text

    def _append(self, pcm: np.ndarray) -> None:
        """Grow the mel buffer by recomputing it over all audio received so far.

        Feeding `append_audio` one slice at a time looks cheaper, but the mel
        preprocessor pads each call independently, so every slice boundary gets a
        window artifact — that cost us a word ("川は" decoded as "かは") in
        testing. Recomputing from the accumulated PCM keeps the frames identical
        to the offline path; mel over a few seconds of audio is negligible next
        to the encoder step.
        """
        import torch

        self._pcm = np.concatenate([self._pcm, np.ascontiguousarray(pcm, dtype=np.float32)])
        processed, processed_len = self._buffer.preprocess_audio(self._pcm)

        # Replace the buffer contents in place, preserving buffer_idx/step so the
        # already-decoded prefix is not re-fed to the encoder.
        self._buffer.buffer = processed
        self._buffer.streams_length = torch.tensor(
            [processed.size(-1)], device=processed.device
        )
        self._stream_id = 0

    def _drain(self, final: bool) -> None:
        import torch

        from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis

        while True:
            remaining = self._buffer.buffer.size(-1) - self._buffer.buffer_idx
            if remaining <= 0:
                return
            if not final and remaining < self._chunk_size():
                return  # wait for more audio rather than desync the cache

            try:
                chunk_audio, chunk_lengths = next(iter(self._buffer))
            except StopIteration:
                return

            drop = 0 if self._step_num == 0 else self._model.encoder.streaming_cfg.drop_extra_pre_encoded
            with self._lock, torch.inference_mode():
                (
                    self._pred_out_stream,
                    transcribed_texts,
                    self._cache_last_channel,
                    self._cache_last_time,
                    self._cache_last_channel_len,
                    self._previous_hypotheses,
                ) = self._model.conformer_stream_step(
                    processed_signal=chunk_audio,
                    processed_signal_length=chunk_lengths,
                    cache_last_channel=self._cache_last_channel,
                    cache_last_time=self._cache_last_time,
                    cache_last_channel_len=self._cache_last_channel_len,
                    keep_all_outputs=self._buffer.is_buffer_empty(),
                    previous_hypotheses=self._previous_hypotheses,
                    previous_pred_out=self._pred_out_stream,
                    drop_extra_pre_encoded=drop,
                    return_transcription=True,
                )
            self._step_num += 1

            hyp = transcribed_texts[0] if transcribed_texts else ""
            if isinstance(hyp, Hypothesis):
                hyp = hyp.text
            # The model appends "<xx-XX>" after the terminal punctuation, same as
            # in the batch path — strip it here too or it reaches the LLM.
            self._text = LANG_TAG_RE.sub("", hyp or "").strip()

    def _chunk_size(self) -> int:
        """Mirror of CacheAwareStreamingAudioBuffer.__iter__ (pad_and_drop_preencoded=False)."""
        cs = self._buffer.streaming_cfg.chunk_size
        if not isinstance(cs, list):
            return cs
        return cs[0] if self._buffer.buffer_idx == 0 else cs[1]
