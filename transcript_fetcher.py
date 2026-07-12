"""
transcript_fetcher.py
----------------------
Extracts the video ID from a YouTube URL and pulls the full transcript
using youtube-transcript-api. Handles videos with auto-generated or
manually uploaded captions.
"""

import re
from youtube_transcript_api import YouTubeTranscriptApi

try:
    from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
except ImportError:
    from youtube_transcript_api import TranscriptsDisabled, NoTranscriptFound


def extract_video_id(url_or_id: str) -> str:
    """Accepts a full YouTube URL or a bare video ID and returns the video ID."""
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"youtu\.be\/([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    # Fallback: assume it's already a bare video ID
    if len(url_or_id) == 11:
        return url_or_id
    raise ValueError(f"Could not extract a valid video ID from: {url_or_id}")


def _fetch_raw_transcript(video_id: str, languages: list):
    """
    Handles both the old (<1.0) static API and the new (>=1.0) instance-based API,
    since youtube-transcript-api changed its interface between versions.
    """
    try:
        # Old API (<1.0): static method
        return YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
    except AttributeError:
        # New API (>=1.0): instance-based
        ytt_api = YouTubeTranscriptApi()
        fetched = ytt_api.fetch(video_id, languages=languages)
        return [{"text": snippet.text} for snippet in fetched]


def _list_transcripts(video_id: str):
    """Handles both old and new API for listing all available transcripts."""
    try:
        return YouTubeTranscriptApi.list_transcripts(video_id)
    except AttributeError:
        ytt_api = YouTubeTranscriptApi()
        return ytt_api.list(video_id)


def fetch_transcript(url_or_id: str, languages=("en",)) -> str:
    """
    Fetches transcript text for a given video.
    Returns a single concatenated string of the full transcript.

    Fallback order:
      1. Manually created or auto-generated transcript in a requested language
      2. Any available transcript, translated to English if translation is supported
      3. Raises a clear error if neither is possible
    """
    video_id = extract_video_id(url_or_id)

    try:
        transcript_list = _fetch_raw_transcript(video_id, list(languages))
        return " ".join(segment["text"] for segment in transcript_list)
    except (TranscriptsDisabled, NoTranscriptFound):
        pass  # fall through to the broader search below

    # Broader search: look at every transcript available for this video
    try:
        transcript_catalog = _list_transcripts(video_id)
    except TranscriptsDisabled as e:
        raise RuntimeError(f"Transcripts are disabled for video {video_id}: {e}")

    # Try translating any translatable transcript to English
    for transcript in transcript_catalog:
        if transcript.is_translatable:
            try:
                translated = transcript.translate("en").fetch()
                return " ".join(
                    (s.text if hasattr(s, "text") else s["text"]) for s in translated
                )
            except Exception:
                continue

    # Last resort: just take whatever transcript exists, in its original language
    for transcript in transcript_catalog:
        fetched = transcript.fetch()
        return " ".join(
            (s.text if hasattr(s, "text") else s["text"]) for s in fetched
        )

    raise RuntimeError(f"No transcript could be retrieved for video {video_id}.")


def fetch_transcript_with_timestamps(url_or_id: str, languages=("en",)):
    """
    Returns transcript segments with timestamps intact.
    Useful if you later want to link blog sections back to video moments.
    """
    video_id = extract_video_id(url_or_id)
    return YouTubeTranscriptApi.get_transcript(video_id, languages=list(languages))


if __name__ == "__main__":
    # Quick manual test
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    print("Video ID:", extract_video_id(test_url))