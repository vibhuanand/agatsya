import json
from pathlib import Path

base_payload_path = Path("samples/create_episode_payload_base.json")
transcript_path = Path("input/test_transcript.txt")
output_path = Path("samples/create_episode_payload.json")

payload = json.loads(base_payload_path.read_text(encoding="utf-8"))
payload["raw_transcript"] = transcript_path.read_text(encoding="utf-8")

output_path.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print(f"Created {output_path}")
print(f"Transcript chars: {len(payload['raw_transcript'])}")

#import json
# from pathlib import Path

# transcript_path = Path("input/meika_transcript.txt")
# output_path = Path("samples/create_episode_full_payload.json")

# if not transcript_path.exists():
#     raise FileNotFoundError(f"Transcript not found: {transcript_path}")

# transcript = transcript_path.read_text(encoding="utf-8")

# payload = {
#     "youtube_url": "https://www.youtube.com/watch?v=5bttM6SYuLE",
#     "episode_number": "001",
#     "case_hint": "Meika Jordan",
#     "target_duration_min": 22,
#     "cost_mode": "premium",
#     "package_level": "script_first",
#     "style": "Agatsya Anand pure Hindi respectful dark true crime",
#     "enable_gpt_review": False,
#     "hinglish_level": 2,
#     "raw_transcript": transcript
# }

# output_path.parent.mkdir(parents=True, exist_ok=True)
# output_path.write_text(
#     json.dumps(payload, ensure_ascii=False, indent=2),
#     encoding="utf-8"
# )

# print(f"Created: {output_path}")
# print(f"Transcript characters: {len(transcript)}")
# print(f"Payload characters: {len(output_path.read_text(encoding='utf-8'))}")