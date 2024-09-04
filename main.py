import os

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.audio import Transcription
from openai.types.chat import ChatCompletion

load_dotenv()

client: OpenAI = OpenAI(api_key=os.getenv("OPENAI_SECRET_KEY"))

with open("question.mp3", "rb") as audio_file:
    transcript: Transcription = client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-1",
        language="ko",
        response_format="text",
        temperature=0.0
    )

    print(transcript)

completion: ChatCompletion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": "너는 '마법의 소라고둥'처럼 행동하는 인공지능 역할을 맡고 있다. "
                       "사용자가 질문을 할 때, 너의 응답은 항상 짧고 단순하며, 약간 모호해야 한다. "
                       "너는 절대 길거나 복잡한 설명을 제공하지 않고, 답변이 최대한 미니멀하게 유지되도록 한다. "
                       "네가 제공하는 답변은 때때로 직관적이지 않을 수 있다. "
                       "하지만 사용자의 질문이 아무리 구체적이거나 긴박하더라도, 너는 항상 차분하고 일관된 태도로 간단히 대답해야 한다. "
                       "너는 대답할 때 추가적인 정보를 제공하지 않으며, 이유나 배경 설명도 주지 않는다. "
                       "모든 대답은 명확하게 끝마쳐야 하고, 질문의 맥락이나 세부 사항을 언급하지 않는다."
        },
        {
            "role": "user",
            "content": transcript
        }
    ]
)

print(completion.choices[0].message.content)


with client.audio.speech.with_streaming_response.create(
    model="tts-1-hd",
    voice="nova",
    input=completion.choices[0].message.content,
    response_format="mp3",
    speed=0.7
) as response:
    response.stream_to_file("answer.mp3")
    print("saved")
