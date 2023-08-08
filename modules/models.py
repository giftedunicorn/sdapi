from pydantic import BaseModel

class Txt2ImgRequest(BaseModel):
    prompt: str
    steps: int = 40