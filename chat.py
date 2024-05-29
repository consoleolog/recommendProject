from langchain.chat_models import ChatOpenAI
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts import PromptTemplate
from openai import OpenAI
from prompt import examples
import uuid
import urllib.request


prompt = PromptTemplate(
    input_variables=["input","output"],
    template="입력: {input}\n출력: {output}"
)
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=prompt,
    prefix="say just one word",
    suffix="입력: {input_string}\n출력:",
    input_variables=["input_string"],
)
formatted_prompt = few_shot_prompt.format(
    input_string="think about attractive hashtag for instagram feed for today"
)

# 여기서 받을거는 해시태그랑
chat = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    streaming=True,
)
result = chat.predict(formatted_prompt)


make_hashtag_prompt = f"show me draft plan for instagram feed about {result}"

client = OpenAI()

response = client.images.generate(
  model="dall-e-3",
  prompt=make_hashtag_prompt,
  size="1024x1024",
  quality="standard",
  n=1,
)

# 유저 정보
username = ""
follower_count = 15
following_count = 15
media_count = 0

predict_data = open('./data/predict_data.csv','a',encoding='utf-8')
# url 받아서
image_url = response.data[0].url
random_name = uuid.uuid4()
urllib.request.urlretrieve(image_url, "./predict_image/"+str(random_name)+".jpg")
predict_data.write(f'\n"{str(username)}","{str(follower_count)}","{str(following_count)}","{str(media_count)}",{str(random_name)}')
predict_data.close()

