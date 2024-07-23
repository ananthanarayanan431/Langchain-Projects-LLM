
from deepeval.metrics.answer_relevancy import AnswerRelevancy
from langchain_community.callbacks.confident_callback import DeepEvalCallbackHandler
from langchain_openai.llms.base import OpenAI
import constant,os

os.environ['OPENAI_API_KEY']=constant.OPENAI_API_KEY

answer_relevancy=AnswerRelevancyMetric(threshold=0.5)

deepeval_callback=DeepEvalCallbackHandler(
    implementation_name="langchainQuickstart",
    metrics=[answer_relevancy],
)

llm=OpenAI(
    temperature=0.5,
    callbacks=[deepeval_callback],
    verbose=True,
)

output=llm.generate((['What is the capital of India? (no bias at all)']))
print(output)


print(answer_relevancy.is_successful())