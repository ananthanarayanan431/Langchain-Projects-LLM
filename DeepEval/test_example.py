
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric,HallucinationMetric
from deepeval.metrics import ContextualRecallMetric,ContextualRelevancyMetric

import constant,os

os.environ['OPENAI_API_KEY']=constant.OPENAI_API_KEY

arm=AnswerRelevancyMetric(threshold=0.5)
ha=HallucinationMetric(threshold=0.5)
cr=ContextualRecallMetric(threshold=0.5)
cm=ContextualRelevancyMetric(threshold=0.5)

test_case=LLMTestCase(
    input="What if there shoes don't fit?",
    actual_output="We don't sell shoes and we cricket IPL team agency",
    expected_output="We offer a 30-day full refund at no extra cost",
    retrieval_context=['All the Customer are eligible for 30-days refund policy at any cost'],
    context=['All the Customer are eligible for 30-days refund policy at any cost'],
)

evaluate([test_case],[ha,cm])
